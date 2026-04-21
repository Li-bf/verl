# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
from functools import wraps
from collections import OrderedDict
from typing import Optional, Union, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils import hf_tokenizer
from verl.utils.chat_template import extract_system_prompt_and_generation
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.vision_utils import process_image, process_video
from verl.utils.fs import copy_local_path_from_hdfs

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _uses_qwen_vl_mrope(processor: Optional[ProcessorMixin]) -> bool:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return False

    image_processor_name = image_processor.__class__.__name__
    return any(
        name in image_processor_name
        for name in ("Qwen2VLImageProcessor", "Qwen2_5_VLImageProcessor", "Qwen3VLImageProcessor")
    )


def _build_text_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    processor: Optional[ProcessorMixin],
) -> torch.Tensor:
    text_position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)
    if not _uses_qwen_vl_mrope(processor):
        return text_position_ids

    vision_position_ids = get_rope_index(
        processor,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    return torch.cat((text_position_ids.unsqueeze(0), vision_position_ids), dim=0)


def once(func):
    """Decorator to ensure a function runs only once. Subsequent calls do nothing."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, "called"):
            wrapper.called = True
            return func(*args, **kwargs)

    return wrapper

class PretrainDataset(Dataset):
    """
    Dataset for pretraining large language models on text data.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
        max_samples (int, optional): Limit the number of samples. Defaults to -1 (use all).
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "no_padding"], (
            f"Expect pad_mode to be 'right' or 'no_padding'. Got {self.pad_mode}"
        )
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        # for right padding
        self.max_length = config.get("max_length", 1024)
        self.text_key = config.get("text_key", "text")
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.max_samples = max_samples
        self.ignore_input_ids_mismatch = config.get("ignore_input_ids_mismatch", False)
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # default loader loads some list as np.ndarray, which fails the tokenizer
            dataframe = pd.read_parquet(parquet_file, dtype_backend="pyarrow")
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            print(f"selected {self.max_samples} random samples out of {total}")

        self.texts = self.dataframe[self.text_key].tolist()
        logger.debug(f"self.texts[0]: {self.texts[0]}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        logger.debug(f"row_dict: {row_dict}")
        text = row_dict[self.text_key]
        logger.debug(f"text: {text}")

        # 1. tokenize each message
        input_ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.ones_like(input_ids)
        multi_modal_inputs = {}
        assert input_ids.shape == loss_mask.shape == attention_mask.shape, (
            f"Shape mismatch: {input_ids.shape}, {loss_mask.shape}, {attention_mask.shape}"
        )

        # Since the tokenizer may return user-customized results, we need to filter out inconsistent tensor shapes
        keys_to_remove = []
        for k, v in multi_modal_inputs.items():
            if len(v) > 0 and v[0] is not None and isinstance(v[0], torch.Tensor):
                # Check if all tensors in the list have the same shape
                first_shape = v[0].shape[1:]
                if not all(tensor.shape[1:] == first_shape for tensor in v):
                    keys_to_remove.append(k)

        for k in keys_to_remove:
            del multi_modal_inputs[k]

        for k, v in multi_modal_inputs.items():
            multi_modal_inputs[k] = torch.concat(v, dim=0)

        position_ids = _build_text_position_ids(input_ids=input_ids, attention_mask=attention_mask, processor=self.processor)

        # 2. handle padding
        sequence_length = input_ids.shape[0]
        # Handle sequence length
        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                # Pad sequences
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
                padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
                padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
                position_ids = F.pad(position_ids, (0, self.max_length - sequence_length), value=0)
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length :]
                    attention_mask = attention_mask[-self.max_length :]
                    loss_mask = loss_mask[-self.max_length :]
                    position_ids = position_ids[..., -self.max_length :]
                elif self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    attention_mask = attention_mask[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                    position_ids = position_ids[..., : self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

            res = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res
        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            # truncate input_ids if it is longer than max_length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                position_ids = position_ids[..., : self.max_length]

            # return nested tensor with out padding
            res = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")

    def maybe_filter_out_long_prompts(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.filter_overlong_prompts:
            return df

        texts = df[self.text_key].astype("string").fillna("").tolist()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        
        enc = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length + 1,
            padding=False,
            return_attention_mask=False,
        )

        lengths = [len(x) for x in enc["input_ids"]]
        mask = [L <= self.max_length for L in lengths]

        filtered = df[mask].reset_index(drop=True)
        print(f"filter dataset len: {len(filtered)} / {len(df)}")
        return filtered


class RowGroupLRUCache:
    def __init__(self, capacity: int = 4):
        self.capacity = max(int(capacity), 0)
        self._od = OrderedDict()

    def get(self, key):
        if self.capacity <= 0:
            return None
        if key not in self._od:
            return None
        self._od.move_to_end(key)
        return self._od[key]

    def put(self, key, value):
        if self.capacity <= 0:
            return
        if key in self._od:
            self._od.move_to_end(key)
        self._od[key] = value
        if len(self._od) > self.capacity:
            self._od.popitem(last=False)

class PretrainDatasetRowGroupLazy(Dataset):
    "通过(file_id, row_group_id, row_in_group)懒加载数据, 避免全量读入oom"
    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer,
        config: Optional[dict] = None,
        processor=None,
        max_samples: int = -1,
    ):
        config = config or {}
        self.config = config
        self.pad_mode = config.get("pad_mode", "right")
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        # for right padding
        self.max_length = config.get("max_length", 1024)
        self.text_key = config.get("text_key", "text")
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.max_samples = max_samples
        self.ignore_input_ids_mismatch = config.get("ignore_input_ids_mismatch", False)

        assert self.pad_mode in ["right", "no_padding"], (
            f"Expect pad_mode to be 'right' or 'no_padding'. Got {self.pad_mode}"
        )
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor

        self._pfs: List[pq.ParquetFile] = [pq.ParquetFile(p) for p in self.parquet_files]

        self.per_file_rg_ends: List[List[int]] = []
        self.file_ends: List[int] = []
        for pf in self._pfs:
            ends = []
            acc = 0
            for rg in range(pf.num_row_groups):
                acc += pf.metadata.row_group(rg).num_rows
                ends.append(acc)
            self.per_file_rg_ends.append(ends)
            self.file_ends.append(acc)

        for i in range(1, len(self.file_ends)):
            self.file_ends[i] += self.file_ends[i - 1]
        self.total_rows = self.file_ends[-1]

        cache_cap = int(self.config.get("cache_rowgroups", 4))
        self.rg_cache = RowGroupLRUCache(capacity=cache_cap)

        self._setup_index_mapping()

    def _iter_row_groups_texts(self):
        """yield: (file_id, rg_id, texts_list, global_start_row)"""
        global_base = 0
        for file_id, pf in enumerate(self._pfs):
            rg_ends = self.per_file_rg_ends[file_id]
            prev_end = 0
            for rg_id, rg_end in enumerate(rg_ends):
                # 该 row group 在本文件内的起始行
                local_start = prev_end
                prev_end = rg_end

                table = pf.read_row_group(rg_id, columns=[self.text_key])
                texts = table[self.text_key].to_pylist()
                # row group 在全局的起始行
                global_start = global_base + local_start
                yield file_id, rg_id, texts, global_start

            global_base += self.file_ends[file_id] - (self.file_ends[file_id - 1] if file_id > 0 else 0)

    def _build_valid_indices(self) -> np.ndarray:
        batch_size = int(self.config.get("filter_batch_size", 2048))
        max_len_plus = int(self.max_length) + 1

        valid = []
        total = int(self.total_rows)

        desc = f"Filter overlong (max_len={self.max_length})"
        with tqdm(total=total, desc=desc, unit="rows", dynamic_ncols=True) as pbar:
            for _, _, texts, global_start in self._iter_row_groups_texts():
                texts = [("" if t is None else str(t)) for t in texts]

                for offset in range(0, len(texts), batch_size):
                    chunk = texts[offset : offset + batch_size]

                    enc = self.tokenizer(
                        chunk,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=max_len_plus,
                        padding=False,
                        return_attention_mask=False,
                    )

                    lengths = [len(x) for x in enc["input_ids"]]
                    for i, L in enumerate(lengths):
                        if L <= self.max_length:
                            valid.append(global_start + offset + i)

                    pbar.update(len(chunk))
                    
        valid = np.asarray(valid, dtype=np.int64)
        print(f"filter dataset len: {len(valid)} / {self.total_rows}")
        return valid

    def _setup_index_mapping(self):
        if self.filter_overlong_prompts:
            self.valid_global_indices = self._build_valid_indices()
        else:
            self.valid_global_indices = np.arange(self.total_rows, dtype=np.int64)

        N = int(self.valid_global_indices.shape[0])
        rng = np.random.default_rng(self.seed) if self.seed is not None else np.random.default_rng()

        if self.shuffle:
            self.index_map = rng.permutation(N).astype(np.int64)
        else:
            self.index_map = np.arange(N, dtype=np.int64)

        max_samples = int(self.max_samples) if self.max_samples is not None else -1
        self.effective_len = min(max_samples, N) if max_samples > 0 else N


    def _global_index(self, idx: int) -> int:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range (len={len(self)})")
        return int(self.valid_global_indices[self.index_map[idx]])

    def __len__(self):
        return int(getattr(self, "effective_len", self.total_rows))

    @staticmethod
    def _binary_search(a: List[int], x: int) -> int:
        l, r = 0, len(a)
        while l < r:
            mid = (l + r) // 2
            if x < a[mid]:
                r = mid
            else:
                l = mid + 1
        return l

    def _locate_file(self, global_idx: int) -> Tuple[int, int]:
        if global_idx < 0:
            global_idx += self.total_rows
        if global_idx < 0 or global_idx >= self.total_rows:
            raise IndexError(f"Index {global_idx} out of range (len={self.total_rows})")

        file_id = self._binary_search(self.file_ends, global_idx)
        prev_end = 0 if file_id == 0 else self.file_ends[file_id - 1]
        local_idx = global_idx - prev_end
        return file_id, local_idx

    def _locate_row_group(self, file_id: int, local_idx: int) -> Tuple[int, int]:
        rg_ends = self.per_file_rg_ends[file_id]
        rg_id = self._binary_search(rg_ends, local_idx)
        prev_end = 0 if rg_id == 0 else rg_ends[rg_id - 1]
        row_in_group = local_idx - prev_end
        return rg_id, row_in_group

    def _load_texts_for_row_group(self, file_id: int, rg_id: int) -> List[str]:
        key = (file_id, rg_id, self.text_key)
        cached = self.rg_cache.get(key)
        if cached is not None:
            return cached

        pf = self._pfs[file_id]
        table = pf.read_row_group(rg_id, columns=[self.text_key])
        texts = table[self.text_key].to_pylist()

        self.rg_cache.put(key, texts)
        return texts

    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        input_ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.ones_like(input_ids)
        position_ids = _build_text_position_ids(input_ids=input_ids, attention_mask=attention_mask, processor=self.processor)

        # truncation
        seq_len = input_ids.shape[0]
        if seq_len > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
                position_ids = position_ids[-self.max_length:]
            elif self.truncation == "right":
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
                position_ids = position_ids[:self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"sequence_length={seq_len} > max_length={self.max_length}")

        # padding
        if self.pad_mode == DatasetPadMode.RIGHT:
            seq_len = input_ids.shape[0]
            if seq_len < self.max_length:
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                pad_len = self.max_length - seq_len
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)])
                attention_mask = torch.cat([attention_mask, torch.zeros((pad_len,), dtype=attention_mask.dtype)])
                loss_mask = torch.cat([loss_mask, torch.zeros((pad_len,), dtype=loss_mask.dtype)])
                position_ids = F.pad(position_ids, (0, pad_len), value=0)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        else:
            # no_padding
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_idx = self._global_index(idx)
        file_id, local_idx = self._locate_file(global_idx)
        rg_id, row_in_group = self._locate_row_group(file_id, local_idx)

        texts = self._load_texts_for_row_group(file_id, rg_id)
        text = texts[row_in_group]

        return self._encode_text(text)
