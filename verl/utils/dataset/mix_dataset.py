# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mixed SFT dataset that supports:
1. pretrain-style samples with a plain `text` field.
2. multi-turn SFT samples with `messages` and optional `tools`.

Input files support parquet only.
"""

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import ListConfig

from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.chat_template import extract_system_prompt_and_generation
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.multiturn_sft_dataset import convert_nested_value_to_list_recursive, print_assembled_message
from verl.utils.dataset.pretrain_dataset import PretrainDatasetRowGroupLazy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MixDataset(PretrainDatasetRowGroupLazy):
    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer,
        config,
        processor=None,
        max_samples: int = -1,
    ):
        config = dict(config or {})
        
        if config.get("filter_overlong_prompts", False):
            logger.warning("The 'filter_overlong_prompts' option is not supported in MixDataset and will be ignored.")
            config["filter_overlong_prompts"] = False

        self.messages_key = config.get("messages_key", "messages")
        self.tools_key = config.get("tools_key", "tools")
        self.enable_thinking_key = config.get("enable_thinking_key", "enable_thinking")
        self.enable_thinking_default = self._parse_enable_thinking_default(config.get("enable_thinking_default", None))
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        normalized_files = self._normalize_input_files(parquet_files)

        super().__init__(
            parquet_files=normalized_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(self.tokenizer)

    def _normalize_input_files(self, files: str | list[str]) -> list[str]:
        if not isinstance(files, list | ListConfig):
            files = [files]
        normalized: list[str] = []

        for path in files:
            lower = str(path).lower()
            if not lower.endswith(".parquet"):
                raise ValueError(f"Unsupported mix dataset file format: {path}. Only .parquet is supported.")
            normalized.append(path)

        return normalized

    @staticmethod
    def _has_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, list) and len(value) == 0:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        try:
            if pd.isna(value):
                return False
        except Exception:
            pass
        return True

    def _load_mix_columns_for_row_group(
        self, file_id: int, rg_id: int
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        key = (file_id, rg_id, "__mix_messages_text_tools_enable_thinking__")
        cached = self.rg_cache.get(key)
        if cached is not None:
            return cached

        pf = self._pfs[file_id]
        row_group_meta = pf.metadata.row_group(rg_id)
        num_rows = row_group_meta.num_rows
        schema_cols = set(pf.schema_arrow.names)
        wanted = [self.messages_key, self.text_key, self.tools_key, self.enable_thinking_key]
        present = [col for col in wanted if col in schema_cols]
        table = pf.read_row_group(rg_id, columns=present) if present else None

        # Parquet files mixed from different pipelines may miss one or two
        # schema columns entirely. Keep row-wise logic strict, but treat
        # missing columns as all-None at file level.
        loaded = {col: table[col].to_pylist() for col in present} if table is not None else {}

        messages_col = loaded[self.messages_key] if self.messages_key in loaded else [None] * num_rows
        text_col = loaded[self.text_key] if self.text_key in loaded else [None] * num_rows
        tools_col = loaded[self.tools_key] if self.tools_key in loaded else [None] * num_rows
        enable_thinking_col = (
            loaded[self.enable_thinking_key] if self.enable_thinking_key in loaded else [None] * num_rows
        )

        self.rg_cache.put(key, (messages_col, text_col, tools_col, enable_thinking_col))
        return messages_col, text_col, tools_col, enable_thinking_col

    @staticmethod
    def _normalize_enable_thinking(value: Any) -> Optional[bool]:
        if not MixDataset._has_value(value):
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            if int(value) in (0, 1):
                return bool(value)
            return None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
            return None
        return None

    @staticmethod
    def _parse_enable_thinking_default(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            if int(value) in (0, 1):
                return bool(value)
            raise ValueError(f"Invalid enable_thinking_default: {value}. Expect 0/1 for integer values.")
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
            if lowered in {"", "none", "null"}:
                return None
            raise ValueError(
                f"Invalid enable_thinking_default: {value}. "
                "Expect one of true/false/1/0/yes/no/null/none."
            )
        raise ValueError(
            f"Invalid enable_thinking_default type: {type(value)}. "
            "Expect bool/int/str/None."
        )

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ):
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking

        inputs = processor.apply_chat_template(
            [message],
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]

        if index != 0 and message["role"] != "system":
            input_ids = input_ids[len(self.system_prompt) :]
            attention_mask = attention_mask[len(self.system_prompt) :]

        if message["role"] == "assistant":
            loss_mask = torch.ones_like(attention_mask)
            loss_mask[: len(self.generation_prompt)] = 0
        else:
            loss_mask = torch.zeros_like(attention_mask)

        return input_ids, loss_mask, attention_mask, inputs

    def _encode_messages(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        enable_thinking: Optional[bool],
    ) -> dict[str, Any]:
        input_ids, loss_mask, attention_mask, multi_modal_inputs = [], [], [], {}
        for i, message in enumerate(messages):
            _input_ids, _loss_mask, _attention_mask, _inputs = self._process_single_message(
                index=i,
                message=message,
                tools=tools if i == 0 else None,
                enable_thinking=enable_thinking,
            )
            input_ids.append(_input_ids)
            loss_mask.append(_loss_mask)
            attention_mask.append(_attention_mask)
            for k, v in _inputs.items():
                multi_modal_inputs.setdefault(k, []).append(v)

        input_ids = torch.cat(input_ids, dim=0)
        loss_mask = torch.cat(loss_mask, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)

        print_assembled_message(self.tokenizer, messages, input_ids, loss_mask, attention_mask, tools)
        self.sanity_check(input_ids, messages, tools, enable_thinking)

        keys_to_remove = []
        for k, v in multi_modal_inputs.items():
            if len(v) > 0 and v[0] is not None and isinstance(v[0], torch.Tensor):
                first_shape = v[0].shape[1:]
                if not all(tensor.shape[1:] == first_shape for tensor in v):
                    keys_to_remove.append(k)

        for k in keys_to_remove:
            del multi_modal_inputs[k]
        for k, v in multi_modal_inputs.items():
            multi_modal_inputs[k] = torch.concat(v, dim=0)

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            image_grid_thw = multi_modal_inputs.get("image_grid_thw", None)
            video_grid_thw = multi_modal_inputs.get("video_grid_thw", None)
            second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts", None)
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

        sequence_length = input_ids.shape[0]
        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                pad_len = self.max_length - sequence_length
                input_ids = torch.cat((input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)))
                attention_mask = torch.cat((attention_mask, torch.zeros((pad_len,), dtype=attention_mask.dtype)))
                loss_mask = torch.cat((loss_mask, torch.zeros((pad_len,), dtype=loss_mask.dtype)))
                position_ids = F.pad(position_ids, (0, pad_len), value=0)
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

        if self.pad_mode == DatasetPadMode.NO_PADDING:
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                position_ids = position_ids[..., : self.max_length]
            res = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res

        raise ValueError(f"Unknown pad mode {self.pad_mode}")

    def sanity_check(
        self,
        input_ids: torch.Tensor,
        messages: list[dict],
        tools: Optional[list[dict]],
        enable_thinking: Optional[bool],
    ):
        """Check concatenated per-turn input_ids equals one-shot chat-template input_ids."""
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking
        inputs = processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        error_message = (
            "MixDataset applies chat template to each turn separately and concatenates `input_ids`, "
            "which may not equal applying chat template to all messages at once.\n"
            "Set `ignore_input_ids_mismatch=True` to ignore this mismatch and keep concatenated `input_ids`."
        )
        if not torch.equal(input_ids, inputs["input_ids"].squeeze(0)):
            if self.ignore_input_ids_mismatch:
                warn_fn = getattr(logger, "warning_once", logger.warning)
                warn_fn(error_message)
            else:
                raise AssertionError(error_message)

    def __getitem__(self, item):
        global_idx = self._global_index(item)
        file_id, local_idx = self._locate_file(global_idx)
        rg_id, row_in_group = self._locate_row_group(file_id, local_idx)
        messages_col, text_col, tools_col, enable_thinking_col = self._load_mix_columns_for_row_group(file_id, rg_id)
        raw_messages = messages_col[row_in_group]
        raw_text = text_col[row_in_group]

        has_messages = self._has_value(raw_messages)
        has_text = self._has_value(raw_text)
        if has_messages == has_text:
            raise ValueError(
                f"Invalid sample at global index {global_idx}: exactly one of "
                f"`{self.messages_key}` and `{self.text_key}` must be non-empty, "
                f"but got has_messages={has_messages}, has_text={has_text}."
            )

        if has_messages:
            messages = convert_nested_value_to_list_recursive(raw_messages)
            raw_tools = tools_col[row_in_group]
            tools = convert_nested_value_to_list_recursive(raw_tools) if self._has_value(raw_tools) else None
            raw_enable_thinking = enable_thinking_col[row_in_group]
            enable_thinking = self._normalize_enable_thinking(raw_enable_thinking)
            if enable_thinking is None:
                enable_thinking = self.enable_thinking_default
            return self._encode_messages(messages=messages, tools=tools, enable_thinking=enable_thinking)

        return self._encode_text(raw_text)
