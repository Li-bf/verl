# Copyright 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum

import torch
from tensordict.tensorclass import NonTensorData


class DatasetPadMode(str, Enum):
    """Padding mode for dataset"""

    RIGHT = "right"
    LEFT_RIGHT = "left_right"
    NO_PADDING = "no_padding"


class SFTTensorCollator:
    """
    A custom collate_fn that handles batching of sequences.
    1. for variable-length sequences, convert them into NestedTensors.
    2. for fixed-length sequences, use default_collate.
    """

    def __init__(self, pad_mode: DatasetPadMode = DatasetPadMode.LEFT_RIGHT):
        self.pad_mode = pad_mode

    @staticmethod
    def _get_first_value(batch: list[dict[str, any]], key: str):
        for item in batch:
            if key in item:
                return item[key]
        raise KeyError(f"Key {key} is missing from every sample in the batch")

    @staticmethod
    def _normalize_position_ids(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        dims = {tensor.dim() for tensor in tensors}
        if len(dims) <= 1:
            return tensors

        if not dims.issubset({1, 2}):
            raise RuntimeError(
                f"Unsupported mixed position_ids ranks in batch: {[tensor.dim() for tensor in tensors]}"
            )

        target_rows = {tensor.shape[0] for tensor in tensors if tensor.dim() == 2}
        if len(target_rows) != 1:
            raise RuntimeError(
                "Mixed multi-dimensional position_ids must share the same leading dimension, "
                f"got {sorted(target_rows)}"
            )

        target_rows = target_rows.pop()
        return [tensor.unsqueeze(0).expand(target_rows, -1) if tensor.dim() == 1 else tensor for tensor in tensors]

    def __call__(self, batch: list[dict[str, any]]) -> dict[str, any]:
        if self.pad_mode == DatasetPadMode.NO_PADDING:
            return self.collate_variable_batch(batch)
        elif self.pad_mode in [DatasetPadMode.RIGHT, DatasetPadMode.LEFT_RIGHT]:
            from torch.utils.data import default_collate

            return default_collate(batch)
        else:
            raise NotImplementedError(f"pad_mode {self.pad_mode} not implemented")

    def collate_variable_batch(self, batch: list[dict[str, any]]) -> dict[str, any]:
        """
        Collates a list of samples into a single batch.

        Args:
            batch: A list of dictionary samples from the dataset.

        Returns:
            A dictionary representing the batched data, with variable-length
            sequences converted to NestedTensors.
        """

        final_batch = {}

        tensor_keys = set().union(*(d.keys() for d in batch))

        # Handle tensor values by creating a NestedTensor.
        for key in tensor_keys:
            sample_value = self._get_first_value(batch, key)
            if isinstance(sample_value, torch.Tensor):
                missing_indices = [index for index, item in enumerate(batch) if key not in item]
                if missing_indices:
                    raise KeyError(f"Tensor key {key} is missing from samples {missing_indices}")

                tensors = [item[key] for item in batch]
                if key == "position_ids":
                    tensors = self._normalize_position_ids(tensors)
                if tensors[0].dim() >= 2:
                    # For multi-dim tensors (e.g., 3D position_ids with shape (num_heads, seq_len)),
                    # use nested_tensor_from_jagged with explicit jagged_dim to avoid ambiguity
                    # when all samples share the same seq_len.
                    values = torch.cat(tensors, dim=-1)
                    lengths = torch.tensor([t.shape[-1] for t in tensors])
                    offsets = torch.zeros(len(tensors) + 1, dtype=torch.long)
                    torch.cumsum(lengths, dim=0, out=offsets[1:])
                    final_batch[key] = torch.nested.nested_tensor_from_jagged(values, offsets=offsets)
                    final_batch[key]._ragged_idx = 2
                else:
                    final_batch[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
            else:
                tensors = [NonTensorData(item.get(key)) for item in batch]
                final_batch[key] = torch.stack(tensors, dim=0)

        return final_batch
