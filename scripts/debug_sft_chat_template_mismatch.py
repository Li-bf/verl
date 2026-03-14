#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Debug chat-template length mismatches for multiturn SFT data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

# Ensure imports resolve to the current checkout instead of another installed `verl`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.utils import hf_processor, hf_tokenizer
from verl.utils.chat_template import apply_chat_template, extract_system_prompt_and_generation
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.py_functional import convert_nested_value_to_list_recursive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF model/tokenizer path")
    parser.add_argument("--parquet", required=True, nargs="+", help="Parquet file(s)")
    parser.add_argument("--max-length", type=int, default=16384, help="Max sequence length")
    parser.add_argument("--start", type=int, default=0, help="Start row index")
    parser.add_argument("--limit", type=int, default=100, help="Number of rows to inspect")
    parser.add_argument("--show-all", action="store_true", help="Print all inspected rows, not only mismatches")
    parser.add_argument(
        "--stop-after",
        type=int,
        default=20,
        help="Stop after printing this many mismatched rows. <=0 means no limit",
    )
    parser.add_argument(
        "--enable-thinking",
        default=None,
        choices=["true", "false", "none"],
        help="Pass enable_thinking into chat template. Default: do not pass",
    )
    parser.add_argument(
        "--apply-chat-template-kwargs",
        default="{}",
        help="JSON dict forwarded to apply_chat_template, e.g. '{\"enable_thinking\": false}'",
    )
    parser.add_argument("--messages-key", default="messages", help="Messages column name")
    parser.add_argument("--tools-key", default="tools", help="Tools column name")
    parser.add_argument(
        "--ignore-input-ids-mismatch",
        action="store_true",
        help="Match training config behavior and continue even if whole/per-turn token ids differ",
    )
    return parser.parse_args()


def parse_optional_bool(value: str | None) -> bool | None:
    if value is None or value == "none":
        return None
    return value == "true"


def load_dataframe(parquet_files: list[str]) -> pd.DataFrame:
    dataframes = [pd.read_parquet(parquet_file, dtype_backend="pyarrow") for parquet_file in parquet_files]
    return pd.concat(dataframes).reset_index(drop=True)


def decode_preview(tokenizer, token_ids: torch.Tensor, max_chars: int = 2000) -> str:
    text = tokenizer.decode(token_ids.tolist(), skip_special_tokens=False)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>..."
    return text


def stringify_messages(messages: list[dict[str, Any]], max_chars: int = 4000) -> str:
    text = json.dumps(messages, ensure_ascii=False, indent=2)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>..."
    return text


def stringify_object(value: Any, max_chars: int = 4000) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>..."
    return text


def should_use_processor(messages: list[dict[str, Any]], processor) -> bool:
    if processor is None:
        return False

    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            return True
    return False


def maybe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def normalize_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if tools is None:
        return None

    normalized_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            normalized_tools.append(tool)
            continue

        normalized_tool = dict(tool)
        function = normalized_tool.get("function")
        if isinstance(function, dict):
            normalized_function = dict(function)
            normalized_function["parameters"] = maybe_json_loads(normalized_function.get("parameters"))
            normalized_tool["function"] = normalized_function

        normalized_tools.append(normalized_tool)

    return normalized_tools


def build_per_turn_tokens(
    tokenizer,
    processor,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    apply_kwargs: dict[str, Any],
    ignore_input_ids_mismatch: bool,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    dataset = MultiTurnSFTDataset.__new__(MultiTurnSFTDataset)
    dataset.tokenizer = tokenizer
    dataset.processor = processor
    dataset.apply_chat_template_kwargs = apply_kwargs
    dataset.ignore_input_ids_mismatch = ignore_input_ids_mismatch
    dataset.system_prompt, dataset.generation_prompt = extract_system_prompt_and_generation(tokenizer)

    input_ids = []
    turn_infos = []
    for i, message in enumerate(messages):
        turn_tools = tools if i == 0 else None
        turn_input_ids, turn_loss_mask, turn_attention_mask, _ = dataset._process_single_message(
            index=i,
            message=message,
            full_message=messages,
            tools=turn_tools,
            enable_thinking=apply_kwargs.get("enable_thinking"),
        )
        input_ids.append(turn_input_ids)
        turn_infos.append(
            {
                "turn_index": i,
                "role": message.get("role"),
                "token_len": int(turn_input_ids.shape[0]),
                "loss_tokens": int(turn_loss_mask.sum().item()),
                "attn_tokens": int(turn_attention_mask.sum().item()),
                "decoded": decode_preview(tokenizer, turn_input_ids),
            }
        )

    per_turn_ids = torch.cat(input_ids, dim=0)
    return per_turn_ids, turn_infos, {
        "system_prompt_len": len(dataset.system_prompt),
        "generation_prompt_len": len(dataset.generation_prompt),
    }


def main() -> None:
    args = parse_args()
    apply_kwargs = json.loads(args.apply_chat_template_kwargs)
    enable_thinking = parse_optional_bool(args.enable_thinking)
    if enable_thinking is not None:
        apply_kwargs["enable_thinking"] = enable_thinking

    tokenizer = hf_tokenizer(args.model_path, trust_remote_code=True)
    loaded_processor = hf_processor(args.model_path, trust_remote_code=True)
    df = load_dataframe(args.parquet)
    end = min(len(df), args.start + args.limit)

    print(f"model_path={args.model_path}")
    print(f"rows={len(df)} inspect_range=[{args.start}, {end}) max_length={args.max_length}")
    print(f"apply_chat_template_kwargs={json.dumps(apply_kwargs, ensure_ascii=False, sort_keys=True)}")

    mismatch_count = 0
    overlong_count = 0
    error_count = 0

    for row_idx in tqdm(range(args.start, end), desc="Inspect rows", total=end - args.start):
        row = df.iloc[row_idx].to_dict()
        messages = convert_nested_value_to_list_recursive(row[args.messages_key])
        tools = None
        if args.tools_key in row and row[args.tools_key] is not None:
            tools = convert_nested_value_to_list_recursive(row[args.tools_key])
            tools = normalize_tools(tools)
        processor = loaded_processor if should_use_processor(messages, loaded_processor) else None

        try:
            whole_text = apply_chat_template(
                processor if processor is not None else tokenizer,
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                **apply_kwargs,
            )
            whole_enc = tokenizer(
                whole_text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
            )
            whole_ids = torch.tensor(whole_enc["input_ids"], dtype=torch.long)
        except Exception as e:
            error_count += 1
            print("=" * 120)
            print(f"row={row_idx} stage=whole_text_with_generation error={type(e).__name__}: {e}")
            print("messages_json:")
            print(stringify_messages(messages))
            if tools is not None:
                print("tools_json:")
                print(stringify_object(tools))
            if args.stop_after > 0 and error_count >= args.stop_after:
                break
            continue

        try:
            whole_inputs = apply_chat_template(
                processor if processor is not None else tokenizer,
                messages,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **apply_kwargs,
            )
            whole_ids_no_gen = whole_inputs["input_ids"].squeeze(0).cpu()
        except Exception as e:
            error_count += 1
            print("=" * 120)
            print(f"row={row_idx} stage=whole_inputs_no_generation error={type(e).__name__}: {e}")
            print("messages_json:")
            print(stringify_messages(messages))
            if tools is not None:
                print("tools_json:")
                print(stringify_object(tools))
            print("whole_rendered_text_with_generation_prompt:")
            print(whole_text[:4000] + ("\n...<truncated>..." if len(whole_text) > 4000 else ""))
            if args.stop_after > 0 and error_count >= args.stop_after:
                break
            continue

        try:
            per_turn_ids, turn_infos, prompt_info = build_per_turn_tokens(
                tokenizer=tokenizer,
                processor=processor,
                messages=messages,
                tools=tools,
                apply_kwargs=apply_kwargs,
                ignore_input_ids_mismatch=args.ignore_input_ids_mismatch,
            )
        except Exception as e:
            error_count += 1
            print("=" * 120)
            print(f"row={row_idx} stage=per_turn_concat error={type(e).__name__}: {e}")
            print("messages_json:")
            print(stringify_messages(messages))
            if tools is not None:
                print("tools_json:")
                print(stringify_object(tools))
            print("whole_rendered_text_with_generation_prompt:")
            print(whole_text[:4000] + ("\n...<truncated>..." if len(whole_text) > 4000 else ""))
            print("whole_decoded_no_generation_prompt:")
            print(decode_preview(tokenizer, whole_ids_no_gen, max_chars=4000))
            if args.stop_after > 0 and error_count >= args.stop_after:
                break
            continue

        token_mismatch = not torch.equal(per_turn_ids, whole_ids_no_gen)
        filter_keep = len(whole_ids) <= args.max_length
        train_keep = len(per_turn_ids) <= args.max_length
        filtered_but_train_overlong = filter_keep and not train_keep
        if filtered_but_train_overlong:
            overlong_count += 1

        should_print = args.show_all or token_mismatch or filtered_but_train_overlong
        if not should_print:
            continue

        mismatch_count += 1
        print("=" * 120)
        print(
            f"row={row_idx} whole_len_with_gen={len(whole_ids)} "
            f"whole_len_no_gen={len(whole_ids_no_gen)} per_turn_len={len(per_turn_ids)} "
            f"filter_keep={filter_keep} train_keep={train_keep} token_mismatch={token_mismatch}"
        )
        print(
            f"system_prompt_len={prompt_info['system_prompt_len']} "
            f"generation_prompt_len={prompt_info['generation_prompt_len']}"
        )
        print("messages_json:")
        print(stringify_messages(messages))

        if tools is not None:
            print("tools_json:")
            print(json.dumps(tools, ensure_ascii=False, indent=2))

        print("whole_rendered_text_with_generation_prompt:")
        print(whole_text[:4000] + ("\n...<truncated>..." if len(whole_text) > 4000 else ""))

        print("whole_decoded_no_generation_prompt:")
        print(decode_preview(tokenizer, whole_ids_no_gen, max_chars=4000))

        print("per_turn_decoded_concat:")
        print(decode_preview(tokenizer, per_turn_ids, max_chars=4000))

        if token_mismatch:
            min_len = min(len(whole_ids_no_gen), len(per_turn_ids))
            diff_pos = next(
                (i for i in range(min_len) if int(whole_ids_no_gen[i]) != int(per_turn_ids[i])),
                min_len,
            )
            print(f"first_diff_token_pos={diff_pos}")
            if diff_pos < min_len:
                print(
                    "token_window:"
                    f" whole={whole_ids_no_gen[max(0, diff_pos - 10): diff_pos + 10].tolist()}"
                    f" per_turn={per_turn_ids[max(0, diff_pos - 10): diff_pos + 10].tolist()}"
                )

        print("turn_breakdown:")
        for turn in turn_infos:
            print(
                f"  turn={turn['turn_index']} role={turn['role']} "
                f"token_len={turn['token_len']} loss_tokens={turn['loss_tokens']} attn_tokens={turn['attn_tokens']}"
            )
            print(turn["decoded"])
            print("-" * 80)

        if args.stop_after > 0 and mismatch_count >= args.stop_after:
            break

    print("=" * 120)
    print(
        f"done inspected={end - args.start} printed={mismatch_count} "
        f"filtered_but_train_overlong={overlong_count} errors={error_count}"
    )


if __name__ == "__main__":
    main()
