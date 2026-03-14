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
from verl.utils.chat_template import extract_system_prompt_and_generation
from verl.utils.dataset.multiturn_sft_dataset import (
    MultiTurnSFTDataset,
    _normalize_message_tool_calls,
    _normalize_tool_schemas,
)
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


def should_retry_with_dummy_user(messages: list[dict[str, Any]], error: Exception) -> bool:
    if any(message.get("role") == "user" for message in messages):
        return False

    error_text = str(error).lower()
    return "user message" in error_text or "at least one user" in error_text


def debug_apply_chat_template(
    processor_or_tokenizer,
    messages: list[dict[str, Any]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    tools=None,
    return_dict: bool = False,
    **kwargs,
):
    try:
        output = processor_or_tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
        return output, None, None
    except Exception as raw_error:
        if not should_retry_with_dummy_user(messages, raw_error):
            raise RuntimeError(str(raw_error)) from raw_error

        dummy_user_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        try:
            dummy_user_prefix = processor_or_tokenizer.apply_chat_template(
                dummy_user_message,
                tokenize=tokenize,
                add_generation_prompt=False,
                tools=tools,
                return_dict=return_dict,
                **kwargs,
            )
            output = processor_or_tokenizer.apply_chat_template(
                dummy_user_message + messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                return_dict=return_dict,
                **kwargs,
            )
        except Exception as fallback_error:
            raise RuntimeError(f"raw_error={raw_error}; fallback_error={fallback_error}") from fallback_error

        if not tokenize:
            output = output[len(dummy_user_prefix) :]
        elif not return_dict:
            if isinstance(output[0], list):
                assert len(output) == 1, "output must be a list[int] or list[list[int]]"
                dummy_user_prefix = dummy_user_prefix[0]
                output = output[0]
            output = output[len(dummy_user_prefix) :]
        else:
            dummy_user_prefix = dict(dummy_user_prefix)
            output = dict(output)
            prefix_len = dummy_user_prefix["input_ids"].shape[1]
            output["input_ids"] = output["input_ids"][:, prefix_len:]
            output["attention_mask"] = output["attention_mask"][:, prefix_len:]
            if "mm_token_type_ids" in output:
                output["mm_token_type_ids"] = output["mm_token_type_ids"][:, prefix_len:]

        return output, raw_error, None


def build_per_chunk_tokens(
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
    chunk_infos = []
    message_chunks = dataset._group_messages_for_template(messages)
    for i, message_chunk in enumerate(message_chunks):
        chunk_tools = tools if i == 0 else None
        chunk_input_ids, chunk_loss_mask, chunk_attention_mask, _ = dataset._process_message_chunk(
            chunk_index=i,
            message_chunk=message_chunk,
            full_message=messages,
            tools=chunk_tools,
            enable_thinking=apply_kwargs.get("enable_thinking"),
        )
        input_ids.append(chunk_input_ids)
        chunk_infos.append(
            {
                "chunk_index": i,
                "roles": [message.get("role") for message in message_chunk],
                "token_len": int(chunk_input_ids.shape[0]),
                "loss_tokens": int(chunk_loss_mask.sum().item()),
                "attn_tokens": int(chunk_attention_mask.sum().item()),
                "decoded": decode_preview(tokenizer, chunk_input_ids),
            }
        )

    per_chunk_ids = torch.cat(input_ids, dim=0)
    return per_chunk_ids, chunk_infos, {
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
        messages = _normalize_message_tool_calls(messages)
        tools = None
        if args.tools_key in row and row[args.tools_key] is not None:
            tools = convert_nested_value_to_list_recursive(row[args.tools_key])
            tools = _normalize_tool_schemas(tools)
        processor = loaded_processor if should_use_processor(messages, loaded_processor) else None

        try:
            whole_text, whole_text_raw_error, _ = debug_apply_chat_template(
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
            whole_inputs, whole_inputs_raw_error, _ = debug_apply_chat_template(
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
            if whole_text_raw_error is not None:
                print(f"whole_text_raw_error={type(whole_text_raw_error).__name__}: {whole_text_raw_error}")
            print("whole_rendered_text_with_generation_prompt:")
            print(whole_text[:4000] + ("\n...<truncated>..." if len(whole_text) > 4000 else ""))
            if args.stop_after > 0 and error_count >= args.stop_after:
                break
            continue

        try:
            per_chunk_ids, chunk_infos, prompt_info = build_per_chunk_tokens(
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
            print(f"row={row_idx} stage=per_chunk_concat error={type(e).__name__}: {e}")
            print("messages_json:")
            print(stringify_messages(messages))
            if tools is not None:
                print("tools_json:")
                print(stringify_object(tools))
            if whole_text_raw_error is not None:
                print(f"whole_text_raw_error={type(whole_text_raw_error).__name__}: {whole_text_raw_error}")
            if whole_inputs_raw_error is not None:
                print(f"whole_inputs_raw_error={type(whole_inputs_raw_error).__name__}: {whole_inputs_raw_error}")
            print("whole_rendered_text_with_generation_prompt:")
            print(whole_text[:4000] + ("\n...<truncated>..." if len(whole_text) > 4000 else ""))
            print("whole_decoded_no_generation_prompt:")
            print(decode_preview(tokenizer, whole_ids_no_gen, max_chars=4000))
            if args.stop_after > 0 and error_count >= args.stop_after:
                break
            continue

        token_mismatch = not torch.equal(per_chunk_ids, whole_ids_no_gen)
        filter_keep = len(whole_ids) <= args.max_length
        train_keep = len(per_chunk_ids) <= args.max_length
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
            f"whole_len_no_gen={len(whole_ids_no_gen)} per_chunk_len={len(per_chunk_ids)} "
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

        print("per_chunk_decoded_concat:")
        print(decode_preview(tokenizer, per_chunk_ids, max_chars=4000))

        if token_mismatch:
            min_len = min(len(whole_ids_no_gen), len(per_chunk_ids))
            diff_pos = next(
                (i for i in range(min_len) if int(whole_ids_no_gen[i]) != int(per_chunk_ids[i])),
                min_len,
            )
            print(f"first_diff_token_pos={diff_pos}")
            if diff_pos < min_len:
                print(
                    "token_window:"
                    f" whole={whole_ids_no_gen[max(0, diff_pos - 10): diff_pos + 10].tolist()}"
                    f" per_chunk={per_chunk_ids[max(0, diff_pos - 10): diff_pos + 10].tolist()}"
                )

        print("chunk_breakdown:")
        for chunk in chunk_infos:
            print(
                f"  chunk={chunk['chunk_index']} roles={chunk['roles']} "
                f"token_len={chunk['token_len']} loss_tokens={chunk['loss_tokens']} attn_tokens={chunk['attn_tokens']}"
            )
            print(chunk["decoded"])
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
