# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os
from copy import deepcopy

from transformers import PreTrainedTokenizerBase, ProcessorMixin

from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _has_user_message(messages: list[dict]) -> bool:
    return any(message.get("role") == "user" for message in messages if isinstance(message, dict))


def _is_missing_user_error(error: Exception) -> bool:
    error_text = str(error).lower()
    return "user message" in error_text or "at least one user" in error_text or "no user query found" in error_text


def _strip_prefix_or_suffix_from_output(
    output,
    extra_render,
    *,
    tokenize: bool,
    return_dict: bool,
    strip_from: str,
):
    assert strip_from in ["prefix", "suffix"]

    if not tokenize:
        if strip_from == "prefix":
            return output[len(extra_render) :]
        return output[: -len(extra_render)] if len(extra_render) > 0 else output

    if not return_dict:
        if isinstance(output[0], list):  # transformers>=5
            assert len(output) == 1, "output must be a list[int] or list[list[int]]"
            output = output[0]
        if isinstance(extra_render[0], list):
            assert len(extra_render) == 1, "extra_render must be a list[int] or list[list[int]]"
            extra_render = extra_render[0]
        if strip_from == "prefix":
            return output[len(extra_render) :]
        return output[: -len(extra_render)] if len(extra_render) > 0 else output

    extra_render = dict(extra_render)
    output = dict(output)
    extra_len = extra_render["input_ids"].shape[1]
    if strip_from == "prefix":
        output["input_ids"] = output["input_ids"][:, extra_len:]
        output["attention_mask"] = output["attention_mask"][:, extra_len:]
        if "mm_token_type_ids" in output:
            output["mm_token_type_ids"] = output["mm_token_type_ids"][:, extra_len:]
    else:
        output["input_ids"] = output["input_ids"][:, :-extra_len] if extra_len > 0 else output["input_ids"]
        output["attention_mask"] = (
            output["attention_mask"][:, :-extra_len] if extra_len > 0 else output["attention_mask"]
        )
        if "mm_token_type_ids" in output:
            output["mm_token_type_ids"] = (
                output["mm_token_type_ids"][:, :-extra_len] if extra_len > 0 else output["mm_token_type_ids"]
            )
    return output


def _single_message_fallback(
    processor: PreTrainedTokenizerBase | ProcessorMixin,
    message: dict,
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    tools=None,
    return_dict: bool = False,
    **kwargs,
):
    role = message.get("role")
    dummy_user_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]

    if role == "system":
        suffix = processor.apply_chat_template(
            dummy_user_message,
            tokenize=tokenize,
            add_generation_prompt=False,
            tools=None,
            return_dict=return_dict,
            **kwargs,
        )
        output = processor.apply_chat_template(
            [deepcopy(message)] + dummy_user_message,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
        return _strip_prefix_or_suffix_from_output(
            output, suffix, tokenize=tokenize, return_dict=return_dict, strip_from="suffix"
        )

    prefix = processor.apply_chat_template(
        dummy_user_message,
        tokenize=tokenize,
        add_generation_prompt=False,
        tools=tools,
        return_dict=return_dict,
        **kwargs,
    )
    output = processor.apply_chat_template(
        dummy_user_message + [deepcopy(message)],
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        return_dict=return_dict,
        **kwargs,
    )
    return _strip_prefix_or_suffix_from_output(
        output, prefix, tokenize=tokenize, return_dict=return_dict, strip_from="prefix"
    )


def _multi_message_missing_user_fallback(
    processor: PreTrainedTokenizerBase | ProcessorMixin,
    messages: list[dict],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    tools=None,
    return_dict: bool = False,
    **kwargs,
):
    dummy_user_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    prefix = processor.apply_chat_template(
        dummy_user_message,
        tokenize=tokenize,
        add_generation_prompt=False,
        tools=tools,
        return_dict=return_dict,
        **kwargs,
    )
    output = processor.apply_chat_template(
        dummy_user_message + [deepcopy(message) for message in messages],
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        return_dict=return_dict,
        **kwargs,
    )
    return _strip_prefix_or_suffix_from_output(
        output, prefix, tokenize=tokenize, return_dict=return_dict, strip_from="prefix"
    )


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True)
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True)
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def extract_system_prompt_and_generation(tokenizer):
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True)
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True)
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    )
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt


def apply_chat_template(
    processor: PreTrainedTokenizerBase | ProcessorMixin,
    messages: list[dict],
    *,
    tokenize: bool = True,
    add_generation_prompt: bool = True,
    tools=None,
    return_dict: bool = False,
    **kwargs,
) -> list[int] | str:
    """apply_chat_template to messages with special attention to template requiring
    at least one user message, e.g. Qwen3.5.

    Args:
        processor: tokenizer or processor.
        messages: list[dict], messages.
        tokenize: bool, whether to tokenize the output.
        add_generation_prompt: bool, whether to add generation prompt.
        tools: list[dict], tools schema.
        return_dict: bool, whether to return a dict.
        **kwargs: additional arguments for apply_chat_template.

    Returns:
        list[int] | str: tokenized ids or text string.
    """
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
    except Exception as raw_error:
        if len(messages) == 1 and not _has_user_message(messages):
            return _single_message_fallback(
                processor,
                messages[0],
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                return_dict=return_dict,
                **kwargs,
            )

        if _has_user_message(messages) or not _is_missing_user_error(raw_error):
            raise

        return _multi_message_missing_user_fallback(
            processor,
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
