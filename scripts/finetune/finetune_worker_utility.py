"""Utility helpers used by the fine-tuning worker."""

from __future__ import annotations

import json
import math
import sys
from typing import Any

from finetune_config_utility import (
    WORKER_REQUIRED_CONFIG_KEYS,
    load_worker_config,
    validate_early_stop_keys,
    validate_method_keys,
    validate_required_keys,
)
from finetune_constants import *
from finetune_kld import KLD_WORKER_FILE_KEYS


def as_bool(value: Any) -> bool:
    """Parse bools from YAML-native values or common strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def as_int(value: Any) -> int:
    """Parse an int from YAML-native values or strings."""
    return int(value)


def as_float(value: Any) -> float:
    """Parse a float from YAML-native values or strings."""
    return float(value)


def load_config() -> dict:
    """Load config from job-class JSON params, falling back to mounted YAML."""
    if len(sys.argv) > 1:
        config = json.loads(sys.argv[1])
        validate_required_keys(config, WORKER_REQUIRED_CONFIG_KEYS, "Worker params")
        validate_early_stop_keys(config, "Worker params")
        validate_method_keys(
            config,
            "Worker params",
            required_method_file_keys=KLD_WORKER_FILE_KEYS,
        )
        return config

    return load_worker_config()


def download_jsonl(ow, file_id: str) -> list[dict[str, Any]]:
    """Download and parse a JSONL file from OpenWeights."""
    content = ow.files.content(file_id)
    text = content.decode("utf-8") if isinstance(content, bytes) else content
    records = []
    # Split on "\n" only: str.splitlines() also breaks on U+2028, U+0085, form feeds etc., which appear
    # unescaped inside JSON strings (e.g. terminal output in agent trajectories).
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in OpenWeights file {file_id}:{line_number}: {exc}") from exc
    return records


def validate_message_records(records: list[dict[str, Any]], label: str, multi_turn: bool = False) -> None:
    """Validate SFT rows: one user and one assistant turn, or (multi_turn) alternating turns ending on assistant."""
    if not records:
        raise ValueError(f"{label} dataset is empty")

    if multi_turn:
        for idx, record in enumerate(records):
            validate_multi_turn_messages(record.get("messages"), f"{label} row {idx}")
        return

    for idx, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list) or not (2 <= len(messages) <= 3):
            raise ValueError(f"{label} row {idx} must contain exactly two or three messages")

        roles = [message.get("role") for message in messages]
        if roles != ["system", "user", "assistant"][-len(messages):]:
            raise ValueError(f"{label} row {idx} must have roles ['user', 'assistant'] or ['system', 'user', 'assistant']; got {roles}")

        for message_idx, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, str) or not content:
                raise ValueError(f"{label} row {idx} message {message_idx} must contain non-empty text content")


def validate_multi_turn_messages(messages: Any, label: str) -> None:
    """Optional system turn, then user/assistant alternating, ending with an assistant turn."""
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{label} must contain at least two messages")
    roles = [message.get("role") for message in messages]
    body = roles[1:] if roles[0] == "system" else roles
    expected = ["user", "assistant"] * (len(body) // 2 + 1)
    if not body or body != expected[:len(body)] or body[-1] != "assistant":
        raise ValueError(f"{label} must alternate user/assistant and end with assistant; got {roles}")
    for message_idx, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError(f"{label} message {message_idx} must contain non-empty text content")


def estimate_total_steps(config: dict, train_rows: int) -> int:
    """Estimate optimizer steps so warmup_steps can accept percentages."""
    batch_size = as_int(config[CONFIG_KEY_PER_DEVICE_TRAIN_BATCH_SIZE])
    grad_accum = as_int(config[CONFIG_KEY_GRADIENT_ACCUMULATION_STEPS])
    epochs = as_float(config[CONFIG_KEY_EPOCHS])
    batches_per_epoch = math.ceil(train_rows / batch_size)
    steps_per_epoch = max(1, math.ceil(batches_per_epoch / grad_accum))
    return max(1, math.ceil(steps_per_epoch * epochs))


def resolve_warmup_steps(value: Any, total_steps: int) -> int:
    """Resolve an integer warmup step count from an int or percent string."""
    if value is None:
        return 0
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            fraction = float(stripped[:-1]) / 100.0
            return max(0, math.ceil(total_steps * fraction))
        return int(stripped)
    return int(value)


def render_chat(tokenizer, messages: list[dict[str, Any]], add_generation_prompt: bool = False) -> str:
    """Render messages to text using the tokenizer chat template."""
    try:
        # enable_thinking=False: Qwen3.x templates then open the generation prompt with the same empty
        # "<think>\n\n</think>\n\n" block they render before an assistant answer, so the prompt stays a
        # token prefix of the full conversation. Templates without the variable ignore it.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        if add_generation_prompt:
            raise TypeError("Tokenizer chat template must support add_generation_prompt=True")
        return tokenizer.apply_chat_template(messages, tokenize=False)


_GENERATION_EXTRA: dict[tuple[int, bool], tuple[str, str]] = {}


def generation_header_extra(tokenizer, history: bool = False) -> tuple[str, str]:
    """(assistant header as rendered in a finished turn, extra text the generation prompt adds after it).

    The generation prompt can open the assistant turn with text that the rendering of a finished turn
    omits: Gemma 4 (31B) adds an empty thought channel ("<|channel>thought\n<channel|>") to every turn, and
    Qwen3.x adds the empty "<think>\n\n</think>\n\n" block (enable_thinking=False) to the last turn only,
    so earlier turns of a multi-turn conversation lack it. Training on the plain rendering would teach the
    model to answer from a context it never sees at inference (and breaks the prompt-prefix property).
    history=False probes the last assistant turn, history=True an earlier one. ("", "") when they agree.
    """
    key = (id(tokenizer), history)
    if key not in _GENERATION_EXTRA:
        user = [{"role": "user", "content": "PROBE_USER"}]
        base = render_chat(tokenizer, user)
        gen = render_chat(tokenizer, user, add_generation_prompt=True)
        turns = user + [{"role": "assistant", "content": "PROBE_ANSWER"}]
        if history:
            turns += [{"role": "user", "content": "PROBE_NEXT"}, {"role": "assistant", "content": "PROBE_LAST"}]
        full = render_chat(tokenizer, turns)
        result = ("", "")
        if gen.startswith(base) and full.startswith(base) and "PROBE_ANSWER" in full:
            gen_header = gen[len(base):]
            full_header = full[len(base):full.index("PROBE_ANSWER", len(base))]
            if gen_header != full_header and gen_header.startswith(full_header) and full_header:
                result = (full_header, gen_header[len(full_header):])
        _GENERATION_EXTRA[key] = result
    return _GENERATION_EXTRA[key]


def tokenize_text(tokenizer, text: str, max_seq_length: int | None = None) -> list[int]:
    """Tokenize text with no extra special tokens."""
    kwargs = {"add_special_tokens": False}
    if max_seq_length is not None:
        kwargs.update({"truncation": True, "max_length": max_seq_length})

    try:
        encoded = tokenizer(text, **kwargs)
    except TypeError:
        encoded = tokenizer(text)

    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def build_response_only_labels(
    input_ids: list[int],
    prompt_ids: list[int],
    row_idx: int,
) -> list[int]:
    """Mask the prompt prefix and train only on assistant tokens."""
    shared_len = min(len(prompt_ids), len(input_ids))
    if input_ids[:shared_len] != prompt_ids[:shared_len]:
        raise ValueError(
            f"Tokenized row {row_idx} prompt is not a prefix of the full conversation"
        )
    if len(prompt_ids) >= len(input_ids):
        raise ValueError(
            f"Tokenized row {row_idx} has no assistant tokens after truncation; "
            "increase max_seq_length"
        )

    return [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]


# End-of-turn markers that chat templates append after assistant content; they are trained on
# like in the single-turn path, where the label span runs to the end of the rendered text.
END_OF_TURN_MARKERS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<turn|>", "<|end|>", "<|endoftext|>", "</s>")


def assistant_char_spans(tokenizer, messages: list[dict[str, Any]], full_text: str) -> list[tuple[int, int]]:
    """Character spans of each assistant turn (content plus its end-of-turn marker) in the rendered chat.

    Template-agnostic: locates each message's (stripped) content in order, so it does not rely on
    rendered prompts being exact prefixes of the full conversation (multi-turn templates often
    rewrite earlier turns).
    """
    markers = tuple(m for m in (getattr(tokenizer, "eos_token", None),) + END_OF_TURN_MARKERS if m)
    spans, cursor = [], 0
    for message in messages:
        content = message["content"].strip()
        pos = full_text.find(content, cursor)
        if pos < 0:
            raise ValueError(f"{message['role']} content not found verbatim in the rendered chat template")
        end = pos + len(content)
        if message["role"] == "assistant":
            rest = full_text[end:]
            stripped = rest.lstrip()
            for marker in markers:
                if stripped.startswith(marker):
                    end += len(rest) - len(stripped) + len(marker)
                    break
            spans.append((pos, end))
        cursor = end
    return spans


def build_multi_turn_row(record: dict[str, Any], tokenizer, max_seq_length: int, filter_overlong: bool):
    """Tokenize a multi-turn conversation with loss on assistant turns only; None if dropped."""
    messages = record["messages"]
    full_text = render_chat(tokenizer, messages)
    # Insert the generation-only header text before assistant turns rendered without it, so every turn is
    # trained after the exact prompt it is generated from (see generation_header_extra). A turn that already
    # carries it (Qwen3.x's last turn) no longer ends with the bare header and is left alone.
    for header, extra in {generation_header_extra(tokenizer), generation_header_extra(tokenizer, history=True)}:
        if not extra:
            continue
        for start, _ in reversed(assistant_char_spans(tokenizer, messages, full_text)):
            if full_text[:start].endswith(header) and not full_text[:start].endswith(header + extra):
                full_text = full_text[:start] + extra + full_text[start:]
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids, offsets = list(encoded["input_ids"]), encoded["offset_mapping"]
    spans = assistant_char_spans(tokenizer, messages, full_text)
    labels = [
        token_id if any(start <= tok_start < end for start, end in spans) else -100
        for token_id, (tok_start, _) in zip(input_ids, offsets)
    ]
    if len(input_ids) > max_seq_length:
        if filter_overlong:
            return None
        input_ids, labels = input_ids[:max_seq_length], labels[:max_seq_length]
    if all(label == -100 for label in labels):
        return None
    return {"input_ids": input_ids, "labels": labels}


def build_tokenized_rows(
    records: list[dict[str, Any]],
    tokenizer,
    config: dict,
) -> list[dict[str, Any]]:
    """Render and tokenize eval-task chat rows before handing them to TRL."""
    max_seq_length = as_int(config[CONFIG_KEY_MAX_SEQ_LENGTH])
    train_on_responses_only = as_bool(config[CONFIG_KEY_TRAIN_ON_RESPONSES_ONLY])
    filter_overlong = as_bool(config.get(CONFIG_KEY_FILTER_OVERLONG, False))
    if as_bool(config.get(CONFIG_KEY_MULTI_TURN, False)):
        if not train_on_responses_only:
            raise ValueError("multi_turn requires train_on_responses_only")
        rows = [build_multi_turn_row(record, tokenizer, max_seq_length, filter_overlong) for record in records]
        kept = [row for row in rows if row is not None]
        print(f"multi-turn tokenization: kept {len(kept)} / {len(rows)} rows (max_seq_length={max_seq_length})")
        if not kept:
            raise ValueError("No multi-turn rows left after tokenization")
        return kept

    rows = []
    for idx, record in enumerate(records):
        messages = record["messages"] # Assume "assistant" is the last role, "system" is optional
        prompt_text = render_chat(tokenizer, messages[:-1], add_generation_prompt=True) # prompt excluding "assistant" turn
        full_text = render_chat(tokenizer, messages) # full conversation including "assistant" turn
        header, extra = generation_header_extra(tokenizer)
        if extra and not full_text.startswith(prompt_text) and prompt_text.endswith(extra):
            # Train the answer after the exact generation prompt used at inference (see generation_header_extra).
            head = prompt_text[: -len(extra)]
            if full_text.startswith(head):
                full_text = prompt_text + full_text[len(head):]

        prompt_ids = tokenize_text(tokenizer, prompt_text)
        input_ids = tokenize_text(tokenizer, full_text, max_seq_length=max_seq_length)
        if not input_ids:
            raise ValueError(f"Tokenized row {idx} is empty")

        row = {"input_ids": input_ids}
        if train_on_responses_only:
            row["labels"] = build_response_only_labels(input_ids, prompt_ids, idx)
        else:
            row["labels"] = list(input_ids)
        rows.append(row)

    return rows


def build_tokenized_dataset(
    records: list[dict[str, Any]],
    tokenizer,
    config: dict,
):
    """Build a pre-tokenized Hugging Face Dataset for SFTTrainer."""
    from datasets import Dataset

    return Dataset.from_list(
        build_tokenized_rows(
            records,
            tokenizer,
            config,
        )
    )
