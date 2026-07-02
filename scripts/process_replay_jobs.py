#!/usr/bin/env python3
"""Build deterministic, text-only replay jobs from extracted Gateway traces.

The extracted jobs may contain truncated tool results, multimodal payloads, and
entire agent execution histories. Those traces are neither safe nor meaningful
inputs for comparing text-generation models. This stage reduces every accepted
job to the system instruction and the final contiguous block of non-empty user
messages, then validates the resulting conversation against explicit limits.

Example:
    python scripts/process_replay_jobs.py \
        --input data/eval/replay-v1/jobs.jsonl \
        --output data/eval/replay-v2/jobs_v2.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

TRUNCATION_MARKERS = (
    "litellm_truncated",
    "truncation is a db storage safeguard",
)
DATA_URL_RE = re.compile(r"data:(?:image|audio|video|application)/[^;\s]+;base64,", re.I)
BASE64_RUN_RE = re.compile(r"[A-Za-z0-9+/]{2048,}={0,2}")
ALLOWED_SAMPLING_KEYS = ("temperature", "top_p", "max_tokens")
REPLAY_V1_DIR = Path("data/eval/replay-v1")
REPLAY_V2_DIR = Path("data/eval/replay-v2")


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, {"_parse_error": str(exc)}
                continue
            if not isinstance(value, dict):
                yield line_number, {"_parse_error": "record is not a JSON object"}
                continue
            yield line_number, value


def text_content(content: Any) -> str:
    """Extract text without carrying structured or binary content into replay."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in (None, "text", "input_text"):
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _clean_text(value: str) -> str:
    return value.replace("\x00", "").strip()


def canonical_messages(messages: Any) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Return system + final contiguous user block; never retain tool state."""
    if not isinstance(messages, list):
        return [], {}

    normalized: list[tuple[str, str]] = []
    removed = Counter()
    for message in messages:
        if not isinstance(message, dict):
            removed["invalid_messages"] += 1
            continue
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            removed["unsupported_messages"] += 1
            continue
        text = _clean_text(text_content(message.get("content")))
        normalized.append((role, text))

    user_indexes = [i for i, (role, text) in enumerate(normalized) if role == "user" and text]
    if not user_indexes:
        return [], dict(removed)

    # Consecutive user messages are commonly split context/instruction pairs.
    # Once an assistant or tool turn intervenes, older user turns are history
    # rather than the current task and are deliberately excluded.
    first_user = last_user = user_indexes[-1]
    while first_user > 0 and normalized[first_user - 1][0] == "user":
        first_user -= 1
    user_texts = [
        text
        for role, text in normalized[first_user : last_user + 1]
        if role == "user" and text
    ]

    system_texts = [text for role, text in normalized[: first_user + 1] if role == "system" and text]
    result = []
    if system_texts:
        result.append({"role": "system", "content": "\n\n".join(system_texts)})
    result.append({"role": "user", "content": "\n\n".join(user_texts)})

    kept_positions = set(range(first_user, last_user + 1))
    removed["history_messages"] += sum(
        role in {"user", "assistant", "tool"}
        for i, (role, _) in enumerate(normalized)
        if i not in kept_positions
    )
    removed["tool_messages"] += sum(role == "tool" for role, _ in normalized)
    return result, dict(removed)


def estimated_tokens(messages: list[dict[str, str]]) -> int:
    """Conservative tokenizer-independent estimate suitable for filtering."""
    text = "\n".join(message["content"] for message in messages)
    byte_estimate = math.ceil(len(text.encode("utf-8")) / 3)
    word_estimate = math.ceil(len(text.split()) * 1.5)
    return max(byte_estimate, word_estimate) + 16 * len(messages)


def rejection_reason(
    messages: list[dict[str, str]], max_chars: int, max_estimated_tokens: int
) -> tuple[str | None, dict[str, Any]]:
    if not messages or messages[-1]["role"] != "user" or not messages[-1]["content"]:
        return "no_user_text", {}
    text = "\n".join(message["content"] for message in messages)
    lowered = text.lower()
    if any(marker in lowered for marker in TRUNCATION_MARKERS):
        return "truncated_content", {}
    if DATA_URL_RE.search(text) or BASE64_RUN_RE.search(text):
        return "binary_artifact", {}
    char_count = len(text)
    token_estimate = estimated_tokens(messages)
    detail = {"canonical_chars": char_count, "estimated_prompt_tokens": token_estimate}
    if char_count > max_chars:
        return "character_limit", detail
    if token_estimate > max_estimated_tokens:
        return "token_limit", detail
    return None, detail


def process_job(
    job: dict[str, Any], max_chars: int, max_estimated_tokens: int, max_output_tokens: int
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    if "_parse_error" in job:
        return None, "invalid_json", {"error": job["_parse_error"]}
    prompt_id = job.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        return None, "missing_prompt_id", {}

    messages, removed = canonical_messages(job.get("messages"))
    reason, detail = rejection_reason(messages, max_chars, max_estimated_tokens)
    detail["removed_messages"] = removed
    if reason:
        return None, reason, detail

    sampling_source = job.get("sampling") if isinstance(job.get("sampling"), dict) else {}
    sampling = {
        key: sampling_source[key]
        for key in ALLOWED_SAMPLING_KEYS
        if key in sampling_source
    }
    sampling["max_tokens"] = min(
        max(1, int(sampling.get("max_tokens") or max_output_tokens)),
        max_output_tokens,
    )

    result = copy.deepcopy(job)
    result["messages"] = messages
    result["system_prompt"] = messages[0]["content"] if messages[0]["role"] == "system" else ""
    result["prompt_text"] = messages[-1]["content"]
    result["sampling"] = sampling
    result["source_prompt_tokens"] = result.get("prompt_tokens", 0)
    result["prompt_tokens"] = detail["estimated_prompt_tokens"]
    result["processing"] = {
        "version": 2,
        "strategy": "system_and_final_user_block",
        "canonical_chars": detail["canonical_chars"],
        "removed_messages": removed,
    }
    return result, None, detail


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    drop_path = Path(args.drop_log) if args.drop_log else output_path.with_name(
        f"{output_path.stem}_dropped.jsonl"
    )
    report_path = Path(args.report) if args.report else output_path.with_name(
        f"{output_path.stem}_report.json"
    )
    for path in (output_path, drop_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    removed_counts = Counter()
    seen = set()
    with output_path.open("w", encoding="utf-8") as accepted, drop_path.open(
        "w", encoding="utf-8"
    ) as dropped:
        for line_number, job in iter_jsonl(input_path):
            counts["input"] += 1
            prompt_id = job.get("prompt_id")
            if isinstance(prompt_id, str) and prompt_id in seen:
                result, reason, detail = None, "duplicate_prompt_id", {}
            else:
                result, reason, detail = process_job(
                    job, args.max_chars, args.max_estimated_tokens, args.max_output_tokens
                )
            if isinstance(prompt_id, str):
                seen.add(prompt_id)

            removed_counts.update(detail.get("removed_messages", {}))
            if result is None:
                counts[f"dropped:{reason}"] += 1
                dropped.write(
                    json.dumps(
                        {
                            "line": line_number,
                            "prompt_id": prompt_id,
                            "reason": reason,
                            **{k: v for k, v in detail.items() if k != "removed_messages"},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue
            counts["accepted"] += 1
            accepted.write(json.dumps(result, ensure_ascii=False) + "\n")

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "drop_log": str(drop_path),
        "limits": {
            "max_chars": args.max_chars,
            "max_estimated_tokens": args.max_estimated_tokens,
            "max_output_tokens": args.max_output_tokens,
        },
        "counts": dict(sorted(counts.items())),
        "removed_messages": dict(sorted(removed_counts.items())),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(REPLAY_V1_DIR / "jobs.jsonl"))
    parser.add_argument("--output", default=str(REPLAY_V2_DIR / "jobs_v2.jsonl"))
    parser.add_argument("--drop-log")
    parser.add_argument("--report")
    parser.add_argument("--max-chars", type=int, default=60_000)
    parser.add_argument("--max-estimated-tokens", type=int, default=16_384)
    parser.add_argument("--max-output-tokens", type=int, default=4_096)
    args = parser.parse_args()
    report = build_dataset(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
