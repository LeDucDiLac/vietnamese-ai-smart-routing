"""Label Vietnamese prompts with DeepSeek v4 Flash via OpenCode AI.

Two modes:

  relabel-v1   Re-label the Sonnet-labeled rows (data/raw/labeled.jsonl) with
               DeepSeek and save a comparison report vs Sonnet ground truth.
               Use this first to calibrate agreement before committing to the
               full 56k new rows.

  label-new    Label rows in data/raw/crawl.jsonl that are NOT yet in
               data/raw/labeled.jsonl (the ~56k rows from vi-alpaca + uitnlp).

Features:
  - Batch calls (8 texts per request) to reduce cost vs one-at-a-time
  - System prompt with decision rules + Sonnet-derived few-shot examples
  - Prompt caching: system prompt is constant → backend caches it after call 1
  - Per-run token tracking: input / output / cached tokens + estimated cost
  - Resume: skips rows already in the output file

Needs DEEPSEEK_API_KEY in the environment.

    # Dry-run to estimate cost:
    DEEPSEEK_API_KEY=... python -m data.label_deepseek --mode relabel-v1 --dry-run

    # Smoke test on 50 rows:
    DEEPSEEK_API_KEY=... python -m data.label_deepseek --mode relabel-v1 --limit 50

    # Full v1 comparison:
    DEEPSEEK_API_KEY=... python -m data.label_deepseek --mode relabel-v1

    # Label new rows:
    DEEPSEEK_API_KEY=... python -m data.label_deepseek --mode label-new
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_label_schema, task_type_id, LabelSchema


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------


class TaskLabel(BaseModel):
    """Structured output for one labeled prompt."""

    model_config = ConfigDict(extra="ignore")  # silently drops 'id' the model echoes back

    task_type: str
    creativity_scope: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: float = Field(default=0.0, ge=0.0, le=1.0)
    contextual_knowledge: float = Field(default=0.0, ge=0.0, le=1.0)
    domain_knowledge: float = Field(default=0.0, ge=0.0, le=1.0)
    constraint_ct: float = Field(default=0.0, ge=0.0, le=1.0)
    number_of_few_shots: float = Field(default=0.0, ge=0.0, le=1.0)


class BatchResponse(BaseModel):
    """Wrapper so the model always returns a top-level object (not a bare array)."""

    results: list[TaskLabel]


def _normalize(label: TaskLabel, schema: LabelSchema) -> dict[str, Any]:
    """Map a TaskLabel → canonical schema dict, normalizing task_type spelling."""
    tid = task_type_id(label.task_type)
    task = next((t for t in schema.task_types if task_type_id(t) == tid), "Other")
    return {
        "task_type": task,
        **{dim: getattr(label, dim, 0.0) for dim in schema.complexity_dimensions},
    }

OPENCODE_BASE_URL = "https://opencode.ai/zen/go/v1"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_BATCH_SIZE = 8

# DeepSeek pricing defaults (USD per 1M tokens).
# Override with --input-cost / --output-cost / --cache-cost if OpenCode differs.
DEFAULT_INPUT_COST_PER_1M = 0.27
DEFAULT_CACHED_COST_PER_1M = 0.07
DEFAULT_OUTPUT_COST_PER_1M = 1.10

CHARS_PER_TOKEN_APPROX = 4


# ---------------------------------------------------------------------------
# Usage / cost tracking
# ---------------------------------------------------------------------------


@dataclass
class UsageStats:
    input_tokens: int = 0
    cached_tokens: int = 0   # subset of input that hit the cache
    output_tokens: int = 0
    api_calls: int = 0
    fallback_calls: int = 0  # single-row fallback calls (batch parse failure)

    # pricing in USD / 1M tokens
    input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M
    cached_cost_per_1m: float = DEFAULT_CACHED_COST_PER_1M
    output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, inp: int, cached: int, out: int, calls: int = 1, fallbacks: int = 0) -> None:
        with self._lock:
            self.input_tokens += inp
            self.cached_tokens += cached
            self.output_tokens += out
            self.api_calls += calls
            self.fallback_calls += fallbacks

    @property
    def uncached_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_tokens)

    @property
    def estimated_cost(self) -> float:
        return (
            self.uncached_tokens / 1_000_000 * self.input_cost_per_1m
            + self.cached_tokens / 1_000_000 * self.cached_cost_per_1m
            + self.output_tokens / 1_000_000 * self.output_cost_per_1m
        )

    def print_summary(self) -> None:
        cache_pct = self.cached_tokens / max(1, self.input_tokens) * 100
        print("\n=== Token Usage & Cost ===")
        print(f"  API calls      : {self.api_calls:,}  (batch fallbacks: {self.fallback_calls:,})")
        print(f"  Input tokens   : {self.input_tokens:,}")
        print(f"    cached       : {self.cached_tokens:,}  ({cache_pct:.1f}%)")
        print(f"    uncached     : {self.uncached_tokens:,}")
        print(f"  Output tokens  : {self.output_tokens:,}")
        print(f"  Estimated cost : ${self.estimated_cost:.4f} USD")
        print(f"    @ ${self.input_cost_per_1m}/1M input (uncached)")
        print(f"    @ ${self.cached_cost_per_1m}/1M input (cached)")
        print(f"    @ ${self.output_cost_per_1m}/1M output")


def _extract_usage(response) -> tuple[int, int, int]:
    """Return (input_tokens, cached_tokens, output_tokens) from an API response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    inp = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0)
    out = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0)
    # cached tokens: OpenAI SDK stores in prompt_tokens_details.cached_tokens
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details else 0
    return int(inp), int(cached), int(out)


# ---------------------------------------------------------------------------
# Few-shot example selection
# ---------------------------------------------------------------------------


# Decision rules appended to each class description.
_DECISION_RULES = {
    "Open QA": (
        "Câu hỏi có câu trả lời thông tin/giải thích, dù không có một đáp án duy nhất. "
        "Khác Text Generation ở chỗ: Open QA *hỏi* điều gì đó; Text Generation *yêu cầu tạo/viết* nội dung mới."
    ),
    "Text Generation": (
        "Yêu cầu tạo ra nội dung mới (viết bài, soạn thảo, mô tả, hướng dẫn). "
        "Nếu prompt bắt đầu bằng 'viết', 'soạn', 'tạo bài', 'mô tả' → Text Generation. "
        "Nếu bắt đầu bằng câu hỏi ('là gì?', 'tại sao?', 'như thế nào?') → Open QA."
    ),
    "Brainstorming": (
        "Yêu cầu liệt kê NHIỀU ý tưởng, gợi ý, phương án, hoặc khả năng. "
        "Dấu hiệu: 'gợi ý', 'đề xuất', 'liệt kê', 'cho tôi N ý tưởng', 'có thể là gì'. "
        "Khác Open QA: Brainstorming cần đầu ra là *danh sách ý tưởng đa dạng*, không phải một câu trả lời."
    ),
    "Other": (
        "Chỉ dùng khi prompt KHÔNG thuộc bất kỳ loại nào ở trên. "
        "Ưu tiên gán nhãn cụ thể hơn — chỉ chọn Other khi thực sự không rõ."
    ),
    "Closed QA": (
        "Câu hỏi có một đáp án xác định, thường dựa trên ngữ cảnh hoặc kiến thức tra cứu được. "
        "Ví dụ: bài toán, câu hỏi lịch sử, giải phương trình."
    ),
    "Extraction": (
        "Yêu cầu rút trích thông tin cụ thể từ một đoạn văn bản cho trước "
        "(tên, ngày, địa chỉ, danh sách, điểm chung/khác biệt)."
    ),
    "Rewrite": (
        "Có văn bản gốc cụ thể và yêu cầu chỉnh sửa/viết lại nó "
        "(đổi giọng văn, sửa lỗi, rút gọn, dịch phong cách)."
    ),
    "Code Generation": (
        "Yêu cầu viết, hoàn thiện, hoặc giải thích mã nguồn / thuật toán lập trình."
    ),
    "Classification": (
        "Yêu cầu gán nhãn, phân nhóm, hoặc xác định loại của một đối tượng/nội dung."
    ),
    "Chatbot": (
        "Hội thoại thông thường, chào hỏi, hoặc câu lệnh không thuộc tác vụ cụ thể nào."
    ),
    "Summarization": (
        "Yêu cầu tóm tắt, rút gọn, hoặc liệt kê điểm chính của một văn bản dài."
    ),
}


def _select_fewshot_examples(
    labeled_path: Path,
    schema: LabelSchema,
    *,
    min_chars: int = 35,
    max_chars: int = 160,
    per_class: int = 2,
) -> dict[str, list[str]]:
    """Pick deterministic few-shot examples from labeled.jsonl.

    For each class, selects the ``per_class`` shortest texts in [min_chars, max_chars]
    that Sonnet labeled for that class. Deterministic (sorted by length) so the system
    prompt stays identical across runs → cache hits on every call after the first.
    """
    buckets: dict[str, list[str]] = collections.defaultdict(list)
    with labeled_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r.get("text", "")
            cls = r.get("task_type", "")
            if cls and min_chars <= len(t) <= max_chars:
                buckets[cls].append(t)

    result: dict[str, list[str]] = {}
    for cls in schema.task_types:
        candidates = sorted(buckets.get(cls, []), key=len)
        result[cls] = candidates[:per_class]
    return result


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_system_prompt(schema: LabelSchema, fewshot: dict[str, list[str]]) -> str:
    # Class list with decision rules
    class_lines = []
    for t in schema.task_types:
        rule = _DECISION_RULES.get(t, "")
        vi_gloss = schema.task_types_vi.get(t, "")
        class_lines.append(f"- {t}: {vi_gloss}. {rule}".strip().rstrip(".") + ".")

    dim_lines = [
        f"- {d}: {schema.complexity_dimensions_vi.get(d, '')}"
        for d in schema.complexity_dimensions
    ]

    # Few-shot block
    example_lines = ["## Ví dụ từ dữ liệu thực (KHÔNG phân loại — chỉ tham khảo):"]
    for cls in schema.task_types:
        examples = fewshot.get(cls, [])
        for ex in examples:
            example_lines.append(f'[{cls}] "{ex}"')

    return (
        "Bạn là chuyên gia phân loại prompt tiếng Việt theo chuẩn NVIDIA.\n\n"
        "## Các loại nhiệm vụ (chọn ĐÚNG MỘT):\n"
        + "\n".join(class_lines)
        + "\n\n"
        + "\n".join(example_lines)
        + "\n\n"
        "## Chiều độ phức tạp (chấm mỗi chiều một số thực trong [0,1]):\n"
        + "\n".join(dim_lines)
        + "\n\n"
        "## Định dạng đầu ra:\n"
        "Khi nhận được nhiều prompt được đánh số [1], [2], ..., [N], trả về "
        "DUY NHẤT một JSON object với key \"results\" là mảng đúng N phần tử, "
        "theo đúng thứ tự. Mỗi phần tử có các key:\n"
        "  id (số nguyên, bắt đầu từ 1), task_type (tên loại chính xác như trên), "
        "creativity_scope, reasoning, contextual_knowledge, domain_knowledge, "
        "constraint_ct, number_of_few_shots."
    )


def _build_user_message(texts: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


def _get_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "label_deepseek needs `uv pip install openai` (or `pip install openai`)."
        ) from exc
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Run: export DEEPSEEK_API_KEY=your_key"
        )
    from openai import OpenAI
    return OpenAI(base_url=OPENCODE_BASE_URL, api_key=api_key)


def _is_rate_limit(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) in (429, 529):
        return True
    return "rate limit" in str(exc).lower() or "429" in str(exc)


def _call_with_retry(fn, *, max_retries: int = 5) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_retries:
                raise
            rate_limited = _is_rate_limit(exc)
            if not rate_limited and attempt >= 2:
                raise
            delay = min(60.0, 2.0 * (2 ** attempt)) + random.uniform(0, 1.0)
            print(f"  {'rate-limited' if rate_limited else 'error'} ({exc}); "
                  f"retry {attempt + 1}/{max_retries} in {delay:.1f}s")
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Batch labeling — Pydantic v2 structured output
# ---------------------------------------------------------------------------


def _call_structured(
    client, system_prompt: str, texts: list[str]
) -> tuple[BatchResponse | None, tuple[int, int, int]]:
    """Try client.beta.chat.completions.parse() — proper JSON-schema enforcement.

    Returns (BatchResponse | None, (inp, cached, out)).  Returns None immediately
    on 400 (endpoint doesn't support json_schema format) so callers skip straight
    to the json_object fallback without wasting retries.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_user_message(texts)},
    ]
    try:
        resp = client.beta.chat.completions.parse(
            model=DEEPSEEK_MODEL,
            messages=messages,
            response_format=BatchResponse,
            temperature=0.0,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is not None and isinstance(parsed, BatchResponse):
            return parsed, _extract_usage(resp)
        return None, _extract_usage(resp)
    except Exception as exc:
        # 400 = endpoint doesn't support json_schema structured output; skip silently.
        # Any other error also falls through — json_object path will retry properly.
        if getattr(exc, "status_code", None) == 400:
            return None, (0, 0, 0)
        return None, (0, 0, 0)


def _call_json_object(
    client, system_prompt: str, texts: list[str]
) -> tuple[BatchResponse | None, tuple[int, int, int]]:
    """Fallback: json_object mode + Pydantic model_validate_json()."""
    def _call():
        return client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_user_message(texts)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

    resp = _call_with_retry(_call)
    raw = resp.choices[0].message.content or "{}"
    usage = _extract_usage(resp)

    # Strip markdown fences if present
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1].lstrip("json").strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        s = s[start:end + 1]

    try:
        parsed = BatchResponse.model_validate_json(s)
        return parsed, usage
    except Exception:
        return None, usage


def _call_single_fallback(
    client, system_prompt: str, text: str, schema: LabelSchema
) -> tuple[dict[str, Any], tuple[int, int, int]]:
    """Per-row last resort when batch parsing fails entirely."""
    batch, usage = _call_json_object(client, system_prompt, [text])
    if batch and batch.results:
        return _normalize(batch.results[0], schema), usage
    return {"task_type": "Other", **{d: 0.0 for d in schema.complexity_dimensions}}, usage


def label_batch(
    client,
    system_prompt: str,
    rows: list[dict[str, Any]],
    schema: LabelSchema,
    stats: UsageStats,
) -> list[dict[str, Any]]:
    """Label one batch. Strategy (in order):

    1. beta.parse() with BatchResponse schema  — full structured output
    2. json_object + BatchResponse.model_validate_json()  — Pydantic parsing
    3. Per-row json_object + single TaskLabel validation  — last resort
    """
    texts = [r["text"] for r in rows]
    n = len(texts)

    # 1. Structured output
    batch, usage = _call_structured(client, system_prompt, texts)
    if batch is not None and len(batch.results) == n:
        stats.add(*usage, calls=1)
        return [_normalize(lbl, schema) for lbl in batch.results]

    # 2. json_object + Pydantic validation
    batch, usage = _call_json_object(client, system_prompt, texts)
    if batch is not None and len(batch.results) == n:
        stats.add(*usage, calls=1)
        return [_normalize(lbl, schema) for lbl in batch.results]

    # 3. Per-row fallback
    print(f"  [warn] batch failed for {n} rows; falling back to per-row calls")
    stats.add(*usage, calls=1)
    results = []
    for row in rows:
        rec, fb_usage = _call_single_fallback(client, system_prompt, row["text"], schema)
        stats.add(*fb_usage, calls=1, fallbacks=1)
        results.append(rec)
    return results


# ---------------------------------------------------------------------------
# Main labeling loop
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def label_rows(
    rows: list[dict[str, Any]],
    out_path: Path,
    schema: LabelSchema,
    system_prompt: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = 4,
    label_source: str = "teacher_deepseek",
    cost_config: dict[str, float] | None = None,
) -> tuple[int, UsageStats]:
    """Label rows and append to out_path. Returns (rows_written, UsageStats)."""
    client = _get_client()

    stats = UsageStats()
    if cost_config:
        stats.input_cost_per_1m = cost_config.get("input", DEFAULT_INPUT_COST_PER_1M)
        stats.cached_cost_per_1m = cost_config.get("cached", DEFAULT_CACHED_COST_PER_1M)
        stats.output_cost_per_1m = cost_config.get("output", DEFAULT_OUTPUT_COST_PER_1M)

    # Resume: skip rows already in output
    done: set[str] = set()
    if out_path.exists():
        for prev in _read_jsonl(out_path):
            t = prev.get("text")
            if t:
                done.add(t)
    todo = [r for r in rows if r["text"] not in done]
    if done:
        print(f"  resuming: {len(done)} already labeled, {len(todo)} remaining")
    if not todo:
        print("  all rows already labeled; nothing to do")
        return 0, stats

    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    print(f"  {len(todo)} rows → {len(batches)} batches of ≤{batch_size}")

    write_lock = threading.Lock()
    written = 0
    total = len(todo)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _process_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        labels = label_batch(client, system_prompt, batch, schema, stats)
        results = []
        for src_row, lbl in zip(batch, labels):
            rec: dict[str, Any] = {
                "text": src_row["text"],
                "task_type": lbl["task_type"],
                "label_source": label_source,
            }
            if src_row.get("source"):
                rec["hf_source"] = src_row["source"]
            elif src_row.get("hf_source"):
                rec["hf_source"] = src_row["hf_source"]
            for dim in schema.complexity_dimensions:
                rec[dim] = lbl.get(dim, 0.0)
            results.append(rec)
        return results

    with out_path.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_batch, b): b for b in batches}
            for fut in as_completed(futures):
                try:
                    batch_results = fut.result()
                except Exception as exc:
                    print(f"  batch failed after retries: {exc}; skipping")
                    continue
                with write_lock:
                    for rec in batch_results:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    written += len(batch_results)
                    if written % 500 == 0 or written == total:
                        print(f"  labeled {written}/{total}  "
                              f"(~${stats.estimated_cost:.3f} so far)")
    return written, stats


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


def compare_with_sonnet(
    sonnet_path: Path,
    deepseek_path: Path,
    schema: LabelSchema,
    report_path: Path,
) -> None:
    sonnet_rows = _read_jsonl(sonnet_path)
    deepseek_rows = _read_jsonl(deepseek_path)

    deepseek_map: dict[str, str] = {
        r["text"]: r["task_type"] for r in deepseek_rows
        if r.get("text") and r.get("task_type")
    }

    total = agreed = 0
    confusion: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    per_class_n: dict[str, int] = collections.defaultdict(int)
    per_class_ok: dict[str, int] = collections.defaultdict(int)

    for row in sonnet_rows:
        ds = deepseek_map.get(row.get("text", ""))
        if ds is None:
            continue
        sonnet = row.get("task_type", "")
        total += 1
        per_class_n[sonnet] += 1
        confusion[sonnet][ds] += 1
        if task_type_id(sonnet) == task_type_id(ds):
            agreed += 1
            per_class_ok[sonnet] += 1

    rate = agreed / total if total else 0.0
    per_class: dict[str, Any] = {}
    for cls in schema.task_types:
        n = per_class_n.get(cls, 0)
        ok = per_class_ok.get(cls, 0)
        top_conf = sorted(
            [(k, v) for k, v in confusion.get(cls, {}).items()
             if task_type_id(k) != task_type_id(cls)],
            key=lambda x: -x[1],
        )[:3]
        per_class[cls] = {
            "sonnet_count": n,
            "agreed": ok,
            "agreement_rate": round(ok / n, 4) if n else None,
            "top_confusions": top_conf,
        }

    report = {
        "total_compared": total,
        "agreed": agreed,
        "overall_agreement_rate": round(rate, 4),
        "per_class": per_class,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n=== DeepSeek vs Sonnet Agreement ===")
    print(f"Compared: {total}   Agreed: {agreed}   Overall: {rate:.1%}\n")
    print(f"{'Class':<22} {'N':>6} {'Agreed':>8} {'Rate':>7}  Top confusions")
    print("-" * 80)
    for cls in schema.task_types:
        pc = per_class[cls]
        n, ok = pc["sonnet_count"], pc["agreed"]
        r_str = f"{pc['agreement_rate']:.1%}" if pc["agreement_rate"] is not None else "  n/a"
        conf_str = ", ".join(f"{k}({v})" for k, v in pc["top_confusions"])
        print(f"{cls:<22} {n:>6} {ok:>8} {r_str:>7}  {conf_str}")
    print(f"\nFull report → {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Label Vietnamese prompts with DeepSeek v4 Flash"
    )
    ap.add_argument(
        "--mode", choices=["relabel-v1", "label-new"], required=True,
    )
    ap.add_argument("--labeled", default="data/raw/labeled.jsonl")
    ap.add_argument("--crawl", default="data/raw/crawl.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default="data/raw/deepseek_vs_sonnet.json")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Count rows and estimate cost without calling the API")
    # Cost overrides (USD per 1M tokens)
    ap.add_argument("--input-cost", type=float, default=DEFAULT_INPUT_COST_PER_1M,
                    help=f"Input token cost per 1M (default: {DEFAULT_INPUT_COST_PER_1M})")
    ap.add_argument("--cache-cost", type=float, default=DEFAULT_CACHED_COST_PER_1M,
                    help=f"Cached input cost per 1M (default: {DEFAULT_CACHED_COST_PER_1M})")
    ap.add_argument("--output-cost", type=float, default=DEFAULT_OUTPUT_COST_PER_1M,
                    help=f"Output token cost per 1M (default: {DEFAULT_OUTPUT_COST_PER_1M})")
    args = ap.parse_args()

    schema = load_label_schema()
    labeled_path = Path(args.labeled)
    cost_config = {
        "input": args.input_cost,
        "cached": args.cache_cost,
        "output": args.output_cost,
    }

    # Build system prompt with few-shot examples from labeled.jsonl
    print("Loading few-shot examples from Sonnet labels…")
    fewshot = _select_fewshot_examples(labeled_path, schema)
    system_prompt = _build_system_prompt(schema, fewshot)
    print(f"System prompt: {len(system_prompt)} chars  "
          f"(~{len(system_prompt) // CHARS_PER_TOKEN_APPROX} tokens)\n")

    if args.mode == "relabel-v1":
        rows = _read_jsonl(labeled_path)
        if not rows:
            print(f"No rows at {labeled_path}"); sys.exit(1)
        if args.limit:
            rows = rows[:args.limit]
        out_path = Path(args.out) if args.out else Path("data/raw/labeled_deepseek.jsonl")

        est_chars = sum(len(r["text"]) for r in rows)
        est_input_tokens = (est_chars // CHARS_PER_TOKEN_APPROX
                            + len(system_prompt) // CHARS_PER_TOKEN_APPROX * len(rows) // args.batch_size)
        est_output_tokens = len(rows) * 80  # ~80 tokens per label JSON
        est_cost = (est_input_tokens / 1_000_000 * args.input_cost
                    + est_output_tokens / 1_000_000 * args.output_cost)

        print(f"Mode          : relabel-v1")
        print(f"Rows          : {len(rows):,}")
        print(f"Est. input tk : ~{est_input_tokens:,}")
        print(f"Est. output tk: ~{est_output_tokens:,}")
        print(f"Est. cost     : ~${est_cost:.3f} USD  (before cache savings)")
        print(f"Output        : {out_path}")

        if args.dry_run:
            print("\n[dry-run] No API calls made.")
            return

        n, stats = label_rows(
            rows, out_path, schema, system_prompt,
            batch_size=args.batch_size, workers=args.workers,
            label_source="teacher_deepseek", cost_config=cost_config,
        )
        print(f"\nWrote {n} rows to {out_path}")
        stats.print_summary()

        if _read_jsonl(out_path):
            compare_with_sonnet(labeled_path, out_path, schema, Path(args.report))

    elif args.mode == "label-new":
        labeled_rows = _read_jsonl(labeled_path)
        labeled_texts = {r["text"] for r in labeled_rows if r.get("text")}
        crawl_rows = _read_jsonl(Path(args.crawl))
        new_rows = [r for r in crawl_rows if r.get("text") and r["text"] not in labeled_texts]
        if args.limit:
            new_rows = new_rows[:args.limit]
        out_path = Path(args.out) if args.out else Path("data/raw/labeled_deepseek_new.jsonl")

        est_chars = sum(len(r["text"]) for r in new_rows)
        est_input_tokens = (est_chars // CHARS_PER_TOKEN_APPROX
                            + len(system_prompt) // CHARS_PER_TOKEN_APPROX * len(new_rows) // args.batch_size)
        est_output_tokens = len(new_rows) * 80
        est_cost = (est_input_tokens / 1_000_000 * args.input_cost
                    + est_output_tokens / 1_000_000 * args.output_cost)

        print(f"Mode          : label-new")
        print(f"Total crawl   : {len(crawl_rows):,}")
        print(f"Already in v1 : {len(labeled_texts):,}")
        print(f"New rows      : {len(new_rows):,}")
        print(f"Est. input tk : ~{est_input_tokens:,}")
        print(f"Est. output tk: ~{est_output_tokens:,}")
        print(f"Est. cost     : ~${est_cost:.3f} USD  (before cache savings)")
        print(f"Output        : {out_path}")

        if args.dry_run:
            print("\n[dry-run] No API calls made.")
            return

        n, stats = label_rows(
            new_rows, out_path, schema, system_prompt,
            batch_size=args.batch_size, workers=args.workers,
            label_source="teacher_deepseek", cost_config=cost_config,
        )
        print(f"\nWrote {n} rows to {out_path}")
        stats.print_summary()


if __name__ == "__main__":
    main()
