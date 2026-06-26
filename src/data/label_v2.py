"""Label Vietnamese prompts with v2 schema (6 task types / 3 complexity dims).

Two teacher backends, each labelable independently then reconciled:

  anthropic   Label with Claude (Anthropic API) using the v2 rubric.
  deepseek    Label with DeepSeek v4 Flash (OpenCode AI) using the v2 rubric.
  compare     Reconcile two pre-computed label files, print agreement stats,
              write silver (agreed) and gold-queue (disagreed) rows.

Typical workflow
────────────────
    # 1. Label with both teachers (can run in parallel on separate machines):
    ANTHROPIC_AUTH_TOKEN=... python -m data.label_v2 --mode anthropic \\
        --in data/processed/v1.5/train.jsonl --out data/raw/v2_anthropic.jsonl

    DEEPSEEK_API_KEY=... python -m data.label_v2 --mode deepseek \\
        --in data/processed/v1.5/train.jsonl --out data/raw/v2_deepseek.jsonl

    # 2. Reconcile and check agreement:
    python -m data.label_v2 --mode compare \\
        --teacher-a data/raw/v2_anthropic.jsonl \\
        --teacher-b data/raw/v2_deepseek.jsonl \\
        --silver-out data/raw/v2_silver.jsonl \\
        --gold-out   data/raw/v2_gold_queue.jsonl

    # Smoke test (50 rows):
    ANTHROPIC_AUTH_TOKEN=... python -m data.label_v2 --mode anthropic \\
        --in data/processed/v1.5/train.jsonl --out /tmp/v2_test.jsonl --limit 50

Features
────────
  - Resume: rows whose text is already in --out are skipped on restart
  - Batching (DeepSeek): 8 prompts per request for cost efficiency
  - Prompt caching (Anthropic): system prompt marked ephemeral → cache hit after call 1
  - Per-run token / cost tracking (DeepSeek mode)
  - Exponential backoff + jitter on rate-limit / transient errors
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LabelSchema, load_label_schema, task_type_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENCODE_BASE_URL = "https://opencode.ai/zen/go/v1"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_BATCH_SIZE = 8

DEFAULT_INPUT_COST_PER_1M  = 0.27
DEFAULT_CACHED_COST_PER_1M = 0.07
DEFAULT_OUTPUT_COST_PER_1M = 1.10

# ---------------------------------------------------------------------------
# Pydantic output models (v2 dims only)
# ---------------------------------------------------------------------------


class TeacherLabelV2(BaseModel):
    """Structured output for one labeled prompt — v2 schema."""

    model_config = ConfigDict(extra="ignore")

    task_type:            str
    reasoning_depth:      float = Field(ge=0.0, le=1.0)
    domain_knowledge:     float = Field(ge=0.0, le=1.0)
    instruction_precision: float = Field(ge=0.0, le=1.0)

    @field_validator("task_type")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("task_type is empty")
        return str(v).strip()

    @classmethod
    def from_payload(cls, data: dict[str, Any], schema: LabelSchema) -> "TeacherLabelV2":
        """Validate and resolve task_type to a known v2 schema label.

        Raises ValueError on unknown task_type or out-of-range dims.
        """
        obj = cls(**data)
        tid = task_type_id(obj.task_type)
        for display in schema.task_types:
            if task_type_id(display) == tid:
                obj.task_type = display
                return obj
        raise ValueError(f"unknown task_type: {obj.task_type!r}")


class BatchResponseV2(BaseModel):
    results: list[TeacherLabelV2]


# ---------------------------------------------------------------------------
# Decision rules (Vietnamese) for v2 task types
# ---------------------------------------------------------------------------

_DECISION_RULES_V2: dict[str, str] = {
    "Knowledge Retrieval / QA": (
        "Câu hỏi có đáp án TỒN TẠI và có thể tra cứu được. Bao gồm: hỏi đáp thông tin, "
        "câu hỏi dựa trên văn bản cho sẵn, phân loại nội dung, trích xuất thông tin từ văn bản. "
        "Đáp án có sẵn — không cần suy luận phức tạp hay tạo nội dung mới."
    ),
    "Reasoning / Problem Solving": (
        "Cần suy luận NHIỀU BƯỚC, tính toán, phân tích logic hoặc lập kế hoạch. "
        "Đáp án phải được suy ra, không thể tra cứu trực tiếp. "
        "Dấu hiệu: toán học, chứng minh, so sánh phân tích, lập kế hoạch nhiều bước."
    ),
    "Summarization": (
        "Có VĂN BẢN DÀI CHO SẴN trong prompt và yêu cầu tóm tắt, rút gọn hoặc trình bày lại. "
        "LUÔN có văn bản đầu vào. Nếu không có văn bản đầu vào rõ ràng → không phải Summarization."
    ),
    "Content Creation": (
        "Yêu cầu TẠO NỘI DUNG MỚI: viết bài, sáng tác, liệt kê ý tưởng, viết lại/cải thiện văn bản. "
        "Đầu ra là văn bản sáng tạo, không bị ràng buộc bởi một đáp án duy nhất. "
        "Dấu hiệu: 'viết', 'soạn', 'tạo', 'gợi ý', 'đề xuất', 'viết lại'."
    ),
    "Code": (
        "Liên quan đến LẬP TRÌNH: viết code, gỡ lỗi, giải thích code, thiết kế thuật toán, viết script. "
        "Bất kỳ prompt nào đề cập đến ngôn ngữ lập trình, hàm, class, SQL, hoặc mã nguồn."
    ),
    "Conversation": (
        "Hội thoại thông thường, chào hỏi, tán gẫu, hoặc yêu cầu rất đơn giản không cần kiến thức. "
        "Câu trả lời ngắn, không cần tra cứu hay suy luận. Dấu hiệu: 'xin chào', 'cảm ơn', "
        "'bạn có thể...?', câu hỏi nhẹ nhàng không có nội dung chuyên sâu."
    ),
}

# Scoring rubric for the 3 complexity dims
_DIM_RUBRIC_V2 = """
reasoning_depth — Độ sâu suy luận:
  0.0 = tra cứu trực tiếp, không cần suy luận
  0.5 = cần phân tích hoặc so sánh đơn giản
  1.0 = suy luận nhiều bước, chứng minh, lập kế hoạch phức tạp

domain_knowledge — Kiến thức chuyên sâu:
  0.0 = kiến thức phổ thông, ai cũng biết
  0.5 = cần hiểu biết nhất định về lĩnh vực (kỹ thuật, lịch sử, ...)
  1.0 = chuyên môn sâu (y tế, luật, toán cao cấp, kỹ thuật chuyên biệt)

instruction_precision — Độ chính xác hướng dẫn:
  0.0 = hoàn toàn tự do, không có ràng buộc
  0.5 = có một số hướng dẫn về định dạng hoặc độ dài
  1.0 = ràng buộc chặt chẽ đa chiều: định dạng JSON/bảng cụ thể, giới hạn từ,
        quy tắc phong cách + ngôn ngữ + cấu trúc cùng lúc
"""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_system_prompt(schema: LabelSchema) -> str:
    """Build the shared system prompt for v2 labeling (both Anthropic and DeepSeek)."""
    class_lines = []
    for t in schema.task_types:
        rule = _DECISION_RULES_V2.get(t, "")
        vi_gloss = schema.task_types_vi.get(t, "")
        line = f"- {t}: {vi_gloss}."
        if rule:
            line += f" {rule}"
        class_lines.append(line.rstrip(".") + ".")

    dim_lines = [
        f"- {d}: {schema.complexity_dimensions_vi.get(d, '')}"
        for d in schema.complexity_dimensions
    ]

    return (
        "Bạn là chuyên gia phân loại prompt AI theo schema v2.\n\n"
        "## Các loại nhiệm vụ (chọn ĐÚNG MỘT):\n"
        + "\n".join(class_lines)
        + "\n\n"
        "## Chiều độ phức tạp:\n"
        + _DIM_RUBRIC_V2.strip()
        + "\n\n"
        "## Định dạng đầu ra:\n"
        "Khi nhận được nhiều prompt được đánh số [1], [2], ..., [N], trả về "
        "DUY NHẤT một JSON object với key \"results\" là mảng đúng N phần tử, "
        "theo đúng thứ tự. Mỗi phần tử có các key:\n"
        "  id (số nguyên, bắt đầu từ 1), task_type (tên loại chính xác như trên), "
        "reasoning_depth, domain_knowledge, instruction_precision."
    )


def _build_user_message(texts: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _normalize(label: TeacherLabelV2, schema: LabelSchema) -> dict[str, Any]:
    tid = task_type_id(label.task_type)
    task = next((t for t in schema.task_types if task_type_id(t) == tid), schema.task_types[-1])
    return {
        "task_type": task,
        **{dim: getattr(label, dim, 0.0) for dim in schema.complexity_dimensions},
    }


# ---------------------------------------------------------------------------
# Rate limiter (Anthropic path)
# ---------------------------------------------------------------------------


class _RateLimiter:
    def __init__(self, rpm: float):
        self._interval = 60.0 / max(rpm, 1.0)
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


def _is_rate_limit(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) in (429, 529):
        return True
    return "rate limit" in str(exc).lower() or "429" in str(exc)


def _call_with_retry(fn, *, max_retries: int = 6, base_delay: float = 2.0) -> Any:
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            attempt += 1
            rate_limited = _is_rate_limit(exc)
            if attempt > max_retries:
                raise
            if not rate_limited and attempt > 2:
                raise
            delay = min(60.0, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 1.0)
            kind = "rate-limited" if rate_limited else "transient error"
            print(f"  {kind} ({exc}); retry {attempt}/{max_retries} in {delay:.1f}s")
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Usage tracking (DeepSeek path)
# ---------------------------------------------------------------------------


@dataclass
class UsageStats:
    input_tokens:  int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    api_calls:     int = 0
    fallback_calls: int = 0
    input_cost_per_1m:  float = DEFAULT_INPUT_COST_PER_1M
    cached_cost_per_1m: float = DEFAULT_CACHED_COST_PER_1M
    output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, inp: int, cached: int, out: int, calls: int = 1, fallbacks: int = 0) -> None:
        with self._lock:
            self.input_tokens  += inp
            self.cached_tokens += cached
            self.output_tokens += out
            self.api_calls     += calls
            self.fallback_calls += fallbacks

    @property
    def estimated_cost(self) -> float:
        uncached = max(0, self.input_tokens - self.cached_tokens)
        return (
            uncached / 1_000_000 * self.input_cost_per_1m
            + self.cached_tokens / 1_000_000 * self.cached_cost_per_1m
            + self.output_tokens / 1_000_000 * self.output_cost_per_1m
        )

    def print_summary(self) -> None:
        cache_pct = self.cached_tokens / max(1, self.input_tokens) * 100
        print("\n=== Token Usage & Cost ===")
        print(f"  API calls      : {self.api_calls:,}  (fallbacks: {self.fallback_calls:,})")
        print(f"  Input tokens   : {self.input_tokens:,}")
        print(f"    cached       : {self.cached_tokens:,}  ({cache_pct:.1f}%)")
        print(f"  Output tokens  : {self.output_tokens:,}")
        print(f"  Estimated cost : ${self.estimated_cost:.4f} USD")


def _extract_usage(response) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    inp = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0)
    out = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details else 0
    return int(inp), int(cached), int(out)


# ---------------------------------------------------------------------------
# Anthropic teacher
# ---------------------------------------------------------------------------


def label_with_anthropic(
    rows: list[dict[str, Any]],
    schema: LabelSchema,
    *,
    model: str = "kr/claude-sonnet-4.6",
    out_path: Path,
    rpm: float = 90.0,
    workers: int = 8,
    max_retries: int = 6,
) -> int:
    """Label rows with Anthropic (Claude) using the v2 rubric.

    Reads ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN from env.
    Resumes from out_path if it already exists.
    """
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic mode needs `pip install anthropic`") from exc

    client = Anthropic(
        base_url=os.environ.get("ANTHROPIC_BASE_URL") or None,
        auth_token=os.environ.get("ANTHROPIC_AUTH_TOKEN") or None,
    )
    system_prompt = _build_system_prompt(schema)
    limiter = _RateLimiter(rpm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        for prev in _read_jsonl(out_path):
            if t := prev.get("text"):
                done.add(t)
    todo = [r for r in rows if r["text"] not in done]
    if done:
        print(f"  resuming: {len(done)} already labeled, {len(todo)} remaining")

    write_lock = threading.Lock()
    written = 0
    total = len(todo)

    def _label(row: dict[str, Any]) -> dict[str, Any] | None:
        def _call():
            limiter.acquire()
            return _anthropic_label_one(client, model, system_prompt, row["text"], schema)
        try:
            rec = _call_with_retry(_call, max_retries=max_retries)
        except Exception as exc:
            print(f"  giving up on row ({exc}); skipping", file=sys.stderr)
            return None
        rec["text"] = row["text"]
        rec["label_source"] = "teacher_anthropic_v2"
        if row.get("hf_source"):
            rec["hf_source"] = row["hf_source"]
        elif row.get("source"):
            rec["hf_source"] = row["source"]
        return rec

    with out_path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_label, r): r for r in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            if rec is None:
                continue
            with write_lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                written += 1
                if written % 100 == 0:
                    print(f"  labeled {written}/{total} (this run)")
    return written


def _extract_json(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        return s[start:end + 1]
    return s


def _anthropic_label_one(
    client, model: str, system_prompt: str, text: str, schema: LabelSchema
) -> dict[str, Any]:
    resp = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=0.0,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": f"Prompt:\n{text}"}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text")
    data = json.loads(_extract_json(raw))
    label = TeacherLabelV2.from_payload(data, schema)
    rec: dict[str, Any] = {"task_type": label.task_type}
    for dim in schema.complexity_dimensions:
        rec[dim] = getattr(label, dim)
    return rec


# ---------------------------------------------------------------------------
# DeepSeek teacher
# ---------------------------------------------------------------------------


def _get_deepseek_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("deepseek mode needs `pip install openai`") from exc
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set. Run: export DEEPSEEK_API_KEY=your_key")
    from openai import OpenAI
    return OpenAI(base_url=OPENCODE_BASE_URL, api_key=api_key)


_NO_THINK = {"thinking": {"type": "disabled"}}


def _call_structured_v2(client, system_prompt: str, texts: list[str]):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": _build_user_message(texts)},
    ]
    try:
        resp = client.beta.chat.completions.parse(
            model=DEEPSEEK_MODEL,
            messages=messages,
            response_format=BatchResponseV2,
            temperature=0.0,
            extra_body=_NO_THINK,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is not None and isinstance(parsed, BatchResponseV2):
            return parsed, _extract_usage(resp)
        return None, _extract_usage(resp)
    except Exception as exc:
        if getattr(exc, "status_code", None) == 400:
            return None, (0, 0, 0)
        return None, (0, 0, 0)


def _call_json_object_v2(client, system_prompt: str, texts: list[str]):
    def _call():
        return client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": _build_user_message(texts)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            extra_body=_NO_THINK,
        )
    resp = _call_with_retry(_call)
    raw   = resp.choices[0].message.content or "{}"
    usage = _extract_usage(resp)
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1].lstrip("json").strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        s = s[start:end + 1]
    try:
        return BatchResponseV2.model_validate_json(s), usage
    except Exception:
        return None, usage


def _label_batch_deepseek(
    client,
    system_prompt: str,
    texts: list[str],
    schema: LabelSchema,
    stats: UsageStats,
) -> list[dict[str, Any]]:
    batch, usage = _call_structured_v2(client, system_prompt, texts)
    if batch is None:
        batch, usage = _call_json_object_v2(client, system_prompt, texts)
    stats.add(*usage)

    if batch and len(batch.results) == len(texts):
        return [_normalize(r, schema) for r in batch.results]

    # Fallback: one row at a time
    results = []
    for text in texts:
        b, u = _call_json_object_v2(client, system_prompt, [text])
        stats.add(*u, fallbacks=1)
        if b and b.results:
            results.append(_normalize(b.results[0], schema))
        else:
            results.append({"task_type": schema.task_types[-1],
                            **{d: 0.0 for d in schema.complexity_dimensions}})
    return results


def label_with_deepseek(
    rows: list[dict[str, Any]],
    schema: LabelSchema,
    *,
    out_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = 1,
    limit: int | None = None,
) -> None:
    """Label rows with DeepSeek using the v2 rubric.

    workers > 1 fires that many batches concurrently — each worker gets its own
    OpenAI client instance to avoid shared-state issues.  The bottleneck is output
    generation (~400 tok/batch), not network RTT, so linear speedup holds up to
    the API's undocumented concurrency ceiling (empirically ~20 safe).
    """
    system_prompt = _build_system_prompt(schema)
    stats = UsageStats()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        for prev in _read_jsonl(out_path):
            if t := prev.get("text"):
                done.add(t)
    todo = [r for r in rows if r["text"] not in done]
    if limit:
        todo = todo[:limit]
    if done:
        print(f"  resuming: {len(done)} already labeled, {len(todo)} remaining")

    # Slice todo into fixed-size batches upfront
    batches = [todo[s:s + batch_size] for s in range(0, len(todo), batch_size)]
    total = len(todo)
    write_lock = threading.Lock()
    written = 0

    def _process_batch(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Each thread gets its own client (OpenAI client is not thread-safe for reuse)
        client = _get_deepseek_client()
        texts = [r["text"] for r in chunk]
        labels = _label_batch_deepseek(client, system_prompt, texts, schema, stats)
        recs = []
        for src_row, lbl in zip(chunk, labels):
            rec = {"text": src_row["text"], **lbl, "label_source": "teacher_deepseek_v2"}
            if src_row.get("hf_source"):
                rec["hf_source"] = src_row["hf_source"]
            elif src_row.get("source"):
                rec["hf_source"] = src_row["source"]
            recs.append(rec)
        return recs

    with out_path.open("a", encoding="utf-8") as fh, \
         ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process_batch, b): b for b in batches}
        for fut in as_completed(futures):
            recs = fut.result()
            with write_lock:
                for rec in recs:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                written += len(recs)
                if written % 200 == 0 or written >= total:
                    print(f"  labeled {written}/{total}")

    stats.print_summary()


# ---------------------------------------------------------------------------
# Reconcile two teachers + agreement report
# ---------------------------------------------------------------------------


def reconcile(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    schema: LabelSchema,
    *,
    dim_tol: float = 0.25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge two teachers' v2 labels.

    Agreement: task_type matches AND all 3 dims within dim_tol.
    Agreed rows take the mean of dims; disagreed rows go to gold queue.
    Returns (silver, gold_queue).
    """
    by_text_b = {r["text"]: r for r in rows_b}
    silver, gold = [], []

    for a in rows_a:
        b = by_text_b.get(a["text"])
        if b is None:
            silver.append({**a, "label_source": "teacher_a_only_v2"})
            continue

        same_task = task_type_id(a["task_type"]) == task_type_id(b["task_type"])
        within = all(
            abs(float(a.get(d, 0.0)) - float(b.get(d, 0.0))) <= dim_tol
            for d in schema.complexity_dimensions
        )
        if same_task and within:
            merged: dict[str, Any] = {
                "text":         a["text"],
                "task_type":    a["task_type"],
                "label_source": "teacher_agreed_v2",
            }
            for src in ("hf_source",):
                v = a.get(src) or b.get(src)
                if v:
                    merged[src] = v
            for d in schema.complexity_dimensions:
                merged[d] = (float(a.get(d, 0.0)) + float(b.get(d, 0.0))) / 2.0
            silver.append(merged)
        else:
            gold.append({
                "text":    a["text"],
                "teacher_a": a,
                "teacher_b": b,
                "reason": "task_mismatch" if not same_task else "dim_mismatch",
            })

    return silver, gold


def print_agreement_report(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    silver: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    schema: LabelSchema,
) -> None:
    total = len(silver) + len(gold)
    agree_rate = len(silver) / max(1, total) * 100

    print(f"\n{'═'*60}")
    print(f"Agreement report — v2 schema")
    print(f"{'═'*60}")
    print(f"  Total compared   : {total:,}")
    print(f"  Agreed (silver)  : {len(silver):,}  ({agree_rate:.1f}%)")
    print(f"  Disagreed (gold) : {len(gold):,}  ({100-agree_rate:.1f}%)")

    # Task-type agreement per class
    task_total:  Counter = Counter()
    task_agree:  Counter = Counter()
    by_text_b = {r["text"]: r for r in rows_b}
    for a in rows_a:
        b = by_text_b.get(a["text"])
        if b is None:
            continue
        task_total[a["task_type"]] += 1
        if task_type_id(a["task_type"]) == task_type_id(b["task_type"]):
            task_agree[a["task_type"]] += 1

    print(f"\n  Per-class task_type agreement (teacher_a view):")
    for cls in schema.task_types:
        n = task_total.get(cls, 0)
        if n == 0:
            continue
        pct = task_agree.get(cls, 0) / n * 100
        print(f"    {cls:<35} {task_agree.get(cls,0):>5}/{n:<5}  {pct:.1f}%")

    # Dim agreement (mean absolute delta)
    print(f"\n  Mean |delta| per complexity dim (over all compared rows):")
    by_text_b2 = {r["text"]: r for r in rows_b}
    dim_deltas: dict[str, list[float]] = defaultdict(list)
    for a in rows_a:
        b = by_text_b2.get(a["text"])
        if b is None:
            continue
        for d in schema.complexity_dimensions:
            dim_deltas[d].append(abs(float(a.get(d, 0.0)) - float(b.get(d, 0.0))))
    for d in schema.complexity_dimensions:
        deltas = dim_deltas.get(d, [])
        mean_d = sum(deltas) / max(1, len(deltas))
        within_tol = sum(1 for x in deltas if x <= 0.25) / max(1, len(deltas)) * 100
        print(f"    {d:<25} mean_delta={mean_d:.3f}  within_0.25={within_tol:.1f}%")

    # Disagreement breakdown
    task_mismatch = sum(1 for r in gold if r.get("reason") == "task_mismatch")
    dim_mismatch  = sum(1 for r in gold if r.get("reason") == "dim_mismatch")
    print(f"\n  Disagreement breakdown:")
    print(f"    task_mismatch : {task_mismatch:,}")
    print(f"    dim_mismatch  : {dim_mismatch:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Label Vietnamese prompts with v2 schema")
    ap.add_argument("--mode", choices=["anthropic", "deepseek", "compare"], required=True)

    # anthropic / deepseek modes
    ap.add_argument("--in",   dest="inp",  help="input JSONL (v1.5 or raw)")
    ap.add_argument("--out",  help="output labeled JSONL")
    ap.add_argument("--limit", type=int, default=None, help="cap rows (smoke test)")
    ap.add_argument("--model", default=None, help="model override (anthropic mode)")
    ap.add_argument("--rpm",     type=float, default=90.0,  help="max req/min (anthropic)")
    ap.add_argument("--workers", type=int,   default=1,
                    help="concurrent API calls (anthropic: default 8; deepseek: default 1, safe up to ~20)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="rows per request (deepseek)")

    # compare mode
    ap.add_argument("--teacher-a",  help="labeled JSONL from teacher A (compare mode)")
    ap.add_argument("--teacher-b",  help="labeled JSONL from teacher B (compare mode)")
    ap.add_argument("--silver-out", help="agreed rows output (compare mode)")
    ap.add_argument("--gold-out",   help="disagreed rows output (compare mode)")
    ap.add_argument("--dim-tol", type=float, default=0.25, help="dim agreement tolerance")

    args = ap.parse_args()
    schema = load_label_schema(version="v2")

    if args.mode in ("anthropic", "deepseek"):
        if not args.inp:
            ap.error("--in is required for anthropic/deepseek mode")
        if not args.out:
            ap.error("--out is required for anthropic/deepseek mode")

        rows = list(_read_jsonl(Path(args.inp)))
        if args.limit:
            rows = rows[: args.limit]
        print(f"Loaded {len(rows)} rows from {args.inp}")

        if args.mode == "anthropic":
            n = label_with_anthropic(
                rows, schema,
                model=args.model or "kr/claude-sonnet-4.6",
                out_path=Path(args.out),
                rpm=args.rpm,
                workers=args.workers if args.workers != 1 else 8,
            )
            print(f"Done. {n} rows written to {args.out}")

        else:  # deepseek
            label_with_deepseek(
                rows, schema,
                out_path=Path(args.out),
                batch_size=args.batch_size,
                workers=args.workers,
                limit=args.limit,
            )
            print(f"Done. Output: {args.out}")

    elif args.mode == "compare":
        for flag, name in [("--teacher-a", args.teacher_a), ("--teacher-b", args.teacher_b)]:
            if not name:
                ap.error(f"{flag} is required for compare mode")

        rows_a = list(_read_jsonl(Path(args.teacher_a)))
        rows_b = list(_read_jsonl(Path(args.teacher_b)))
        print(f"Teacher A: {len(rows_a):,} rows | Teacher B: {len(rows_b):,} rows")

        silver, gold = reconcile(rows_a, rows_b, schema, dim_tol=args.dim_tol)
        print_agreement_report(rows_a, rows_b, silver, gold, schema)

        if args.silver_out:
            Path(args.silver_out).parent.mkdir(parents=True, exist_ok=True)
            with Path(args.silver_out).open("w", encoding="utf-8") as fh:
                for r in silver:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"\nSilver written → {args.silver_out}  ({len(silver):,} rows)")

        if args.gold_out:
            Path(args.gold_out).parent.mkdir(parents=True, exist_ok=True)
            with Path(args.gold_out).open("w", encoding="utf-8") as fh:
                for r in gold:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Gold queue written → {args.gold_out}  ({len(gold):,} rows)")


if __name__ == "__main__":
    main()
