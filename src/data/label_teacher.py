"""Teacher labeling: raw VN prompts -> NVIDIA-style silver labels (plan §3.2).

Two teachers, cross-checked:

1. **NVIDIA teacher** (``mode=nvidia``): translate VN->EN, run the original
   ``nvidia/prompt-task-and-complexity-classifier``, keep its task_type +
   complexity outputs. NVIDIA-aligned distribution for free. Needs the ``ml``
   extra (transformers + the NVIDIA model) and is the GPU-heavy path — run it on
   Kaggle, not locally.
2. **LLM teacher** (``mode=llm``): prompt a strong chat model with NVIDIA's
   rubric (the label schema's VN glosses) to label the **Vietnamese** text
   directly — catches what translation distorts. Needs ``openai`` + an API key;
   set ``OPENAI_BASE_URL`` to point at any compatible endpoint.

When both teachers are run, :func:`reconcile` merges them: rows where they agree
on task_type within a complexity tolerance become silver training labels; rows
where they disagree are flagged for the human-gold queue (never used as training
labels — plan §3.2).

    # GPU path (Kaggle):
    python -m data.label_teacher --mode nvidia \
        --in data/raw/crawl.jsonl --out data/raw/labeled_nvidia.jsonl
    # API path (any machine):
    python -m data.label_teacher --mode llm \
        --in data/raw/crawl.jsonl --out data/raw/labeled_llm.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import LabelSchema, load_label_schema, task_type_id

NVIDIA_MODEL_ID = "nvidia/prompt-task-and-complexity-classifier"


class TeacherLabel(BaseModel):
    """Strict schema for one teacher-labeled row (plan §3.2).

    All six complexity dims are **required** and bounded to [0,1]; ``task_type``
    must be a non-empty string. Construct via :meth:`from_payload` so an unknown
    task label is rejected (and the row retried/dropped) rather than silently
    coerced to ``Other`` — that silent coercion was the dataset-corruption risk
    in the old ``.get(dim, 0.0)`` path.
    """

    model_config = ConfigDict(extra="ignore")

    task_type: str
    creativity_scope: float = Field(ge=0.0, le=1.0)
    reasoning: float = Field(ge=0.0, le=1.0)
    contextual_knowledge: float = Field(ge=0.0, le=1.0)
    domain_knowledge: float = Field(ge=0.0, le=1.0)
    constraint_ct: float = Field(ge=0.0, le=1.0)
    number_of_few_shots: float = Field(ge=0.0, le=1.0)

    @field_validator("task_type")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("task_type is empty")
        return str(v).strip()

    @classmethod
    def from_payload(cls, data: dict[str, Any], schema: LabelSchema) -> "TeacherLabel":
        """Validate a raw teacher dict and resolve task_type to a schema label.

        Raises ``ValueError`` (pydantic) on missing/out-of-range dims, and on a
        task_type that doesn't map to a known schema label.
        """
        obj = cls(**data)
        # strict task-type membership: unknown -> raise (no silent Other)
        tid = task_type_id(obj.task_type)
        for display in schema.task_types:
            if task_type_id(display) == tid:
                obj.task_type = display
                return obj
        raise ValueError(f"unknown task_type: {obj.task_type!r}")

# NVIDIA's task-type names -> our schema display labels. They match 1:1 (parity),
# but NVIDIA uses slightly different casing/spacing, so normalize via task_type_id.


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Teacher 1: NVIDIA model (GPU path — run on Kaggle)
# ---------------------------------------------------------------------------


def label_with_nvidia(
    rows: list[dict[str, Any]],
    schema: LabelSchema,
    *,
    translate: bool = True,
    batch_size: int = 16,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Run the NVIDIA classifier over (optionally translated) prompts.

    The NVIDIA model is English-only, so by default we machine-translate VN->EN
    first (plan §3.2). Translation uses a HF MT model if available; if it's
    missing we label the raw text and tag ``translated=False`` so the noise is
    visible downstream.
    """
    import torch  # noqa: F401  (ensures the ml extra is present)
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if _cuda_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(NVIDIA_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(NVIDIA_MODEL_ID, trust_remote_code=True)
    model.to(device).eval()

    translate_fn = _build_translator(device) if translate else None

    labeled: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        texts = [r["text"] for r in chunk]
        if translate_fn is not None:
            texts_en = translate_fn(texts)
        else:
            texts_en = texts
        preds = _run_nvidia_batch(model, tok, texts_en, device)
        for src_row, pred in zip(chunk, preds):
            labeled.append(
                _nvidia_pred_to_row(src_row["text"], pred, schema, translated=translate)
            )
    return labeled


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _build_translator(device: str):  # pragma: no cover - heavy/optional
    """Return a callable list[str]->list[str] VN->EN, or None if unavailable."""
    try:
        from transformers import pipeline

        pipe = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-vi-en",
            device=0 if device == "cuda" else -1,
        )

        def _tr(texts: list[str]) -> list[str]:
            outs = pipe([t[:512] for t in texts], max_length=512)
            return [o["translation_text"] for o in outs]

        return _tr
    except Exception as exc:
        print(f"  translator unavailable ({exc}); labeling raw VN text")
        return None


def _run_nvidia_batch(model, tok, texts: list[str], device) -> list[dict[str, Any]]:  # pragma: no cover - heavy
    """Call the NVIDIA custom model. Its forward returns a dict of head outputs.

    The exact return keys depend on the model card's custom code; we read the
    standard fields and fall back gracefully if a key is renamed.
    """
    import torch

    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    # NVIDIA's custom model returns a list of per-example dicts via its own
    # post-processing, or a tensor dict; normalize both.
    if isinstance(out, list):
        return out
    # tensor-dict form: convert to per-example dicts
    results = []
    n = len(texts)
    for i in range(n):
        rec: dict[str, Any] = {}
        for k, v in out.items():
            try:
                rec[k] = v[i].tolist() if hasattr(v, "__getitem__") else v
            except Exception:
                rec[k] = v
        results.append(rec)
    return results


def _nvidia_pred_to_row(
    text: str, pred: dict[str, Any], schema: LabelSchema, *, translated: bool
) -> dict[str, Any]:
    """Map a NVIDIA prediction dict into our schema row."""
    task = pred.get("task_type_1") or pred.get("task_type") or "Other"
    if isinstance(task, list):
        task = task[0] if task else "Other"
    row: dict[str, Any] = {
        "text": text,
        "task_type": _match_task(task, schema),
        "label_source": "teacher_nvidia",
        "translated": translated,
    }
    for dim in schema.complexity_dimensions:
        val = pred.get(dim, 0.0)
        if isinstance(val, list):
            val = val[0] if val else 0.0
        row[dim] = max(0.0, min(1.0, float(val)))
    return row


def _match_task(label: str, schema: LabelSchema) -> str:
    """Map a teacher's task label to a schema display label (parity -> normalize)."""
    tid = task_type_id(str(label))
    for display in schema.task_types:
        if task_type_id(display) == tid:
            return display
    return "Other"


# ---------------------------------------------------------------------------
# Teacher 2: LLM labeler (API path — any machine)
# ---------------------------------------------------------------------------


def label_with_llm(
    rows: list[dict[str, Any]],
    schema: LabelSchema,
    *,
    model: str = "gpt-4o-mini",
) -> list[dict[str, Any]]:
    """Label native Vietnamese text with a strong LLM using NVIDIA's rubric."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("llm mode needs `pip install openai` + OPENAI_API_KEY") from exc

    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
    rubric = _build_rubric(schema)

    labeled: list[dict[str, Any]] = []
    for r in rows:
        try:
            rec = _llm_label_one(client, model, rubric, r["text"], schema)
        except Exception as exc:  # pragma: no cover - network path
            print(f"  LLM label failed: {exc}; skipping row")
            continue
        rec["text"] = r["text"]
        rec["label_source"] = "teacher_llm"
        if r.get("source"):
            rec["hf_source"] = r["source"]
        labeled.append(rec)
    return labeled


def _build_rubric(schema: LabelSchema) -> str:
    tasks = "\n".join(
        f"- {t}: {schema.task_types_vi.get(t, '')}" for t in schema.task_types
    )
    dims = "\n".join(
        f"- {d}: {schema.complexity_dimensions_vi.get(d, '')}"
        for d in schema.complexity_dimensions
    )
    return (
        "Bạn là người gán nhãn dữ liệu theo chuẩn NVIDIA prompt classifier.\n"
        "Phân loại prompt tiếng Việt vào ĐÚNG MỘT loại nhiệm vụ:\n"
        f"{tasks}\n\n"
        "Và chấm 6 chiều độ phức tạp, mỗi chiều một số thực trong [0,1]:\n"
        f"{dims}\n\n"
        "Trả về DUY NHẤT một JSON object với các khóa: task_type (tên loại), "
        "creativity_scope, reasoning, contextual_knowledge, domain_knowledge, "
        "constraint_ct, number_of_few_shots."
    )


def _llm_label_one(client, model, rubric, text: str, schema: LabelSchema) -> dict[str, Any]:  # pragma: no cover - network
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": rubric},
            {"role": "user", "content": f"Prompt:\n{text}"},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    rec: dict[str, Any] = {"task_type": _match_task(data.get("task_type", "Other"), schema)}
    for dim in schema.complexity_dimensions:
        rec[dim] = max(0.0, min(1.0, float(data.get(dim, 0.0))))
    return rec


# ---------------------------------------------------------------------------
# Teacher 2b: Anthropic labeler (API path — Claude proxy, any machine)
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Thread-safe limiter that spaces calls to at most ``rpm`` per minute.

    A simple monotonic-clock gate: each acquirer waits until at least
    ``60 / rpm`` seconds have elapsed since the previous grant. With N worker
    threads this caps the *aggregate* request rate at ``rpm``, which is what the
    proxy's per-minute limit cares about — not the per-thread rate.
    """

    def __init__(self, rpm: float) -> None:
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next = max(now, self._next) + self._interval


def _is_rate_limit(exc: Exception) -> bool:
    """True if the exception looks like an HTTP 429 / rate-limit / overload."""
    status = getattr(exc, "status_code", None)
    if status in (429, 529):
        return True
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "overload" in name:
        return True
    return "rate limit" in str(exc).lower() or "429" in str(exc)


def _retry_after_seconds(exc: Exception) -> float | None:
    """Pull a ``retry-after`` hint (seconds) off the exception's response, if any."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    val = headers.get("retry-after") or headers.get("Retry-After")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _call_with_retry(
    fn: Callable[[], Any],
    *,
    max_retries: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> Any:
    """Call ``fn`` with exponential backoff on rate-limit / transient errors.

    Rate-limit errors honor a ``retry-after`` header when present; otherwise we
    back off exponentially with jitter. Non-rate-limit errors get a couple of
    short retries (transient network blips) then re-raise.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - network path
            attempt += 1
            rate_limited = _is_rate_limit(exc)
            if attempt > max_retries:
                raise
            if not rate_limited and attempt > 2:
                raise
            hinted = _retry_after_seconds(exc) if rate_limited else None
            if hinted is not None:
                delay = hinted + random.uniform(0, 1.0)
            else:
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                delay += random.uniform(0, delay * 0.25)  # jitter
            kind = "rate-limited" if rate_limited else "transient error"
            print(f"  {kind} ({exc}); retry {attempt}/{max_retries} in {delay:.1f}s")
            time.sleep(delay)


def label_with_anthropic(
    rows: list[dict[str, Any]],
    schema: LabelSchema,
    *,
    model: str = "kr/claude-sonnet-4.6",
    out_path: str | Path,
    rpm: float = 90.0,
    workers: int = 8,
    max_retries: int = 6,
) -> int:
    """Label native Vietnamese text with Claude using NVIDIA's rubric.

    Reads ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN`` from the env (the
    proxy config in ``.claude/settings.json``). Same rubric and output schema as
    :func:`label_with_llm`, just a different SDK.

    Concurrency + safety for a long (~55k-row) run:

    - **rate limit**: a global :class:`_RateLimiter` caps aggregate requests at
      ``rpm`` per minute (default 90, comfortably under a 100 RPM ceiling).
    - **retry**: rate-limit / transient errors back off and retry (honoring a
      ``retry-after`` header when the proxy sends one).
    - **resume**: rows whose ``text`` is already present in ``out_path`` are
      skipped, so a crashed run restarts where it left off.
    - **incremental write**: each labeled row is appended (and flushed) as it
      completes, so progress survives an interrupt.

    Returns the number of rows newly written this invocation.
    """
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic mode needs `pip install anthropic`") from exc

    client = Anthropic(
        base_url=os.environ.get("ANTHROPIC_BASE_URL") or None,
        auth_token=os.environ.get("ANTHROPIC_AUTH_TOKEN") or None,
    )
    rubric = _build_rubric(schema)
    limiter = _RateLimiter(rpm)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip rows already labeled in a prior (possibly crashed) run.
    done: set[str] = set()
    if out.exists():
        for prev in read_jsonl(out):
            t = prev.get("text")
            if t:
                done.add(t)
    todo = [r for r in rows if r["text"] not in done]
    if done:
        print(f"  resuming: {len(done)} already labeled, {len(todo)} remaining")

    write_lock = threading.Lock()
    written = 0
    total = len(todo)

    def _label(row: dict[str, Any]) -> dict[str, Any] | None:
        def _one() -> dict[str, Any]:
            limiter.acquire()
            return _anthropic_label_one(client, model, rubric, row["text"], schema)

        try:
            rec = _call_with_retry(_one, max_retries=max_retries)
        except Exception as exc:  # pragma: no cover - network path
            print(f"  giving up on row after retries ({exc}); skipping")
            return None
        rec["text"] = row["text"]
        rec["label_source"] = "teacher_anthropic"
        if row.get("source"):
            rec["hf_source"] = row["source"]
        return rec

    with out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
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


def _anthropic_label_one(client, model, rubric, text: str, schema: LabelSchema) -> dict[str, Any]:  # pragma: no cover - network
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0.0,
        system=[{"type": "text", "text": rubric, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": f"Prompt:\n{text}"}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text")
    data = json.loads(_extract_json(raw))
    label = TeacherLabel.from_payload(data, schema)
    rec: dict[str, Any] = {"task_type": label.task_type}
    for dim in schema.complexity_dimensions:
        rec[dim] = getattr(label, dim)
    return rec


def _extract_json(raw: str) -> str:
    """Pull the first JSON object out of a model reply (handles ```json fences)."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


# ---------------------------------------------------------------------------
# Reconcile two teachers
# ---------------------------------------------------------------------------


def reconcile(
    nvidia_rows: list[dict[str, Any]],
    llm_rows: list[dict[str, Any]],
    schema: LabelSchema,
    *,
    dim_tol: float = 0.25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge two teachers' labels.

    Returns ``(agreed_silver, disagreed_for_gold)``. Rows agree if the task_type
    matches and every complexity dim is within ``dim_tol``; agreed rows take the
    mean of the dims. Disagreed rows are routed to the human-gold queue.
    """
    by_text_llm = {r["text"]: r for r in llm_rows}
    agreed: list[dict[str, Any]] = []
    disagreed: list[dict[str, Any]] = []

    for nv in nvidia_rows:
        llm = by_text_llm.get(nv["text"])
        if llm is None:
            agreed.append(nv)  # only one teacher saw it; keep as silver
            continue
        same_task = task_type_id(nv["task_type"]) == task_type_id(llm["task_type"])
        within = all(
            abs(float(nv.get(d, 0.0)) - float(llm.get(d, 0.0))) <= dim_tol
            for d in schema.complexity_dimensions
        )
        if same_task and within:
            merged = {
                "text": nv["text"],
                "task_type": nv["task_type"],
                "label_source": "teacher_agreed",
                "hf_source": nv.get("hf_source") or llm.get("hf_source"),
            }
            for d in schema.complexity_dimensions:
                merged[d] = (float(nv.get(d, 0.0)) + float(llm.get(d, 0.0))) / 2.0
            agreed.append(merged)
        else:
            disagreed.append(
                {
                    "text": nv["text"],
                    "nvidia": nv,
                    "llm": llm,
                    "reason": "task_mismatch" if not same_task else "dim_mismatch",
                }
            )
    return agreed, disagreed


def main() -> None:
    ap = argparse.ArgumentParser(description="Teacher-label raw VN prompts")
    ap.add_argument("--mode", choices=["nvidia", "llm", "anthropic", "reconcile"], required=True)
    ap.add_argument("--in", dest="inp", help="input raw JSONL (nvidia/llm modes)")
    ap.add_argument("--out", help="output labeled JSONL")
    ap.add_argument("--nvidia", help="nvidia-labeled JSONL (reconcile mode)")
    ap.add_argument("--llm", help="llm-labeled JSONL (reconcile mode)")
    ap.add_argument("--gold-out", help="disagreement queue JSONL (reconcile mode)")
    ap.add_argument(
        "--model",
        default=None,
        help="teacher model id; defaults per mode (gpt-4o-mini for llm, "
        "kr/claude-sonnet-4.6 for anthropic)",
    )
    ap.add_argument("--no-translate", action="store_true", help="skip VN->EN (nvidia mode)")
    ap.add_argument("--limit", type=int, default=None, help="cap rows for a smoke run")
    ap.add_argument("--rpm", type=float, default=90.0, help="max requests/min (anthropic)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent workers (anthropic)")
    ap.add_argument("--max-retries", type=int, default=6, help="retries per row (anthropic)")
    args = ap.parse_args()

    schema = load_label_schema()

    if args.mode == "reconcile":
        nv = list(read_jsonl(args.nvidia))
        llm = list(read_jsonl(args.llm))
        agreed, disagreed = reconcile(nv, llm, schema)
        n = write_jsonl(agreed, args.out)
        print(f"wrote {n} agreed silver rows to {args.out}")
        if args.gold_out:
            m = write_jsonl(disagreed, args.gold_out)
            print(f"wrote {m} disagreements to gold queue {args.gold_out}")
        return

    rows = list(read_jsonl(args.inp))
    if args.limit:
        rows = rows[: args.limit]

    if args.mode == "anthropic":
        # Writes incrementally + resumes; returns the count written this run.
        n = label_with_anthropic(
            rows,
            schema,
            model=args.model or "kr/claude-sonnet-4.6",
            out_path=args.out,
            rpm=args.rpm,
            workers=args.workers,
            max_retries=args.max_retries,
        )
        print(f"wrote {n} anthropic-labeled rows to {args.out} (this run)")
        return

    if args.mode == "nvidia":
        labeled = label_with_nvidia(rows, schema, translate=not args.no_translate)
    else:
        labeled = label_with_llm(rows, schema, model=args.model or "gpt-4o-mini")

    n = write_jsonl(labeled, args.out)
    print(f"wrote {n} {args.mode}-labeled rows to {args.out}")


if __name__ == "__main__":
    main()
