"""Synthetic Vietnamese prompt generation (plan §3.3).

Produces Vietnamese prompts deliberately spanning all 11 task types x
low/med/high complexity, so the training set isn't skewed toward whatever the
crawl happened to contain.

Two modes:

- **template** (default, no deps, no network): expands hand-written Vietnamese
  templates per task type with complexity-graded fillers. Deterministic, free,
  and enough to produce a runnable smoke dataset on any machine (including the
  Kaggle build step before training).
- **llm**: calls an OpenAI-compatible chat endpoint (set ``--mode llm`` plus the
  ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` env vars) prompted with the label
  schema's Vietnamese task glosses to generate richer, more natural prompts.

Each emitted row carries silver complexity labels derived from the complexity
tier it was generated for, so the output feeds straight into
``data.build_dataset`` as silver training data.

    python -m data.synth_gen --out data/synthetic/synth.jsonl --per-cell 30
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from config import LabelSchema, load_label_schema, task_type_id

# ---------------------------------------------------------------------------
# Complexity tiers -> silver dimension labels
# ---------------------------------------------------------------------------
# Per tier, a base value for each complexity dimension in [0,1]. The template
# generator stamps these onto every row it makes for that tier; the LLM mode
# uses them as the target it asks the model to write to.

TIERS: dict[str, dict[str, float]] = {
    "low": {
        "creativity_scope": 0.15,
        "reasoning": 0.15,
        "contextual_knowledge": 0.10,
        "domain_knowledge": 0.15,
        "constraint_ct": 0.10,
        "number_of_few_shots": 0.0,
    },
    "med": {
        "creativity_scope": 0.45,
        "reasoning": 0.50,
        "contextual_knowledge": 0.40,
        "domain_knowledge": 0.45,
        "constraint_ct": 0.40,
        "number_of_few_shots": 0.20,
    },
    "high": {
        "creativity_scope": 0.80,
        "reasoning": 0.85,
        "contextual_knowledge": 0.75,
        "domain_knowledge": 0.80,
        "constraint_ct": 0.75,
        "number_of_few_shots": 0.50,
    },
}

# Topic fillers — mixed VN with some EN/technical terms (matches the ~70/30 mix).
TOPICS = [
    "trí tuệ nhân tạo",
    "biến đổi khí hậu",
    "lịch sử Việt Nam",
    "kinh tế vĩ mô",
    "machine learning",
    "an toàn thông tin",
    "sức khỏe cộng đồng",
    "giáo dục trực tuyến",
    "phát triển web",
    "cơ sở dữ liệu phân tán",
    "du lịch bền vững",
    "năng lượng tái tạo",
]

# Per task type: templates keyed by complexity tier. ``{topic}`` is filled in.
# Higher tiers add constraints / reasoning / few-shot framing so the silver
# complexity label matches the surface form.
TEMPLATES: dict[str, dict[str, list[str]]] = {
    "open_qa": {
        "low": ["{topic} là gì?", "Giải thích ngắn gọn về {topic}."],
        "med": [
            "Phân tích những lợi ích và rủi ro chính của {topic} trong bối cảnh hiện nay.",
            "Tại sao {topic} lại quan trọng? Hãy nêu ít nhất ba lý do.",
        ],
        "high": [
            "Đánh giá toàn diện tác động của {topic} đến xã hội, kinh tế và môi "
            "trường trong 20 năm tới, có lập luận và dẫn chứng cho từng khía cạnh.",
        ],
    },
    "closed_qa": {
        "low": [
            "Dựa vào đoạn văn sau, thủ đô được nhắc đến là gì? '{topic} bắt nguồn từ Hà Nội.'",
        ],
        "med": [
            "Đọc đoạn văn về {topic} dưới đây và trả lời: sự kiện nào xảy ra đầu "
            "tiên và vì sao? (đoạn văn: ...)",
        ],
        "high": [
            "Cho ba đoạn tài liệu về {topic}. Chỉ dựa trên các đoạn này, hãy xác "
            "định mâu thuẫn giữa chúng và kết luận thông tin nào đáng tin cậy hơn.",
        ],
    },
    "summarization": {
        "low": ["Tóm tắt đoạn văn sau về {topic} trong một câu."],
        "med": [
            "Tóm tắt bài viết về {topic} thành 3 ý chính, giữ nguyên các số liệu quan trọng.",
        ],
        "high": [
            "Tóm tắt báo cáo dài về {topic} thành đúng 5 gạch đầu dòng, mỗi dòng "
            "tối đa 15 từ, không được bỏ sót kết luận và khuyến nghị.",
        ],
    },
    "text_generation": {
        "low": ["Viết một câu khẩu hiệu về {topic}."],
        "med": [
            "Viết một bài đăng mạng xã hội khoảng 100 từ về {topic} với giọng văn tích cực.",
        ],
        "high": [
            "Viết một bài luận 800 từ về {topic} theo cấu trúc mở-thân-kết, có "
            "ít nhất hai dẫn chứng cụ thể và một phản biện, giọng văn trang trọng.",
        ],
    },
    "code_generation": {
        "low": ["Viết hàm Python in ra 'Hello {topic}'."],
        "med": [
            "Viết một hàm Python nhận danh sách số và trả về trung bình, có xử lý "
            "trường hợp danh sách rỗng. Chủ đề ví dụ: {topic}.",
        ],
        "high": [
            "Thiết kế một REST API bằng FastAPI cho hệ thống {topic}: gồm CRUD, "
            "xác thực JWT, phân trang và viết unit test. Trả về code đầy đủ kèm giải thích.",
        ],
    },
    "chatbot": {
        "low": ["Chào bạn, hôm nay thế nào?", "Bạn có thể nói chuyện về {topic} không?"],
        "med": [
            "Mình đang băn khoăn về {topic}, bạn tư vấn giúp mình với nhé?",
        ],
        "high": [
            "Mình muốn trò chuyện sâu về {topic}: hãy đóng vai chuyên gia, hỏi lại "
            "mình vài câu để hiểu nhu cầu rồi mới đưa lời khuyên phù hợp.",
        ],
    },
    "classification": {
        "low": ["Phân loại câu sau là tích cực hay tiêu cực: 'Tôi rất thích {topic}.'"],
        "med": [
            "Phân loại đoạn đánh giá về {topic} vào một trong các nhãn: tích cực, "
            "trung lập, tiêu cực, và giải thích ngắn gọn.",
        ],
        "high": [
            "Xây dựng tiêu chí phân loại đa nhãn cho các văn bản về {topic} theo "
            "chủ đề, sắc thái và độ tin cậy, rồi áp dụng cho 5 ví dụ.",
        ],
    },
    "rewrite": {
        "low": ["Viết lại câu sau cho hay hơn: 'Cái {topic} này tốt.'"],
        "med": [
            "Viết lại đoạn văn về {topic} sang giọng văn trang trọng, giữ nguyên ý.",
        ],
        "high": [
            "Viết lại bài viết về {topic} cho đối tượng học sinh cấp 2: đơn giản hóa "
            "thuật ngữ, thêm ví dụ minh họa, giữ độ dài tương đương và không sai lệch nội dung.",
        ],
    },
    "brainstorming": {
        "low": ["Gợi ý vài ý tưởng về {topic}."],
        "med": [
            "Đưa ra 10 ý tưởng sáng tạo liên quan đến {topic}, mỗi ý một dòng.",
        ],
        "high": [
            "Động não 15 ý tưởng đột phá về {topic}, nhóm chúng theo tính khả thi "
            "và chi phí, rồi đề xuất 3 ý tưởng đáng theo đuổi nhất kèm lý do.",
        ],
    },
    "extraction": {
        "low": ["Trích xuất tên riêng trong câu: 'Công ty {topic} ở Hà Nội.'"],
        "med": [
            "Trích xuất tất cả ngày tháng và số tiền trong đoạn văn về {topic}.",
        ],
        "high": [
            "Trích xuất các thực thể (người, tổ chức, địa điểm, thời gian) từ tài "
            "liệu về {topic} và trả về dưới dạng JSON đúng định dạng schema cho trước.",
        ],
    },
    "other": {
        "low": ["{topic}.", "Ừm, {topic} à?"],
        "med": ["Mình không chắc nên hỏi gì về {topic}, bạn nghĩ sao?"],
        "high": [
            "Đây là một yêu cầu hỗn hợp về {topic} không thuộc loại rõ ràng nào: "
            "vừa cần phân tích vừa cần sáng tác vừa cần trích dẫn.",
        ],
    },
}

FEWSHOT_PREFIX = (
    "Ví dụ 1: ... -> ...\nVí dụ 2: ... -> ...\nVí dụ 3: ... -> ...\nBây giờ: "
)


def _gen_template(
    schema: LabelSchema, per_cell: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for tid in schema.task_type_ids:
        cell = TEMPLATES.get(tid, TEMPLATES["other"])
        for tier, dims in TIERS.items():
            templates = cell.get(tier) or cell.get("med") or next(iter(cell.values()))
            for _ in range(per_cell):
                tmpl = rng.choice(templates)
                topic = rng.choice(TOPICS)
                text = tmpl.format(topic=topic)
                row_dims = dict(dims)
                # high-tier rows sometimes carry an explicit few-shot block
                if tier == "high" and rng.random() < 0.4:
                    text = FEWSHOT_PREFIX + text
                    row_dims["number_of_few_shots"] = 0.6
                rows.append(
                    {
                        "text": text,
                        "task_type": tid,
                        **row_dims,
                        "source": f"synth_template:{tier}",
                    }
                )
    rng.shuffle(rows)
    return rows


def _gen_llm(
    schema: LabelSchema, per_cell: int, seed: int, model: str
) -> list[dict[str, Any]]:
    """Generate via an OpenAI-compatible chat endpoint.

    Requires ``openai`` installed and ``OPENAI_API_KEY`` set. ``OPENAI_BASE_URL``
    may point at any compatible gateway (so this works against Viettel's own
    endpoint later). Falls back to raising a clear error if the dep/key is absent.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError(
            "llm mode needs `pip install openai` and OPENAI_API_KEY set"
        ) from exc

    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for tid in schema.task_type_ids:
        display = next(
            (t for t in schema.task_types if task_type_id(t) == tid), tid
        )
        gloss = schema.task_types_vi.get(display, display)
        for tier, dims in TIERS.items():
            sys = (
                "Bạn là người tạo dữ liệu huấn luyện. Hãy viết các prompt tiếng "
                "Việt tự nhiên, đa dạng."
            )
            user = (
                f"Viết {per_cell} prompt tiếng Việt khác nhau thuộc loại nhiệm vụ "
                f"'{display}' ({gloss}), ở mức độ phức tạp '{tier}'. "
                "Mỗi prompt một dòng, không đánh số."
            )
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.9,
                )
                lines = [
                    ln.strip(" -•\t")
                    for ln in (resp.choices[0].message.content or "").splitlines()
                    if ln.strip()
                ]
            except Exception as exc:  # pragma: no cover - network path
                print(f"  LLM call failed for {tid}/{tier}: {exc}; skipping cell")
                continue
            for text in lines:
                rows.append(
                    {
                        "text": text,
                        "task_type": tid,
                        **dims,
                        "source": f"synth_llm:{tier}",
                    }
                )
    rng.shuffle(rows)
    return rows


def generate(
    out_path: str | Path,
    *,
    per_cell: int = 30,
    seed: int = 7,
    mode: str = "template",
    model: str = "gpt-4o-mini",
) -> int:
    schema = load_label_schema()
    if mode == "llm":
        rows = _gen_llm(schema, per_cell, seed, model)
    else:
        rows = _gen_template(schema, per_cell, seed)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic VN prompts")
    ap.add_argument("--out", default="data/synthetic/synth.jsonl")
    ap.add_argument("--per-cell", type=int, default=30, help="rows per task x tier")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mode", choices=["template", "llm"], default="template")
    ap.add_argument("--model", default="gpt-4o-mini", help="LLM model id (llm mode)")
    args = ap.parse_args()

    n = generate(
        args.out,
        per_cell=args.per_cell,
        seed=args.seed,
        mode=args.mode,
        model=args.model,
    )
    print(f"wrote {n} synthetic rows to {args.out}")


if __name__ == "__main__":
    main()
