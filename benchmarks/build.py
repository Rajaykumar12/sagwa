"""Builds Sagwa's real golden sets and calibration labels from public
benchmarks. Deterministic (fixed seed), so re-running reproduces the same
files byte for byte.

    python -m benchmarks.build

Writes:
- golden_sets/hotpotqa.jsonl      50 HotpotQA (distractor, validation) rag_qa cases,
                                   25 bridge + 25 comparison
- benchmarks/hotpotqa_corpus.jsonl the pooled paragraphs of those 50 questions
                                   (500 paragraphs) — the RAG app's knowledge base
- golden_sets/cnndm.jsonl         30 CNN/DailyMail (3.0.0, test) summarization cases
- calibration/helpsteer2_human_labels.jsonl
                                   200 HelpSteer2 (validation) prompt/response pairs
                                   with human helpfulness ratings, 100 pass / 100 fail
"""
import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
SEED = 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows -> {path.relative_to(ROOT)}")


def build_hotpotqa(per_type: int = 25) -> None:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").shuffle(seed=SEED)
    cases, corpus = [], []
    for qtype in ("bridge", "comparison"):
        for row in ds.filter(lambda r: r["type"] == qtype).select(range(per_type)):
            cases.append({
                "id": f"hotpotqa-{row['id']}",
                "input": row["question"],
                "expected_output": row["answer"],
                "task_type": "rag_qa",
                "tags": [row["type"], row["level"]],
            })
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
                corpus.append({"title": title, "text": "".join(sentences).strip()})
    _write_jsonl(ROOT / "golden_sets/hotpotqa.jsonl", cases)
    _write_jsonl(ROOT / "benchmarks/hotpotqa_corpus.jsonl", corpus)


def build_cnndm(n: int = 30, max_words: int = 600) -> None:
    # Short articles only: Groq's free tier caps each model at 8K tokens/min.
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test").shuffle(seed=SEED)
    ds = ds.filter(lambda r: len(r["article"].split()) <= max_words).select(range(n))
    cases = [
        {
            "id": f"cnndm-{row['id']}",
            "input": row["article"],
            "expected_output": row["highlights"],
            "task_type": "summarization",
            "tags": ["news"],
        }
        for row in ds
    ]
    _write_jsonl(ROOT / "golden_sets/cnndm.jsonl", cases)


def build_helpsteer2(per_class: int = 100, max_chars: int = 2000) -> None:
    # Single-turn only (multi-turn prompts contain <extra_id_1> turn markers).
    # Human label: helpfulness >= 3 on HelpSteer2's 0-4 scale counts as pass.
    ds = load_dataset("nvidia/HelpSteer2", split="validation").shuffle(seed=SEED)
    ds = ds.filter(
        lambda r: "<extra_id_1>" not in r["prompt"] and len(r["prompt"]) + len(r["response"]) <= max_chars
    )
    rows = []
    for label, keep in (("pass", lambda h: h >= 3), ("fail", lambda h: h < 3)):
        picked = ds.filter(lambda r: keep(int(r["helpfulness"]))).select(range(per_class))
        rows += [
            {
                "id": f"helpsteer2-{label}-{i:03d}",
                "prompt": r["prompt"],
                "response": r["response"],
                "helpfulness": int(r["helpfulness"]),
                "human_label": label,
            }
            for i, r in enumerate(picked)
        ]
    _write_jsonl(ROOT / "calibration/helpsteer2_human_labels.jsonl", rows)


if __name__ == "__main__":
    build_hotpotqa()
    build_cnndm()
    build_helpsteer2()
