"""Target pipelines built on public benchmarks, wrapped as Sagwa adapters.

- HotpotV1: RAG-QA over the pooled HotpotQA paragraphs (dense retrieval, top_k=3,
  grounded short-answer prompt). The baseline.
- HotpotV2: a plausible "cost-cutting" PR — retrieve 1 paragraph instead of
  3 and drop the "say I don't know" instruction. HotpotQA questions are
  multi-hop (the answer needs 2 paragraphs), so this should regress, most of
  all on `bridge` questions.
- CnnDmSummarizer: news summarization, the second target type (PRD FR-3c).

Target apps call `openai/gpt-oss-120b`, not the judge's `gpt-oss-20b`:
Groq's free tier rate-limits per model, so this keeps the target's calls
from eating the judge/RAGAS quota.
"""
import json
import os
import time
from pathlib import Path

from sagwa.adapters.base import AdapterResult

# Per-run override: Groq rate-limits tokens per minute PER MODEL, so a target
# with no baseline to match can be moved onto an otherwise-idle model.
# NEVER override this for a run being compared against an existing baseline —
# a different target model would conflate "we changed the model" with whatever
# change is actually under test.
MODEL = os.environ.get("SAGWA_TARGET_MODEL", "openai/gpt-oss-120b")
CORPUS = Path(__file__).parent / "hotpotqa_corpus.jsonl"

PROMPT_STRICT = """Answer the question using ONLY the context below.
Reply with the short answer only (a name, date, number, or yes/no) — no explanation.
If the context does not contain the answer, reply exactly: "I don't know".

Context:
{context}

Question: {question}
Answer:"""

PROMPT_LOOSE = """Answer the question briefly.

Context:
{context}

Question: {question}
Answer:"""

PROMPT_SUMMARY = """Summarize the news article below in 3-4 short sentences covering the key facts.

Article:
{article}

Summary:"""


class Retriever:
    """Paragraph-level dense retrieval over the pooled HotpotQA corpus, using
    Sagwa's shared embedding model. Measured on the 50 golden questions, top_k=3
    finds both supporting paragraphs for 32/50 (TF-IDF: 4/50, BM25: 23/50)."""

    def __init__(self, corpus_path: Path = CORPUS):
        from sagwa._embedding import get_embedding_model

        rows = [json.loads(line) for line in corpus_path.read_text().splitlines()]
        self.chunks = [f"{r['title']}: {r['text']}" for r in rows]
        self.model = get_embedding_model()
        self.embeddings = self.model.encode(self.chunks, normalize_embeddings=True)

    def search(self, query: str, top_k: int) -> list[str]:
        scores = self.embeddings @ self.model.encode([query], normalize_embeddings=True)[0]
        return [self.chunks[i] for i in scores.argsort()[::-1][:top_k]]


def _complete(client, prompt: str) -> tuple[str, int]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=800,
        reasoning_effort="low",
    )
    return response.choices[0].message.content.strip(), response.usage.total_tokens


def _groq():
    from groq import Groq

    # timeout: a laptop suspend kills the open connection, and without one
    # the resumed process waits on a socket that will never answer.
    return Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=6, timeout=90.0)


class _HotpotAdapter:
    top_k: int
    prompt: str

    def __init__(self):
        self.retriever = Retriever()
        self.client = _groq()
        self.model_label = f"{MODEL} top_k={self.top_k}"

    def run(self, case_input: str) -> AdapterResult:
        start = time.time()
        context = "\n\n".join(self.retriever.search(case_input, self.top_k))
        answer, tokens = _complete(self.client, self.prompt.format(context=context, question=case_input))
        return AdapterResult(
            answer=answer,
            context=context,
            latency_ms=int((time.time() - start) * 1000),
            tokens=tokens,
            cost_usd=None,
        )


class HotpotV1(_HotpotAdapter):
    name = "hotpotqa-v1"
    top_k = 3
    prompt = PROMPT_STRICT


class HotpotV2(_HotpotAdapter):
    name = "hotpotqa-v2"
    top_k = 1
    prompt = PROMPT_LOOSE


class CnnDmSummarizer:
    name = "cnndm-summarizer"

    def __init__(self):
        self.client = _groq()
        self.model_label = MODEL

    def run(self, case_input: str) -> AdapterResult:
        start = time.time()
        answer, tokens = _complete(self.client, PROMPT_SUMMARY.format(article=case_input))
        return AdapterResult(
            answer=answer,
            context=None,
            latency_ms=int((time.time() - start) * 1000),
            tokens=tokens,
            cost_usd=None,
        )
