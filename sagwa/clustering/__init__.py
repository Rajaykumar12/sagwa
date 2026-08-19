"""Failure clustering (PRD FR-20-FR-22): embeds a run's failing cases,
clusters them with HDBSCAN (density-based, no fixed cluster count), and
auto-generates a natural-language label per cluster.

"Failing" reuses the same pass/fail predicate diff uses (`sagwa.gate.
_predicate.case_passes`, driven by `config/gates.yaml`) — one shared
definition across diff/gate/clustering, not a clustering-specific one.

Labels (FR-21) are LLM-generated via the existing judge harness's
`groq_llm_call()` when `GROQ_API_KEY` is configured, capped to the
top-`label_top_n` clusters by size to bound cost/latency; clusters beyond
that cap, and all clusters when no key is configured, get a cheap
keyword-frequency label instead — mirrors the "degrade to a fallback
rather than crash" pattern used throughout the metrics layer.

Requires the `clustering` extra (`sentence-transformers`, `hdbscan`).
`min_cluster_size=3` is a provisional default — real tuning needs a real
golden set (currently only a 3-case example exists, see
docs/GAP_ANALYSIS.md), not the toy fixtures this module's own tests use.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from sagwa.gate._predicate import case_passes
from sagwa.storage import Result

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on",
    "for", "and", "or", "it", "this", "that", "with", "as", "at", "by",
    "be", "not", "no", "i", "you", "your", "answer", "question",
}


@dataclass
class FailureCluster:
    cluster_id: int  # HDBSCAN label; -1 is the reserved noise/outlier bucket
    case_ids: list[str]
    label: str
    size: int

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "case_ids": self.case_ids,
            "label": self.label,
            "size": self.size,
        }


def failing_results(session, run_id: str, gates_config: dict) -> list[Result]:
    """Cases from `run_id` that fail per `case_passes` (`True`-passing and
    indeterminate/`None` cases are excluded — clustering is only meaningful
    over cases confidently known to have failed)."""
    results = session.query(Result).filter(Result.run_id == run_id).all()
    return [r for r in results if case_passes(r, gates_config) is False]


def embed_texts(texts: list[str]):
    """Embeds `texts` with the shared sentence-transformers model (see
    `sagwa/_embedding.py`)."""
    from sagwa._embedding import get_embedding_model

    model = get_embedding_model()
    return model.encode(texts)


def cluster_failures(results: list[Result], min_cluster_size: int = 3) -> list[FailureCluster]:
    """HDBSCAN over each failing case's `input + output` text. `-1` (HDBSCAN's
    noise label) is kept as its own cluster rather than dropped, so every
    case is accounted for."""
    import hdbscan

    if min_cluster_size < 2:
        # hdbscan itself raises ValueError("Min cluster size must be greater
        # than one") for min_cluster_size == 1 — caught here so the CLI can
        # report it cleanly instead of a raw traceback (a single failing
        # case is never its own "cluster" anyway; that's just one case).
        raise ValueError(f"min_cluster_size must be >= 2, got {min_cluster_size}")

    if not results:
        return []

    texts = [f"{r.input}\n{r.output}" for r in results]
    embeddings = embed_texts(texts)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)

    grouped: dict[int, list[str]] = {}
    for result, label in zip(results, labels):
        grouped.setdefault(int(label), []).append(result.case_id)

    return [
        FailureCluster(cluster_id=cluster_id, case_ids=case_ids, label="", size=len(case_ids))
        for cluster_id, case_ids in grouped.items()
    ]


def _keyword_label(case_texts: list[str]) -> str:
    """Cheap, offline fallback: most frequent non-stopword tokens across
    the cluster's texts, joined as a comma-separated label. Not a sentence
    — a keyword summary, used when an LLM call isn't available/afforded."""
    words = []
    for text in case_texts:
        words.extend(w for w in re.findall(r"[a-zA-Z']+", text.lower()) if w not in _STOPWORDS and len(w) > 2)
    top = [word for word, _ in Counter(words).most_common(4)]
    return ", ".join(top) if top else "(no distinguishing keywords)"


def label_cluster(case_texts: list[str]) -> str:
    """Natural-language cluster label (FR-21). Tries the LLM judge harness
    when `GROQ_API_KEY` is configured; falls back to keyword extraction
    otherwise or on any LLM failure — never raises, since a failed label
    shouldn't sink the whole clustering result."""
    import os

    if "GROQ_API_KEY" not in os.environ:
        return _keyword_label(case_texts)

    try:
        from sagwa.judge.harness import groq_llm_call

        llm_call = groq_llm_call()
        sample = "\n---\n".join(case_texts[:5])  # cap prompt size
        prompt = (
            "The following are inputs/outputs from failing test cases in an LLM "
            "evaluation run. In one short phrase (under 10 words), summarize the "
            f"common failure pattern:\n\n{sample}"
        )
        return llm_call(prompt).strip()
    except Exception:
        return _keyword_label(case_texts)


def cluster_run(
    session,
    run_id: str,
    gates_config: dict,
    min_cluster_size: int = 3,
    label_top_n: int = 10,
) -> list[FailureCluster]:
    """Entry point (FR-20..22): failing_results -> embed -> cluster ->
    label, sorted by size descending (serves FR-22's dashboard sort
    requirement for both the CLI and the dashboard). Only the top
    `label_top_n` clusters by size get an LLM-generated label — the rest
    get the cheap keyword fallback, to bound cost on a run with many small
    clusters."""
    results = failing_results(session, run_id, gates_config)
    results_by_case_id = {r.case_id: r for r in results}

    clusters = cluster_failures(results, min_cluster_size=min_cluster_size)
    clusters.sort(key=lambda c: c.size, reverse=True)

    labeled = []
    for i, cluster in enumerate(clusters):
        case_texts = [
            f"{results_by_case_id[cid].input}\n{results_by_case_id[cid].output}" for cid in cluster.case_ids
        ]
        if cluster.cluster_id == -1:
            label = "(unclustered outliers)"
        elif i < label_top_n:
            label = label_cluster(case_texts)
        else:
            label = _keyword_label(case_texts)
        labeled.append(
            FailureCluster(cluster_id=cluster.cluster_id, case_ids=cluster.case_ids, label=label, size=cluster.size)
        )
    return labeled
