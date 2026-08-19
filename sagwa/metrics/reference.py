"""Reference-based metrics (PRD FR-8): scored against a golden case's
`expected_output`. Callers should only invoke this module for cases that
have one — no ground truth means no reference-based score, not a 0.0.
"""
from __future__ import annotations

import difflib
import re


def exact_match(output: str, expected: str) -> float:
    """1.0 if output equals expected after whitespace/case normalization, else 0.0."""
    return 1.0 if _normalize(output) == _normalize(expected) else 0.0


def fuzzy_match(output: str, expected: str) -> float:
    """Character-level similarity ratio in [0, 1], via stdlib difflib —
    no extra fuzzy-matching dependency needed at this granularity."""
    return difflib.SequenceMatcher(None, _normalize(output), _normalize(expected)).ratio()


def rouge_l_f1(output: str, expected: str) -> float:
    """ROUGE-L F1: F1 over the longest-common-subsequence of tokens.
    Hand-rolled (no `rouge-score` dependency) — this is the standard
    LCS-based ROUGE-L definition, not an approximation of it."""
    output_tokens = _tokenize(output)
    expected_tokens = _tokenize(expected)
    if not output_tokens or not expected_tokens:
        return 0.0

    lcs_len = _lcs_length(output_tokens, expected_tokens)
    precision = lcs_len / len(output_tokens)
    recall = lcs_len / len(expected_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def embedding_similarity(output: str, expected: str) -> float | None:
    """Cosine similarity between sentence-transformer embeddings, in [-1, 1].
    Returns `None` — not 0.0, which would misreport as "completely
    dissimilar" — if `sentence-transformers` isn't installed."""
    from sagwa._embedding import get_embedding_model

    try:
        model = get_embedding_model()
    except ImportError:
        return None

    vectors = model.encode([output, expected])
    return _cosine(vectors[0], vectors[1])


def compute_reference_metrics(output: str, expected: str) -> dict:
    """All reference-based metrics for one case (PRD FR-8)."""
    metrics = {
        "exact_match": exact_match(output, expected),
        "fuzzy_match": fuzzy_match(output, expected),
        "rouge_l_f1": rouge_l_f1(output, expected),
    }
    similarity = embedding_similarity(output, expected)
    if similarity is not None:
        metrics["embedding_similarity"] = similarity
    return metrics


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def _lcs_length(a: list[str], b: list[str]) -> int:
    # Standard O(len(a) * len(b)) LCS DP, rolling one row at a time.
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            curr[j] = prev[j - 1] + 1 if token_a == token_b else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _cosine(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0
