"""Shared lazy `sentence-transformers` model loader — used by both
`sagwa/metrics/reference.py` (embedding-similarity, FR-8) and
`sagwa/clustering` (failure embedding, FR-20), so a process that runs
both loads `all-MiniLM-L6-v2` once, not twice. Not a public module (the
leading underscore): callers should go through `reference.py`'s or
`clustering`'s own functions, not this loader directly.
"""
from __future__ import annotations

_model = None


def get_embedding_model():
    """Returns the shared `SentenceTransformer` instance, loading it on
    first use. Raises `ImportError` if `sentence-transformers` isn't
    installed — callers are responsible for degrading gracefully (see
    `reference.embedding_similarity`'s `None`-on-`ImportError` pattern)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model
