"""An offline, deterministic target pipeline, so this repo's own CI can gate a
real pipeline on real data without an API key or a token budget.

LeadSummarizer is Lead-N extractive summarization over the CNN/DailyMail
golden set (golden_sets/cnndm.jsonl): return the article's first N
sentences. Lead-3 is the standard CNN/DM baseline — news puts the key facts
first — so this is a real summarizer with known behaviour, not a fake.

Deterministic by construction: the same commit always produces the same
summaries, so any change in the gate's verdict between main and a PR is
caused by the PR, never by sampling noise.
"""
import re
import time

from sagwa.adapters.base import AdapterResult

# How many leading sentences make the summary. Changing this is the kind of
# one-line "make summaries shorter" PR the CI gate exists to judge.
LEAD_SENTENCES = 3

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def lead_summary(article: str, n: int = LEAD_SENTENCES) -> str:
    return " ".join(_SENTENCE_END.split(article.strip())[:n])


class LeadSummarizer:
    name = "cnndm-lead"

    def __init__(self):
        self.model_label = f"lead-{LEAD_SENTENCES}"

    def run(self, case_input: str) -> AdapterResult:
        start = time.perf_counter()
        summary = lead_summary(case_input)
        return AdapterResult(
            answer=summary,
            context=None,
            latency_ms=int((time.perf_counter() - start) * 1000),
            tokens=len(summary.split()),
            cost_usd=0.0,
        )
