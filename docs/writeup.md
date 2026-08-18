# Sagwa — Writeup

**Status**: stub. Write this after the calibration checkpoint (PRD.md §10, end of Week 6) and the CI-gate checkpoint (end of Week 10), then finalize alongside the recorded demo (end of Week 12).

Planned structure (see PRD.md §2 for positioning rationale):

1. The problem: shipping LLM features on vibes, and why it breaks silently within 6-12 months.
2. System walkthrough: golden set → run → diff → clustered failures → CI gate → dashboard.
3. The calibration study, in detail — this is the load-bearing section (link `calibration/calibration_report.md`).
4. Head-to-head: Sagwa's calibrated judge vs. ringo's existing `backend/eval.py` (PRD §7's headline metric).
5. What broke / what I'd do differently.
