# Judge Calibration Report

**Status**: not started — this is a stub. See PRD.md §5.5, §7 and PLAN.md §9 (Week 5-6), the critical-path deliverable of the whole project.

## What goes here (once the calibration study runs)

1. **Human-labeled set**: size, sampling method, `calibration/human_labels.csv` reference.
2. **Judge prompt version(s)**: each iteration tried, with the reasoning for the change.
3. **Agreement metrics**: accuracy, Cohen's κ, confusion matrix — per judge-prompt version.
4. **Baseline comparison** (PRD FR-15a): the same human-labeled set scored by ringo's existing `backend/eval.py` judge, reported side-by-side with Sagwa's calibrated judge. This delta is the headline proof point for the whole project (PRD §7).
5. **Pass/fail call**: whether κ ≥ 0.70 was reached, and if not, an honest account of the iteration process (PRD §9 Risks).
