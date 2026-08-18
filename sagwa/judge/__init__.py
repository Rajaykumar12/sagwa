"""LLM-as-judge harness + calibration workflow against human labels
(PRD FR-10-FR-15a; PLAN.md Week 5-6)."""
from sagwa.judge.calibration import (
    CalibrationRequired,
    CalibrationResult,
    calibrate,
    cohen_kappa,
    confusion_matrix,
    load_calibration_results,
    require_calibration,
    save_calibration_result,
)
from sagwa.judge.harness import groq_llm_call, score_absolute, score_pairwise

__all__ = [
    "CalibrationRequired",
    "CalibrationResult",
    "calibrate",
    "cohen_kappa",
    "confusion_matrix",
    "load_calibration_results",
    "require_calibration",
    "save_calibration_result",
    "groq_llm_call",
    "score_absolute",
    "score_pairwise",
]
