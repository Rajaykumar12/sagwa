from pathlib import Path

import pytest

from sagwa.datasets import DatasetError, load_golden_set

FIXTURES = Path(__file__).parent / "fixtures"


def test_loads_valid_golden_set():
    cases = load_golden_set(Path(__file__).parent.parent / "golden_sets" / "example.jsonl")
    assert len(cases) == 3
    assert {c.id for c in cases} == {"ex-001", "ex-002", "ex-003"}


def test_rejects_invalid_rows_with_line_numbers():
    with pytest.raises(DatasetError) as exc_info:
        load_golden_set(FIXTURES / "invalid_example.jsonl")
    message = str(exc_info.value)
    assert "line 1" in message  # missing task_type
    assert "line 2" in message  # invalid JSON
    assert "line 4" in message  # duplicate of line 3's id


def test_missing_file_raises():
    with pytest.raises(DatasetError):
        load_golden_set(FIXTURES / "does_not_exist.jsonl")
