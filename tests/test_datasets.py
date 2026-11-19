import json
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


def _write(tmp_path, lines):
    path = tmp_path / "set.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines))
    return path


def test_agentic_case_loads_with_success_criteria(tmp_path):
    path = _write(tmp_path, [{
        "id": "a-1", "input": "How many orders shipped late?", "task_type": "agentic",
        "tags": ["database_tasks"], "expected_tools": ["order_db"],
        "success_criteria": {"answer_contains": ["17"], "required_tools": ["order_db"], "max_steps": 12},
    }])
    (case,) = load_golden_set(path)
    assert case.success_criteria.answer_contains == ["17"]
    assert case.success_criteria.must_not_error is True
    assert case.expected_tools == ["order_db"]


def test_agentic_case_without_success_criteria_fails_loudly_with_its_line(tmp_path):
    path = _write(tmp_path, [
        {"id": "ok", "input": "q", "task_type": "rag_qa"},
        {"id": "a-1", "input": "q", "task_type": "agentic"},
    ])
    with pytest.raises(DatasetError) as exc_info:
        load_golden_set(path)
    assert "line 2" in str(exc_info.value)
    assert "success_criteria" in str(exc_info.value)
