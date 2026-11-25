"""The versioned prompt store (docs/AGENTIC_PRD.md Phase 6): versions are
immutable, a new one never moves `current`, and runs record what they used."""
import json

import pytest
from typer.testing import CliRunner

from sagwa.cli import app
from sagwa.prompts import PromptStore, format_versions
from sagwa.storage import Run, get_session


def test_versions_are_numbered_and_current_defaults_to_the_latest(tmp_path):
    store = PromptStore(tmp_path)
    assert store.add_version("planner", "plan A") == "v1"
    assert store.add_version("planner", "plan B", parent="v1", note="tighter") == "v2"
    assert store.versions("planner") == ["v1", "v2"]
    assert store.current("planner") == "v2" and store.load("planner") == "plan B"
    assert json.loads((tmp_path / "planner" / "v2.json").read_text())["parent"] == "v1"


def test_adding_a_version_never_moves_current_or_rewrites_an_old_one(tmp_path):
    store = PromptStore(tmp_path)
    store.add_version("writer", "original")
    store.set_current("writer", "v1")
    store.add_version("writer", "candidate", parent="v1")
    assert store.current("writer") == "v1" and store.load("writer") == "original"
    assert store.load("writer", "v2") == "candidate"


def test_versions_sort_numerically_not_lexically(tmp_path):
    store = PromptStore(tmp_path)
    for i in range(11):
        store.add_version("p", f"text {i}")
    assert store.versions("p")[-2:] == ["v10", "v11"] and store.current("p") == "v11"


def test_unknown_versions_are_errors(tmp_path):
    store = PromptStore(tmp_path)
    store.add_version("p", "x")
    with pytest.raises(KeyError):
        store.load("p", "v9")
    with pytest.raises(KeyError):
        store.set_current("p", "v9")


def test_format_versions_is_order_independent():
    assert format_versions({"writer": "v2", "planner": "v1"}) == format_versions({"planner": "v1", "writer": "v2"})
    assert format_versions({"writer": "v2", "planner": "v1"}) == "planner=v1,writer=v2"


class _VersionedAdapter:
    name = "versioned"
    prompt_versions = {"planner": "v3", "writer": "v1"}

    def run(self, case_input):
        from sagwa.adapters.base import AdapterResult

        return AdapterResult(answer="a", context=None, latency_ms=1, tokens=1, cost_usd=0.0)


def test_sagwa_run_records_the_adapters_prompt_versions(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("sagwa.cli.compute_metrics", lambda case, answer, context: {})
    monkeypatch.setattr("sagwa.cli._load_adapter_class", lambda target: _VersionedAdapter)
    out = tmp_path / "run.json"
    result = CliRunner().invoke(app, ["run", "--target", "x", "--dataset", "golden_sets/example.jsonl",
                                      "--json", str(out)])
    assert result.exit_code == 0, result.output
    with get_session() as session:
        assert session.get(Run, json.loads(out.read_text())["run_id"]).prompt_version == "planner=v3,writer=v1"
