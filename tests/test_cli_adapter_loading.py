import pytest

from sagwa.adapters.stub import StubAdapter
from sagwa.cli import _load_adapter_class


def test_builtin_stub_resolves():
    assert _load_adapter_class("stub") is StubAdapter


def test_third_party_adapter_resolves_via_module_class_path():
    """The whole point: a class that is not registered anywhere in
    sagwa/'s source still loads, by pointing --target at it directly."""
    factory = _load_adapter_class("tests.fixtures.fake_adapter:FakeAdapter")

    adapter = factory()
    assert adapter.name == "fake"
    result = adapter.run("hello")
    assert result.answer == "fake: hello"


def test_unknown_bare_name_raises_actionable_error():
    with pytest.raises(ValueError, match="module.path:ClassName"):
        _load_adapter_class("nonexistent")


def test_unresolvable_module_path_raises_import_error():
    with pytest.raises(ImportError):
        _load_adapter_class("this.module.does.not.exist:Whatever")
