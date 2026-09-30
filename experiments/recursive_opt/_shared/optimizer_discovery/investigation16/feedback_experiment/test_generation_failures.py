"""AST-only format diagnostics never execute or choose a candidate program."""

import importlib.util
from pathlib import Path

import pytest


def load_helper() -> object:
    """Load the standalone diagnostic without shadowing the frozen experiment module."""
    path = Path(__file__).with_name("generation_failures.py")
    spec = importlib.util.spec_from_file_location("f1_generation_format", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = load_helper()


@pytest.mark.parametrize("content", [None, ""])
def test_no_final_content_is_separate_from_rejected_code(content: str | None) -> None:
    """A missing final channel is not reinterpreted as hidden reasoning or usable source."""
    result = H.inspect_content(content)
    assert not result["final_text_present"]
    assert not result["whole_response_extractable"]
    assert result["recognized_blocks"] == []


def test_multiple_ast_parsable_blocks_do_not_become_one_eligible_candidate() -> None:
    """Strict whole-response rejection is distinct from independently parsable fragments."""
    content = "```python\ndef choose(x):\n return x\n```\n```python\ndef propose(history,bounds,seed):\n return []\n```"
    result = H.inspect_content(content)
    assert not result["whole_response_extractable"]
    assert result["ast_parsable_blocks"] == 2
    assert result["exact_api_blocks"] == 1
    assert result["executed_or_selected_blocks"] == 0


def test_syntax_parser_does_not_execute_top_level_statements() -> None:
    """Infinite source and incomplete snippets remain harmless AST inspection data."""
    result = H.inspect_content(
        "```python\nwhile True: pass\ndef choose(x,bounds,ss,seed):\n return x\n```"
    )
    assert result["whole_response_extractable"]
    assert result["ast_parsable_blocks"] == 1
    assert result["exact_api_blocks"] == 0
    bad = H.inspect_content("```python\nif True:\n```")
    assert bad["recognized_blocks"][0]["syntax_status"] == "SyntaxError"
