"""Read-only format audit of three explicitly designated completed F1 failures."""

from __future__ import annotations

import ast
import re
from collections import Counter
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import feedback_experiment as F
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G

SLOTS = (
    "16301/Aanytime_code/slot_00",
    "16302/Aanytime_code/slot_00",
    "16301/Aanytime_rich/slot_00",
)


def inspect_content(content: str | None) -> dict[str, Any]:
    """Describe code spans by AST only; never repair, execute, or select any span."""
    if content is not None and not isinstance(content, str):
        raise TypeError("final content must be text or absent")
    text = content or ""
    try:
        source = G.parse_program(content)
        rejection = None
    except ValueError as error:
        source, rejection = None, str(error)
    blocks = []
    for match in re.finditer(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL):
        code = match.group(1)
        record = {
            "offset": match.start(),
            "end": match.end(),
            "characters": len(code),
            "source_sha256": B.source_hash(code),
            "syntax_status": "AST_parsable",
        }
        try:
            tree = ast.parse(code)
        except SyntaxError as error:
            record.update(
                {
                    "syntax_status": "SyntaxError",
                    "line": error.lineno,
                    "message": error.msg,
                    "functions": [],
                }
            )
        else:
            functions = [
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            record["functions"] = [
                {
                    "name": node.name,
                    "kind": type(node).__name__,
                    "positional": [arg.arg for arg in node.args.args],
                    "positional_only": [arg.arg for arg in node.args.posonlyargs],
                    "keyword_only": [arg.arg for arg in node.args.kwonlyargs],
                    "defaults": len(node.args.defaults),
                    "vararg": node.args.vararg.arg if node.args.vararg else None,
                    "kwarg": node.args.kwarg.arg if node.args.kwarg else None,
                    "exact_required_api": (
                        isinstance(node, ast.FunctionDef)
                        and node.name == "propose"
                        and [arg.arg for arg in node.args.args]
                        == ["history", "bounds", "seed"]
                        and not node.args.posonlyargs
                        and not node.args.kwonlyargs
                        and not node.args.defaults
                        and node.args.vararg is None
                        and node.args.kwarg is None
                    ),
                }
                for node in functions
            ]
        blocks.append(record)
    return {
        "content_type": type(content).__name__,
        "final_text_present": bool(text.strip()),
        "content_sha256": B.source_hash(content) if content is not None else None,
        "content_characters": len(text),
        "fence_markers": text.count("```"),
        "definition_names_in_visible_text": dict(
            Counter(re.findall(r"(?m)^\s*def\s+(\w+)\s*\(", text))
        ),
        "whole_response_extractable": source is not None,
        "whole_response_rejection": rejection,
        "whole_response_source_sha256": (
            B.source_hash(source) if source is not None else None
        ),
        "recognized_blocks": blocks,
        "ast_parsable_blocks": sum(
            block["syntax_status"] == "AST_parsable" for block in blocks
        ),
        "exact_api_blocks": sum(
            any(function["exact_required_api"] for function in block["functions"])
            for block in blocks
        ),
        "executed_or_selected_blocks": 0,
        "interpretation": "AST parsability is not executable contract validity; no block is a scientific replacement proposal",
    }


def analyze_designated_failures() -> dict[str, Any]:
    """Read only the specified requests, final responses and safe provider receipts."""
    frozen = F.preflight()
    requests = {request["slot_id"]: request for request in frozen["requests"]}
    rows = []
    for slot in SLOTS:
        request = requests[slot]
        response = F._response(request)
        receipt_path = F.ROOT / "raw" / slot / "provider_generation.json"
        receipt = E.read(receipt_path) if E.exists(receipt_path) else None
        if receipt is not None and receipt.get("id") != response["id"]:
            raise RuntimeError("format audit receipt identity mismatch")
        if response["source"] != "" or response["parse_status"] != "unparsable":
            raise RuntimeError(
                "designated format failure differs from the preserved response"
            )
        rows.append(
            {
                "slot_id": slot,
                "response_id": response["id"],
                "request_canonical_json_sha256": B.digest(request),
                "response_canonical_json_sha256": B.digest(response),
                "parent_sha256": request["parent_sha256"],
                "extracted_source_sha256": response["source_sha256"],
                "model": response["model"],
                "finish_reason": response["finish_reason"],
                "configured_cap": request["settings"]["max_tokens"],
                "request_characters": [
                    len(message["content"]) for message in request["messages"]
                ],
                "prompt_explicit_signature": "exporting exactly propose(history, bounds, seed)"
                in request["messages"][0]["content"],
                "prompt_explicit_one_block": "Return exactly one Python code block with the complete file; no explanations."
                in request["messages"][0]["content"],
                "usage": response["usage"],
                "wall_s": response["wall_s"],
                "format": inspect_content(response["content"]),
                "receipt": receipt,
            }
        )
    return {
        "stage": "F1_DESIGNATED_GENERATION_FAILURES",
        "scope": "format and feasibility only; no task-score comparison",
        "rows": rows,
        "new_generations": 0,
        "candidate_executions": 0,
        "raw_or_frozen_source_edits": 0,
        "reasoning_channel_limit": "The recorder retains _optimizer_response_text(message.content), not a raw reasoning payload; absent final text does not establish absence of internal reasoning or internal code",
    }


def main() -> None:
    """Persist a bounded diagnostic without opening training/validation outcome files."""
    result = analyze_designated_failures()
    I.persist(F.ROOT / "generation_failure_audit.json", result)
    print({"designated_responses": len(result["rows"]), "candidate_executions": 0})


if __name__ == "__main__":
    main()
