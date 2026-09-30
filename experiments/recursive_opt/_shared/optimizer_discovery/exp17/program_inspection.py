"""Inspect/export sealed successor programs without importing an evaluator or LLM.

Only saved source, selection, analysis and observation records are read. Candidate
code is parsed as AST for descriptive counts and is never executed or repaired.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery.investigation16.production import (
    program_inspection as OLD,
)


class _Reader:
    """Authenticate exact input bytes without retaining every decoded candidate pool."""

    def __init__(self, root: Path) -> None:
        """Bind a completed run and record all physical file hashes as they are read."""
        self.root = root.resolve()
        self.checked: dict[str, str] = {}

    def exists(self, relative: str) -> bool:
        """Recognize ordinary and losslessly compressed JSON records."""
        path = self.root / relative
        return path.exists() or path.with_suffix(path.suffix + ".gz").exists()

    def read(self, relative: str) -> Any:
        """Reject path escape and track exact bytes for a final immutability check."""
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("inspection input escaped its run directory")
        compressed = path.with_suffix(path.suffix + ".gz")
        if path.exists() and compressed.exists():
            raise ValueError("ambiguous JSON and compressed inspection input")
        if not path.exists():
            path = compressed
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("inspection input escaped its run directory")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if str(path) in self.checked and self.checked[str(path)] != digest:
            raise ValueError("input changed during inspection")
        self.checked[str(path)] = digest
        return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)

    def verify_unchanged(self) -> None:
        """Prove that this read-only derivation preserved every inspected input."""
        for name, digest in self.checked.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError("input changed during inspection")


def _same_number(left: float | None, right: float | None) -> bool:
    """Compare descriptive recomputations without altering saved scientific values."""
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def _ancestry(
    records: list[dict[str, Any]], pool: list[dict[str, Any]], selected: int
) -> list[dict[str, Any]]:
    """Preserve all possible earlier source origins rather than invent object ancestry."""
    pending, visited = [selected], set()
    while pending:
        index = pending.pop()
        if index < 0 or index in visited:
            continue
        visited.add(index)
        pending.extend(records[index]["parent_origin_slots"])
    result = []
    for index in sorted(visited):
        record = records[index]
        origins = record["parent_origin_slots"]
        parent = pool[origins[0] + 1]["source"]
        child = pool[index + 1]["source"]
        lines = list(
            difflib.unified_diff(
                parent.splitlines(),
                child.splitlines(),
                fromfile="parent_" + record["parent_sha256"][:12],
                tofile=f"slot_{index}",
                lineterm="",
            )
        )
        result.append(
            {
                "slot": index,
                "source_sha256": record["source_sha256"],
                "parent_source_sha256": record["parent_sha256"],
                "parent_origin_slots": origins,
                "declared_parent_origin": (record.get("pareto_decision") or {}).get(
                    "selected_index"
                ),
                "depth_range": record["depth_range"],
                "added_lines": sum(
                    line.startswith("+") and not line.startswith("+++")
                    for line in lines
                ),
                "removed_lines": sum(
                    line.startswith("-") and not line.startswith("---")
                    for line in lines
                ),
                "diff": "\n".join(lines) + ("\n" if lines else ""),
                "diff_preview": "\n".join(lines[:80]),
                "preview_omitted_lines": max(0, len(lines) - 80),
            }
        )
    return result


def inspect(root: Path) -> dict[str, Any]:
    """Inspect every registered selected artifact only after complete primary analysis."""
    reader = _Reader(root)
    if not reader.exists("analysis_results.json"):
        raise ValueError("program inspection requires completed analysis")
    if not reader.exists("selections_frozen.json"):
        raise ValueError("program inspection requires every selection to be frozen")
    frozen = reader.read("freeze.json")
    frozen_hash = reader.read("freeze_sha256.json")["sha256"]
    if (
        OLD.digest(frozen) != frozen_hash
        or OLD.source_hash(frozen["seed_source"]) != frozen["seed_sha256"]
    ):
        raise ValueError("inspection freeze/source integrity mismatch")
    config = frozen["config"]
    seeds, arms, slots = config["outer_seeds"], config["arms"], config["slots"]
    expected_arms = {"exp17": ["I", "C"], "exp18": ["L", "M", "P", "PM"]}
    if (
        arms != expected_arms.get(config["analysis_kind"])
        or not seeds
        or len(set(seeds)) != len(seeds)
        or type(slots) is not int
        or slots < 1
    ):
        raise ValueError("inspection requires a complete registered arm/seed grid")
    generation, selections, analysis = (
        reader.read(name)
        for name in (
            "generation_frozen.json",
            "selections_frozen.json",
            "analysis_results.json",
        )
    )
    selection_paths = {
        f"raw/{outer}/{arm}/selection.json" for outer in seeds for arm in arms
    }
    if set(selections["hashes"]) != selection_paths:
        raise ValueError("barrier must contain every selection exactly once")
    if (
        analysis.get("schema") != "optimizer_discovery.shared_analysis.v1"
        or analysis.get("namespace") != config["namespace"]
        or analysis.get("analysis_kind") != config["analysis_kind"]
        or analysis.get("outer_seeds") != seeds
        or set(analysis.get("per_seed", {})) != {str(seed) for seed in seeds}
    ):
        raise ValueError("completed analysis does not cover this frozen study")
    all_arms = {"A0", "B2", *arms}
    if config["audit_controls"] != ["A0", "B2"] or any(
        set(analysis["per_seed"][str(seed)]) != all_arms for seed in seeds
    ):
        raise ValueError("completed analysis does not cover every registered arm")
    for barrier in (generation, selections):
        for relative, digest in barrier["hashes"].items():
            if OLD.digest(reader.read(relative)) != digest:
                raise ValueError("sealed evidence changed before inspection")
    audit_start, audit = reader.read("audit_started.json"), reader.read(
        "audit_results.json"
    )
    if not (
        generation["completed_ns"]
        < selections["completed_ns"]
        < audit_start["wall_ns"]
        < audit["completed_ns"]
        and audit["selections_frozen_ns"] == selections["completed_ns"]
        and set(audit["per_seed"]) == {str(seed) for seed in seeds}
        and all(set(audit["per_seed"][str(seed)]) == all_arms for seed in seeds)
    ):
        raise ValueError("selection/audit chronology or outer-seed coverage violation")
    seed_hash = frozen["seed_sha256"]
    result: dict[str, Any] = {
        "schema": "optimizer_successor.program_inspection.v1",
        "experiment": config["experiment"],
        "namespace": config["namespace"],
        "freeze_sha256": frozen_hash,
        "outer_seeds": seeds,
        "arms": arms,
        "selected": {},
        "searches": {},
        "representatives": {},
        "executed_candidates": 0,
        "objective_calls": 0,
        "llm_calls": 0,
        "lineage_scope": "Source-origin DAG; repeated source bytes can have several earlier origins. Pareto declared origins are canonical source origins, not proof of unique copied object ancestry.",
    }
    selected_records = {}
    for outer in seeds:
        for control, source in (
            ("A0", frozen["seed_source"]),
            ("B2", frozen["fixed_controls"]["B2"]["source"]),
        ):
            actual = audit["per_seed"][str(outer)][control]
            if (
                actual["source_sha256"] != OLD.source_hash(source)
                or any(
                    row["source_sha256"] != actual["source_sha256"]
                    for row in actual["rows"]
                )
                or not _same_number(actual["auc"], OLD._auc(actual["rows"]))
                or not _same_number(
                    actual["auc"], analysis["per_seed"][str(outer)][control]["auc"]
                )
            ):
                raise ValueError(
                    "completed control analysis does not match saved evidence"
                )
        result["selected"][str(outer)] = {}
        for arm in arms:
            directory = f"raw/{outer}/{arm}"
            pool = reader.read(directory + "/pool.json")
            if [item["index"] for item in pool] != list(range(-1, slots)):
                raise ValueError("candidate pool omits or reorders allocated slots")
            for item in pool:
                if OLD.source_hash(item["source"]) != item["source_sha256"]:
                    raise ValueError("candidate pool source integrity mismatch")
                valid = all(
                    row["valid"]
                    for split in ("train", "validation")
                    for row in item[split]
                )
                auc = OLD._auc(item["validation"]) if valid else None
                if valid != item["eligible"] or not _same_number(
                    item["validation_auc"], auc
                ):
                    raise ValueError("saved eligibility/validation criterion mismatch")
            if not pool[0]["eligible"] or pool[0]["source_sha256"] != seed_hash:
                raise ValueError("trusted seed is absent or invalid")
            expected = min(
                (item for item in pool if item["eligible"]),
                key=lambda item: (item["validation_auc"], item["index"]),
            )
            chosen = reader.read(directory + "/selection.json")
            if (
                any(
                    chosen[key] != expected[key]
                    for key in ("index", "source", "source_sha256", "validation_auc")
                )
                or not generation["completed_ns"]
                < chosen["selected_ns"]
                < selections["completed_ns"]
            ):
                raise ValueError(
                    "selected source violates the frozen validation-only rule"
                )
            selected_records[f"{outer}/{arm}"] = chosen
            records = []
            for slot in range(slots):
                folder = directory + f"/slot_{slot:02d}"
                request, response = reader.read(folder + "/request.json"), reader.read(
                    folder + "/response.json"
                )
                if any(
                    folder + f"/{name}.json" not in generation["hashes"]
                    for name in ("request", "response")
                ):
                    raise ValueError(
                        "generation barrier omits allocated request/source evidence"
                    )
                if (
                    (request["outer"], request["arm"], request["slot"])
                    != (outer, arm, slot)
                    or request["freeze_sha256"] != frozen_hash
                    or response["source"] != pool[slot + 1]["source"]
                    or response["source_sha256"] != pool[slot + 1]["source_sha256"]
                    or response["completed_ns"] >= generation["completed_ns"]
                ):
                    raise ValueError("raw source/request provenance mismatch")
                parent_hash = request["parent_sha256"]
                if arm == "I" and (
                    parent_hash != seed_hash or len(request["messages"]) != 1
                ):
                    raise ValueError("independent arm received an evolving parent")
                record = {
                    "index": slot,
                    "source_sha256": response["source_sha256"],
                    "parent_sha256": parent_hash,
                    "eligible": pool[slot + 1]["eligible"],
                    "train_auc": OLD._auc(pool[slot + 1]["train"]),
                    "response_record": folder + "/response.json",
                    "pareto_decision": None,
                }
                if arm in {"P", "PM"}:
                    path = directory + f"/parent_decisions/slot_{slot:02d}.json"
                    decision = reader.read(path)
                    index = decision["selected_index"]
                    if (
                        path not in generation["hashes"]
                        or type(index) is not int
                        or not -1 <= index < slot
                        or decision["next_slot"] != slot
                        or not decision["will_generate"]
                        or decision["selected_source_sha256"] != parent_hash
                        or pool[index + 1]["source_sha256"] != parent_hash
                    ):
                        raise ValueError(
                            "actual Pareto decision and request parent differ"
                        )
                    record["pareto_decision"] = decision
                records.append(record)
            records = OLD.lineage(records, seed_hash)
            terminal = None
            if arm in {"P", "PM"}:
                path = directory + f"/parent_decisions/slot_{slots:02d}.json"
                terminal = reader.read(path)
                index = terminal["selected_index"]
                if (
                    path not in generation["hashes"]
                    or terminal["next_slot"] != slots
                    or terminal["will_generate"]
                    or type(index) is not int
                    or not -1 <= index < slots
                    or pool[index + 1]["source_sha256"]
                    != terminal["selected_source_sha256"]
                ):
                    raise ValueError(
                        "terminal Pareto selection must remain an unused authenticated decision"
                    )
            actual = audit["per_seed"][str(outer)][arm]
            analyzed = analysis["per_seed"][str(outer)][arm]
            if (
                actual["source_sha256"] != chosen["source_sha256"]
                or any(
                    row["source_sha256"] != chosen["source_sha256"]
                    for row in actual["rows"]
                )
                or any(
                    analyzed["selection"][key] != chosen[key]
                    for key in ("index", "source_sha256", "validation_auc")
                )
                or not _same_number(actual["auc"], OLD._auc(actual["rows"]))
                or not _same_number(analyzed["auc"], actual["auc"])
            ):
                raise ValueError(
                    "completed analysis/audit does not match the selected source"
                )
            chain = _ancestry(records, pool, chosen["index"])
            tree = ast.parse(chosen["source"])
            result["selected"][str(outer)][arm] = {
                **chosen,
                "selected_seed_source": chosen["source_sha256"] == seed_hash,
                "source_record": directory + "/selection.json:source",
                "source_bytes": len(chosen["source"].encode()),
                "source_lines": len(chosen["source"].splitlines()),
                "ast_nodes": sum(1 for _ in ast.walk(tree)),
                "depth_range": (
                    [0, 0]
                    if chosen["index"] == -1
                    else records[chosen["index"]]["depth_range"]
                ),
                "lineage": chain,
                "lineage_ambiguous": any(
                    len(edge["parent_origin_slots"]) > 1 for edge in chain
                ),
                "validation_behavior": OLD.behavior(expected["validation"]),
                "audit_behavior": OLD.behavior(actual["rows"]),
            }
            result["searches"][f"{outer}/{arm}"] = {
                "records": records,
                "selected_slot": chosen["index"],
                "eligible_generated": sum(item["eligible"] for item in pool[1:]),
                "ineligible_generated": sum(not item["eligible"] for item in pool[1:]),
                "distinct_generated_sources": len(
                    {item["source_sha256"] for item in pool[1:]}
                ),
                "parent_counts": dict(
                    Counter(item["parent_sha256"] for item in records)
                ),
                "terminal_parent_decision": terminal,
            }
    for arm in arms:
        outer = min(
            seeds,
            key=lambda seed: (
                selected_records[f"{seed}/{arm}"]["validation_auc"],
                seeds.index(seed),
            ),
        )
        expected = {"outer": outer, "selection": selected_records[f"{outer}/{arm}"]}
        if selections["representatives"][arm] != expected:
            raise ValueError(
                "representative differs from the frozen validation-only rule"
            )
        result["representatives"][arm] = {
            "arm": arm,
            "outer": outer,
            **result["selected"][str(outer)][arm],
        }
    representative = result["representatives"][config["representative_arm"]]
    if (
        selections["representative_outer"] != representative["outer"]
        or selections["representative"]
        != selected_records[f"{representative['outer']}/{representative['arm']}"]
        or any(
            analysis["representative"][key] != representative[key]
            for key in ("arm", "outer", "index", "source_sha256", "validation_auc")
        )
    ):
        raise ValueError(
            "overall representative differs from frozen validation selection"
        )
    result["representative"] = representative
    reader.verify_unchanged()
    result["input_file_sha256"] = reader.checked
    result["input_files_unchanged"] = len(reader.checked)
    return result


def _report(data: dict[str, Any]) -> str:
    """Render concise descriptive metadata without inferring algorithm novelty."""
    representative = data["representative"]
    lines = [
        f"# {data['experiment']} selected program inspection",
        "",
        "Selection and the representative were frozen using validation before audit. This report reads completed analysis and existing observations only; no candidate, objective or model call was executed.",
        "",
        f"Representative: **{representative['arm']}, outer {representative['outer']}, slot {representative['index']}**. SHA256 `{representative['source_sha256']}`.",
        (
            "The representative is the unchanged seed."
            if representative["selected_seed_source"]
            else "The representative is a preserved generated artifact."
        ),
        "",
        "| Outer | Arm | Selected slot | Source SHA256 | Lineage depth range | Seed source |",
        "|---|---|---:|---|---|---|",
    ]
    for outer in data["outer_seeds"]:
        for arm in data["arms"]:
            item = data["selected"][str(outer)][arm]
            lines.append(
                f"| {outer} | {arm} | {item['index']} | `{item['source_sha256']}` | {item['depth_range']} | {item['selected_seed_source']} |"
            )
    lines += [
        "",
        "Exact selected Python source is packaged as `.py.txt` without formatting or repair. The extension identifies verbatim scientific evidence; the bytes can be copied directly to an executor's `optimizer.py`. Full source-origin DAGs, concise diff previews, complete diff files and observed behavior counts accompany the JSON report. Duplicate source origins remain explicit, including canonical source-origin choices logged by the Pareto selector.",
        "",
        "Observed boundary/repetition/improvement counts describe saved trajectories. They do not prove unobserved branch choices, algorithmic novelty, recursion-depth benefits or amortization.",
        "",
    ]
    return "\n".join(lines)


def export(root: Path, output: Path) -> dict[str, Any]:
    """Preserve exact source/diff/report bytes and refuse conflicting re-exports."""
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("inspection input and output directories must be disjoint")
    data = inspect(root)
    payloads: dict[str, bytes] = {}
    for outer, arms in data["selected"].items():
        for arm, selected in arms.items():
            relative = f"selected/{arm}/{outer}_optimizer.py.txt"
            payloads[relative] = selected["source"].encode()
            for edge in selected["lineage"]:
                name = f"selected/{arm}/{outer}_slot_{edge['slot']}.diff.json"
                payloads[name] = (
                    json.dumps(edge, indent=2, allow_nan=False) + "\n"
                ).encode()
    payloads["PROGRAM_INSPECTION.md"] = _report(data).encode()
    data["export_sha256"] = {
        relative: hashlib.sha256(raw).hexdigest() for relative, raw in payloads.items()
    }
    payloads["program_inspection.json"] = (
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    ).encode()
    for relative, raw in payloads.items():
        path = output / relative
        if not path.resolve().is_relative_to(output):
            raise ValueError("inspection export target escaped its output directory")
        if path.exists() and path.read_bytes() != raw:
            raise RuntimeError(
                "refusing to replace nonidentical program inspection output"
            )
    for relative, raw in payloads.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(raw)
    return {
        "output": str(output),
        "files": len(payloads),
        "selected_artifacts": sum(len(arms) for arms in data["selected"].values()),
        "report_sha256": hashlib.sha256(
            payloads["program_inspection.json"]
        ).hexdigest(),
    }


def main() -> None:
    """Export a completed run only when explicitly supplied input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.output)))


if __name__ == "__main__":
    main()
