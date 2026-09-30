"""Static export must preserve sealed validation choices and exact source bytes."""

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery.exp17 import program_inspection as P

SEED = 'def propose(history, bounds, seed):\n    raise RuntimeError("never execute")\n'
FIRST = SEED + "\n# first generated source\n"
LAST = SEED + "\n# selected generated source\n"


def write(root: Path, relative: str, value: Any) -> None:
    """Write a disposable synthetic evidence record with stable formatting."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def row(source: str, value: float) -> dict[str, Any]:
    """Provide saved observations; no task function or candidate is invoked."""
    return {
        "source_sha256": P.OLD.source_hash(source),
        "valid": True,
        "candidate_valid": True,
        "fallback_used": False,
        "stratum": "sphere/2",
        "metrics": {"auc": value},
        "observations": [{"x": [0.0, 0.0], "value": value}],
    }


def prepared(root: Path, *, mechanism: bool = False) -> Path:
    """Build two complete outer replications, including a seed-selected representative."""
    arms = ["L", "M", "P", "PM"] if mechanism else ["I", "C"]
    representative_arm = "PM" if mechanism else "C"
    config = {
        "experiment": "EXP-18" if mechanism else "EXP-17",
        "namespace": "UNIT-INSPECTION",
        "analysis_kind": "exp18" if mechanism else "exp17",
        "outer_seeds": [11, 23],
        "arms": arms,
        "slots": 3,
        "representative_arm": representative_arm,
        "audit_controls": ["A0", "B2"],
    }
    frozen = {
        "config": config,
        "seed_source": SEED,
        "seed_sha256": P.OLD.source_hash(SEED),
        "fixed_controls": {
            "B2": {"source": FIRST, "source_sha256": P.OLD.source_hash(FIRST)}
        },
    }
    write(root, "freeze.json", frozen)
    write(root, "freeze_sha256.json", {"sha256": P.OLD.digest(frozen)})
    generation_hashes, selection_hashes, chosen = {}, {}, {}
    audit, analyzed = {}, {}
    for outer in config["outer_seeds"]:
        audit[str(outer)], analyzed[str(outer)] = {}, {}
        for control, source in (("A0", SEED), ("B2", FIRST)):
            audit[str(outer)][control] = {
                "source_sha256": P.OLD.source_hash(source),
                "rows": [row(source, 0.5)],
                "auc": 0.5,
            }
            analyzed[str(outer)][control] = {"auc": 0.5}
        for arm in arms:
            directory = f"raw/{outer}/{arm}"
            seed_auc = 1.0 if outer == 11 else 0.01
            pool = []
            for index, source in enumerate((SEED, FIRST, FIRST, LAST), -1):
                value = seed_auc if index == -1 else 0.1 if index == 2 else 0.5
                pool.append(
                    {
                        "index": index,
                        "source": source,
                        "source_sha256": P.OLD.source_hash(source),
                        "eligible": True,
                        "validation_auc": value,
                        "train": [row(source, value)],
                        "validation": [row(source, value)],
                    }
                )
            write(root, directory + "/pool.json", pool)
            best = min(pool, key=lambda item: (item["validation_auc"], item["index"]))
            selection = {
                key: best[key]
                for key in ("index", "source", "source_sha256", "validation_auc")
            }
            selection["selected_ns"] = 150
            selected_path = directory + "/selection.json"
            write(root, selected_path, selection)
            chosen[f"{outer}/{arm}"] = selection
            selection_hashes[selected_path] = P.OLD.digest(selection)
            for slot in range(3):
                parent = FIRST if slot == 2 and arm != "I" else SEED
                source = pool[slot + 1]["source"]
                request = {
                    "outer": outer,
                    "arm": arm,
                    "slot": slot,
                    "parent_sha256": P.OLD.source_hash(parent),
                    "freeze_sha256": P.OLD.digest(frozen),
                    "messages": [{"role": "user", "content": "invariant"}],
                }
                response = {
                    "source": source,
                    "source_sha256": P.OLD.source_hash(source),
                    "completed_ns": 50 + slot,
                }
                for name, value in (("request", request), ("response", response)):
                    path = directory + f"/slot_{slot:02d}/{name}.json"
                    write(root, path, value)
                    generation_hashes[path] = P.OLD.digest(value)
                if arm in {"P", "PM"}:
                    decision = {
                        "next_slot": slot,
                        "selected_index": 0 if slot == 2 else -1,
                        "selected_source_sha256": P.OLD.source_hash(parent),
                        "will_generate": True,
                        "frontier_indices": [-1, 0] if slot else [-1],
                        "scalar_best_source_sha256": P.OLD.source_hash(SEED),
                        "trainer_runtime_alias": "unit-pareto",
                    }
                    path = directory + f"/parent_decisions/slot_{slot:02d}.json"
                    write(root, path, decision)
                    generation_hashes[path] = P.OLD.digest(decision)
            if arm in {"P", "PM"}:
                terminal = {
                    **decision,
                    "next_slot": 3,
                    "selected_index": 2,
                    "selected_source_sha256": P.OLD.source_hash(LAST),
                    "will_generate": False,
                }
                path = directory + "/parent_decisions/slot_03.json"
                write(root, path, terminal)
                generation_hashes[path] = P.OLD.digest(terminal)
            # Holdout ordering deliberately disagrees with validation ordering.
            score = 0.001 if outer == 11 else 100.0
            audit[str(outer)][arm] = {
                "source_sha256": best["source_sha256"],
                "rows": [row(best["source"], score)],
                "auc": score,
            }
            analyzed[str(outer)][arm] = {
                "auc": score,
                "selection": {
                    key: selection[key]
                    for key in ("index", "source_sha256", "validation_auc")
                },
            }
    representatives = {
        arm: {"outer": 23, "selection": chosen[f"23/{arm}"]} for arm in arms
    }
    write(
        root,
        "generation_frozen.json",
        {"completed_ns": 100, "hashes": generation_hashes},
    )
    write(
        root,
        "selections_frozen.json",
        {
            "completed_ns": 200,
            "hashes": selection_hashes,
            "representatives": representatives,
            "representative_outer": 23,
            "representative": chosen[f"23/{representative_arm}"],
        },
    )
    write(root, "audit_started.json", {"wall_ns": 250})
    write(
        root,
        "audit_results.json",
        {"completed_ns": 300, "selections_frozen_ns": 200, "per_seed": audit},
    )
    write(
        root,
        "analysis_results.json",
        {
            "schema": "optimizer_discovery.shared_analysis.v1",
            "namespace": config["namespace"],
            "analysis_kind": config["analysis_kind"],
            "outer_seeds": config["outer_seeds"],
            "per_seed": analyzed,
            "representative": {
                "arm": representative_arm,
                "outer": 23,
                **{
                    key: chosen[f"23/{representative_arm}"][key]
                    for key in ("index", "source_sha256", "validation_auc")
                },
            },
        },
    )
    return root


@pytest.mark.parametrize("mechanism", [False, True])
def test_all_seeds_validation_representative_and_exact_exports(
    tmp_path: Path, mechanism: bool
) -> None:
    """The representative remains the validation-selected seed despite a poor audit score."""
    root = prepared(tmp_path / "run", mechanism=mechanism)
    data = P.inspect(root)
    assert data["representative"]["outer"] == 23
    assert data["representative"]["selected_seed_source"]
    assert set(data["selected"]) == {"11", "23"}
    assert (
        data["llm_calls"] == data["objective_calls"] == data["executed_candidates"] == 0
    )
    out = tmp_path / "out"
    result = P.export(root, out)
    for outer, arms in data["selected"].items():
        for arm, selection in arms.items():
            source_path = out / f"selected/{arm}/{outer}_optimizer.py.txt"
            assert source_path.read_bytes() == selection["source"].encode()
    assert P.export(root, out) == result
    assert (out / "PROGRAM_INSPECTION.md").exists()


def test_duplicate_source_ancestry_is_retained_without_inventing_one_parent(
    tmp_path: Path,
) -> None:
    """The selected generated artifact has two possible earlier source origins."""
    data = P.inspect(prepared(tmp_path / "run"))
    chosen = data["selected"]["11"]["C"]
    assert chosen["lineage_ambiguous"]
    assert chosen["lineage"][-1]["parent_origin_slots"] == [0, 1]
    assert chosen["depth_range"] == [2, 2]
    assert chosen["lineage"][-1]["diff"]


def test_missing_analysis_blocks_inspection_before_audit_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No early holdout inspection is enabled by having only frozen selections."""
    root = prepared(tmp_path / "run")
    (root / "analysis_results.json").unlink()
    original = Path.read_bytes
    accessed = []

    def read(path: Path) -> bytes:
        """Observe file accesses without changing their contents."""
        accessed.append(path.name)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(ValueError, match="completed analysis"):
        P.inspect(root)
    assert "audit_results.json" not in accessed


def test_early_audit_or_missing_selection_is_rejected(tmp_path: Path) -> None:
    """A valid-looking subset cannot stand in for the complete selection freeze."""
    root = prepared(tmp_path / "run")
    write(root, "audit_started.json", {"wall_ns": 190})
    with pytest.raises(ValueError, match="chronology"):
        P.inspect(root)
    write(root, "audit_started.json", {"wall_ns": 250})
    barrier = json.loads((root / "selections_frozen.json").read_text())
    barrier["hashes"].pop("raw/23/C/selection.json")
    write(root, "selections_frozen.json", barrier)
    with pytest.raises(ValueError, match="every selection"):
        P.inspect(root)


def test_changed_raw_source_and_conflicting_export_are_rejected(tmp_path: Path) -> None:
    """Scientific source bytes cannot be replaced by a repaired presentation copy."""
    root = prepared(tmp_path / "run")
    out = tmp_path / "out"
    P.export(root, out)
    (out / "selected/C/11_optimizer.py.txt").write_text("changed source")
    with pytest.raises(RuntimeError, match="replace"):
        P.export(root, out)
    response = json.loads((root / "raw/11/C/slot_02/response.json").read_text())
    response["source"] += "# edited\n"
    write(root, "raw/11/C/slot_02/response.json", response)
    with pytest.raises(ValueError, match="sealed|source"):
        P.inspect(root)


def test_actual_pareto_parent_decisions_are_crosslinked(tmp_path: Path) -> None:
    """Recorded parent decisions must name the source actually received by the next request."""
    data = P.inspect(prepared(tmp_path / "run", mechanism=True))
    record = data["searches"]["11/P"]["records"][2]
    assert record["pareto_decision"]["selected_index"] == 0
    assert (
        record["pareto_decision"]["selected_source_sha256"] == record["parent_sha256"]
    )
    assert (
        data["searches"]["11/P"]["terminal_parent_decision"]["will_generate"] is False
    )


def test_incomplete_analysis_controls_or_changed_control_score_are_rejected(
    tmp_path: Path,
) -> None:
    """Completed analysis must include the unchanged controls and their saved scores."""
    root = prepared(tmp_path / "run")
    original = json.loads((root / "analysis_results.json").read_text())
    partial = json.loads(json.dumps(original))
    partial["per_seed"]["23"].pop("B2")
    write(root, "analysis_results.json", partial)
    with pytest.raises(ValueError, match="complete.*arm"):
        P.inspect(root)
    original["per_seed"]["23"]["B2"]["auc"] = 0.01
    write(root, "analysis_results.json", original)
    with pytest.raises(ValueError, match="control"):
        P.inspect(root)


def test_invalid_sources_and_realized_fallback_remain_visible(tmp_path: Path) -> None:
    """An invalid generated slot is retained without parsing or scoring its source."""
    root = prepared(tmp_path / "run")
    source = "this is not Python code !!\n"
    pool = json.loads((root / "raw/11/C/pool.json").read_text())
    invalid = pool[2]
    invalid.update(
        source=source,
        source_sha256=P.OLD.source_hash(source),
        eligible=False,
        validation_auc=None,
    )
    for split in ("train", "validation"):
        invalid[split] = [
            {
                **row(source, 0.5),
                "valid": False,
                "candidate_valid": False,
                "metrics": None,
                "observations": [],
            }
        ]
    write(root, "raw/11/C/pool.json", pool)
    relative = "raw/11/C/slot_01/response.json"
    response = json.loads((root / relative).read_text())
    response.update(source=source, source_sha256=P.OLD.source_hash(source))
    write(root, relative, response)
    barrier = json.loads((root / "generation_frozen.json").read_text())
    barrier["hashes"][relative] = P.OLD.digest(response)
    write(root, "generation_frozen.json", barrier)
    audit = json.loads((root / "audit_results.json").read_text())
    audit["per_seed"]["11"]["C"]["rows"][0].update(
        candidate_valid=False, fallback_used=True
    )
    write(root, "audit_results.json", audit)
    data = P.inspect(root)
    assert data["searches"]["11/C"]["ineligible_generated"] == 1
    assert data["searches"]["11/C"]["records"][1]["train_auc"] is None
    assert data["selected"]["11"]["C"]["audit_behavior"]["fallback_trajectories"] == 1


def test_compressed_source_evidence_is_supported_without_path_escape(
    tmp_path: Path,
) -> None:
    """Lossless compression preserves source integrity; a compressed symlink cannot escape."""
    root = prepared(tmp_path / "run")
    path = root / "raw/11/C/slot_02/response.json"
    compressed = path.with_suffix(".json.gz")
    compressed.write_bytes(gzip.compress(path.read_bytes()))
    path.unlink()
    assert P.inspect(root)["selected"]["11"]["C"]["source"] == LAST
    outside = tmp_path / "outside.gz"
    compressed.rename(outside)
    compressed.symlink_to(outside)
    with pytest.raises(ValueError, match="escaped"):
        P.inspect(root)


def snapshot(root: Path) -> dict[str, bytes | str | None]:
    """Capture disposable files, links and directories before a rejected export."""
    return {
        str(path.relative_to(root)): (
            str(path.readlink())
            if path.is_symlink()
            else path.read_bytes() if path.is_file() else None
        )
        for path in root.rglob("*")
    }


@pytest.mark.parametrize("conflicting", [False, True])
def test_dual_json_representations_are_rejected_without_writes(
    tmp_path: Path, conflicting: bool
) -> None:
    """Identical and conflicting JSON/gzip pairs are both ambiguous input evidence."""
    root = prepared(tmp_path / "run")
    path = root / "raw/11/C/slot_02/response.json"
    raw = b'{"source": "conflicting source"}' if conflicting else path.read_bytes()
    path.with_suffix(".json.gz").write_bytes(gzip.compress(raw))
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="ambiguous"):
        P.export(root, tmp_path / "out")
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("relation", ["same", "inside", "ancestor", "symlink"])
def test_export_requires_disjoint_resolved_roots_without_writes(
    tmp_path: Path, relation: str
) -> None:
    """An export can neither enter scientific data nor surround its input directory."""
    root = prepared(tmp_path / "run")
    output = {"same": root, "inside": root / "cache", "ancestor": tmp_path}.get(
        relation, tmp_path / "alias"
    )
    if relation == "symlink":
        output.symlink_to(root, target_is_directory=True)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        P.export(root, output)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("file_link", [False, True])
def test_export_rejects_nested_symlink_escape_before_any_write(
    tmp_path: Path, file_link: bool
) -> None:
    """A nominally separate output cannot redirect any export into the input tree."""
    root = prepared(tmp_path / "run")
    output = tmp_path / "out"
    output.mkdir()
    if file_link:
        (output / "PROGRAM_INSPECTION.md").symlink_to(root / "new_report.md")
    else:
        (output / "selected").symlink_to(root, target_is_directory=True)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="escaped"):
        P.export(root, output)
    assert snapshot(tmp_path) == before
