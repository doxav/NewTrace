"""Verify the P1 supplement with standard-library CSV and OpenXML readers only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def verify_equal(actual: Any, expected: Any, label: str) -> None:
    """Check typed values without treating a missing number as zero."""
    if expected is None:
        assert actual is None, label
    elif isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float)), (label, actual, expected)
        assert math.isfinite(actual), label
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (
            label,
            actual,
            expected,
        )
    else:
        assert actual == expected, (label, actual, expected)


def load_cells(archive: zipfile.ZipFile) -> dict[str, dict[str, dict[str, Any]]]:
    """Read exact stored values and formula caches from every worksheet."""
    relationships = {
        item.attrib["Id"]: item.attrib["Target"].lstrip("/")
        for item in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    }
    shared = [
        "".join(element.itertext())
        for element in ET.fromstring(archive.read("xl/sharedStrings.xml"))
    ]
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for sheet in ET.fromstring(archive.read("xl/workbook.xml")).findall(
        "x:sheets/x:sheet", NS
    ):
        cells: dict[str, dict[str, Any]] = {}
        tree = ET.fromstring(archive.read(relationships[sheet.attrib[RID]]))
        for element in tree.findall("x:sheetData/x:row/x:c", NS):
            kind = element.attrib.get("t", "n")
            assert kind != "e", (sheet.attrib["name"], element.attrib["r"])
            value_node = element.find("x:v", NS)
            value: Any = None if value_node is None else value_node.text
            if value is not None:
                if kind == "s":
                    value = shared[int(value)]
                elif kind == "n":
                    value = float(value)
                elif kind == "b":
                    value = bool(int(value))
            if kind == "inlineStr":
                value = "".join(element.find("x:is", NS).itertext())
            formula = element.find("x:f", NS)
            cells[element.attrib["r"]] = {
                "value": value,
                "formula": None if formula is None else formula.text,
            }
        result[sheet.attrib["name"]] = cells
    return result


def main() -> None:
    """Compare all projected records, formulas, provenance, and retained outcomes."""
    data = json.loads((HERE / "data.json").read_text())
    checks = json.loads((HERE / "p1_workbook_checks.json").read_text())
    output = Path(checks["output"])
    assert hashlib.sha256(output.read_bytes()).hexdigest() == checks["sha256"]
    for filename, digest in checks["input_hashes"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest
    csv_cells = 0
    for name in ("per_seed", "contrasts", "searches", "usage"):
        with (HERE / f"{name}.csv").open(newline="") as handle:
            records = list(csv.DictReader(handle))
        assert len(records) == len(data[name])
        for index, (record, expected) in enumerate(
            zip(records, data[name], strict=True)
        ):
            assert set(record) == set(expected)
            for key, value in expected.items():
                parsed: Any = record[key]
                if value is None:
                    assert parsed == ""
                    parsed = None
                elif isinstance(value, (int, float)):
                    parsed = float(parsed)
                verify_equal(parsed, value, f"{name}[{index}].{key}")
                csv_cells += 1
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        cells = load_cells(archive)
    assert list(cells) == ["Données", "Contrastes", "Recherches", "Usage", "Sources"]
    formula_cells = {
        (sheet, address)
        for sheet, values in cells.items()
        for address, cell in values.items()
        if cell["formula"] is not None
    }
    assert formula_cells == {
        (item["sheet"], item["cell"]) for item in checks["formula_checks"]
    }
    for item in checks["formula_checks"]:
        actual = cells[item["sheet"]][item["cell"]]
        assert actual["formula"] == item["formula"].removeprefix("=")
        verify_equal(
            actual["value"], item["expected"], f"{item['sheet']}!{item['cell']}"
        )

    checked_values = 0

    def cell_equal(sheet: str, address: str, expected: Any) -> None:
        """Count independent comparisons against the authoritative projection."""
        nonlocal checked_values
        actual = cells[sheet].get(address, {"value": None})["value"]
        verify_equal(actual, expected, f"{sheet}!{address}")
        checked_values += 1

    seeds = [16411, 16423, 16437, 16441, 16453, 16467]
    arms = ["A0", "I", "C", "R", "W", "B2"]
    records = sorted(
        data["per_seed"],
        key=lambda row: (seeds.index(row["outer_seed"]), arms.index(row["arm"])),
    )
    keys = [
        "outer_seed",
        "arm",
        "auc",
        "final_regret",
        "target_attainment",
        "capped_target_evaluations",
        "candidate_invalid_trajectories",
        "fallback_trajectories",
        "selection_index",
        "source_sha256",
    ]
    for number, record in enumerate(records, 6):
        for col, key in zip("ABCDEFGHIJ", keys, strict=True):
            cell_equal("Données", f"{col}{number}", record[key])
    for index, contrast in enumerate(data["contrasts"]):
        for col, key in (
            ("A", "contrast"),
            ("C", "mean"),
            ("D", "median"),
            ("E", "ci_low"),
            ("F", "ci_high"),
        ):
            cell_equal("Contrastes", f"{col}{6 + index}", contrast[key])
        for row, seed in enumerate(seeds, 16):
            cell_equal(
                "Contrastes", f"{chr(66 + index)}{row}", contrast[f"delta_{seed}"]
            )
    generated_arms = arms[1:5]
    searches = sorted(
        data["searches"],
        key=lambda row: (
            seeds.index(row["outer_seed"]),
            generated_arms.index(row["arm"]),
        ),
    )
    for row, record in enumerate(searches, 6):
        for col, key in (
            ("A", "outer_seed"),
            ("B", "arm"),
            ("C", "allocated_responses"),
            ("D", "eligible_generated"),
            ("E", "ineligible_generated"),
            ("G", "selected_seed"),
            ("H", "train_allocations"),
            ("I", "validation_allocations"),
        ):
            cell_equal("Recherches", f"{col}{row}", record[key])
        cell_equal(
            "Recherches",
            f"F{row}",
            record["eligible_generated"] / record["allocated_responses"],
        )
    for row, record in enumerate(data["usage"], 6):
        for col, key in zip(
            "ABCDEFG",
            [
                "arm",
                "responses",
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "total_tokens",
                "cost_usd",
            ],
            strict=True,
        ):
            cell_equal("Usage", f"{col}{row}", record[key])
    repo = HERE.parents[6]
    for row in range(6, 15):
        filename = cells["Sources"][f"B{row}"]["value"]
        digest = cells["Sources"][f"D{row}"]["value"]
        assert hashlib.sha256((repo / filename).read_bytes()).hexdigest() == digest
    assert len({r["outer_seed"] for r in records}) == 6
    assert sum(r["allocated_responses"] for r in searches) == 192
    assert sum(r["ineligible_generated"] for r in searches) == 18
    assert sum(r["selected_seed"] for r in searches) == 1
    assert sum(r["fallback_trajectories"] for r in records) == 0
    assert (
        sum(cells["Contrastes"][f"B{row}"]["value"] > 0 for row in range(16, 22)) == 4
    )
    assert len(formula_cells) == 138
    scan = (HERE / "p1_formula_error_scan.ndjson").read_text()
    assert "Cell search matched 0 entries." in scan
    report = {
        "status": "PASS",
        "xlsx_sha256": checks["sha256"],
        "csv_json_cell_comparisons": csv_cells,
        "xlsx_projection_cell_comparisons": checked_values,
        "formula_cache_comparisons": len(formula_cells),
        "sheets": list(cells),
        "sources_verified": 9,
        "outer_seeds": seeds,
        "per_seed_rows": 36,
        "contrasts": 6,
        "pools": 24,
        "completed_responses": 192,
        "ineligible_generations": 18,
        "seed_selections": 1,
        "audit_fallback_trajectories": 0,
        "R_minus_I_losses_preserved": 4,
        "B2_selection_index_blanks_preserved": 6,
        "formula_errors": 0,
        "new_model_calls": 0,
        "new_objective_calls": 0,
        "method": "Python standard-library CSV/ZIP/XML; no spreadsheet authoring or scientific reevaluation",
    }
    (HERE / "p1_workbook_independent_checks.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
