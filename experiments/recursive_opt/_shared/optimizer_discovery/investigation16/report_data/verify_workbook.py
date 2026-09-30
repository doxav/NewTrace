"""Independently inspect only the completed-diagnostic workbook with stdlib readers."""

from __future__ import annotations

import argparse
import collections
import datetime
import gzip
import hashlib
import json
import math
import operator
import posixpath
import re
import statistics
import xml.etree.ElementTree as ET
import zipfile
from itertools import pairwise
from pathlib import Path
from typing import Any

REPO = Path("/home/xav/code/Trace")
ROOT = REPO / "experiments/recursive_opt/_shared/optimizer_discovery/investigation16"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output",
    type=Path,
    help="Write a new review JSON; existing paths are refused. Default: verify without writing.",
)
args = parser.parse_args()
META = ROOT / "report_data/workbook_checks.json"
BUILDER = ROOT / "report_data/build_workbook.mjs"
meta = json.loads(META.read_text())
book = Path(meta["output"])
expected_book = (
    REPO / "experiments/recursive_opt/EXP16/presentation/optimizer_discovery_data.xlsx"
)
assert book == expected_book


def sha(path: Path) -> str:
    """Hash exact saved bytes without interpreting executable content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


book_before = sha(book)
builder_before = sha(BUILDER)
assert book_before == meta["output_sha256"]
assert builder_before == meta["builder_sha256"]
allowed_dirs = [
    ROOT / name
    for name in [
        "statistics",
        "generation",
        "benchmark",
        "selection",
        "throughput",
        "feedback_experiment",
    ]
]
source_values: dict[Path, Any] = {}
source_hashes: dict[str, str] = {}
for filename, expected in meta["input_byte_hashes"].items():
    source = Path(filename).resolve()
    assert source == REPO / "experiments/recursive_opt/_shared/optimizer_discovery/exp15_results.json" or any(
        source.is_relative_to(p) for p in allowed_dirs
    ), filename
    assert (
        "production" not in str(source.relative_to(ROOT))
        if source.is_relative_to(ROOT)
        else True
    )
    actual = sha(source)
    assert actual == expected, filename
    source_hashes[str(source)] = actual
    content = source.read_bytes()
    source_values[source] = json.loads(
        gzip.decompress(content) if source.suffix == ".gz" else content
    )
assert len(source_hashes) == meta["sources"] == 204

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
cells: dict[str, dict[str, Any]] = {}
formulas: dict[tuple[str, str], str] = {}
worksheet_counts: dict[str, int] = {}
table_definitions: list[dict[str, Any]] = []
with zipfile.ZipFile(book) as archive:
    members = archive.namelist()
    assert len(members) == len(set(members))
    assert archive.testzip() is None
    assert all(not n.startswith("/") and ".." not in Path(n).parts for n in members)
    assert all(not info.flag_bits & 1 for info in archive.infolist())
    parsed = {
        n: ET.fromstring(archive.read(n))
        for n in members
        if n.endswith((".xml", ".rels"))
    }
    unsafe_parts = [
        n
        for n in members
        if any(
            w in n.lower()
            for w in [
                "externallink",
                "vbaproject",
                "vbasignature",
                "macrosheet",
                "connections.xml",
                "embeddings/",
            ]
        )
    ]
    assert not unsafe_parts, unsafe_parts
    external_relationships = []
    relationship_count = 0
    for name, xml in parsed.items():
        if not name.endswith(".rels"):
            continue
        base = "" if name == "_rels/.rels" else name.split("/_rels/")[0]
        for relationship in xml.findall(REL + "Relationship"):
            relationship_count += 1
            target = relationship.attrib["Target"]
            if relationship.get("TargetMode") == "External":
                external_relationships.append((name, target))
            else:
                resolved = (
                    target.lstrip("/")
                    if target.startswith("/")
                    else posixpath.normpath(posixpath.join(base, target))
                )
                assert resolved in members, (name, target, resolved)
    assert not external_relationships
    types = archive.read("[Content_Types].xml").decode("utf-8-sig").lower()
    assert not any(x in types for x in ["macroenabled", "vbaproject", "macrosheet"])
    wb = parsed["xl/workbook.xml"]
    relationships = {
        r.attrib["Id"]: r.attrib["Target"].lstrip("/")
        for r in parsed["xl/_rels/workbook.xml.rels"]
    }
    registered_sheets = wb.findall("s:sheets/s:sheet", NS)
    names = [s.attrib["name"] for s in registered_sheets]
    assert names == meta["sheets"] and len(names) == 17
    assert len({s.attrib["sheetId"] for s in registered_sheets}) == 17
    assert all(s.get("state", "visible") == "visible" for s in registered_sheets)
    shared = [
        "".join(node.itertext())
        for node in parsed["xl/sharedStrings.xml"].findall("s:si", NS)
    ]
    errors = []
    formula_type_counts = collections.Counter()
    for sheet in registered_sheets:
        name = sheet.attrib["name"]
        sheet_path = relationships[sheet.attrib[RID]]
        xml = parsed[sheet_path]
        base = posixpath.dirname(sheet_path)
        sheet_rels = base + "/_rels/" + posixpath.basename(sheet_path) + ".rels"
        table_links = (
            {r.get("Id"): r.attrib["Target"] for r in parsed[sheet_rels]}
            if sheet_rels in parsed
            else {}
        )
        for part in xml.findall("s:tableParts/s:tablePart", NS):
            target = table_links[part.attrib[RID]]
            target = (
                target.lstrip("/")
                if target.startswith("/")
                else posixpath.normpath(posixpath.join(base, target))
            )
            table = parsed[target]
            columns = table.find("s:tableColumns", NS)
            assert columns is not None
            table_definitions.append(
                {
                    "sheet": name,
                    "ref": table.get("ref"),
                    "columns": len(columns),
                    "header_rows": int(table.get("headerRowCount", "1")),
                }
            )
        values: dict[str, Any] = {}
        for cell in xml.findall("s:sheetData/s:row/s:c", NS):
            address = cell.attrib["r"]
            assert address not in values, (name, address)
            kind = cell.get("t", "n")
            v = cell.find("s:v", NS)
            if kind == "inlineStr":
                value = "".join(cell.find("s:is", NS).itertext())
            elif v is None:
                value = None
            elif kind == "s":
                value = shared[int(v.text)]
            elif kind in {"str", "e"}:
                value = v.text or ""
            elif kind == "b":
                value = int(v.text)
            else:
                value = float(v.text)
                assert math.isfinite(value), (name, address)
            if (
                kind == "e"
                or isinstance(value, str)
                and re.search(
                    r"#(?:REF!|DIV/0!|VALUE!|NAME\?|N/A|NUM!|NULL!|SPILL!|CALC!)", value
                )
            ):
                errors.append((name, address, value))
            values[address] = value
            formula = cell.find("s:f", NS)
            if formula is not None:
                assert formula.text and v is not None and value is not None
                assert "[" not in formula.text and "]" not in formula.text
                formulas[name, address] = formula.text
                formula_type_counts[kind] += 1
        assert not any(
            row.get("hidden") == "1" for row in xml.findall("s:sheetData/s:row", NS)
        )
        cells[name] = values
        worksheet_counts[name] = len(values)
    assert not errors
    archive_counts = {
        "members": len(members),
        "xml_or_relationship_parts_parsed": len(parsed),
        "relationships_resolved": relationship_count,
        "worksheets": len(names),
        "tables": len(
            [n for n in members if n.startswith("xl/tables/") and n.endswith(".xml")]
        ),
        "crc_error": None,
        "external_relationships": 0,
        "unsafe_or_macro_parts": unsafe_parts,
        "cell_error_count": 0,
    }

computed: dict[tuple[str, str], Any] = {}
visiting: set[tuple[str, str]] = set()
functions_used: collections.Counter[str] = collections.Counter()


def close(actual: Any, expected: Any) -> bool:
    """Compare export numbers at the builder's fixed tolerance, keeping missingness strict."""
    if actual is None or expected is None:
        return actual is expected
    if isinstance(expected, (float, int, bool)):
        return (
            isinstance(actual, (float, int, bool))
            and math.isfinite(float(actual))
            and abs(actual - expected) <= 1e-12 * max(1, abs(expected))
        )
    return actual == expected


def split_top(text: str, operators: str) -> list[int]:
    """Find unquoted operators outside nested function arguments."""
    depth = 0
    quote = ""
    positions = []
    for i, char in enumerate(text):
        if quote:
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and char in operators:
            positions.append(i)
    assert depth == 0 and not quote
    return positions


def arguments(text: str) -> list[str]:
    """Split exact comma-separated formula arguments without evaluating source code."""
    separators = [-1, *split_top(text, ","), len(text)]
    return [text[a + 1 : b] for a, b in pairwise(separators)]


def flatten(values: list[Any]) -> list[Any]:
    """Flatten evaluated ranges for spreadsheet aggregate functions."""
    return [v for item in values for v in (item if isinstance(item, list) else [item])]


def column_index(label: str) -> int:
    """Decode A1 column labels for rectangular ranges."""
    result = 0
    for char in label:
        result = result * 26 + ord(char) - 64
    return result


def column_label(index: int) -> str:
    """Encode a one-based column number without spreadsheet dependencies."""
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def cell_value(sheet: str, address: str) -> Any:
    """Compute formula dependencies independently of cached results."""
    key = sheet, address.replace("$", "")
    if key in computed:
        return computed[key]
    if key not in formulas:
        return cells[sheet].get(key[1])
    assert key not in visiting, ("cyclic formula", key)
    visiting.add(key)
    value = expression(sheet, formulas[key])
    visiting.remove(key)
    computed[key] = value
    return value


def expression(sheet: str, text: str) -> Any:
    """Evaluate only the explicit arithmetic/range/aggregate grammar used in this export."""
    text = text.strip()
    match = re.fullmatch(r"([A-Z]+)\((.*)\)", text)
    if match:
        function, content = match.groups()
        functions_used[function] += 1
        raw_args = arguments(content)
        if function == "IF":
            assert len(raw_args) == 3
            return expression(
                sheet, raw_args[1] if expression(sheet, raw_args[0]) else raw_args[2]
            )
        args = [expression(sheet, item) for item in raw_args]
        if function in {"SUM", "AVERAGE", "MEDIAN"}:
            values = [v for v in flatten(args) if isinstance(v, (int, float))]
            assert values
            return (
                sum(values)
                if function == "SUM"
                else (
                    statistics.mean(values)
                    if function == "AVERAGE"
                    else statistics.median(values)
                )
            )
        if function in {"COUNTIF", "COUNTIFS", "SUMIF", "AVERAGEIF", "AVERAGEIFS"}:
            if function in {"SUMIF", "AVERAGEIF"}:
                assert len(args) == 3
                checks, data = [(args[0], args[1])], args[2]
            elif function == "AVERAGEIFS":
                data, tail = args[0], args[1:]
                assert len(tail) % 2 == 0
                checks = list(zip(tail[::2], tail[1::2]))
            else:
                assert len(args) % 2 == 0
                checks = list(zip(args[::2], args[1::2]))
                data = [1] * len(checks[0][0])
            assert all(len(rng) == len(data) for rng, _ in checks)
            selected = [
                v
                for i, v in enumerate(data)
                if all(rng[i] == criterion for rng, criterion in checks)
            ]
            if function.startswith("COUNT"):
                return len(selected)
            return sum(selected) if function == "SUMIF" else statistics.mean(selected)
        raise AssertionError(("unhandled function", function))
    for operators in ["<>", "+-", "*/"]:
        positions = split_top(text, operators)
        if positions:
            i = positions[-1]
            assert i > 0
            a, b = expression(sheet, text[:i]), expression(sheet, text[i + 1 :])
            return {
                "<": operator.lt,
                ">": operator.gt,
                "+": operator.add,
                "-": operator.sub,
                "*": operator.mul,
                "/": operator.truediv,
            }[text[i]](a, b)
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    match = re.fullmatch(
        r"(?:'([^']+)'!)?(\$?[A-Z]+\$?\d+)(?::(\$?[A-Z]+\$?\d+))?", text
    )
    if match:
        other, first, last = match.groups()
        target = other or sheet
        if last is None:
            return cell_value(target, first)
        parts = [
            re.fullmatch(r"\$?([A-Z]+)\$?(\d+)", token).groups()
            for token in [first, last]
        ]
        return [
            cell_value(target, f"{column_label(col)}{row}")
            for row in range(int(parts[0][1]), int(parts[1][1]) + 1)
            for col in range(column_index(parts[0][0]), column_index(parts[1][0]) + 1)
        ]
    return float(text)


for key in formulas:
    actual = cell_value(*key)
    assert close(actual, cells[key[0]][key[1]]), (key, actual, cells[key[0]][key[1]])
assert len(computed) == len(formulas) == 557
assert len([v for v in computed.values() if isinstance(v, str)]) == 192
expected_tables = [
    {
        "sheet": s["sheet"],
        "ref": f"A{s['start']}:{column_label(s['columns'])}{s['end']}",
        "columns": s["columns"],
        "header_rows": 1,
    }
    for s in meta["sections"]
]
assert len(table_definitions) == 26 and table_definitions == expected_tables
formula_receipts_checked = 0
for check in meta["numeric_checks"]:
    assert abs(check["actual"] - check["expected"]) <= check["tolerance"] * max(
        1, abs(check["expected"])
    )
    match = re.fullmatch(r"([^!]+)!([A-Z]+\d+)", check["name"])
    if match:
        key = match.groups()
        assert key in formulas and close(computed[key], check["expected"])
        formula_receipts_checked += 1
assert formula_receipts_checked == meta["formula_checks"] == 332

source_map: dict[Path, str] = {}
for row in range(6, 210):
    entry = [cells["Sources"].get(f"{col}{row}") for col in "ABCDEF"]
    identity, logical_name, section, expected_hash, kind, purpose = entry
    assert identity == f"S{row - 5:03d}" and all(
        isinstance(v, str) and v for v in entry
    )
    logical = REPO / logical_name
    physical = (
        logical if logical.exists() else logical.with_suffix(logical.suffix + ".gz")
    )
    physical = physical.resolve()
    assert source_hashes[str(physical)] == expected_hash
    assert ("gzip" in kind) == (physical.suffix == ".gz")
    assert physical not in source_map
    source_map[physical] = identity
assert set(map(str, source_map)) == set(source_hashes)


def data(relative: str) -> tuple[Any, str]:
    """Read only already-hashed and allowlisted completed-study records."""
    logical = REPO / relative
    physical = (
        logical if logical.exists() else logical.with_suffix(logical.suffix + ".gz")
    )
    return source_values[physical.resolve()], source_map[physical.resolve()]


row_comparisons: collections.Counter[str] = collections.Counter()


def verify_rows(sheet: str, expected: list[list[Any]], start: int = 6) -> None:
    """Compare every required imported value and computed cell with its independent source projection."""
    for offset, values in enumerate(expected):
        for index, value in enumerate(values, 1):
            address = f"{column_label(index)}{start + offset}"
            assert close(cell_value(sheet, address), value), (
                sheet,
                address,
                cell_value(sheet, address),
                value,
            )
            row_comparisons[sheet] += 1


exp, exp_id = data("experiments/recursive_opt/_shared/optimizer_discovery/exp15_results.json")
s0, s0_id = data(
    "experiments/recursive_opt/_shared/optimizer_discovery/investigation16/statistics/diagnostics.json"
)
g1, g1_id = data(
    "experiments/recursive_opt/_shared/optimizer_discovery/investigation16/generation/analysis_results.json"
)
b2, b2_id = data(
    "experiments/recursive_opt/_shared/optimizer_discovery/investigation16/benchmark/b2/results.json"
)
b2freeze, _ = data(
    "experiments/recursive_opt/_shared/optimizer_discovery/investigation16/benchmark/b2/freeze.json"
)
f1, f1_id = data(
    "experiments/recursive_opt/_shared/optimizer_discovery/investigation16/feedback_experiment/analysis_results.json"
)
assert (
    len(s0["responses"]) == 80
    and len(g1["rows"]) == 24
    and len(b2["paired_rows"]) == 96
    and len(f1["rows"]) == 24
)
assert len({(r["outer_seed"], r["arm"], r["slot"]) for r in s0["responses"]}) == 80
assert len({(r["block"], r["context"], r["cap"]) for r in g1["rows"]}) == 24
assert len({r["id"] for r in b2["paired_rows"]}) == 96
assert len({(r["block"], r["condition"]) for r in f1["rows"]}) == 24
verify_rows(
    "EXP15_generations",
    [
        [
            r["outer_seed"],
            r["arm"],
            r["slot"],
            r["eligible"],
            r["source_status"],
            r["finish_reason"],
            r["provider"],
            r["prompt_tokens"],
            r["completion_tokens"],
            r["reasoning_tokens"],
            r["cost_usd"],
            r["source_bytes"],
            r["source_sha256"],
            s0_id,
        ]
        for r in s0["responses"]
    ],
)
verify_rows(
    "EXP15",
    [
        [
            block["outer_seed"],
            arm,
            r["auc"],
            r["final_regret"],
            r["target_attainment"],
            r["capped_target_evaluations"],
            r["fallback_trajectories"],
            r["candidate_valid_trajectories"],
            exp_id,
        ]
        for block in exp["per_seed"]
        for arm in ["A0", "A1", "A2"]
        for r in [block[arm]]
    ],
)
verify_rows(
    "G1",
    [
        [
            r["block"],
            r["context"],
            r["cap"],
            r["eligible"],
            r["source_status"],
            r["finish_reason"],
            r["auc"],
            r["usage"]["prompt_tokens"],
            r["usage"]["completion_tokens"],
            r["usage"]["total_tokens"],
            r["usage"]["cost_usd"],
            r["wall_s"],
            r["transport_attempts"],
            (r.get("receipt") or {}).get("provider_name"),
            r["source_sha256"],
            g1_id,
        ]
        for r in g1["rows"]
    ],
)
allocs = {r["id"]: r for r in b2freeze["allocations"]}
b2rows = []
for pair in b2["paired_rows"]:
    a = allocs[pair["id"]]
    c = pair["comparison"]
    sm, vm = c["seed"]["metrics"], c["variant"]["metrics"]
    source_ids = []
    for kind, name in [
        (
            "seed",
            "experiments/recursive_opt/_shared/optimizer_discovery/investigation16/benchmark/"
            + a["control_path"],
        ),
        (
            "variant",
            "experiments/recursive_opt/_shared/optimizer_discovery/investigation16/benchmark/b2/raw/"
            + pair["id"]
            + ".json",
        ),
    ]:
        raw, identity = data(name)
        assert raw["valid"] and raw["candidate_valid"] and not raw["fallback_used"]
        assert close(raw["metrics"]["auc"], c[kind]["metrics"]["auc"]) and close(
            raw["metrics"]["final_regret"], c[kind]["metrics"]["final_regret"]
        )
        source_ids.append(identity)
    auc_delta, final_delta = (
        vm["auc"] - sm["auc"],
        vm["final_regret"] - sm["final_regret"],
    )
    labels = [
        "gain" if value < 0 else "perte" if value > 0 else "égalité"
        for value in [auc_delta, final_delta]
    ]
    b2rows.append(
        [
            pair["id"],
            a["condition"],
            f"{a['task']['family']}/{a['task']['dimension']}",
            a["local_seed"],
            sm["auc"],
            vm["auc"],
            auc_delta,
            sm["final_regret"],
            vm["final_regret"],
            final_delta,
            *labels,
            c["seed"]["valid"] == 1,
            c["variant"]["valid"] == 1,
            *source_ids,
            pair["control_sha256"],
            pair["variant_sha256"],
        ]
    )
verify_rows("B2", b2rows)
conditions = ["legacy_code", "anytime_code", "anytime_sparse", "anytime_rich"]
frows = sorted(f1["rows"], key=lambda r: (r["block"], conditions.index(r["condition"])))
verify_rows(
    "F1",
    [
        [
            r["block"],
            r["condition"],
            r["parent_kind"],
            r["validation_auc"],
            r["final_regret"],
            r["target_attainment"],
            r["capped_target_evaluations"],
            r["train_valid"],
            r["candidate_valid_trajectories"],
            r["fallback_trajectories"],
            r["source_status"],
            r["finish_reason"],
            r["parent_auc"],
            r["seed_auc"],
            r["source_sha256"],
            f1_id,
        ]
        for r in frows
    ],
)
verify_rows(
    "F1_usage",
    [
        [
            r["block"],
            r["condition"],
            r["source_status"],
            r["finish_reason"],
            r["usage"]["prompt_tokens"],
            r["usage"]["completion_tokens"],
            r["usage"]["reasoning_tokens"],
            r["usage"]["total_tokens"],
            r["usage"]["cost_usd"],
            (r.get("receipt") or {}).get("total_cost"),
            r["transport_attempts"],
            r["possible_remote_completion_attempts"],
            r["wall_s"],
            (r.get("receipt") or {}).get("provider_name"),
            r["actual_objective_calls"],
            r["unused_objective_allocations"],
            r["subprocess_executions"],
            " ; ".join(r["usage_issues"]) or None,
            f1_id,
        ]
        for r in frows
    ],
)
verify_rows(
    "F1_contrastes",
    [
        [
            name,
            *r["deltas"],
            r["mean"],
            r["median"],
            *r["paired_bootstrap_95"],
            r["interpretation"],
            f1_id,
        ]
        for name, r in f1["contrasts"].items()
    ],
)
assert all(
    cells["G1"].get(f"G{i+6}") is None
    for i, r in enumerate(g1["rows"])
    if not r["eligible"]
)
retention = {
    "EXP15_generations": {
        "rows": 80,
        "ineligible": sum(not r["eligible"] for r in s0["responses"]),
        "length": sum(r["finish_reason"] == "length" for r in s0["responses"]),
    },
    "G1": {
        "rows": 24,
        "ineligible": sum(not r["eligible"] for r in g1["rows"]),
        "invalid_auc_cells_blank": sum(not r["eligible"] for r in g1["rows"]),
    },
    "B2": {
        "pairs": 96,
        "auc_losses": sum(row[6] > 0 for row in b2rows),
        "final_regret_losses": sum(row[9] > 0 for row in b2rows),
        "all_controls_and_variants_valid": True,
    },
    "F1": {
        "rows": 24,
        "ineligible": sum(not r["train_valid"] for r in frows),
        "fallback_trajectories": sum(r["fallback_trajectories"] for r in frows),
        "provider_error_responses": sum(r["finish_reason"] == "error" for r in frows),
        "transport_attempts": sum(r["transport_attempts"] for r in frows),
        "all_four_contrasts_retained": len(f1["contrasts"]) == 4,
    },
}
assert retention["B2"]["auc_losses"] == 26
assert (
    retention["F1"]["ineligible"] == 4
    and retention["F1"]["fallback_trajectories"] == 24
)
assert retention["EXP15_generations"]["length"] == 16
for filename, expected in source_hashes.items():
    assert sha(Path(filename)) == expected
assert sha(book) == book_before and sha(BUILDER) == builder_before
report = {
    "schema": "investigation16.independent_workbook_review.v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "workbook": str(book.relative_to(REPO)),
    "workbook_sha256": book_before,
    "builder_sha256": builder_before,
    "builder_checks_sha256": sha(META),
    "scope": {
        "method": "Python standard-library ZIP/XML read and independent restricted formula evaluator; no spreadsheet or research runner imported",
        "model_calls": 0,
        "objective_calls": 0,
        "candidate_executions": 0,
        "p1_data_reads": 0,
        "credential_loads": 0,
        "network_calls": 0,
        "scientific_raw_or_workbook_changes": 0,
    },
    "archive": archive_counts,
    "sheet_names": names,
    "worksheet_cell_counts": worksheet_counts,
    "formula_checks": {
        "formulas": len(formulas),
        "all_have_cached_values": True,
        "independently_recomputed": len(computed),
        "numeric_results": formula_type_counts["n"],
        "string_results": formula_type_counts["str"],
        "function_calls_evaluated": dict(functions_used),
        "builder_expected_formula_checks_reconciled": formula_receipts_checked,
        "builder_numeric_checks_consistent": len(meta["numeric_checks"]),
        "scaled_absolute_tolerance": 1e-12,
        "external_references": 0,
    },
    "source_checks": {
        "registered_sources": len(source_hashes),
        "all204_exact_byte_hashes_match_registry_and_source_sheet": True,
        "all204_unchanged_before_after": True,
        "output_and_builder_unchanged_before_after": True,
        "source_paths_allowlisted_before_read": True,
        "input_byte_hashes": source_hashes,
    },
    "table_checks": {
        "exact_registered_table_rectangles": 26,
        "all_table_rows_and_columns_match": True,
        "tables": table_definitions,
        "primary_data_row_counts": {
            "EXP15_generations": 80,
            "G1": 24,
            "B2": 96,
            "F1": 24,
        },
        "source_data_rows": 204,
    },
    "retention": retention,
    "cells_compared_to_source_projections": dict(row_comparisons),
    "limitations": [
        "No Excel or LibreOffice application was started; the restricted evaluator verifies the exact formula grammar present, not universal spreadsheet compatibility.",
        "Visual layout, print pagination and accessibility are outside this XML review and handled separately by the coordinator.",
        "Source hashes prove byte identity; source projections verify the named exported tables. No objective, candidate or statistical bootstrap was rerun.",
        "The review does not independently project every imported cell of B1/S1/S0/T1; all formula caches and all204 source identities were verified, and the builder records additional source-value checks.",
        "P1 is deliberately excluded. Completion of this workbook does not complete the active research inquiry.",
    ],
    "verification_program_sha256": sha(Path(__file__)),
    "verification_program_path": str(Path(__file__).resolve().relative_to(REPO)),
}
if args.output is not None:
    assert not args.output.exists(), "Existing review evidence cannot be overwritten."
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(
    json.dumps(
        {
            k: report[k]
            for k in [
                "status",
                "archive",
                "formula_checks",
                "retention",
                "cells_compared_to_source_projections",
            ]
        },
        ensure_ascii=False,
    )
)
