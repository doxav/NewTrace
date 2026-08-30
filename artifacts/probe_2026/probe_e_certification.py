"""Probe E - certify a candidate task pool before spending an experiment on it.

Runs `measurement.certify_task` over a spread of task families. The point is not to
find winners but to make the instrument's limits explicit: a task that is broken,
saturated, or too noisy to resolve the target effect is rejected BEFORE it consumes an
experiment budget. `internal:multiobjective_bbeh` is included deliberately as a known
negative control - it must come back `broken`.
"""
import json, time
from pathlib import Path

from opto.features.recursive_opt import measurement as M

TASKS = [
    "internal:multiobjective_gsm8k",   # known: quiet at temp 0.2
    "hf:qasper",                       # known: graded score
    "internal:multiobjective_bbeh",    # known BROKEN - negative control
    "internal:multi_param",            # used by example A
    "internal:code_param",             # used by example B
    "hf:drop",
    "llm4ad:optimization/online_bin_packing",   # deterministic evaluator
    "veribench:binary_search",                  # deterministic evaluator
]
TARGET_DELTA, TARGET_N, REPEATS, MAX_EXAMPLES = 0.05, 5, 3, 4

t0 = time.time()
certs = M.certify_pool(TASKS, repeats=REPEATS, max_examples=MAX_EXAMPLES,
                       target_delta=TARGET_DELTA, target_n=TARGET_N, max_workers=8)
print(M.format_certificates(certs, target_n=TARGET_N))
print(f"\nwall={time.time()-t0:.0f}s")

usable = [t for t, c in certs.items() if c.usable]
print(f"\nCERTIFIED ({len(usable)}/{len(TASKS)}): {usable}")
for task, c in sorted(certs.items()):
    if not c.usable:
        print(f"  rejected {task:42s} {c.verdict:10s} {(c.reasons[0] if c.reasons else '')[:90]}")

Path(__file__).with_name("probe_e_results.json").write_text(
    json.dumps({t: c.to_dict() for t, c in certs.items()}, indent=2, default=str))
print("\nWROTE probe_e_results.json")
