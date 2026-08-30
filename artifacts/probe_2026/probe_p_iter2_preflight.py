"""Iteration 2 pre-flight — can bbeh_object_counting satisfy Experiment-0's constraint?

HYPOTHESIS: on bbeh_object_counting with `invalid_rate <= 0.02` measured POOLED rather
than zero-per-unit, the frozen 40-unit matrix can complete.

KILL CONDITION: the measured invalid rate makes P(all 40 units pass) < 0.8 under the
constraint as written.

Section 16.2 showed the original design could not complete whatever the optimizer did:
`invalid_rate <= 0` on 24 samples per unit gives P(all 40 pass) = 0.0001 at a 1% rate.
This measures the replacement task's rate at the concurrency the run will use - Probe K
showed a sequential noise floor is not the floor an experiment sees.
"""
import json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
PKG = Path(__file__).resolve().parents[2] / "experiments/recursive_opt/multiobjective_reasoning"

from experiments.recursive_opt.multiobjective_reasoning import components, datasets, evaluator  # noqa: E402
from opto.features.recursive_opt.measurement import wilson_interval  # noqa: E402
from opto.features.recursive_opt.runmode import make_live_llm  # noqa: E402

FROZEN = json.loads((PKG / "manifests/preregistration_frozen.json").read_text())
FWD, MODEL = FROZEN["model_profiles"]["forward"], FROZEN["model_profiles"]["resolved_model"]
TASK = "bbeh_object_counting"
WORKERS = int(os.environ.get("PROBE_WORKERS", "8"))
N = int(os.environ.get("PROBE_N", "160"))
UNITS, HOLDOUT = 40, 24

instruction = FROZEN["initial_artifact"]
if isinstance(instruction, dict):
    instruction = next(iter(instruction.values()))


def evaluate(item):
    llm = make_live_llm(MODEL, budget_resource=None, empty_response_retries=0)
    try:
        p1 = components._analysis_prompt(instruction, str(item["question"]))
        a = llm(messages=[{"role": "user", "content": str(getattr(p1, "data", p1))}],
                temperature=FWD["temperature"], max_tokens=FWD["max_tokens"],
                **FWD.get("request_params", {}))
        p2 = components._answer_prompt(instruction, str(item["question"]),
                                       {"content": components._response_text(a)})
        b = llm(messages=[{"role": "user", "content": str(getattr(p2, "data", p2))}],
                temperature=FWD["temperature"], max_tokens=FWD["max_tokens"],
                **FWD.get("request_params", {}))
        text = components._response_text(b)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:110]}"}
    produced = evaluator._extract(text, "numeric")
    return {"error": None, "invalid": not produced,
            "correct": bool(produced) and produced == evaluator._extract(str(item["expected"]), "numeric"),
            "raw_excerpt": "" if produced else text[:200]}


pool = []
for split in ("holdout", "train", "validation"):
    pool += datasets._resolve_v2(TASK, split, {})
pool = pool[:N]
print(f"task={TASK} model={MODEL} n={len(pool)} concurrency={WORKERS}", flush=True)

t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    rows = list(ex.map(evaluate, pool))
wall = time.time() - t0

ok = [r for r in rows if not r["error"]]
k_inv = sum(r["invalid"] for r in ok)
k_acc = sum(r["correct"] for r in ok)
n = len(ok)
ilo, ihi = wilson_interval(k_inv, n)
alo, ahi = wilson_interval(k_acc, n)

print(f"\nwall={wall:.0f}s  evaluated={n}  transport errors={len(rows)-n}")
print(f"invalid  : {k_inv}/{n} = {k_inv/n:.4f}   95% CI ({ilo:.4f}, {ihi:.4f})")
print(f"accuracy : {k_acc}/{n} = {k_acc/n:.4f}   95% CI ({alo:.4f}, {ahi:.4f})")
print(f"headroom : {1-k_acc/n:.4f}")

print(f"\nP(all {UNITS} units pass) with a {HOLDOUT}-sample holdout:")
verdict = {}
for label, tol in (("invalid_rate <= 0 (as frozen)", 0.0), ("invalid_rate <= 0.02 (proposed)", 0.02)):
    # per-unit pass probability: observed rate must not exceed the tolerance
    allowed = math.floor(tol * HOLDOUT)
    p = ihi  # use the pessimistic bound, not the point estimate
    per_unit = sum(math.comb(HOLDOUT, j) * p**j * (1-p)**(HOLDOUT-j) for j in range(allowed+1))
    verdict[label] = per_unit ** UNITS
    print(f"   {label:34s} allows {allowed} invalid/unit -> "
          f"P(unit)={per_unit:.3f}  P(all {UNITS})={per_unit**UNITS:.4f}")

passes = verdict["invalid_rate <= 0.02 (proposed)"] >= 0.8
print(f"\nKILL CONDITION (P(all 40) < 0.8 under the proposed constraint): "
      f"{'NOT TRIGGERED - proceed' if passes else 'TRIGGERED - constraint still too tight'}")
for r in (r for r in ok if r["invalid"]):
    print(f"   invalid excerpt: {r['raw_excerpt'][:120]!r}")

Path(__file__).with_name("probe_p_results.json").write_text(json.dumps(
    {"task": TASK, "n": n, "invalid": k_inv, "invalid_ci": [ilo, ihi],
     "accuracy": k_acc / n if n else None, "accuracy_ci": [alo, ahi],
     "wall_s": wall, "workers": WORKERS, "p_all_units": verdict}, indent=2))
print("\nWROTE probe_p_results.json")
