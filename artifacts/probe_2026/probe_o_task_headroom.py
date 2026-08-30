"""Probe O — which task gives Experiment-0 room to measure anything?

Section 16.3: arm A, the FIXED baseline with no optimizer, already scores 98.4% on GSM8K.
An optimizer cannot demonstrate a gain in 1.6 points, and the design provides 15x too few
samples to resolve even that. Replacing the task is therefore step 1, and the choice must
be made on measured headroom rather than on a task's reputation for difficulty.

Measures, for each candidate, the baseline accuracy under the FROZEN forward profile, and
reports the headroom plus the samples-per-arm needed to detect a plausible gain.
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
WORKERS = int(os.environ.get("PROBE_WORKERS", "8"))
N = int(os.environ.get("PROBE_N", "48"))
TASKS = ["gsm8k", "bbeh_object_counting", "bbeh_boolean_expressions"]

instruction = FROZEN["initial_artifact"]
if isinstance(instruction, dict):
    instruction = next(iter(instruction.values()))


def n_per_arm(p1, gain):
    """Samples per arm for 80% power at alpha=0.05 to detect `gain` over p1."""
    p2 = min(0.9999, p1 + gain)
    if p2 <= p1:
        return None
    za, zb, pbar = 1.959963985, 0.841621234, (p1 + p2) / 2
    return math.ceil((za * math.sqrt(2 * pbar * (1 - pbar)) +
                      zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2)


def evaluate(job):
    task, item = job
    kind = "choice" if "boolean" in task else "numeric"
    llm = make_live_llm(MODEL, budget_resource=None, empty_response_retries=0)
    try:
        p1 = components._analysis_prompt(instruction, str(item["question"]))
        a = llm(messages=[{"role": "user", "content": str(getattr(p1, "data", p1))}],
                temperature=FWD["temperature"], max_tokens=FWD["max_tokens"],
                **FWD.get("request_params", {}))
        acall = {"content": components._response_text(a)}
        p2 = components._answer_prompt(instruction, str(item["question"]), acall)
        b = llm(messages=[{"role": "user", "content": str(getattr(p2, "data", p2))}],
                temperature=FWD["temperature"], max_tokens=FWD["max_tokens"],
                **FWD.get("request_params", {}))
        text = components._response_text(b)
    except Exception as exc:
        return {"task": task, "error": f"{type(exc).__name__}: {str(exc)[:100]}"}
    produced = evaluator._extract(text, kind)
    return {"task": task, "error": None, "invalid": not produced,
            "correct": bool(produced) and produced == evaluator._extract(str(item["expected"]), kind)}


jobs = []
for task in TASKS:
    try:
        pool = datasets._resolve_v2(task, "holdout", {}) + datasets._resolve_v2(task, "train", {})
        jobs += [(task, item) for item in pool[:N]]
    except Exception as exc:
        print(f"  {task}: pool unavailable ({type(exc).__name__}: {exc})", flush=True)

print(f"model={MODEL}  candidates={TASKS}  n<={N} each  workers={WORKERS}", flush=True)
t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    rows = list(ex.map(evaluate, jobs))
wall = time.time() - t0

out = {}
print(f"\nwall={wall:.0f}s\n")
print(f"{'task':28s} {'accuracy':>16s} {'headroom':>9s} {'invalid':>9s}  n/arm for +5pp")
print("-" * 84)
for task in TASKS:
    sub = [r for r in rows if r["task"] == task and not r["error"]]
    if not sub:
        continue
    k, n = sum(r["correct"] for r in sub), len(sub)
    p = k / n
    lo, hi = wilson_interval(k, n)
    inv = sum(r["invalid"] for r in sub)
    need = n_per_arm(p, 0.05)
    out[task] = {"accuracy": p, "ci": [lo, hi], "n": n, "invalid": inv,
                 "headroom": 1 - p, "n_per_arm_5pp": need}
    print(f"{task:28s} {k:3d}/{n:<3d}={p:.3f} ({lo:.2f}-{hi:.2f}) {1-p:>8.3f} "
          f"{inv:>4d}/{n:<4d} {str(need):>14s}")

best = [t for t, v in out.items() if 0.05 <= v["accuracy"] <= 0.9]
print(f"\nusable headroom (accuracy in 5%-90%): {best or 'NONE of the candidates'}")
Path(__file__).with_name("probe_o_results.json").write_text(json.dumps(
    {"model": MODEL, "wall_s": wall, "tasks": out}, indent=2, default=str))
print("WROTE probe_o_results.json")
