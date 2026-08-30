"""Probe N — measure Experiment-0's invalid-extraction rate to a usable precision.

The frozen main run stopped on `invalid_rate <= 0` after 1 empty extraction in 24.
Probe M's replication saw 0 in 24, whose Wilson interval (0.000, 0.138) contains the
original 0.042 — so it settled nothing. This measures the rate properly, and separates
two causes that demand different fixes:

  * SAMPLE-SPECIFIC: certain questions reliably defeat the extraction regex. Repeats of
    the same sample then agree, and the fix is the extractor or the prompt.
  * TRANSIENT: the provider intermittently returns unusable content. Repeats of the same
    sample then disagree, and the fix is retry/validation at the transport.

Part A repeats the frozen 24-sample holdout to expose sample-level determinism.
Part B draws fresh GSM8K test samples, disjoint from every frozen pool, for an
independent-observation estimate of the overall rate.

Diagnostic only: no optimizer runs, nothing is fitted, and no frozen artifact is touched.
"""
import json, os, statistics as st, sys, time
from collections import Counter
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
REPEATS = int(os.environ.get("PROBE_REPEATS", "4"))
FRESH_N = int(os.environ.get("PROBE_FRESH", "150"))

instruction = FROZEN["initial_artifact"]
if isinstance(instruction, dict):
    instruction = next(iter(instruction.values()))


def _call(llm, prompt):
    response = llm(messages=[{"role": "user", "content": prompt}],
                   temperature=FWD["temperature"], max_tokens=FWD["max_tokens"],
                   **FWD.get("request_params", {}))
    return {"content": components._response_text(response),
            "provider": components._provider_metadata(response),
            "usage": components._usage_dict(response)}


def evaluate(job):
    """Run the two-call workflow once; a failure is recorded, never scored."""
    label, sample_id, question, expected, rep = job
    llm = make_live_llm(MODEL, budget_resource=None, empty_response_retries=0)
    try:
        p1 = components._analysis_prompt(instruction, question)
        analysis = _call(llm, str(getattr(p1, "data", p1)))
        p2 = components._answer_prompt(instruction, question, analysis)
        answer = _call(llm, str(getattr(p2, "data", p2)))
    except Exception as exc:
        return {"part": label, "id": sample_id, "rep": rep, "error": True,
                "reason": f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"}
    produced = evaluator._extract(answer["content"], "numeric")
    return {"part": label, "id": sample_id, "rep": rep, "error": False,
            "invalid": not produced,
            "correct": bool(produced) and produced == evaluator._extract(str(expected), "numeric"),
            "finish_reason": answer["provider"].get("finish_reason"),
            "completion_tokens": answer["usage"].get("completion_tokens"),
            "raw_len": len(answer["content"]),
            "raw_excerpt": answer["content"][:300] if not produced else ""}


holdout = datasets._resolve_v2("gsm8k", "holdout", {})
frozen_idx = set()
for split in ("train", "validation", "holdout"):
    frozen_idx |= {r["source_index"] for r in datasets._resolve_v2("gsm8k", split, {})}
source = datasets._source_rows("gsm8k", "test")
fresh = [(i, source[i]) for i in range(len(source)) if i not in frozen_idx][:FRESH_N]

jobs = [("A_holdout", r["id"], r["question"], r["expected"], k)
        for r in holdout for k in range(REPEATS)]
jobs += [("B_fresh", f"gsm8k:test:{i}", row["question"], row["expected"], 0)
         for i, row in fresh]
print(f"model={MODEL} temp={FWD['temperature']} max_tokens={FWD['max_tokens']}")
print(f"Part A: {len(holdout)} frozen holdout x {REPEATS} repeats | "
      f"Part B: {len(fresh)} fresh disjoint samples")
print(f"total evaluations={len(jobs)} forward calls={2*len(jobs)} workers={WORKERS}", flush=True)

t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    rows = list(ex.map(evaluate, jobs))
wall = time.time() - t0

ok = [r for r in rows if not r["error"]]
errors = [r for r in rows if r["error"]]
def rate(subset):
    n = len(subset)
    k = sum(r["invalid"] for r in subset)
    lo, hi = wilson_interval(k, n)
    return k, n, (k / n if n else 0.0), lo, hi

print(f"\nwall={wall:.0f}s  evaluated={len(ok)}  transport errors={len(errors)}")
for label in ("A_holdout", "B_fresh"):
    sub = [r for r in ok if r["part"] == label]
    if not sub:
        continue
    k, n, p, lo, hi = rate(sub)
    print(f"  {label:10s} invalid {k}/{n} = {p:.4f}  95% CI ({lo:.4f}, {hi:.4f})  "
          f"accuracy {sum(r['correct'] for r in sub)}/{n}")
k, n, p, lo, hi = rate(ok)
print(f"  {'POOLED':10s} invalid {k}/{n} = {p:.4f}  95% CI ({lo:.4f}, {hi:.4f})")

# sample-specific vs transient
per_sample = {}
for r in (r for r in ok if r["part"] == "A_holdout"):
    per_sample.setdefault(r["id"], []).append(r["invalid"])
disagreeing = {s: v for s, v in per_sample.items() if 0 < sum(v) < len(v)}
always = {s: v for s, v in per_sample.items() if v and all(v)}
print(f"\nPart A repeat structure ({REPEATS} repeats each):")
print(f"  samples ALWAYS invalid    : {len(always)}  -> sample-specific (extractor/prompt)")
print(f"  samples SOMETIMES invalid : {len(disagreeing)} -> transient (transport)")
for s, v in list(disagreeing.items())[:5]:
    print(f"     {s}: {sum(v)}/{len(v)} invalid")

for r in (r for r in ok if r["invalid"])  :
    print(f"\n  INVALID {r['id']} rep{r['rep']} finish_reason={r['finish_reason']} "
          f"tokens={r['completion_tokens']} raw_len={r['raw_len']}")
    print(f"     raw: {r['raw_excerpt'][:200]!r}")

Path(__file__).with_name("probe_n_results.json").write_text(json.dumps(
    {"model": MODEL, "forward_profile": FWD, "wall_s": wall, "repeats": REPEATS,
     "pooled": {"invalid": k, "n": n, "rate": p, "ci": [lo, hi]},
     "transport_errors": len(errors), "rows": rows}, indent=2, default=str))
print("\nWROTE probe_n_results.json")
