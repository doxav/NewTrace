"""Probe M — diagnostic replication of Experiment-0's first main unit.

The 2026-08-24 frozen main run stopped on its FIRST unit (seed 0, budget 6, arm A)
against the hard constraint `invalid_rate <= 0`: one of 24 holdout samples
(`gsm8k:test:915`, expected 23) produced an empty deterministic extraction. Its own
report recorded that "the raw provider response text is not persisted, so the exact
upstream formatting cause is unknown and is not inferred".

This is NOT a resumption of that frozen run, and must not be reported as one. Changing
`evaluator.py` to persist the raw text on failure changes the Experiment-0 source hash,
so `_load_main_lock` correctly refuses to continue the frozen matrix. Re-freezing would
invalidate the preregistration. This is a debugging replication that answers one
question: *why did that extraction come back empty?*

It uses the frozen model profile exactly (temperature 0, max_tokens 384, reasoning
disabled) and runs the holdout in parallel, capturing the raw text for EVERY sample.
"""
import json, os, statistics as st, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
PKG = Path(__file__).resolve().parents[2] / "experiments/recursive_opt/multiobjective_reasoning"
sys.path.insert(0, str(PKG.parents[1]))

from experiments.recursive_opt.multiobjective_reasoning import components, datasets, evaluator  # noqa: E402
from opto.features.recursive_opt.runmode import make_live_llm  # noqa: E402

FROZEN = json.loads((PKG / "manifests/preregistration_frozen.json").read_text())
PROFILE = FROZEN["model_profiles"]
FWD = PROFILE["forward"]
HOLDOUT_N = FROZEN["main"]["split_limits"]["holdout"]
MODEL = PROFILE["resolved_model"]
WORKERS = int(os.environ.get("PROBE_WORKERS", "8"))

print(f"model={MODEL} temperature={FWD['temperature']} max_tokens={FWD['max_tokens']} "
      f"request_params={FWD.get('request_params')}", flush=True)

# _resolve() serves the small default indices (2/4/2); the frozen run used the v2 pools.
holdout = datasets._resolve_v2("gsm8k", "holdout", {})[:HOLDOUT_N]
print(f"holdout samples: {len(holdout)}", flush=True)

instruction = FROZEN["initial_artifact"]
if isinstance(instruction, dict):
    instruction = next(iter(instruction.values()))


def _call(llm, prompt):
    started = time.time()
    response = llm(messages=[{"role": "user", "content": prompt}],
                   temperature=FWD["temperature"], max_tokens=FWD["max_tokens"],
                   **FWD.get("request_params", {}))
    text = components._response_text(response)
    meta = components._provider_metadata(response)
    return {"content": text, "latency_s": time.time() - started, "provider": meta,
            "usage": components._usage_dict(response)}


def run_sample(item):
    """Reproduce the two-call compound workflow for one holdout sample."""
    llm = make_live_llm(MODEL, budget_resource=None, empty_response_retries=0)
    question = str(item["question"])
    try:
        # both prompt builders are @bundle()-decorated and return traced nodes
        analysis_prompt = components._analysis_prompt(instruction, question)
        analysis = _call(llm, str(getattr(analysis_prompt, "data", analysis_prompt)))
        answer_prompt = components._answer_prompt(instruction, question, analysis)
        answer = _call(llm, str(getattr(answer_prompt, "data", answer_prompt)))
    except Exception as exc:
        return {"id": item["id"], "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    produced = evaluator._extract(answer["content"], str(item.get("task_kind", "numeric")))
    expected = evaluator._extract(str(item["expected"]), str(item.get("task_kind", "numeric")))
    return {"id": item["id"], "produced": produced, "expected": expected,
            "invalid": not produced, "correct": bool(produced) and produced == expected,
            "answer_finish_reason": answer["provider"].get("finish_reason"),
            "answer_tokens": answer["usage"].get("completion_tokens"),
            "answer_raw": answer["content"],
            "analysis_finish_reason": analysis["provider"].get("finish_reason"),
            "analysis_tokens": analysis["usage"].get("completion_tokens")}


t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    rows = list(ex.map(run_sample, holdout))
wall = time.time() - t0

ok = [r for r in rows if "error" not in r]
invalid = [r for r in ok if r["invalid"]]
truncated = [r for r in ok if r.get("answer_finish_reason") == "length"]
print(f"\nwall={wall:.0f}s  ({WORKERS}-way parallel; the frozen sequential run took 199s)")
print(f"samples={len(rows)} evaluated={len(ok)} errors={len(rows)-len(ok)}")
print(f"accuracy    : {sum(r['correct'] for r in ok)}/{len(ok)}")
print(f"invalid_rate: {len(invalid)}/{len(ok)}   (hard constraint requires 0)")
print(f"truncated at max_tokens={FWD['max_tokens']}: {len(truncated)}/{len(ok)}")

for r in invalid:
    print(f"\n--- INVALID {r['id']} (expected {r['expected']}) ---")
    print(f"    answer finish_reason={r['answer_finish_reason']} "
          f"completion_tokens={r['answer_tokens']}")
    print(f"    analysis finish_reason={r['analysis_finish_reason']} "
          f"completion_tokens={r['analysis_tokens']}")
    print(f"    raw answer ({len(r['answer_raw'])} chars): {r['answer_raw'][:400]!r}")

Path(__file__).with_name("probe_m_results.json").write_text(
    json.dumps({"model": MODEL, "forward_profile": FWD, "wall_s": wall,
                "workers": WORKERS, "rows": rows}, indent=2, default=str))
print("\nWROTE probe_m_results.json")
