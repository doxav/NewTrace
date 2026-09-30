"""Probe A — is the config->score surface non-flat, or is it evaluator noise?

The whole recursive_opt programme presupposes that changing `starting_artifact`
moves the benchmark score. This measures, on the two tasks used by the flagship
UC4 experiment:

  * BETWEEN-prompt spread : max-min over 5 distinct prompts (the claimed signal)
  * WITHIN-prompt noise   : stdev of repeated runs of the SAME prompt (the floor)

A claimed "gain" is only real if it exceeds the noise floor.
inner_steps=0 => no nested trainer, no optimizer LLM calls: this isolates the
evaluation surface itself.
"""
import json, statistics, sys, time
from pathlib import Path

from opto.features.recursive_opt import tracebench as TB
from opto.features.recursive_opt.levels import LevelConfig

# The three probes score_spread() itself uses by default.
ART_MENU = [
    "",
    "Answer directly.",
    "Plan step by step, then verify the answer before replying.",
]
TASKS = ["internal:multiobjective_gsm8k", "hf:qasper"]
REPEATS = 3
MAX_EXAMPLES = 2
RUN_TIMEOUT_S = 150   # a provider call with no timeout stalled the first attempt


class _RunTimeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _RunTimeout(f"evaluation exceeded {RUN_TIMEOUT_S}s")


import signal
signal.signal(signal.SIGALRM, _alarm)

adapter = TB.TraceBenchTaskAdapter(
    max_examples=MAX_EXAMPLES, inner_steps=0, eval_kwargs={"timeout_seconds": 90}
)
TB.register_task_adapter(adapter)

out = {"model": "openrouter/deepseek/deepseek-v4-flash-0731",
       "max_examples": MAX_EXAMPLES, "repeats": REPEATS, "inner_steps": 0,
       "tasks": {}}

for task in TASKS:
    per_prompt = {}
    for prompt in ART_MENU:
        runs = []
        for r in range(REPEATS):
            t0 = time.time()
            signal.alarm(RUN_TIMEOUT_S)
            try:
                s, _fb = adapter.run_task(LevelConfig(starting_artifact=prompt), task)
                runs.append(float(s))
            except Exception as exc:
                runs.append(None)
                print(f"  !! {task} r{r}: {type(exc).__name__}: {exc}", flush=True)
            finally:
                signal.alarm(0)
            print(f"  {task:32s} prompt[{ART_MENU.index(prompt)}] r{r} "
                  f"score={runs[-1]} ({time.time()-t0:.1f}s)", flush=True)
        ok = [v for v in runs if v is not None]
        per_prompt[prompt or "<empty>"] = {
            "runs": runs,
            "mean": statistics.mean(ok) if ok else None,
            "within_stdev": statistics.pstdev(ok) if len(ok) > 1 else None,
        }
    means = [v["mean"] for v in per_prompt.values() if v["mean"] is not None]
    noises = [v["within_stdev"] for v in per_prompt.values()
              if v["within_stdev"] is not None]
    out["tasks"][task] = {
        "per_prompt": per_prompt,
        "between_prompt_spread": (max(means) - min(means)) if means else None,
        "mean_within_prompt_noise": statistics.mean(noises) if noises else None,
        "pooled_mean": statistics.mean(means) if means else None,
    }
    t = out["tasks"][task]
    print(f"== {task}: spread={t['between_prompt_spread']} "
          f"noise={t['mean_within_prompt_noise']}", flush=True)

Path(__file__).with_name("probe_a_results.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print("\nWROTE probe_a_results.json")
