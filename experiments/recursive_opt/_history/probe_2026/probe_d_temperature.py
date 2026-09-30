"""Probe D — how much of the noise floor is sampling temperature?

Stage-0 showed the evaluation example set is already byte-stable across bundle loads,
so the run-to-run variance in §7.1 is not example difficulty: it is response sampling.
`learner.call_llm` passes no `temperature`, so every measurement runs at the provider
default (~1.0) — maximal sampling noise on what is supposed to be a measurement.

A first attempt at temperature=0.0 stalled indefinitely (greedy decoding sending this
model into degenerate repetition), which is itself worth knowing: an unbounded request
turns a measurement into a hang. Every arm here is therefore bounded by max_tokens and
a request timeout, so the comparison is like-for-like and terminates.
"""
import json, statistics as st, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from opto.features.recursive_opt import tracebench as TB
from opto.features.recursive_opt.levels import LevelConfig

PROMPT = "Answer directly."
TASKS = ["internal:multiobjective_gsm8k", "hf:qasper"]
REPEATS = 3
MAX_EXAMPLES = 4
TEMPERATURES = (("provider_default", None), ("temperature_0_2", 0.2))
MAX_TOKENS, REQUEST_TIMEOUT = 512, 60
MAX_WORKERS = 6


class BoundedLLM:
    """Bound an evaluation call: explicit temperature, token cap, request timeout."""

    def __init__(self, inner, temperature, max_tokens=MAX_TOKENS, timeout=REQUEST_TIMEOUT):
        self._inner, self._t = inner, temperature
        self._max_tokens, self._timeout = max_tokens, timeout

    def __call__(self, *a, **kw):
        if self._t is not None:
            kw.setdefault("temperature", self._t)
        kw.setdefault("max_tokens", self._max_tokens)
        kw.setdefault("timeout", self._timeout)
        return self._inner(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def one_run(task, temperature, _rep):
    """Score one PRIVATE bundle; never share a bundle across threads."""
    ad = TB.TraceBenchTaskAdapter(max_examples=MAX_EXAMPLES, inner_steps=0,
                                  eval_kwargs={"timeout_seconds": 90})
    b = ad._load_bundle(task, fresh=True)
    ad._apply_starting_artifact(b, LevelConfig(starting_artifact=PROMPT))
    param = b["param"]
    if hasattr(param, "llm"):           # bound BOTH arms, so only temperature differs
        param.llm = BoundedLLM(param.llm, temperature)
    try:
        score, _fb = TB._score_bundle(b, MAX_EXAMPLES)
        return float(score)
    except Exception as exc:            # a failure is not a score
        print(f"    !! {task} temp={temperature}: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:110]}", flush=True)
        return None


out = {}
jobs = [(t, label, temp, r) for t in TASKS for label, temp in TEMPERATURES for r in range(REPEATS)]
t_all = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    results = list(ex.map(lambda j: (j, one_run(j[0], j[2], j[3])), jobs))

for (task, label, temp, _r), value in results:
    out.setdefault(task, {}).setdefault(label, {"runs": [], "temperature": temp})
    out[task][label]["runs"].append(value)

for task, arms in out.items():
    print(f"\n=== {task} ===")
    for label, d in arms.items():
        ok = [v for v in d["runs"] if v is not None]
        d["n_valid"], d["n_failed"] = len(ok), len(d["runs"]) - len(ok)
        d["mean"] = st.mean(ok) if ok else None
        d["sd"] = st.pstdev(ok) if len(ok) > 1 else None
        print(f"  {label:18s} mean={d['mean'] if d['mean'] is None else round(d['mean'],4)} "
              f"sd={d['sd'] if d['sd'] is None else round(d['sd'],4)} "
              f"runs={[None if v is None else round(v,4) for v in d['runs']]} "
              f"failed={d['n_failed']}")
    a, b = arms["provider_default"]["sd"], arms["temperature_0_2"]["sd"]
    if a and b:
        print(f"  -> noise {a:.4f} -> {b:.4f} ({a/b:.1f}x reduction)" if b > 1e-9
              else f"  -> noise {a:.4f} -> deterministic")

print(f"\ntotal wall={time.time()-t_all:.0f}s")
Path(__file__).with_name("probe_d_results.json").write_text(json.dumps(out, indent=2))
print("WROTE probe_d_results.json")
