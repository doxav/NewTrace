"""Probe AA - was qasper seed 101's 0.5455 a real find, or evaluator noise?

probe_g (n_pairs=2) showed the standard arm reaching 0.5455 on hf:qasper with a real
transfer prompt while the recursive arm stayed at the default and scored 0.1769. Read
naively that is a large standard win. But probe A recorded the EMPTY prompt scoring
0.088 / 0.5488 / 0.107 across three repeats - so 0.5455 sits inside the range the empty
prompt already produces on its own.

Design: score both conditions n times each, INTERLEAVED in one process, so any endpoint
load or drift hits both arms equally. A between-process A/B would confound the contrast
with load, which is the error that spoiled an earlier pair of runs.

Decision rule, fixed before running: the find is real only if the found prompt's mean
exceeds the empty prompt's mean by more than the pooled within-condition spread.
"""
import json, signal, statistics as st, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from opto.features.recursive_opt import tracebench as TB           # noqa: E402
from opto.features.recursive_opt.levels import LevelConfig         # noqa: E402
from opto.features.recursive_opt.spec import make_scored_task_runner  # noqa: E402

TASK = "hf:qasper"
N = 6
# Match probe A exactly. Its empty-prompt figures (0.088 / 0.549 / 0.107) are the numbers
# this probe exists to compare against, and they were produced at max_examples=2. A first
# attempt left evaluation UNBOUNDED: the first item alone took 514 s and the full design
# would have taken 10+ hours -- the same unbounded-sampling defect that cost 14.6x
# resolution earlier in this project.
MAX_EXAMPLES = 2
RUN_TIMEOUT_S = 480   # endpoint slower than on probe A's day; 150s now times out


class _RunTimeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _RunTimeout(f"evaluation exceeded {RUN_TIMEOUT_S}s")


signal.signal(signal.SIGALRM, _alarm)
FOUND = Path("/tmp/claude-1000/-home-xav-code-Trace/b9d97baf-4098-451b-9ede-8836c0e3ded6/"
             "scratchpad/found_prompt.txt").read_text()
FOUND = FOUND.split("starting_artifact:", 1)[-1].strip()


def main():
    TB.register_task_adapter(TB.TraceBenchTaskAdapter(
        max_examples=MAX_EXAMPLES, inner_steps=0, eval_kwargs={"timeout_seconds": 90}))
    runner = make_scored_task_runner(None)
    conditions = {"empty": "", "found": FOUND}
    scores = {k: [] for k in conditions}
    order = []
    for i in range(N):
        for name, text in conditions.items():          # interleaved, not blocked
            t0 = time.time()
            signal.alarm(RUN_TIMEOUT_S)
            try:
                s = float(runner(LevelConfig(starting_artifact=text), TASK)[0])
            except Exception as exc:                    # pragma: no cover
                s = None
                print(f"  [{i}] {name}: ERROR {type(exc).__name__}: {exc}", flush=True)
            finally:
                signal.alarm(0)
            scores[name].append(s)
            order.append({"i": i, "condition": name, "score": s,
                          "wall_s": round(time.time() - t0, 1)})
            print(f"  [{i}] {name:6s} -> {s}", flush=True)

    out = {"task": TASK, "n_per_condition": N, "max_examples": MAX_EXAMPLES,
           "design": "interleaved single process; adapter matched to probe A",
           "found_prompt": FOUND, "order": order}
    for k, v in scores.items():
        ok = [x for x in v if x is not None]
        out[k] = {"scores": ok, "n": len(ok),
                  "mean": st.mean(ok) if ok else None,
                  "sd": st.pstdev(ok) if len(ok) > 1 else None,
                  "min": min(ok) if ok else None, "max": max(ok) if ok else None}
    e, f = out["empty"], out["found"]
    if e["n"] and f["n"]:
        pooled = st.mean([e["sd"] or 0.0, f["sd"] or 0.0])
        out["delta_found_minus_empty"] = f["mean"] - e["mean"]
        out["pooled_within_sd"] = pooled
        out["verdict"] = ("REAL FIND" if (f["mean"] - e["mean"]) > pooled
                          else "INSIDE NOISE - not a find")
        out["seed101_0.5455_within_empty_range"] = e["min"] <= 0.5455 <= e["max"]
    Path(__file__).with_name("probe_aa_results.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n" + json.dumps({k: out[k] for k in
          ("delta_found_minus_empty", "pooled_within_sd", "verdict",
           "seed101_0.5455_within_empty_range") if k in out}, indent=1))


if __name__ == "__main__":
    main()
