"""Probe X — the W2 comparison with a REAL optimizer on both arms.

Probe U's `c_std` assumed the standard arm samples a fixed 9-entry menu uniformly. Probe W
showed a cold LLM lands at q_norm 0.96-0.99 in ONE candidate when its code is valid, which
would collapse probe U's saving. But probe W also showed it emits INVALID code most of the
time. So the real W2 question on this family is not "does the prior know the answer" - both
arms do - it is "how many candidates does each arm burn before it holds a VALID, good artifact".

Two things are measured here, both with the same cold model and the same evaluator:

A. STANDARD arm cost. `n` independent first-candidate proposals per target task (no feedback,
   no menu, exactly what the optimizer's first turn sees). Gives p_valid and the exact
   best-of-b curve by enumeration over the sample pool, hence c_std(Q).

B. TRANSFER, measured rather than assumed. Every valid artifact the model wrote for a SOURCE
   task is applied to the TARGET task through `_apply_starting_artifact` and scored. This is
   the recursive arm's actual artifact: code written while optimising a sibling. A code
   artifact can only transfer if it executes on the target - the four routing tasks share the
   entry point `select_next_node` but NOT its arity, so this is the crux and it is measured,
   not asserted. The hand-written probe-T menu is `*args`-tolerant by construction and is
   scored alongside as the ceiling a transfer-aware artifact could reach.

Samples are drawn in parallel threads; the EVALUATOR is re-checked for determinism at that
concurrency (probe V0 measured range 0.0 sequentially) before any of it is believed.
"""
import itertools, json, math, os, re, statistics as st, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
for line in (HERE.parents[2] / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("TRACE_LITELLM_MODEL", "openrouter/deepseek/deepseek-v4-flash-0731")

from opto.features.recursive_opt import tracebench as TB       # noqa: E402
from opto.features.recursive_opt.levels import LevelConfig     # noqa: E402
from opto.utils.llm import LLM                                 # noqa: E402
from probe_t_routing_menu import MENU, ROUTING                 # noqa: E402
from probe_w_cold_llm import SYS, prompt, extract              # noqa: E402

TARGETS = [t for t in ROUTING if not t.endswith("ovrp_construct")]
N = int(os.environ.get("PROBE_X_N", "12"))
CONC = int(os.environ.get("PROBE_X_CONCURRENCY", "6"))
QS = [1.00, 0.99, 0.95]
INVALID = 1e5
_LOCK = threading.Lock()


def score(ad, tid, text):
    """Serialised: the adapter/bundle path is not known to be thread-safe."""
    with _LOCK:
        b = ad._load_bundle(tid, fresh=True)
        if text is not None:
            ad._apply_starting_artifact(b, LevelConfig(starting_artifact=text))
        try:
            return float(TB._score_bundle(b, 2)[0])
        except Exception as exc:
            return f"ERR {type(exc).__name__}: {str(exc)[:100]}"


def sample_one(llm, code, desc, temp):
    resp = llm(messages=[{"role": "system", "content": SYS},
                         {"role": "user", "content": prompt(code, desc)}],
               temperature=temp, max_tokens=6000)
    msg = resp.choices[0].message
    text = msg.content or getattr(msg, "reasoning_content", None) or ""
    if not text.strip():
        raise RuntimeError(f"empty completion ({resp.choices[0].finish_reason})")
    return extract(text)


def best_of_b_curve(qs, m):
    """Exact E[best-of-b] over all C(len(qs), b) subsets of an i.i.d. sample pool."""
    out = {}
    for b in range(1, m + 1):
        vals = [max(c) for c in itertools.combinations(qs, b)]
        out[b] = {"E": sum(vals) / len(vals),
                  "P_ge_095": sum(v >= 0.95 for v in vals) / len(vals),
                  "P_ge_099": sum(v >= 0.99 for v in vals) / len(vals),
                  "P_ge_100": sum(v >= 1.0 - 1e-9 for v in vals) / len(vals)}
    return out


def main():
    TB.ensure_default_task_adapter(require=True)
    ad = TB.current_task_adapter()
    llm = LLM()
    t_start = time.time()

    # ---- menu scale + concurrency determinism re-check --------------------------------
    menu_tbl = {t: {n: score(ad, t, MENU[n]) for n in MENU} for t in TARGETS}
    scale = {t: (min(menu_tbl[t].values()), max(menu_tbl[t].values())) for t in TARGETS}

    def qn(t, s):
        lo, hi = scale[t]
        if not isinstance(s, float) or abs(s) >= INVALID or hi <= lo:
            return None
        return (s - lo) / (hi - lo)

    rep_jobs = [(t, k) for t in TARGETS for k in range(10)]
    with ThreadPoolExecutor(CONC) as ex:
        reps = list(ex.map(lambda j: score(ad, j[0], MENU["nearest"]), rep_jobs))
    conc_range = {}
    for t in TARGETS:
        vals = [v for (j, v) in zip(rep_jobs, reps) if j[0] == t and isinstance(v, float)]
        conc_range[t] = round(max(vals) - min(vals), 12) if vals else None

    # ---- A. cold-LLM proposals, n per target ------------------------------------------
    ctx = {}
    for t in TARGETS:
        b = ad._load_bundle(t, fresh=True)
        node = ad._trainable_node(b["param"])
        ctx[t] = (str(node._data), str(getattr(node, "description", "")))

    jobs = [(t, i) for t in TARGETS for i in range(N)]

    def work(job):
        t, i = job
        code, desc = ctx[t]
        try:
            cand = sample_one(llm, code, desc, 0.0 if i == 0 else 1.0)
        except Exception as exc:
            return {"task": t, "i": i, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        s = score(ad, t, cand)
        return {"task": t, "i": i, "temp": 0.0 if i == 0 else 1.0, "score": s,
                "q_norm": qn(t, s), "code": cand}

    with ThreadPoolExecutor(CONC) as ex:
        samples = list(ex.map(work, jobs))
    for s in samples:
        note = ("ERR " + s["error"][:60]) if "error" in s else f"q_norm={s.get('q_norm')}"
        print(f"  {s['task'].split('/')[-1]:16s} i={s['i']:2d} {note}", flush=True)

    per_task = {}
    for t in TARGETS:
        rows = [s for s in samples if s["task"] == t]
        valid = [s for s in rows if s.get("q_norm") is not None]
        # an invalid proposal still COSTS a candidate: its quality is 0 on the normalised scale
        pool = [(s["q_norm"] if s.get("q_norm") is not None else 0.0)
                for s in rows if "error" not in s]
        per_task[t] = {
            "n": len(rows), "n_api_error": sum("error" in s for s in rows),
            "n_valid": len(valid), "p_valid": len(valid) / max(1, len(pool)),
            "q_norm_valid": sorted(round(s["q_norm"], 4) for s in valid),
            "best_q_norm": max((s["q_norm"] for s in valid), default=None),
            "n_beats_menu_best": sum(s["q_norm"] > 1.0 + 1e-9 for s in valid),
            "curve": best_of_b_curve(pool, min(len(pool), 8)) if pool else {},
        }

    # ---- B. does a source-task artifact EXECUTE on the target? ------------------------
    transfer = {}
    for target in TARGETS:
        rows = []
        for src in TARGETS:
            if src == target:
                continue
            for s in samples:
                if s["task"] != src or s.get("q_norm") is None:
                    continue
                st_ = score(ad, target, s["code"])
                rows.append({"source": src, "i": s["i"],
                             "source_q_norm": round(s["q_norm"], 4),
                             "target_score": st_, "target_q_norm": qn(target, st_)})
        ok = [r for r in rows if r["target_q_norm"] is not None]
        transfer[target] = {
            "n_transferred": len(rows), "n_executes_on_target": len(ok),
            "p_executes": len(ok) / len(rows) if rows else None,
            "mean_target_q_norm": (st.mean(r["target_q_norm"] for r in ok) if ok else None),
            "best_target_q_norm": max((r["target_q_norm"] for r in ok), default=None),
            "rows": rows,
            # ceiling: the hand-written *args-tolerant menu artifact, which does transfer
            "handwritten_transferable_q_norm": qn(target, menu_tbl[target]["nearest"]),
        }

    # ---- c_std / c_rec / K* on the LLM path ------------------------------------------
    breakeven = {}
    for Q in QS:
        key = {1.00: "P_ge_100", 0.99: "P_ge_099", 0.95: "P_ge_095"}[Q]
        cs = {}
        for t in TARGETS:
            c = per_task[t]["curve"]
            # expected candidates until first success, from the per-draw success rate
            p1 = c.get(1, {}).get(key, 0.0)
            cs[t] = {"p_success_per_candidate": p1,
                     "E_candidates_to_success": (math.inf if p1 <= 0 else 1.0 / p1),
                     "b_for_90pct": next((b for b in sorted(c) if c[b][key] >= 0.90), None)}
        breakeven[str(Q)] = cs

    out = {"model": os.environ.get("TRACE_LITELLM_MODEL"), "n_per_target": N,
           "concurrency": CONC,
           "evaluator_replicate_range_at_concurrency": conc_range,
           "menu_scores": menu_tbl, "samples": samples, "per_task": per_task,
           "transfer": transfer, "standard_cost": breakeven,
           "wall_s": round(time.time() - t_start, 1)}
    p = HERE.with_name("probe_x_llm_w2.json")
    p.write_text(json.dumps(out, indent=2, default=str) + "\n")

    print(f"\nevaluator replicate range at concurrency {CONC}: {conc_range}")
    print("\n=== A. cold LLM, per-candidate ===")
    for t, r in per_task.items():
        print(f"{t.split('/')[-1]:16s} n={r['n']} api_err={r['n_api_error']} "
              f"valid={r['n_valid']} p_valid={r['p_valid']:.2f} best_q={r['best_q_norm']} "
              f"beats_menu={r['n_beats_menu_best']} q_valid={r['q_norm_valid']}")
    print("\n=== B. source artifact applied to target ===")
    for t, r in transfer.items():
        print(f"{t.split('/')[-1]:16s} transferred={r['n_transferred']} "
              f"executes={r['n_executes_on_target']} p={r['p_executes']} "
              f"mean_q={r['mean_target_q_norm']} best_q={r['best_target_q_norm']} "
              f"handwritten_ceiling={r['handwritten_transferable_q_norm']}")
    print("\n=== standard-arm cost (cold LLM) ===")
    print(json.dumps(breakeven, indent=2, default=str))
    print(f"\nwrote {p} ({out['wall_s']}s)")


if __name__ == "__main__":
    main()
