"""Probe W — the pre-registered decisive control for probe U.

Probe U's `c_std` assumes the standard arm searches the menu UNINFORMED (uniform draws). A real
LLM optimizer given the TSP/CVRP/VRPTW docstring may well propose nearest-neighbour on its FIRST
candidate, in which case c_std == c_rec == 1, the compute saving vanishes and K* -> infinity.

This measures it. For each target task the model is handed exactly what the optimizer's first
turn sees - the trainable node's current code (template + docstring) and the node's bench-added
description - with no feedback, no score, no examples, and asked for a replacement. Its proposal
is applied through the SAME `_apply_starting_artifact` path and scored by the SAME evaluator, then
placed on probe U's min-max normalised quality scale for that task.

Reaching q_norm >= 1.0 means the cold model matched the best of the 9-entry menu on its first try.
"""
import json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for line in (Path(__file__).resolve().parents[2] / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("TRACE_LITELLM_MODEL", "openrouter/deepseek/deepseek-v4-flash-0731")

from opto.features.recursive_opt import tracebench as TB       # noqa: E402
from opto.features.recursive_opt.levels import LevelConfig     # noqa: E402
from opto.utils.llm import LLM                                 # noqa: E402
from probe_t_routing_menu import MENU, ROUTING                 # noqa: E402

TARGETS = [t for t in ROUTING if not t.endswith("ovrp_construct")]
N_SAMPLES = int(os.environ.get("PROBE_W_SAMPLES", "5"))
TEMPS = [0.0] + [1.0] * (N_SAMPLES - 1)

SYS = ("You are optimizing a Python function inside an algorithm-discovery benchmark. "
       "Reply with ONE fenced ```python block containing the complete replacement module "
       "(imports + the function). No commentary.")


def prompt(code, desc):
    return (f"Current implementation:\n\n```python\n{code}\n```\n\n"
            f"Constraints:\n{desc}\n\n"
            "Propose a better implementation. Maximise the benchmark score.")


def extract(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def score(ad, tid, text):
    b = ad._load_bundle(tid, fresh=True)
    if text is not None:
        ad._apply_starting_artifact(b, LevelConfig(starting_artifact=text))
    try:
        return float(TB._score_bundle(b, 2)[0])
    except Exception as exc:
        return f"ERR {type(exc).__name__}: {str(exc)[:120]}"


def main():
    TB.ensure_default_task_adapter(require=True)
    ad = TB.current_task_adapter()
    llm = LLM()
    menu_tbl = {}
    for tid in TARGETS:
        menu_tbl[tid] = {n: score(ad, tid, MENU[n]) for n in MENU}

    out = {"model": os.environ.get("TRACE_LITELLM_MODEL"), "n_samples": N_SAMPLES,
           "menu_scores": menu_tbl, "targets": {}}
    for tid in TARGETS:
        row = menu_tbl[tid]
        lo, hi = min(row.values()), max(row.values())
        b = ad._load_bundle(tid, fresh=True)
        node = ad._trainable_node(b["param"])
        code, desc = str(node._data), str(getattr(node, "description", ""))
        samples = []
        for i, temp in enumerate(TEMPS):
            t0 = time.time()
            try:
                resp = llm(messages=[{"role": "system", "content": SYS},
                                     {"role": "user", "content": prompt(code, desc)}],
                           temperature=temp, max_tokens=6000)
                msg = resp.choices[0].message
                text = msg.content or getattr(msg, "reasoning_content", None) or ""
                if not text.strip():
                    raise RuntimeError(
                        f"empty completion (finish_reason={resp.choices[0].finish_reason})")
            except Exception as exc:
                samples.append({"i": i, "temp": temp,
                                "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
                continue
            cand = extract(text)
            s = score(ad, tid, cand)
            qn = ((s - lo) / (hi - lo)) if isinstance(s, float) and abs(s) < 1e5 and hi > lo else None
            samples.append({"i": i, "temp": temp, "score": s, "q_norm": qn,
                            "beats_menu_best": isinstance(s, float) and abs(s) < 1e5 and s > hi,
                            "reaches_menu_best": qn is not None and qn >= 1.0 - 1e-9,
                            "wall_s": round(time.time() - t0, 1),
                            "code": cand[:2000]})
            print(f"  {tid.split('/')[-1]:16s} sample{i} T={temp} score={s} "
                  f"q_norm={None if qn is None else round(qn, 4)}", flush=True)
        ok = [x for x in samples if x.get("q_norm") is not None]
        out["targets"][tid] = {
            "menu_min": lo, "menu_max": hi, "samples": samples,
            "n_valid": len(ok), "n_invalid": len(samples) - len(ok),
            "n_reaches_menu_best": sum(x["reaches_menu_best"] for x in ok),
            "n_beats_menu_best": sum(x["beats_menu_best"] for x in ok),
            "mean_q_norm": (sum(x["q_norm"] for x in ok) / len(ok)) if ok else None,
            "greedy_q_norm": next((x["q_norm"] for x in samples if x.get("temp") == 0.0), None),
        }
    p = Path(__file__).with_name("probe_w_cold_llm.json")
    p.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print("\n=== cold LLM, first candidate, no feedback ===")
    for tid, r in out["targets"].items():
        print(f"{tid:40s} valid={r['n_valid']}/{len(r['samples'])} "
              f"reaches_menu_best={r['n_reaches_menu_best']} beats={r['n_beats_menu_best']} "
              f"mean_q_norm={r['mean_q_norm']} greedy_q_norm={r['greedy_q_norm']}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
