"""Probe I — is Probe F's +0.217 an accuracy gain or a brevity gain?

The optimizer's best artifact was "Solve the math problem. Do not show work. Output only
the final numeric answer." On an objective of -(1.0*error) - (0.001*tokens), that is
exactly what a token-minimiser looks like. §11.4 predicted this. Decomposing the gain
into its error and token components settles whether the optimizer improved reasoning or
discovered that the metric pays for silence.
"""
import json, statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from opto.features.recursive_opt import measurement as M

TASK = "internal:multiobjective_gsm8k"
W_ERR, W_TOK = 1.0, 0.001
REPEATS = 3
ARTIFACTS = {
    "initial (empty)": "",
    "optimizer best (terse)": "Solve the math problem. Do not show work. "
                              "Output only the final numeric answer.",
    "optimizer alt (CoT)": "Let's think step by step",
}

jobs = [(label, art, r) for label, art in ARTIFACTS.items() for r in range(REPEATS)]
with ThreadPoolExecutor(max_workers=6) as ex:
    obs = list(ex.map(
        lambda j: (j[0], M.evaluate_once(TASK, artifact=j[1] or None, max_examples=4)), jobs))

out = {}
for label, o in obs:
    if o.get("valid"):
        out.setdefault(label, []).append(o)

print(f"{'arm':26s} {'score':>9s} {'error':>8s} {'tokens':>9s} {'err term':>9s} {'tok term':>9s}")
print("-" * 78)
summary = {}
for label in ARTIFACTS:
    rows = [r for o in out.get(label, []) for r in o["per_example"]]
    if not rows:
        print(f"{label:26s}  (no valid observations)")
        continue
    err = st.mean(r.get("error", 0.0) for r in rows)
    tok = st.mean(r.get("tokens_in", 0.0) + r.get("tokens_out", 0.0) for r in rows)
    score = st.mean(o["score"] for o in out[label])
    summary[label] = {"score": score, "error_rate": err, "tokens": tok,
                      "error_term": -W_ERR * err, "token_term": -W_TOK * tok,
                      "n_examples": len(rows)}
    print(f"{label:26s} {score:+9.4f} {err:8.3f} {tok:9.1f} "
          f"{-W_ERR*err:+9.4f} {-W_TOK*tok:+9.4f}")

base = summary.get("initial (empty)")
if base:
    print()
    for label, s in summary.items():
        if label == "initial (empty)":
            continue
        d_err = s["error_term"] - base["error_term"]
        d_tok = s["token_term"] - base["token_term"]
        total = d_err + d_tok
        share = abs(d_tok) / (abs(d_err) + abs(d_tok)) * 100 if (d_err or d_tok) else 0
        print(f"{label} vs initial:")
        print(f"   from ERROR  : {d_err:+.4f}")
        print(f"   from TOKENS : {d_tok:+.4f}")
        print(f"   total       : {total:+.4f}   ({share:.0f}% of the gain is TOKENS)")

Path(__file__).with_name("probe_i_results.json").write_text(json.dumps(summary, indent=2))
print("\nWROTE probe_i_results.json")
