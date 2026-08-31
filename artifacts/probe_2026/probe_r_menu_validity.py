"""Probe R — the menu is the instrument.

Iteration 3 concluded that `admissible_set` and `code_param` were "flat" surfaces and that
`online_bin_packing`'s deltas were noise. This probe shows the flatness was an artefact of the
MENU, not a property of the task.

`TraceBenchTaskAdapter._apply_starting_artifact` writes the candidate text straight into the
trainable node's `_data` with NO surface check, and returns True. A prose menu entry therefore
REPLACES a 353-char Python `priority()` function (or a float `1.0`) with "Answer directly.",
which then fails to run and scores the invalid sentinel. Only the empty candidate survives, so
the effective menu size is 1 and no optimizer of any kind can express a preference.

Two independent ways a menu can collapse to size 1:
  (a) TYPE-INCOMPATIBLE  - prose on a code/numeric surface -> every candidate invalid.
  (b) RANKING-EQUIVALENT - candidates that are monotone transforms of one another. For
      online_bin_packing only the argmax matters, so `item - bins`, `-(bins - item)`,
      `1/(gap+eps)` and `-(gap**2)` are THE SAME heuristic and must score identically.
Both were present in the Iteration 3 menu. (b) was my own error when first re-testing.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from opto.features.recursive_opt import tracebench as TB          # noqa: E402
from opto.features.recursive_opt.levels import LevelConfig        # noqa: E402
from opto.features.recursive_opt.measurement import detect_surface  # noqa: E402

PROSE_MENU = ["", "Answer directly.", "Plan step by step, then verify the answer before replying."]
BP = "import numpy as np\ndef priority(item, bins):\n    "
CODE_MENUS = {
    "llm4ad:optimization/online_bin_packing": {
        "best_fit  (tightest)":   BP + "return -(bins - item)\n",
        "worst_fit (loosest)":    BP + "return (bins - item)\n",
        "first_fit (low index)":  BP + "return -np.arange(len(bins), dtype=float)\n",
        "last_fit  (high index)": BP + "return np.arange(len(bins), dtype=float)\n",
        "almost_worst":           BP + "g = bins - item\n    return -np.abs(g - np.median(g))\n",
        "exact_then_best":        BP + "g = bins - item\n    return -g + 1e6*(np.abs(g) < 1e-9)\n",
        "ratio":                  BP + "return item / np.maximum(bins, 1e-9)\n",
        "const (tie->first)":     BP + "return np.zeros(len(bins))\n",
    },
    "llm4ad:optimization/admissible_set": {
        "baseline 0.0":    "import numpy as np\ndef priority(el, n=15, w=10):\n    return 0.0\n",
        "sum":             "import numpy as np\ndef priority(el, n=15, w=10):\n    return float(sum(el))\n",
        "neg_sum":         "import numpy as np\ndef priority(el, n=15, w=10):\n    return -float(sum(el))\n",
        "count_nonzero":   "import numpy as np\ndef priority(el, n=15, w=10):\n    return float(sum(1 for x in el if x))\n",
        "weighted_idx":    "import numpy as np\ndef priority(el, n=15, w=10):\n    return float(sum(i*x for i,x in enumerate(el)))\n",
        "mod3":            "import numpy as np\ndef priority(el, n=15, w=10):\n    return float(sum(x for x in el if x % 3 == 0))\n",
    },
}
# Executability controls: prove the candidate really reaches the benchmark.
CONTROLS = {
    "raises":     BP + "raise RuntimeError('EXECUTED')\n",
    "syntax_err": "import numpy as np\ndef priority(item, bins)\n    return 0\n",
    "wrong_name": "import numpy as np\ndef not_priority(item, bins):\n    return item - bins\n",
}


def score(ad, tid, text, max_examples=2):
    b = ad._load_bundle(tid, fresh=True)
    if text is not None:
        ad._apply_starting_artifact(b, LevelConfig(starting_artifact=text))
    try:
        return float(TB._score_bundle(b, max_examples)[0])
    except Exception as exc:                       # pragma: no cover - diagnostic path
        return f"ERR {type(exc).__name__}: {exc}"


def main():
    TB.ensure_default_task_adapter(require=True)
    ad = TB.current_task_adapter()
    out = {"tasks": {}, "controls": {}}

    for tid, menu in CODE_MENUS.items():
        b = ad._load_bundle(tid, fresh=True)
        node = ad._trainable_node(b.get("param"))
        original = str(getattr(node, "_data", ""))
        surface = detect_surface(b)

        # (a) does a prose menu entry destroy the parameter?
        destroyed = []
        for m in PROSE_MENU[1:]:
            b2 = ad._load_bundle(tid, fresh=True)
            applied = ad._apply_starting_artifact(b2, LevelConfig(starting_artifact=m))
            after = str(getattr(ad._trainable_node(b2.get("param")), "_data", ""))
            destroyed.append({"candidate": m, "applied_returned": bool(applied),
                              "param_overwritten": after.strip() == m.strip()})

        prose_scores = [score(ad, tid, m if m else None) for m in PROSE_MENU]
        code_scores = {name: score(ad, tid, code) for name, code in menu.items()}
        valid = [v for v in code_scores.values() if isinstance(v, float) and abs(v) < 1e5]

        out["tasks"][tid] = {
            "surface_kind": surface.kind, "param_name": surface.param_name,
            "calls_llm": surface.calls_llm, "original_len": len(original),
            "prose_menu_destroys_param": destroyed,
            "prose_menu_scores": prose_scores,
            "prose_menu_effective_size": len({v for v in prose_scores
                                              if isinstance(v, float) and abs(v) < 1e5}),
            "code_menu_scores": code_scores,
            "code_menu_distinct": len(set(valid)),
            "code_menu_range": round(max(valid) - min(valid), 4) if valid else 0.0,
            "headroom": len(set(valid)) > 1,
        }

    ctl_tid = "llm4ad:optimization/online_bin_packing"
    out["controls"][ctl_tid] = {k: score(ad, ctl_tid, v) for k, v in CONTROLS.items()}
    out["controls"][ctl_tid]["original"] = score(ad, ctl_tid, None)

    path = Path(__file__).with_name("probe_r_results.json")
    path.write_text(json.dumps(out, indent=2) + "\n")

    for tid, r in out["tasks"].items():
        print(f"\n=== {tid} ({r['surface_kind']}, {r['param_name']}, "
              f"calls_llm={r['calls_llm']}) ===")
        print(f"  prose menu: param overwritten "
              f"{sum(d['param_overwritten'] for d in r['prose_menu_destroys_param'])}/"
              f"{len(r['prose_menu_destroys_param'])}, effective menu size "
              f"{r['prose_menu_effective_size']}  scores={r['prose_menu_scores']}")
        for name, s in r["code_menu_scores"].items():
            print(f"    {name:24s} {s}")
        print(f"  type-correct menu: distinct={r['code_menu_distinct']} "
              f"range={r['code_menu_range']} HEADROOM={'YES' if r['headroom'] else 'NO'}")
    print(f"\ncontrols on {ctl_tid} (proves the candidate reaches the benchmark):")
    for k, v in out["controls"][ctl_tid].items():
        print(f"    {k:12s} {v}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
