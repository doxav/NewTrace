"""Probe S — which llm4ad *_construct tasks share a trainable-parameter signature?

Precondition for W2 (amortisation): a code artifact can only transfer between tasks whose
trainable function has the SAME signature. This dumps the original param source for every
`llm4ad:optimization/*` task (excluding co_bench) so a shared-signature family can be
identified before any compute is spent.
"""
import json, re, sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from opto.features.recursive_opt import tracebench as TB           # noqa: E402
from opto.features.recursive_opt.measurement import detect_surface  # noqa: E402

CANDIDATES = [t for t in TB.list_tasks() if t.startswith("llm4ad:optimization/")
              and "/co_bench/" not in t]


def main():
    TB.ensure_default_task_adapter(require=True)
    ad = TB.current_task_adapter()
    out = {}
    for tid in CANDIDATES:
        try:
            b = ad._load_bundle(tid, fresh=True)
            node = ad._trainable_node(b.get("param"))
            src = str(getattr(node, "_data", ""))
            surf = detect_surface(b)
            sigs = re.findall(r"def\s+(\w+)\s*\(([^)]*)\)", src)
            out[tid] = {"surface": surf.kind, "param_name": surf.param_name,
                        "calls_llm": surf.calls_llm, "len": len(src),
                        "defs": [{"name": n, "args": a.strip()} for n, a in sigs],
                        "source": src}
        except Exception as exc:
            out[tid] = {"error": f"{type(exc).__name__}: {exc}",
                        "tb": traceback.format_exc()[-400:]}
    p = Path(__file__).with_name("probe_s_signatures.json")
    p.write_text(json.dumps(out, indent=2) + "\n")
    for tid, r in out.items():
        if "error" in r:
            print(f"{tid:52s} ERROR {r['error'][:90]}")
        else:
            d = "; ".join(f"{x['name']}({x['args']})" for x in r["defs"])
            print(f"{tid:52s} {r['surface']:8s} llm={r['calls_llm']!s:5s} {d[:110]}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
