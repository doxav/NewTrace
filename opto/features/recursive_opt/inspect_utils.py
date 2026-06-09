"""
opto.features.recursive_opt.inspect_utils
==========================================

Small, dependency-light helpers used by the demo notebook to *analyze* a
recursive-optimization run:

  * ``param_snapshot(module)``  -> {param_name: value} of the trainable params
  * ``code_diff(before, after)`` -> unified diff (great for code/config/text artifacts)
  * ``trace_feedback(node)``     -> the feedback/score carried by an output node
  * ``trace_graph_text(node)``   -> a compact textual view of the execution trace
  * ``summarize(before, after, score_before, score_after)`` -> one-line verdict

These let the notebook show the difference between the INITIAL and FINAL
trained variable/code, plus the trace signal that drove the change.
"""
from __future__ import annotations

import difflib
from typing import Any, Dict, Optional


def param_snapshot(module) -> Dict[str, Any]:
    """Return {parameter_name: current_value} for a trace.Module's params."""
    snap = {}
    try:
        for p in module.parameters():
            snap[getattr(p, "name", repr(p))] = p.data
    except Exception:
        pass
    return snap


def code_diff(before: str, after: str, name: str = "artifact") -> str:
    """Unified diff between the initial and final artifact (code/config/text)."""
    before_l = str(before).splitlines(keepends=True)
    after_l = str(after).splitlines(keepends=True)
    diff = difflib.unified_diff(
        before_l, after_l, fromfile=f"{name} (initial)", tofile=f"{name} (final)"
    )
    out = "".join(diff)
    return out if out.strip() else f"(no change to {name})"


def _unwrap(node):
    return node.data if hasattr(node, "data") else node


def trace_feedback(node) -> Dict[str, Any]:
    """Extract {score, feedback, objectives?} from a level/forward output node."""
    data = _unwrap(node)
    if isinstance(data, dict):
        return {k: data[k] for k in ("score", "feedback", "objectives") if k in data}
    return {"value": data}


def trace_graph_text(node, max_nodes: int = 40) -> str:
    """Compact textual view of the execution trace feeding ``node``.

    Walks the node's parents (the traced computation graph) and lists each
    operator with a short description. Best-effort: Trace's internal graph API
    varies by version, so this degrades gracefully to just the node summary.
    """
    lines = []
    seen = set()

    def visit(n, depth):
        if n is None or id(n) in seen or len(lines) >= max_nodes:
            return
        seen.add(id(n))
        name = getattr(n, "name", None) or type(n).__name__
        desc = getattr(n, "description", "") or ""
        val = repr(_unwrap(n))
        if len(val) > 60:
            val = val[:57] + "..."
        lines.append(f"{'  ' * depth}- {name} {('['+desc+']') if desc else ''} = {val}")
        parents = getattr(n, "parents", None) or getattr(n, "_inputs", None) or []
        try:
            parents = list(parents.values()) if isinstance(parents, dict) else list(parents)
        except Exception:
            parents = []
        for p in parents:
            visit(p, depth + 1)

    visit(node, 0)
    return "\n".join(lines) if lines else f"(no traceable parents for {node})"


def summarize(before: str, after: str, score_before: float, score_after: float,
              name: str = "artifact") -> str:
    changed = str(before).strip() != str(after).strip()
    delta = score_after - score_before
    arrow = "improved" if delta > 0 else ("unchanged" if delta == 0 else "regressed")
    return (f"{name}: score {score_before:.3f} -> {score_after:.3f} "
            f"(Δ={delta:+.3f}, {arrow}); artifact {'changed' if changed else 'unchanged'}.")


def repeat_scores(eval_fn, seeds=(0, 1, 2)):
    """Run ``eval_fn(seed) -> float`` over several seeds and return stats.

    Returns ``{"scores", "mean", "std", "n"}``. Use this so reported numbers come
    with a mean±std over seeds instead of a single (often noisy) run — especially
    important when the eval set is tiny.
    """
    import math

    scores = [float(eval_fn(s)) for s in seeds]
    n = len(scores)
    mean = sum(scores) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in scores) / n if n else 0.0
    return {"scores": scores, "mean": mean, "std": math.sqrt(var), "n": n}


def fmt_mean_std(stats: dict, name: str = "score") -> str:
    """Pretty 'name = mean ± std (n=…)' from ``repeat_scores`` output."""
    return f"{name} = {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['n']})"
