"""
recursive_opt.traces  —  multi-trace substrate  (robustify B.3 / B.4 / B.5)
===========================================================================

Thin, defensive wrappers around the PR #73 IO layer so the recursive stack can
consume *heterogeneous* trace sources uniformly:

    B.3  Graph adapter            -> opto.features.graph.GraphAdapter / LangGraphAdapter
    B.4  OpenTelemetry            -> opto.trace.io.instrument_graph / TelemetrySession
    B.5  Trace / OTEL / Sysmon    -> observers + TGJ (Trace Graph JSON) merge

The whole point of PR #73 is that an arbitrary graph with *named bindings*
becomes a ``trace.Module`` (``GraphAdapter.as_module()``), and that OTEL/Sysmon
spans can be lifted into Trace nodes via ``otlp_traces_to_trace_json`` /
``ingest_tgj``. That single fact lets recursion treat graph workflows exactly
like prompts/code: one ``forward()``, one set of trainable parameters.

All imports are guarded: if PR #73 isn't installed, the helpers degrade to the
internal-trace-only path so the rest of the package still runs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

# --- guarded PR #73 imports ------------------------------------------------ #
try:
    from opto.trace.io import (
        instrument_graph,
        optimize_graph,
        TelemetrySession,
        Binding,
        apply_updates,
        make_dict_binding,
        otlp_traces_to_trace_json,
        ingest_tgj,
        merge_tgj,
    )
    from opto.trace.io.observers import GraphObserver, OTelObserver
    from opto.trace.io.sysmonitoring import SysMonObserver, sysmon_profile_to_tgj
    from opto.features.graph import GraphAdapter, LangGraphAdapter, GraphModule

    HAVE_PR73 = True
except Exception:  # pragma: no cover
    HAVE_PR73 = False


def require_pr73(feature: str = "this feature") -> None:
    """Raise a clear error if PR #73 (graph/OTEL/Sysmon) is not installed.

    Use this at the top of any code path that genuinely needs PR #73, so the
    absence is a LOUD failure rather than a silent no-op.
    """
    if not HAVE_PR73:
        raise RuntimeError(
            f"{feature} requires PR #73 (opto.features.graph + opto.trace.io), "
            "which is NOT installed in this environment. Install/merge PR #73 "
            "before using the graph adapter / OTEL / Sysmon trace backends."
        )


def graph_to_module(build_graph: Callable, bindings: Dict[str, Any]):
    """B.3: turn any graph + named knobs into a trainable ``trace.Module``.

    `bindings` maps param-key -> (getter, setter) via make_dict_binding, exactly
    the PR #73 contract. Returns a GraphModule whose .parameters() are trainable.
    """
    if not HAVE_PR73:
        raise RuntimeError(
            "PR #73 graph adapter not available; "
            "use ArtifactLevel / native nesting instead."
        )
    adapter = LangGraphAdapter(
        build_graph=build_graph, bindings=make_dict_binding(bindings)
    )
    return adapter.as_module()  # plugs straight into opto.trainer.train


def collect_traces(trace_types: List[str]) -> "MultiTraceSession":
    """B.4/B.5: open a unified session emitting the requested trace backends."""
    return MultiTraceSession(trace_types)


class MultiTraceSession:
    """Unifies internal Trace + OTEL + Sysmon into one TGJ feedback bundle.

    trace_types subset of {"internal", "otel", "sysmon"}. The internal Trace is
    always available; OTEL/Sysmon require PR #73. On exit, all enabled backends
    are merged into a single Trace-Graph-JSON dict usable as optimizer feedback.
    """

    def __init__(self, trace_types: List[str]):
        self.trace_types = [t for t in trace_types]
        self._otel = None
        self._sysmon = None
        self._tgj: Dict[str, Any] = {"nodes": [], "edges": [], "sources": []}

    def __enter__(self):
        if "otel" in self.trace_types and HAVE_PR73:
            self._otel = TelemetrySession().__enter__()
            self._tgj["sources"].append("otel")
        if "sysmon" in self.trace_types and HAVE_PR73:
            from opto.trace.io.sysmonitoring import SysMonitoringSession

            self._sysmon = SysMonitoringSession().__enter__()
            self._tgj["sources"].append("sysmon")
        self._tgj["sources"].append("internal")
        return self

    def __exit__(self, *exc):
        if self._otel is not None:
            self._otel.__exit__(*exc)
        if self._sysmon is not None:
            self._sysmon.__exit__(*exc)
        return False

    def record_internal(self, node, max_nodes: int = 100) -> "MultiTraceSession":
        """Normalize the internal Trace graph feeding ``node`` into TGJ nodes/edges.

        Until now the internal trace was only a label in ``sources``; this walks
        the node's parents and adds real ``{id,label,value}`` nodes and ``{src,dst}``
        edges, so multi-trace records actually include the internal view (not just
        OTEL/Sysmon). Best-effort and version-tolerant.
        """
        seen = set()

        def visit(n):
            if n is None or id(n) in seen or len(self._tgj["nodes"]) >= max_nodes:
                return
            seen.add(id(n))
            nid = str(id(n))
            label = getattr(n, "name", None) or type(n).__name__
            val = repr(getattr(n, "data", n))
            self._tgj["nodes"].append(
                {"id": nid, "label": label, "value": val[:80], "source": "internal"}
            )
            parents = getattr(n, "parents", None) or getattr(n, "_inputs", None) or []
            try:
                parents = list(parents.values()) if isinstance(parents, dict) else list(parents)
            except Exception:
                parents = []
            for p in parents:
                self._tgj["edges"].append({"src": str(id(p)), "dst": nid, "source": "internal"})
                visit(p)

        try:
            visit(node)
        except Exception as e:  # never break the optimization loop on tracing
            import warnings
            warnings.warn(f"internal-trace normalization failed: {e!r}", RuntimeWarning)
        return self

    def to_tgj(self) -> Dict[str, Any]:
        """Merge enabled backends into one Trace-Graph-JSON feedback object."""
        if not HAVE_PR73:
            return self._tgj
        if self._otel is not None:
            try:
                self._tgj = merge_tgj(
                    self._tgj, otlp_traces_to_trace_json(self._otel.export())
                )
            except Exception as e:  # don't hide failures behind a clean-looking result
                import warnings
                warnings.warn(f"OTEL->TGJ merge failed: {e!r}", RuntimeWarning)
        if self._sysmon is not None:
            try:
                self._tgj = merge_tgj(
                    self._tgj, sysmon_profile_to_tgj(self._sysmon.profile())
                )
            except Exception as e:
                import warnings
                warnings.warn(f"Sysmon->TGJ merge failed: {e!r}", RuntimeWarning)
        return self._tgj

    def feedback_text(self, base: str = "") -> str:
        """Compact, optimizer-readable summary of all trace sources."""
        tgj = self.to_tgj()
        srcs = ",".join(tgj.get("sources", []))
        return (
            f"{base}\n[traces:{srcs}] "
            f"nodes={len(tgj.get('nodes', []))} edges={len(tgj.get('edges', []))}"
        ).strip()
