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

    def to_tgj(self) -> Dict[str, Any]:
        """Merge enabled backends into one Trace-Graph-JSON feedback object."""
        if not HAVE_PR73:
            return self._tgj
        if self._otel is not None:
            try:
                self._tgj = merge_tgj(
                    self._tgj, otlp_traces_to_trace_json(self._otel.export())
                )
            except Exception:
                pass
        if self._sysmon is not None:
            try:
                self._tgj = merge_tgj(
                    self._tgj, sysmon_profile_to_tgj(self._sysmon.profile())
                )
            except Exception:
                pass
        return self._tgj

    def feedback_text(self, base: str = "") -> str:
        """Compact, optimizer-readable summary of all trace sources."""
        tgj = self.to_tgj()
        srcs = ",".join(tgj.get("sources", []))
        return (
            f"{base}\n[traces:{srcs}] "
            f"nodes={len(tgj.get('nodes', []))} edges={len(tgj.get('edges', []))}"
        ).strip()
