"""
recursive_opt.traces  —  multi-trace substrate  (B.3 / B.4 / B.5)
=================================================================

Thin, defensive wrappers around the optional graph/telemetry IO layer so the
recursive stack can consume *heterogeneous* trace sources uniformly:

    B.3  Graph adapter            -> opto.features.graph.GraphAdapter / LangGraphAdapter
    B.4  OpenTelemetry            -> opto.trace.io.instrument_graph / TelemetrySession
    B.5  Trace / OTEL / Sysmon    -> observers + TGJ (Trace Graph JSON) merge

An arbitrary graph with *named bindings* becomes a ``trace.Module``
(``GraphAdapter.as_module()``), and OTEL/Sysmon spans can be lifted into Trace
nodes via ``otlp_traces_to_trace_json`` / ``ingest_tgj``. That single fact lets
recursion treat graph workflows exactly like prompts/code: one ``forward()``,
one set of trainable parameters.

All imports are guarded: if graph/telemetry modules are unavailable, internal
trace collection still works and feature-specific paths fail with a clear error.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

# --- optional graph/telemetry imports -------------------------------------- #
try:
    from opto.trace.io import (
        instrument_graph,
        optimize_graph,
        TelemetrySession,
        Binding,
        apply_updates,
        make_dict_binding,
    )
    from opto.trace.io.sysmonitoring import sysmon_profile_to_tgj
    from opto.features.graph import GraphAdapter, LangGraphAdapter, GraphModule

    HAVE_TRACE_IO = True
except Exception:  # pragma: no cover
    HAVE_TRACE_IO = False


def require_trace_io(feature: str = "this feature") -> None:
    """Raise a clear error if graph/telemetry backends are unavailable.

    Use this at the top of any code path that genuinely needs those backends, so the
    absence is a LOUD failure rather than a silent no-op.
    """
    if not HAVE_TRACE_IO:
        raise RuntimeError(
            f"{feature} requires graph/telemetry backends "
            "(opto.features.graph + opto.trace.io), which are not importable "
            "in this environment."
        )


def graph_to_module(build_graph: Callable, bindings: Dict[str, Any]):
    """B.3: turn any graph + named knobs into a trainable ``trace.Module``.

    `bindings` maps param-key -> (getter, setter) via make_dict_binding. Returns
    a GraphModule whose .parameters() are trainable.
    """
    if not HAVE_TRACE_IO:
        raise RuntimeError(
            "graph adapter backend is not importable; use ArtifactLevel / "
            "native nesting instead."
        )
    adapter = LangGraphAdapter(
        build_graph=build_graph, bindings=make_dict_binding(bindings)
    )
    return adapter.as_module()  # plugs straight into opto.trainer.train


def collect_traces(
    trace_types: List[str],
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> "MultiTraceSession":
    """B.4/B.5: open a unified session emitting requested trace backends."""
    return MultiTraceSession(trace_types, meta=meta)


class MultiTraceSession:
    """Unifies internal Trace + OTEL + Sysmon into one TGJ feedback bundle.

    trace_types subset of {"internal", "otel", "sysmon"}. The internal Trace is
    always available; OTEL/Sysmon require the optional trace IO backends. On exit,
    all enabled backends are merged into a single Trace-Graph-JSON dict usable as
    optimizer feedback.
    """

    def __init__(self, trace_types: List[str], *, meta: Optional[Dict[str, Any]] = None):
        self.trace_types = [t for t in trace_types]
        self._meta = dict(meta or {})
        self._otel = None
        self._sysmon = None
        self._sysmon_profile: Optional[Dict[str, Any]] = None
        self._otel_flushed = False
        self._sysmon_flushed = False
        self._tgj: Dict[str, Any] = {
            "nodes": [],
            "edges": [],
            "sources": [],
            "documents": [],
        }

    def __enter__(self):
        if "otel" in self.trace_types and HAVE_TRACE_IO:
            self._otel = TelemetrySession()
            self._otel.__enter__()
            self._tgj["sources"].append("otel")
        if "sysmon" in self.trace_types and HAVE_TRACE_IO:
            from opto.trace.io.sysmonitoring import SysMonitoringSession

            self._sysmon = SysMonitoringSession(service_name="recursive-opt-sysmon")
            # Upstream SysMonitoringSession is start/stop based, not a context
            # manager. Pass semantic filters through meta when callers need a
            # bounded profile around a large benchmark run.
            meta = {"service_name": "recursive-opt-sysmon"}
            meta.update(self._meta)
            self._sysmon.start(bindings={}, meta=meta)
            self._tgj["sources"].append("sysmon")
        self._tgj["sources"].append("internal")
        return self

    def __exit__(self, *exc):
        if self._otel is not None:
            self._otel.__exit__(*exc)
        if self._sysmon is not None:
            error = exc[1] if len(exc) > 1 else None
            self._sysmon_profile = self._sysmon.stop(error=error)
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
        if not HAVE_TRACE_IO:
            return self._tgj
        if self._otel is not None and not self._otel_flushed:
            try:
                docs = self._otel.flush_tgj(agent_id_hint="recursive-opt", clear=True)
                self._add_tgj_documents("otel", docs)
                self._otel_flushed = True
            except Exception as e:  # don't hide failures behind a clean-looking result
                import warnings
                warnings.warn(f"OTEL->TGJ merge failed: {e!r}", RuntimeWarning)
        if self._sysmon_profile is not None and not self._sysmon_flushed:
            try:
                doc = sysmon_profile_to_tgj(
                    self._sysmon_profile,
                    run_id="recursive-opt-sysmon",
                    graph_id="sysmon",
                    scope="recursive-opt/sysmon",
                )
                self._add_tgj_documents("sysmon", [doc])
                self._sysmon_flushed = True
            except Exception as e:
                import warnings
                warnings.warn(f"Sysmon->TGJ merge failed: {e!r}", RuntimeWarning)
        return self._tgj

    def _add_tgj_documents(self, source: str, docs: Iterable[Dict[str, Any]]) -> None:
        """Attach backend TGJ documents and add compact nodes for summaries."""
        for doc in docs:
            self._tgj["documents"].append({"source": source, "document": doc})
            nodes = doc.get("nodes", {})
            node_iter = nodes.values() if isinstance(nodes, dict) else nodes
            for rec in node_iter or []:
                node_id = str(rec.get("id") or rec.get("name") or len(self._tgj["nodes"]))
                self._tgj["nodes"].append(
                    {
                        "id": node_id,
                        "label": rec.get("name", node_id),
                        "kind": rec.get("kind", "message"),
                        "source": source,
                    }
                )

    def feedback_text(self, base: str = "") -> str:
        """Compact, optimizer-readable summary of all trace sources."""
        tgj = self.to_tgj()
        srcs = ",".join(tgj.get("sources", []))
        return (
            f"{base}\n[traces:{srcs}] "
            f"nodes={len(tgj.get('nodes', []))} edges={len(tgj.get('edges', []))}"
        ).strip()
