"""
recursive_opt.traces  —  multi-trace substrate  (B.3 / B.4 / B.5)
=================================================================

Thin, defensive wrappers around the optional telemetry IO layer so the
recursive stack can consume *heterogeneous* trace sources uniformly:

    B.4  OpenTelemetry            -> opto.trace.io.instrument_graph / TelemetrySession
    B.5  Trace / OTEL / Sysmon    -> observers + TGJ (Trace Graph JSON) merge

All imports are guarded: if telemetry modules are unavailable, internal
trace collection still works and feature-specific paths fail with a clear error.
"""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

# --- optional telemetry imports ------------------------------------------- #
_TRACE_IO_IMPORT_ERROR: Optional[ImportError] = None
try:
    from opto.trace.io.telemetry_session import TelemetrySession

    HAVE_TRACE_IO = True
except ImportError as exc:  # pragma: no cover - optional integration
    HAVE_TRACE_IO = False
    _TRACE_IO_IMPORT_ERROR = exc


try:
    from opto.trace.io.sysmonitoring import sysmon_profile_to_tgj

    HAVE_SYSMON = hasattr(sys, "monitoring")
except ImportError:
    HAVE_SYSMON = False


def require_trace_io(feature: str = "this feature") -> None:
    """Raise a clear error if graph/telemetry backends are unavailable.

    Use this at the top of any code path that genuinely needs those backends, so the
    absence is a LOUD failure rather than a silent no-op.
    """
    if not HAVE_TRACE_IO:
        error = RuntimeError(
            f"{feature} requires graph/telemetry backends "
            "(opto.trace.io), which are not importable "
            "in this environment."
        )
        if _TRACE_IO_IMPORT_ERROR is not None:
            raise error from _TRACE_IO_IMPORT_ERROR
        raise error


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

    def __init__(
        self,
        trace_types: List[str],
        *,
        meta: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ) -> None:
        self.trace_types = [t for t in trace_types]
        self.strict = strict
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
        if self.strict:
            if "otel" in self.trace_types:
                require_trace_io("requested OTEL capture")
            if "sysmon" in self.trace_types and not HAVE_SYSMON:
                raise RuntimeError("requested sysmon capture requires Python >= 3.12")
        if "otel" in self.trace_types and HAVE_TRACE_IO:
            self._otel = TelemetrySession()
            if hasattr(self._otel, "bundle_spans"):
                self._otel.bundle_spans = replace(
                    self._otel.bundle_spans,
                    max_spans=self._meta.get("max_events", 1000),
                )
            self._otel.__enter__()
            self._tgj["sources"].append("otel")
        if "sysmon" in self.trace_types and (
            HAVE_TRACE_IO or self.strict and HAVE_SYSMON
        ):
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
                parents = (
                    list(parents.values())
                    if isinstance(parents, dict)
                    else list(parents)
                )
            except Exception:
                parents = []
            for p in parents:
                self._tgj["edges"].append(
                    {"src": str(id(p)), "dst": nid, "source": "internal"}
                )
                visit(p)

        try:
            visit(node)
        except Exception as e:  # never break the optimization loop on tracing
            import warnings

            warnings.warn(f"internal-trace normalization failed: {e!r}", RuntimeWarning)
        return self

    def to_tgj(self) -> Dict[str, Any]:
        """Merge enabled backends into one Trace-Graph-JSON feedback object."""
        if not HAVE_TRACE_IO and not self.strict:
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
            node_iter = (
                nodes.items() if isinstance(nodes, dict) else enumerate(nodes or [])
            )
            for key, rec in node_iter:
                node_id = str(rec.get("id") or key)
                self._tgj["nodes"].append(
                    {
                        "id": node_id,
                        "label": rec.get("name", node_id),
                        "kind": rec.get("kind", "message"),
                        "source": source,
                        "value": rec.get("output", {}).get("value", rec.get("data")),
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


def validate_trace_config(config: Mapping[str, Any]) -> None:
    """Reject unsupported capture/projection options rather than silently ignoring them."""
    if not isinstance(config, Mapping):
        raise TypeError("trace_config must be a mapping")
    unknown = set(config) - {
        "mode",
        "detail",
        "credit_horizon",
        "max_nodes",
        "max_chars",
        "semantic_names",
    }
    if unknown:
        raise ValueError(f"unknown trace_config keys: {sorted(unknown)}")
    for key, default, values in (
        ("mode", "internal", {"internal", "otel", "sysmon", "hybrid"}),
        ("detail", "summary", {"summary", "full"}),
        ("credit_horizon", "episode", {"step", "episode", "truncated", "full"}),
    ):
        if config.get(key, default) not in values:
            raise ValueError(f"unsupported trace_config.{key}")
    for key, default, maximum in (
        ("max_nodes", 100, 10000),
        ("max_chars", 4000, 100000),
    ):
        value = config.get(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"trace_config.{key} must be an integer in [1, {maximum}]")
    names = config.get("semantic_names", [])
    if not isinstance(names, (list, tuple)) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("trace_config.semantic_names must contain function names")


def capture_evaluation(
    execute: Callable[[], Tuple[Any, Any]], config: Mapping[str, Any]
) -> Tuple[Any, Any]:
    """Capture actual evaluator execution and expose a bounded projection to the Guide.

    credit_horizon limits the trace projection within this evaluation. It does not
    change optimizer backpropagation or introduce cross-episode memory.
    """
    validate_trace_config(config)
    mode = config.get("mode", "internal")
    types = ["internal", "otel", "sysmon"] if mode == "hybrid" else [mode]
    limit = config.get("max_nodes", 100)
    meta = {
        "semantic_names": config.get(
            "semantic_names", ["execute", "forward", "evaluate", "get_feedback"]
        ),
        "max_events": limit,
    }
    with MultiTraceSession(types, meta=meta, strict=True) as session:
        span = (
            session._otel.tracer.start_as_current_span("recursive_opt.evaluate")
            if session._otel
            else nullcontext()
        )
        with span:
            result, output = execute()
            if "internal" in types:
                roots = output.values() if isinstance(output, Mapping) else [output]
                for root in roots:
                    if hasattr(root, "parents"):
                        session.record_internal(root, max_nodes=limit)
    graph = session.to_tgj()
    graph["sources"] = types
    nodes = graph["nodes"]
    horizon = config.get("credit_horizon", "episode")
    count = (
        1
        if horizon in {"step", "truncated"}
        else min(8, limit) if horizon == "episode" else limit
    )
    # Interleave sources so an internal graph cannot hide all external evidence.
    grouped = [
        [record for record in nodes if record.get("source") == source]
        for source in types
    ]
    projected = [
        group[index]
        for index in range(max((len(group) for group in grouped), default=0))
        for group in grouped
        if index < len(group)
    ][:count]
    if config.get("detail", "summary") == "summary":
        projected = [
            {"label": n.get("label"), "source": n.get("source")} for n in projected
        ]
    rendered = json.dumps(
        {"sources": types, "nodes": projected}, ensure_ascii=False, default=str
    )
    feedback = (
        str(result.feedback)
        + "\nexecution_trace: "
        + rendered[: config.get("max_chars", 4000)]
    )
    return (
        replace(
            result,
            feedback=feedback,
            trace={"evaluation": result.trace, "capture": graph},
        ),
        output,
    )
