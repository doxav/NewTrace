"""Lazy telemetry API: optional graph frontends do not gate low-level observers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "instrument_graph": "opto.trace.io.instrumentation",
    "InstrumentedGraph": "opto.trace.io.instrumentation",
    "SysMonInstrumentedGraph": "opto.trace.io.instrumentation",
    "instrument_trace_graph": "opto.features.graph.graph_instrumentation",
    "TraceGraph": "opto.features.graph.graph_instrumentation",
    "optimize_graph": "opto.trace.io.optimization",
    "EvalResult": "opto.trace.io.optimization",
    "EvalFn": "opto.trace.io.optimization",
    "RunResult": "opto.trace.io.optimization",
    "OptimizationResult": "opto.trace.io.optimization",
    "TelemetrySession": "opto.trace.io.telemetry_session",
    "Binding": "opto.trace.io.bindings",
    "apply_updates": "opto.trace.io.bindings",
    "make_dict_binding": "opto.trace.io.bindings",
    "emit_reward": "opto.trace.io.otel_semconv",
    "emit_agentlightning_reward": "opto.trace.io.otel_semconv",
    "emit_trace": "opto.trace.io.otel_semconv",
    "set_span_attributes": "opto.trace.io.otel_semconv",
    "record_genai_chat": "opto.trace.io.otel_semconv",
    "TracingLLM": "opto.trace.io.otel_runtime",
    "LLMCallError": "opto.trace.io.otel_runtime",
    "InMemorySpanExporter": "opto.trace.io.otel_runtime",
    "init_otel_runtime": "opto.trace.io.otel_runtime",
    "flush_otlp": "opto.trace.io.otel_runtime",
    "extract_eval_metrics_from_otlp": "opto.trace.io.otel_runtime",
    "otlp_traces_to_trace_json": "opto.trace.io.otel_adapter",
    "ingest_tgj": "opto.trace.io.tgj_ingest",
    "merge_tgj": "opto.trace.io.tgj_ingest",
    "ObserverArtifact": "opto.trace.io.observers",
    "GraphObserver": "opto.trace.io.observers",
    "OTelObserver": "opto.trace.io.observers",
    "SysMonitoringSession": "opto.trace.io.sysmonitoring",
    "SysMonObserver": "opto.trace.io.sysmonitoring",
    "sysmon_profile_to_tgj": "opto.trace.io.sysmonitoring",
    "GraphAdapter": "opto.features.graph",
    "LangGraphAdapter": "opto.features.graph",
    "GraphModule": "opto.features.graph",
    "GraphRunSidecar": "opto.features.graph",
    "OTELRunSidecar": "opto.features.graph",
    "GraphCandidateSnapshot": "opto.features.graph",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load only the backend owning the requested public API."""
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
