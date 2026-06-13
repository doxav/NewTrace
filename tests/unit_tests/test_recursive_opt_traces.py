"""Mock-backed integration test for multi-trace merging (internal+OTEL+Sysmon).

traces.py was architecturally right but untested end-to-end; this exercises the
real MultiTraceSession merge path with fake backends so CI never needs the
optional trace-IO dependencies installed.
"""
from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Optional

import pytest

from opto.features.recursive_opt import traces


class _FakeTelemetrySession:
    def __enter__(self) -> "_FakeTelemetrySession":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def flush_tgj(
        self,
        agent_id_hint: Optional[str] = None,
        clear: bool = True,
    ) -> List[Dict[str, Any]]:
        return [{"nodes": [{"id": "otel-1", "name": "otel_node", "kind": "message"}]}]


class _FakeSysMonSession:
    def __init__(self, service_name: str = "") -> None:
        self.service_name = service_name

    def start(
        self,
        bindings: Optional[Any] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.meta = dict(meta or {})

    def stop(self, error: Optional[Any] = None) -> Dict[str, Any]:
        return {"ok": True, "error": error}


def _fake_sysmon_profile_to_tgj(
    profile: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    assert profile["ok"] is True
    return {"nodes": [{"id": "sys-1", "name": "sys_node", "kind": "message"}]}


class _Node:
    def __init__(
        self,
        name: str,
        data: Any,
        parents: Optional[List["_Node"]] = None,
    ) -> None:
        self.name = name
        self.data = data
        self.parents = parents or []


def test_multitrace_merges_internal_otel_and_sysmon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(traces, "HAVE_TRACE_IO", True)
    monkeypatch.setattr(traces, "TelemetrySession", _FakeTelemetrySession, raising=False)
    monkeypatch.setattr(traces, "sysmon_profile_to_tgj", _fake_sysmon_profile_to_tgj, raising=False)
    monkeypatch.setitem(sys.modules, "opto.trace.io.sysmonitoring",
                        types.SimpleNamespace(SysMonitoringSession=_FakeSysMonSession))

    leaf = _Node("leaf", 1)
    root = _Node("root", 2, parents=[leaf])

    with traces.MultiTraceSession(["internal", "otel", "sysmon"]) as sess:
        sess.record_internal(root)
    tgj = sess.to_tgj()

    assert set(tgj["sources"]) == {"internal", "otel", "sysmon"}
    by_source = {n["source"] for n in tgj["nodes"]}
    assert by_source == {"internal", "otel", "sysmon"}     # all three merged
    internal = [n for n in tgj["nodes"] if n["source"] == "internal"]
    assert {n["label"] for n in internal} == {"root", "leaf"}
    assert any(e["source"] == "internal" for e in tgj["edges"])
    assert len(tgj["documents"]) == 2                       # one per backend doc
    text = sess.feedback_text("base")
    assert text.startswith("base") and "[traces:" in text


def test_multitrace_internal_only_without_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(traces, "HAVE_TRACE_IO", False)
    with traces.MultiTraceSession(["internal", "otel", "sysmon"]) as sess:
        sess.record_internal(_Node("only", 0))
    tgj = sess.to_tgj()
    assert tgj["sources"] == ["internal"]                   # graceful degradation
    assert all(n["source"] == "internal" for n in tgj["nodes"])
