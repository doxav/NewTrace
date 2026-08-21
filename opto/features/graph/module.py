"""Trace ``Module`` wrapper for the minimal graph adapter contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from opto.features.graph.adapter import GraphAdapter
from opto.trace.modules import Module
from opto.trace.nodes import Node, ParameterNode


class GraphModule(Module):
    """Expose any registered graph executor as a generic Trace module."""

    def __init__(self, adapter: GraphAdapter) -> None:
        if not isinstance(adapter, GraphAdapter):
            raise TypeError("adapter must be a GraphAdapter")
        self.adapter = adapter

    @property
    def capabilities(self) -> frozenset[str]:
        """Return the adapter capabilities used by engine preflight."""
        return self.adapter.capabilities

    def forward(self, inputs: Any, **kwargs: Any) -> Node:
        """Execute the graph and return its traced output node."""
        return self.adapter.invoke_trace(inputs, **kwargs)

    def invoke(self, inputs: Any, **kwargs: Any) -> Any:
        """Execute the graph and return its unmodified runtime result."""
        return self.adapter.invoke(inputs, **kwargs)

    def parameters(self) -> list[ParameterNode]:
        """Expose graph parameters to Trace optimizers."""
        return self.adapter.parameters()

    def snapshot(self) -> dict[str, Any]:
        """Return the adapter's portable artifact."""
        return self.adapter.snapshot()

    def restore(self, artifact: Mapping[str, Any]) -> None:
        """Restore the adapter from a validated portable artifact."""
        self.adapter.restore(artifact)
