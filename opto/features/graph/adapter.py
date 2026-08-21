"""Minimal graph execution contract with an optional LangGraph adapter."""

from __future__ import annotations

import importlib
import inspect
import json
from functools import wraps
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional

from opto.trace import bundle, node
from opto.trace.nodes import Node, ParameterNode

if TYPE_CHECKING:
    from opto.features.graph.module import GraphModule


GRAPH_ARTIFACT_VERSION = "graph-artifact/v1"
GRAPH_INPUT_CODEC = "graph.codec.state@1"
GRAPH_OUTPUT_CODEC = "graph.codec.output_key@1"


@bundle(description="[graph] Attach graph inputs and parameters to the output trace.")
def _trace_graph_output(value: Any, *_dependencies: Any) -> Any:
    return value


def _json_copy(value: Any, label: str) -> Any:
    """Return a detached JSON value or raise a contextual type error."""
    try:
        return json.loads(json.dumps(value, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be JSON-serializable") from exc


class GraphExecutor(ABC):
    """Backend-neutral contract used by ``GraphAdapter``."""

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[str]:
        """Return stable backend capability names."""

    @abstractmethod
    def invoke(self, state: dict[str, Any], **kwargs: Any) -> Any:
        """Execute one graph state and return its result."""

    @abstractmethod
    def parameters(self) -> Sequence[ParameterNode]:
        """Return the graph's trainable Trace parameters."""

    @abstractmethod
    def snapshot(self) -> Mapping[str, Any]:
        """Return JSON-serializable backend state."""

    @abstractmethod
    def restore(self, artifact: Mapping[str, Any]) -> None:
        """Validate and restore JSON-serializable backend state."""


class GraphAdapter:
    """Apply explicit codecs and persistence around a graph executor."""

    def __init__(
        self,
        executor: GraphExecutor,
        *,
        input_key: str = "query",
        output_key: Optional[str] = None,
        input_codec: str = GRAPH_INPUT_CODEC,
        output_codec: str = GRAPH_OUTPUT_CODEC,
    ) -> None:
        if not isinstance(executor, GraphExecutor):
            raise TypeError("executor must implement GraphExecutor")
        if not isinstance(input_key, str) or not input_key:
            raise ValueError("input_key must be a non-empty string")
        if output_key is not None and (not isinstance(output_key, str) or not output_key):
            raise ValueError("output_key must be None or a non-empty string")
        if input_codec != GRAPH_INPUT_CODEC:
            raise ValueError(f"unsupported graph input codec {input_codec!r}")
        if output_codec != GRAPH_OUTPUT_CODEC:
            raise ValueError(f"unsupported graph output codec {output_codec!r}")
        self.executor = executor
        self.input_key = input_key
        self.output_key = output_key
        self.input_codec = input_codec
        self.output_codec = output_codec

    @property
    def capabilities(self) -> frozenset[str]:
        """Return adapter and backend capabilities used during compilation."""
        return frozenset(self.executor.capabilities) | frozenset(
            {"snapshot", "restore", "trace_module", "input_codec", "output_codec"}
        )

    def parameters(self) -> list[ParameterNode]:
        """Return validated heterogeneous Trace parameters from the executor."""
        parameters = list(self.executor.parameters())
        if not all(isinstance(parameter, ParameterNode) for parameter in parameters):
            raise TypeError("graph executor parameters must be ParameterNode instances")
        return parameters

    def _encode_input(self, inputs: Any) -> dict[str, Any]:
        """Encode a module input into graph state."""
        if isinstance(inputs, Mapping):
            return dict(inputs)
        return {self.input_key: getattr(inputs, "data", inputs)}

    def _decode_output(self, result: Any) -> Any:
        """Decode one graph result using the configured output key."""
        if self.output_key is None:
            return result
        if not isinstance(result, Mapping):
            raise TypeError(
                f"graph output codec expected a mapping containing {self.output_key!r}"
            )
        if self.output_key not in result:
            raise KeyError(f"graph result is missing output key {self.output_key!r}")
        return result[self.output_key]

    def invoke(self, inputs: Any, **kwargs: Any) -> Any:
        """Execute the graph and return its runtime result."""
        return self.executor.invoke(self._encode_input(inputs), **kwargs)

    def invoke_trace(self, inputs: Any, **kwargs: Any) -> Node:
        """Execute the graph and attach its inputs and parameters to a Trace node."""
        state = self._encode_input(inputs)
        result = self.executor.invoke(state, **kwargs)
        output = self._decode_output(result)
        input_node = inputs if isinstance(inputs, Node) else node(inputs, name=self.input_key)
        traced = _trace_graph_output(output, input_node, *self.parameters())
        if not isinstance(traced, Node):
            raise TypeError("graph trace codec did not produce a Trace Node")
        return traced

    def as_module(self) -> "GraphModule":
        """Expose this adapter through the generic Trace ``Module`` contract."""
        from opto.features.graph.module import GraphModule

        return GraphModule(self)

    def snapshot(self) -> dict[str, Any]:
        """Return the explicit JSON artifact for adapter config and backend state."""
        artifact = {
            "schema_version": GRAPH_ARTIFACT_VERSION,
            "config": {
                "input_key": self.input_key,
                "output_key": self.output_key,
                "input_codec": self.input_codec,
                "output_codec": self.output_codec,
            },
            "state": self.executor.snapshot(),
        }
        return _json_copy(artifact, "graph artifact")

    @staticmethod
    def validate_artifact(artifact: Mapping[str, Any]) -> None:
        """Validate the portable outer graph artifact shape."""
        if not isinstance(artifact, Mapping):
            raise TypeError("graph artifact must be a mapping")
        if set(artifact) != {"schema_version", "config", "state"}:
            raise ValueError("graph artifact keys must be schema_version, config, and state")
        if artifact.get("schema_version") != GRAPH_ARTIFACT_VERSION:
            raise ValueError(f"graph artifact schema must be {GRAPH_ARTIFACT_VERSION!r}")
        config = artifact.get("config")
        if not isinstance(config, Mapping) or set(config) != {
            "input_key", "output_key", "input_codec", "output_codec"
        }:
            raise ValueError("graph artifact config has an invalid shape")
        if not isinstance(artifact.get("state"), Mapping):
            raise TypeError("graph artifact state must be a mapping")
        _json_copy(artifact, "graph artifact")

    def restore(self, artifact: Mapping[str, Any]) -> None:
        """Restore backend state only when the artifact config matches this adapter."""
        self.validate_artifact(artifact)
        expected_config = self.snapshot()["config"]
        if dict(artifact["config"]) != expected_config:
            raise ValueError("graph artifact config does not match the target adapter")
        self.executor.restore(dict(artifact["state"]))


class _LangGraphExecutor(GraphExecutor):
    """LangGraph implementation kept behind the backend-neutral contract."""

    def __init__(
        self,
        graph_factory: Callable[..., Any],
        function_targets: Optional[Mapping[str, Callable[..., Any]]],
        graph_knobs: Optional[Mapping[str, Any]],
    ) -> None:
        try:
            importlib.import_module("langgraph")
        except ImportError as exc:
            raise ImportError(
                "LangGraphAdapter requires the optional 'langgraph' dependency"
            ) from exc
        if not callable(graph_factory):
            raise TypeError("graph_factory must be callable")
        targets = dict(function_targets or {})
        if not all(isinstance(name, str) and name and callable(value) for name, value in targets.items()):
            raise TypeError("function_targets must map non-empty names to callables")
        knobs = dict(graph_knobs or {})
        if not all(isinstance(name, str) and name for name in knobs):
            raise TypeError("graph_knobs keys must be non-empty strings")
        self.graph_factory = graph_factory
        self.function_targets = targets
        self.graph_knobs = {
            name: value if isinstance(value, ParameterNode) else node(value, name=name, trainable=True)
            for name, value in knobs.items()
        }

    @property
    def capabilities(self) -> frozenset[str]:
        """Declare only the capabilities implemented by the optional executor."""
        return frozenset({"langgraph", "graph_parameters", "json_snapshot"})

    def _factory_kwargs(
        self, function_targets: Optional[Mapping[str, Callable[..., Any]]] = None
    ) -> dict[str, Any]:
        """Resolve only arguments accepted by the graph factory."""
        available = {
            **dict(function_targets or self.function_targets),
            **{name: parameter.data for name, parameter in self.graph_knobs.items()},
        }
        signature = inspect.signature(self.graph_factory)
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return available
        return {name: value for name, value in available.items() if name in signature.parameters}

    def invoke(self, state: dict[str, Any], **kwargs: Any) -> Any:
        """Build and invoke a LangGraph with the current graph parameters."""
        shadow_state = dict(state)
        shadow_state.update({name: value.data for name, value in self.graph_knobs.items()})
        runtime_targets: dict[str, Callable[..., Any]] = {}
        for name, function in self.function_targets.items():
            runtime_targets[name] = self._state_preserving_wrapper(function, shadow_state)
        graph = self.graph_factory(**self._factory_kwargs(runtime_targets))
        compiled = graph.compile() if callable(getattr(graph, "compile", None)) else graph
        invoke = getattr(compiled, "invoke", None)
        if not callable(invoke):
            raise TypeError("graph_factory must return an object with invoke()")
        return invoke(dict(shadow_state), **kwargs)

    @staticmethod
    def _state_preserving_wrapper(
        function: Callable[..., Any], shadow_state: dict[str, Any]
    ) -> Callable[..., Any]:
        """Preserve mapping state across graph runtimes with replacement semantics."""
        @wraps(function)
        def wrapped(state: Any, *args: Any, **kwargs: Any) -> Any:
            merged = dict(shadow_state)
            if isinstance(state, Mapping):
                merged.update(state)
            result = function(merged, *args, **kwargs)
            if isinstance(result, Mapping):
                shadow_state.update(result)
                return dict(shadow_state)
            return result

        return wrapped

    def parameters(self) -> tuple[ParameterNode, ...]:
        """Return stable graph-knob parameters."""
        return tuple(self.graph_knobs.values())

    def snapshot(self) -> dict[str, Any]:
        """Return JSON state for all graph knobs."""
        return _json_copy(
            {"graph_knobs": {name: value.data for name, value in self.graph_knobs.items()}},
            "LangGraph state",
        )

    def restore(self, artifact: Mapping[str, Any]) -> None:
        """Validate and restore graph knobs without replacing Trace nodes."""
        if set(artifact) != {"graph_knobs"} or not isinstance(
            artifact.get("graph_knobs"), Mapping
        ):
            raise ValueError("LangGraph state must contain only graph_knobs")
        values = dict(artifact["graph_knobs"])
        if set(values) != set(self.graph_knobs):
            raise ValueError("LangGraph artifact graph_knobs do not match the executor")
        _json_copy(values, "LangGraph graph_knobs")
        for name, value in values.items():
            self.graph_knobs[name]._set(value)


class LangGraphAdapter(GraphAdapter):
    """Optional LangGraph frontend for the minimal graph executor contract."""

    def __init__(
        self,
        graph_factory: Callable[..., Any],
        *,
        function_targets: Optional[Mapping[str, Callable[..., Any]]] = None,
        graph_knobs: Optional[Mapping[str, Any]] = None,
        input_key: str = "query",
        output_key: Optional[str] = None,
        input_codec: str = GRAPH_INPUT_CODEC,
        output_codec: str = GRAPH_OUTPUT_CODEC,
        train_graph_agents_functions: bool = False,
    ) -> None:
        executor = _LangGraphExecutor(graph_factory, function_targets, graph_knobs)
        super().__init__(
            executor,
            input_key=input_key,
            output_key=output_key,
            input_codec=input_codec,
            output_codec=output_codec,
        )
        self.graph_factory = graph_factory
        self.function_targets = executor.function_targets
        self.graph_knobs = executor.graph_knobs
        self.train_graph_agents_functions = bool(train_graph_agents_functions)
