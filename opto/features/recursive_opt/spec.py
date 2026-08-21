"""Declarative control plane for ``recursive_opt`` (steps 1+2).

A ``RecursiveSpec`` is a plain ``dict`` (no new dependency) that describes:

* ``families``     - {family_name: [task_ids]} scope,
* ``budget``       - a global :class:`RecursiveOptBudget` (depth/cost guard),
* ``tracebench``   - real adapter bounds (examples, inner steps, timeouts),
* ``scoring``      - optional score normalization/capping for cross-family runs,
* ``prior_promotion`` - M1 -> M3 family-prior promotion policy,
* ``memory_root``  - where the tiered :class:`MemoryLite` persists,
* ``reuse_priors`` - transfer: warm-start each level from (family, level) memory,
* ``levels``       - an **ordered** list; *the ordering is the recursion depth*.

Each level dict selects a ``surface`` (``config`` | ``code`` | ``family_policy``
| ``prior`` | ``custom``) and is compiled into the **existing** level classes,
memory, budget, and :func:`optimize`. Nothing in Trace core is touched; this is a
thin, transparent compiler that returns the built objects so users can inspect
them.

Minimal usage::

    from opto.features.recursive_opt.spec import run_spec
    out = run_spec(my_spec_dict)            # build + optimize each level in order
    out["results"]["o1_setup"]["artifact"] # the learned config/code/policy text
    out["memory"].summary()                 # episodes/artifacts/priors recorded
"""
from __future__ import annotations

import json
import hashlib
import math
import re
import time
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from opto.trace import node
from opto.trace.containers import Map
from opto.trace.modules import Module
from opto.trainer.objectives import (
    EvaluationResult,
    apply_minimize,
    normalize_evaluation_result,
    satisfies_hard_constraints,
    to_scalar_score,
    weighted_scalarize,
)

from .levels import (
    CapabilityArtifact,
    TimedGuide,
    RecursiveGuide,
    LevelConfig,
    MetaLevel,
    FamilyPolicyLevel,
    PriorInductionLevel,
    ComponentSpec,
    CodeArtifactLevel,
    DEFAULT_INVALID_FLOOR,
    best_config_from,
    register_config_values,
    validate_level_config,
)
from .memory import ArtifactRecord, MemoryLite
from .budget import RecursiveOptBudget, reset_budget, make_budget
from .optimize import optimize, current_iterations
from .progress import RecursiveOptProgressLogger, budget_snapshot
from . import tracebench as TB

SURFACES = ("config", "code", "family_policy", "prior", "capability", "custom")

SCHEMA_VERSION = "recursive-opt/v2alpha"
SPEC_KIND = "recursive_optimization"
GEPA_VERSION = "0.1.4"
CANONICAL_SPEC_BLOCKS = (
    "surface",
    "module",
    "engine",
    "runtime",
    "objective",
    "llm_profiles",
    "llm_roles",
    "datasets",
    "knowledge",
    "bindings",
    "outputs",
    "budget",
    "experiment",
)
_TOP_LEVEL_KEYS = {"schema_version", "kind", "fingerprint", "extensions", *CANONICAL_SPEC_BLOCKS}
_LEGACY_TOP_LEVEL_KEYS = {
    "families",
    "budget",
    "tracebench",
    "scoring",
    "prior_promotion",
    "memory_root",
    "reuse_priors",
    "levels",
    "trainer_kwargs",
    "run_id",
    "extensions",
}
_BLOCK_KEYS = {
    "surface": {"kind", "targets", "levels"},
    "module": {"ref", "config", "artifact", "inputs"},
    "engine": {"name", "config"},
    "runtime": {
        "strict_refs", "reproducible", "offline", "resume", "memory_root",
        "reuse_priors", "tracebench", "scoring", "prior_promotion",
        "trainer_kwargs", "run_id", "seed",
    },
    "objective": {
        "evaluator_ref", "intent", "metrics", "directions", "selection", "hard_constraints",
        "aggregation", "feedback_channels",
    },
    "datasets": {"train", "validation", "holdout"},
    "knowledge": {
        "store", "retrieval", "statuses", "scope_fields", "top_k",
        "injection_codec", "promotion_rule", "rollback_rule",
    },
    "outputs": {"directory", "format", "save_artifacts"},
    "budget": {
        "optimizer_llm_calls", "eval_llm_calls", "candidates", "wall_time_s",
        "on_exceed",
    },
    "experiment": {"seeds", "arms", "matrix"},
}
_PROFILE_KEYS = {
    "provider", "model", "resolved_model", "api_key_ref", "fallbacks",
    "temperature", "max_tokens", "base_url",
}
_ROLE_KEYS = {"forward", "optimizer", "feedback", "judge"}
_ROLE_OVERRIDE_KEYS = {"profile", *_PROFILE_KEYS}
_BINDING_KEYS = {"from", "to", "codec", "ordering_only"}
_VERSIONED_REF = re.compile(r"^[A-Za-z0-9_.-]+@[1-9][0-9]*$")
_SECRET_KEYS = {"api_key", "apikey", "access_token", "token", "secret", "password"}


class _FrozenDict(dict):
    """JSON-serializable dictionary that rejects mutation after construction."""

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("normalized recursive-opt specs are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


@dataclass(frozen=True)
class ModuleRegistryEntry:
    """Build and persist one versioned kind of generic ``trace.Module``."""

    build: Callable[[Mapping[str, Any], Mapping[str, Any]], Module]
    snapshot: Callable[[Module], Dict[str, Any]]
    restore: Callable[[Module, Mapping[str, Any]], None]
    validate_artifact: Callable[[Mapping[str, Any]], None]
    capabilities: frozenset[str]


@dataclass(frozen=True)
class EngineRegistryEntry:
    """Run one execution unit and declare supported compile-time capabilities."""

    run: Callable[["_ExecutionUnit", Mapping[str, Any]], "RunResult"]
    capabilities: frozenset[str]


@dataclass(frozen=True)
class _ExecutionUnit:
    """One internal, fully materialized arm/seed/matrix execution unit."""

    unit_id: str
    arm_id: str
    seed: Optional[int]
    matrix: Mapping[str, Any]
    spec: Mapping[str, Any]


@dataclass(frozen=True)
class RunResult:
    """Canonical, JSON-exportable result of one compiled execution unit."""

    unit_id: str
    plan_fingerprint: str
    spec_fingerprint: str
    engine: str
    module_ref: str
    status: str
    valid: bool
    evaluation: EvaluationResult
    artifact: Mapping[str, Any]
    lineage: Tuple[Mapping[str, Any], ...]
    usage: Mapping[str, Any]
    budget: Mapping[str, Any]
    metadata: Mapping[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical result as plain JSON-compatible containers."""
        return {
            "unit_id": self.unit_id,
            "plan_fingerprint": self.plan_fingerprint,
            "spec_fingerprint": self.spec_fingerprint,
            "engine": self.engine,
            "module_ref": self.module_ref,
            "status": self.status,
            "valid": self.valid,
            "evaluation": {
                "valid": self.evaluation.valid,
                "status": self.evaluation.status,
                "metrics": _thaw(self.evaluation.metrics),
                "feedback": _thaw(self.evaluation.feedback),
                "trace": _thaw(self.evaluation.trace),
                "usage": _thaw(self.evaluation.usage),
                "artifacts": _thaw(self.evaluation.artifacts),
                "error": self.evaluation.error,
            },
            "artifact": _thaw(self.artifact),
            "lineage": _thaw(self.lineage),
            "usage": _thaw(self.usage),
            "budget": _thaw(self.budget),
            "metadata": _thaw(self.metadata),
            "error": self.error,
        }


@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable compilation product consumed by engine runners."""

    spec: Mapping[str, Any]
    units: Tuple[_ExecutionUnit, ...]
    fingerprint: str

    def explain(self) -> Dict[str, Any]:
        """Return a compact JSON explanation of this immutable plan."""
        return {
            "fingerprint": self.fingerprint,
            "execution_units": len(self.units),
            "engines": sorted({unit.spec["engine"]["name"] for unit in self.units}),
            "module_refs": sorted({unit.spec["module"]["ref"] for unit in self.units}),
            "unit_ids": [unit.unit_id for unit in self.units],
        }


class DatasetAccess:
    """Capability gate that prevents holdout access during optimization phases."""

    _PHASES = {
        "fit", "proposal", "induction", "candidate_selection",
        "final_evaluation", "promotion", "report",
    }
    _HOLDOUT_PHASES = {"final_evaluation", "promotion", "report"}

    def __init__(self, datasets: Mapping[str, Any]) -> None:
        if not isinstance(datasets, Mapping):
            raise TypeError("datasets must be a mapping")
        unknown = set(datasets) - {"train", "validation", "holdout"}
        if unknown:
            raise ValueError(f"unknown dataset splits: {sorted(unknown)}")
        self._datasets = _freeze({
            split: _thaw(datasets.get(split, []))
            for split in ("train", "validation", "holdout")
        })

    def read(self, split: str, *, phase: str) -> Any:
        """Read one split only when the named execution phase has capability."""
        if phase not in self._PHASES:
            raise ValueError(f"unknown dataset access phase {phase!r}")
        if split not in self._datasets:
            raise ValueError(f"unknown dataset split {split!r}")
        if split == "holdout" and phase not in self._HOLDOUT_PHASES:
            raise PermissionError(f"holdout is inaccessible during {phase}")
        return self._datasets[split]


class _ComponentModule(Module):
    """Small generic multi-component module used by registered workflows."""

    def __init__(self, components: Mapping[str, Any]) -> None:
        if not components:
            raise ValueError("module.config.components must be a non-empty mapping")
        self.components = Map({
            name: node(value, name=name, trainable=True)
            for name, value in components.items()
        })

    def forward(self, inputs: Any) -> Dict[str, Any]:
        """Expose resolved component values alongside the module input."""
        return {
            "inputs": getattr(inputs, "data", inputs),
            "components": {name: value.data for name, value in self.components.items()},
        }


class _LegacyLevelsModule(Module):
    """Compatibility module representing a normalized legacy level pipeline."""

    def __init__(self, levels: Iterable[Mapping[str, Any]]) -> None:
        self.levels = tuple(_freeze(_thaw(level)) for level in levels)

    def forward(self, inputs: Any) -> Dict[str, Any]:
        """Return declarative legacy levels without executing hidden orchestration."""
        return {"inputs": getattr(inputs, "data", inputs), "levels": self.levels}


_MODULE_REGISTRY: Dict[str, ModuleRegistryEntry] = {}
_ENGINE_REGISTRY: Dict[str, EngineRegistryEntry] = {}
_EVALUATOR_REGISTRY: Dict[
    str, Callable[[Module, Any, Mapping[str, Any]], EvaluationResult]
] = {}


@dataclass(frozen=True)
class _CodecEntry:
    """Typed internal codec used by causal bindings."""

    encode: Callable[[Any], Any]
    input_type: Any
    output_type: Any
    input_description: str


_CODEC_REGISTRY: Dict[str, _CodecEntry] = {}


def register_module(ref: str, entry: ModuleRegistryEntry) -> None:
    """Register one exact module reference, rejecting aliases and replacement."""
    if not isinstance(ref, str) or not _VERSIONED_REF.fullmatch(ref):
        raise ValueError("module registry keys must be exact versioned refs")
    if not isinstance(entry, ModuleRegistryEntry):
        raise TypeError("entry must be a ModuleRegistryEntry")
    if ref in _MODULE_REGISTRY and _MODULE_REGISTRY[ref] != entry:
        raise ValueError(f"module ref {ref!r} is already registered")
    _MODULE_REGISTRY[ref] = entry


def register_engine(name: str, entry: EngineRegistryEntry) -> None:
    """Register one engine runner without importing optional dependencies."""
    if not isinstance(name, str) or not name:
        raise ValueError("engine name must be a non-empty string")
    if not isinstance(entry, EngineRegistryEntry):
        raise TypeError("entry must be an EngineRegistryEntry")
    if name in _ENGINE_REGISTRY and _ENGINE_REGISTRY[name] != entry:
        raise ValueError(f"engine {name!r} is already registered")
    _ENGINE_REGISTRY[name] = entry


def register_codec(
    ref: str,
    encode: Callable[[Any], Any],
    *,
    input_type: Any,
    output_type: Any,
    input_description: str,
) -> None:
    """Register an exact typed codec used by explicit causal bindings."""
    if not isinstance(ref, str) or not _VERSIONED_REF.fullmatch(ref):
        raise ValueError("codec registry keys must be exact versioned refs")
    if not callable(encode):
        raise TypeError("codec encode must be callable")
    entry = _CodecEntry(encode, input_type, output_type, input_description)
    if ref in _CODEC_REGISTRY and _CODEC_REGISTRY[ref] != entry:
        raise ValueError(f"codec ref {ref!r} is already registered")
    _CODEC_REGISTRY[ref] = entry


def build_module(spec: Mapping[str, Any], resources: Optional[Mapping[str, Any]] = None) -> Module:
    """Build the generic module selected by a normalized or raw v2alpha spec."""
    normalized = normalize_spec(spec)
    entry = _module_entry(normalized["module"]["ref"])
    module = entry.build(normalized, resources or {})
    if not isinstance(module, Module):
        raise TypeError(f"module builder returned {type(module).__name__}, expected trace.Module")
    return module


def snapshot_module(spec: Mapping[str, Any], module: Module) -> Dict[str, Any]:
    """Create and validate a JSON-serializable artifact for ``module``."""
    normalized = normalize_spec(spec)
    entry = _module_entry(normalized["module"]["ref"])
    artifact = entry.snapshot(module)
    entry.validate_artifact(artifact)
    _validate_no_callables_or_secrets(artifact, "module artifact")
    _canonical_json(artifact)
    return _thaw(artifact)


def restore_module(spec: Mapping[str, Any], module: Module, artifact: Mapping[str, Any]) -> None:
    """Validate and restore a registered module artifact in place."""
    normalized = normalize_spec(spec)
    entry = _module_entry(normalized["module"]["ref"])
    entry.validate_artifact(artifact)
    entry.restore(module, artifact)


def compile_plan(raw_spec: Mapping[str, Any]) -> ExecutionPlan:
    """Normalize, resolve refs, and expand one spec into immutable execution units."""
    normalized = normalize_spec(raw_spec)
    _module_entry(normalized["module"]["ref"])
    _evaluator_entry(normalized["objective"]["evaluator_ref"])
    for binding in normalized["bindings"]:
        if not binding.get("ordering_only", False):
            _codec_entry(binding["codec"])
    units = tuple(_expand_execution_units(normalized))
    if not units:
        raise ValueError("spec expansion produced no execution units")
    for unit in units:
        engine_name = unit.spec["engine"]["name"]
        engine = _engine_entry(engine_name)
        compile_objective(unit.spec["objective"], capabilities=engine.capabilities)
    return ExecutionPlan(spec=normalized, units=units, fingerprint=normalized["fingerprint"])


def execute_plan(
    plan: ExecutionPlan, resources: Optional[Mapping[str, Any]] = None
) -> Tuple[RunResult, ...]:
    """Execute every unit through its registered engine and retain canonical errors."""
    if not isinstance(plan, ExecutionPlan):
        raise TypeError("plan must be an ExecutionPlan")
    runtime_resources = dict(resources or {})
    result_store = runtime_resources.get("result_store")
    if result_store is not None and not isinstance(result_store, MutableMapping):
        raise TypeError("result_store resource must be a mutable mapping")
    results: List[RunResult] = []
    for unit in plan.units:
        engine_name = unit.spec["engine"]["name"]
        resume_key = f"{unit.spec['fingerprint']}:{unit.unit_id}"
        if unit.spec["runtime"]["resume"] and result_store is not None and resume_key in result_store:
            cached = result_store[resume_key]
            if not isinstance(cached, RunResult):
                raise TypeError(f"resume entry {resume_key!r} must be a RunResult")
            if cached.spec_fingerprint != unit.spec["fingerprint"]:
                raise ValueError(f"resume entry {resume_key!r} has a stale fingerprint")
            results.append(cached)
            continue
        started_at = time.monotonic()
        try:
            result = _engine_entry(engine_name).run(unit, runtime_resources)
        except Exception as exc:
            error = _safe_error(exc)
            evaluation = EvaluationResult(
                valid=False,
                status="error",
                metrics={},
                feedback="execution failed",
                error=error,
            )
            result = RunResult(
                unit_id=unit.unit_id,
                plan_fingerprint=plan.fingerprint,
                spec_fingerprint=unit.spec["fingerprint"],
                engine=engine_name,
                module_ref=unit.spec["module"]["ref"],
                status="error",
                valid=False,
                evaluation=evaluation,
                artifact={},
                lineage=(),
                usage=evaluation.usage,
                budget=_account_budget(
                    unit.spec, evaluation, candidates=0, evaluation_runs=0,
                    started_at=started_at,
                ),
                metadata={"engine_capabilities": sorted(_engine_entry(engine_name).capabilities)},
                error=error,
            )
        if result.plan_fingerprint != plan.fingerprint:
            result = RunResult(
                **{**result.__dict__, "plan_fingerprint": plan.fingerprint}
            )
        if result_store is not None:
            result_store[resume_key] = result
        results.append(result)
    return tuple(results)


def apply_bindings(
    spec: Mapping[str, Any],
    outputs: Mapping[str, Any],
    module_inputs: MutableMapping[str, Any],
) -> List[Dict[str, Any]]:
    """Apply typed causal bindings and return lineage for every injected value."""
    normalized = normalize_spec(spec)
    if not isinstance(outputs, Mapping):
        raise TypeError("binding outputs must be a mapping")
    if not isinstance(module_inputs, MutableMapping):
        raise TypeError("module_inputs must be a mutable mapping")
    lineage: List[Dict[str, Any]] = []
    for binding in normalized["bindings"]:
        if binding.get("ordering_only", False):
            continue
        source = _resolve_dotted_path(outputs, binding["from"])
        entry = _codec_entry(binding["codec"])
        if not isinstance(source, entry.input_type):
            raise TypeError(
                f"codec {binding['codec']!r} requires {entry.input_description}; "
                f"got {type(source).__name__}"
            )
        encoded = entry.encode(source)
        if not isinstance(encoded, entry.output_type):
            raise TypeError(f"codec {binding['codec']!r} returned an invalid output type")
        destination = binding["to"].split(".")
        if destination[:2] != ["module", "inputs"] or len(destination) < 3:
            raise ValueError("binding destinations must be below module.inputs")
        _set_nested_value(module_inputs, destination[2:], encoded)
        artifact_id = (
            source.artifact_id
            if isinstance(source, ArtifactRecord)
            else source.get("artifact_id") if isinstance(source, Mapping) else None
        )
        lineage.append({
            "from": binding["from"],
            "to": binding["to"],
            "codec": binding["codec"],
            "artifact_id": artifact_id,
        })
    return lineage


def compile_objective(
    objective: Mapping[str, Any], *, capabilities: Iterable[str]
) -> Dict[str, Any]:
    """Compile a canonical objective with engine capability validation."""
    from opto.trainer.objectives import ObjectiveConfig

    if not isinstance(objective, Mapping):
        raise TypeError("objective must be a mapping")
    selection = objective.get("selection")
    directions = objective.get("directions")
    if not isinstance(selection, Mapping) or not isinstance(directions, Mapping):
        raise TypeError("objective requires selection and directions mappings")
    mode = str(selection.get("mode", "scalar"))
    supported = set(capabilities)
    if mode not in supported:
        raise ValueError(f"engine does not support objective mode {mode!r}")
    minimize = frozenset(
        metric for metric, direction in directions.items() if direction == "minimize"
    )
    config = ObjectiveConfig(
        mode=mode,
        weights=dict(selection.get("weights") or {}),
        minimize=minimize,
        pareto_metrics=(
            tuple(selection["pareto_metrics"])
            if selection.get("pareto_metrics") is not None
            else None
        ),
        tie_break=str(selection.get("tie_break", "weighted")),
        seed=int(selection.get("seed", 0)),
        scalarize_dict=str(selection.get("scalarize_dict", "score")),
        score_key=str(selection.get("score_key", "score")),
    )
    return {
        "config": config,
        "intent": objective.get("intent", ""),
        "metrics": tuple(objective.get("metrics", ())),
        "hard_constraints": tuple(_thaw(objective.get("hard_constraints", ()))),
        "aggregation": _freeze(_thaw(objective.get("aggregation", {}))),
        "feedback_channels": tuple(objective.get("feedback_channels", ())),
    }


def resolve_llm_roles(
    spec: Mapping[str, Any], overrides: Optional[Mapping[str, Any]] = None
) -> Mapping[str, Any]:
    """Resolve all global or level-local LLM role overrides to exact profiles."""
    normalized = normalize_spec(spec)
    roles = _thaw(normalized["llm_roles"])
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise TypeError("llm role overrides must be a mapping")
        _reject_unknown_keys(overrides, _ROLE_KEYS, "level llm_roles")
        roles.update(_thaw(overrides))
    resolved = {
        role: _materialize_role(value, normalized["llm_profiles"], role)
        for role, value in roles.items()
    }
    return _freeze(resolved)


def preflight_llm_profiles(
    spec: Mapping[str, Any], *, checker: Optional[Callable[[str], None]] = None
) -> None:
    """Check every exact model used by a role once, propagating provider errors."""
    if checker is None:
        from .runmode import preflight_model

        checker = preflight_model
    roles = resolve_llm_roles(spec)
    checked: set[str] = set()
    for role in sorted(roles):
        profile = roles[role]
        if profile is None:
            continue
        model = profile["resolved_model"]
        if model not in checked:
            checker(model)
            checked.add(model)


def retrieve_knowledge(
    spec: Mapping[str, Any], memory: MemoryLite, scope: Mapping[str, str]
) -> List[Any]:
    """Retrieve promoted knowledge explicitly in the runner, before module build."""
    normalized = normalize_spec(spec)
    if not isinstance(memory, MemoryLite):
        raise TypeError("knowledge store must be the existing MemoryLite")
    if not isinstance(scope, Mapping):
        raise TypeError("knowledge retrieval scope must be a mapping")
    policy = normalized["knowledge"]
    scoped = {
        field: scope[field]
        for field in policy["scope_fields"]
        if field in scope
    }
    result = memory.retrieve(
        artifact_type="knowledge_card",
        statuses=policy["statuses"],
        scope=scoped,
        topk=policy["top_k"],
        sort=policy["retrieval"],
    )
    return result["artifacts"]


def _module_entry(ref: str) -> ModuleRegistryEntry:
    """Resolve one exact module ref without import fallback."""
    entry = _MODULE_REGISTRY.get(ref)
    if entry is None:
        raise ValueError(f"unregistered module ref {ref!r}")
    return entry


def _engine_entry(name: str) -> EngineRegistryEntry:
    """Resolve one exact engine name without fallback."""
    entry = _ENGINE_REGISTRY.get(name)
    if entry is None:
        raise ValueError(f"unregistered engine {name!r}")
    return entry


def _evaluator_entry(
    ref: str,
) -> Callable[[Module, Any, Mapping[str, Any]], EvaluationResult]:
    """Resolve one exact evaluator ref without dynamic imports."""
    evaluator = _EVALUATOR_REGISTRY.get(ref)
    if evaluator is None:
        raise ValueError(f"unregistered evaluator ref {ref!r}")
    return evaluator


def _codec_entry(ref: str) -> _CodecEntry:
    """Resolve one exact codec ref without import fallback."""
    entry = _CODEC_REGISTRY.get(ref)
    if entry is None:
        raise ValueError(f"unregistered codec ref {ref!r}")
    return entry


def _resolve_dotted_path(value: Mapping[str, Any], path: str) -> Any:
    """Resolve a required dotted source path from execution outputs."""
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"binding source {path!r} is unavailable")
        current = current[part]
    return current


def _set_nested_value(target: MutableMapping[str, Any], parts: List[str], value: Any) -> None:
    """Set a typed binding destination below module.inputs."""
    current = target
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, MutableMapping):
            raise ValueError(f"binding destination segment {part!r} is not a mapping")
        current = child
    current[parts[-1]] = value


def _artifact_to_prior(value: Any) -> Dict[str, Any]:
    """Convert an artifact record/mapping into an injected prior with lineage."""
    if isinstance(value, ArtifactRecord):
        return {"knowledge": value.content}
    if not isinstance(value, Mapping) or "content" not in value:
        raise TypeError("artifact_to_prior requires a mapping artifact with content")
    return {"knowledge": _thaw(value["content"])}


def _component_dict(value: Any) -> Dict[str, Any]:
    """Copy a string-keyed component mapping for module input injection."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError("component_dict requires a string-keyed mapping")
    return _thaw(value)


def _account_budget(
    spec: Mapping[str, Any],
    evaluation: EvaluationResult,
    *,
    candidates: int,
    evaluation_runs: int,
    started_at: float,
) -> Mapping[str, Any]:
    """Attach observed role calls, candidates, and wall time to budget limits."""
    usage = evaluation.usage
    role_calls = {
        role: int(values.get("calls", 0))
        for role, values in usage.items()
    }
    accounted = {
        "optimizer_llm_calls": role_calls.get("optimizer", 0),
        "eval_llm_calls": sum(role_calls.get(role, 0) for role in ("forward", "feedback", "judge")),
        "candidates": candidates,
        "wall_time_s": round(max(0.0, time.monotonic() - started_at), 6),
        "evaluation_runs": evaluation_runs,
        "total_tokens": sum(int(values.get("total_tokens", 0)) for values in usage.values()),
    }
    limits = _thaw(spec["budget"])
    exceeded = [
        name for name in ("optimizer_llm_calls", "eval_llm_calls", "candidates", "wall_time_s")
        if limits[name] is not None and accounted[name] > limits[name]
    ]
    return _freeze({**limits, "accounted": accounted, "exceeded": exceeded})


def _run_fixed_engine(
    unit: _ExecutionUnit, resources: Mapping[str, Any]
) -> RunResult:
    """Evaluate a fixed registered module without fitting it."""
    return _run_module_engine(unit, resources, fit=False)


def _run_trace_engine(
    unit: _ExecutionUnit, resources: Mapping[str, Any]
) -> RunResult:
    """Fit and evaluate an arbitrary registered ``trace.Module``."""
    return _run_module_engine(unit, resources, fit=True)


def _run_gepa_engine(
    unit: _ExecutionUnit, resources: Mapping[str, Any]
) -> RunResult:
    """Adapt GEPA OptimizeAnything to the canonical module/evaluator contracts."""
    started_at = time.monotonic()
    spec = unit.spec
    engine = _engine_entry(spec["engine"]["name"])
    objective = compile_objective(spec["objective"], capabilities=engine.capabilities)
    evaluator = resources.get("evaluator")
    if evaluator is None:
        evaluator = _evaluator_entry(spec["objective"]["evaluator_ref"])
    if not callable(evaluator):
        raise TypeError("GEPA engine requires a canonical evaluator resource")
    seed_module = build_module(spec, resources)
    seed_artifact = snapshot_module(spec, seed_module)
    seed_candidate = seed_artifact.get("components", seed_artifact)
    access = DatasetAccess(spec["datasets"])
    train = list(access.read("train", phase="fit"))
    validation = list(access.read("validation", phase="candidate_selection"))
    holdout = list(access.read("holdout", phase="final_evaluation"))
    evaluation_info: List[Dict[str, Any]] = []

    def gepa_evaluator(candidate: Any, example: Any) -> Tuple[float, Dict[str, Any]]:
        candidate_module = build_module(spec, resources)
        artifact = _candidate_to_artifact(seed_artifact, candidate)
        restore_module(spec, candidate_module, artifact)
        context = {
            "spec": spec,
            "inputs": spec["module"]["inputs"],
            "objective": objective,
            "llm_roles": spec["llm_roles"],
            "engine": spec["engine"]["name"],
        }
        evaluation = normalize_evaluation_result(evaluator(candidate_module, [example], context))
        score, info = _project_for_gepa(evaluation, objective)
        evaluation_info.append(info)
        return score, info

    optimize_anything, config = _resolve_gepa(resources, spec["engine"]["config"])
    gepa_result = optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=gepa_evaluator,
        dataset=train,
        valset=validation,
        test_set=holdout,
        objective=spec["objective"]["intent"],
        config=config,
    )
    best_candidate = _gepa_best_candidate(gepa_result)
    final_module = build_module(spec, resources)
    restore_module(spec, final_module, _candidate_to_artifact(seed_artifact, best_candidate))
    final_dataset = holdout or validation or train
    context = {
        "spec": spec,
        "inputs": spec["module"]["inputs"],
        "objective": objective,
        "llm_roles": spec["llm_roles"],
        "engine": spec["engine"]["name"],
    }
    evaluation = normalize_evaluation_result(evaluator(final_module, final_dataset, context))
    if not satisfies_hard_constraints(evaluation, objective["hard_constraints"]):
        evaluation = EvaluationResult(
            valid=False,
            status="constraint_failed",
            metrics=evaluation.metrics,
            feedback=evaluation.feedback,
            trace=evaluation.trace,
            usage=evaluation.usage,
            artifacts=evaluation.artifacts,
            error="hard constraints not satisfied",
        )
    artifact = snapshot_module(spec, final_module)
    return RunResult(
        unit_id=unit.unit_id,
        plan_fingerprint="",
        spec_fingerprint=spec["fingerprint"],
        engine=spec["engine"]["name"],
        module_ref=spec["module"]["ref"],
        status="success" if evaluation.valid else "invalid",
        valid=evaluation.valid,
        evaluation=evaluation,
        artifact=_freeze(artifact),
        lineage=(),
        usage=_freeze(evaluation.usage),
        budget=_account_budget(
            spec,
            evaluation,
            candidates=max(1, len(evaluation_info) + 1),
            evaluation_runs=len(evaluation_info) + 1,
            started_at=started_at,
        ),
        metadata=_freeze({
            "engine_capabilities": sorted(engine.capabilities),
            "gepa_version": GEPA_VERSION,
            "gepa_evaluations": evaluation_info,
            "objective_projection": objective["config"].mode,
        }),
        error=evaluation.error,
    )


def _candidate_to_artifact(seed_artifact: Mapping[str, Any], candidate: Any) -> Dict[str, Any]:
    """Convert GEPA text/component candidates back to the registered artifact."""
    components = seed_artifact.get("components")
    if not isinstance(components, Mapping):
        if not isinstance(candidate, Mapping):
            raise TypeError("non-component GEPA candidates require a mapping artifact")
        return _thaw(candidate)
    if isinstance(candidate, str):
        if len(components) != 1:
            raise TypeError("text GEPA candidate is valid only for one-component modules")
        candidate = {next(iter(components)): candidate}
    if not isinstance(candidate, Mapping):
        raise TypeError("GEPA candidate must be text or a component mapping")
    return {"components": _thaw(candidate)}


def _project_for_gepa(
    evaluation: EvaluationResult, objective: Mapping[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """Project canonical metrics deterministically while retaining complete info."""
    config = objective["config"]
    feasible = satisfies_hard_constraints(evaluation, objective["hard_constraints"])
    if not feasible:
        score = -1.0e12
    elif config.mode == "weighted":
        metrics = apply_minimize(evaluation.metrics, config.minimize)
        score = weighted_scalarize(metrics, config.weights, config.missing_value)
    elif config.mode == "scalar":
        score = to_scalar_score(evaluation.metrics, config)
    else:
        raise ValueError("GEPA does not support objective mode 'pareto'")
    info = {
        "valid": evaluation.valid and feasible,
        "status": evaluation.status if feasible else "constraint_failed",
        "metrics": _thaw(evaluation.metrics),
        "feedback": _thaw(evaluation.feedback),
        "trace": _thaw(evaluation.trace),
        "usage": _thaw(evaluation.usage),
        "artifacts": _thaw(evaluation.artifacts),
        "error": evaluation.error,
    }
    return float(score), info


def _resolve_gepa(
    resources: Mapping[str, Any], config_values: Mapping[str, Any]
) -> Tuple[Callable[..., Any], Any]:
    """Resolve injected GEPA or import the exact pinned optional dependency."""
    injected = resources.get("gepa_optimize")
    if injected is not None:
        if not callable(injected):
            raise TypeError("gepa_optimize resource must be callable")
        return injected, resources.get("gepa_config")
    try:
        from importlib.metadata import version
        from gepa.optimize_anything import OptimizeAnythingConfig, optimize_anything
    except ImportError as exc:
        raise ImportError(
            f"GEPA engine requires optional dependency gepa=={GEPA_VERSION}; "
            "install trace-opt[gepa]"
        ) from exc
    installed = version("gepa")
    if installed != GEPA_VERSION:
        raise RuntimeError(f"GEPA version must be {GEPA_VERSION}, found {installed}")
    config = OptimizeAnythingConfig(**_thaw(config_values))
    return optimize_anything, config


def _gepa_best_candidate(result: Any) -> Any:
    """Extract the documented best candidate from a GEPA result."""
    if isinstance(result, Mapping) and "best_candidate" in result:
        return result["best_candidate"]
    if hasattr(result, "best_candidate"):
        return result.best_candidate
    raise TypeError("GEPA result does not expose best_candidate")


def _run_module_engine(
    unit: _ExecutionUnit, resources: Mapping[str, Any], *, fit: bool
) -> RunResult:
    """Shared fixed/Trace contract with capability-gated data and rich results."""
    started_at = time.monotonic()
    spec = unit.spec
    engine = _engine_entry(spec["engine"]["name"])
    objective = compile_objective(spec["objective"], capabilities=engine.capabilities)
    module = build_module(spec, resources)
    access = DatasetAccess(spec["datasets"])
    module_inputs = _thaw(spec["module"]["inputs"])
    outputs = _thaw(resources.get("outputs", {}))
    memory = resources.get("memory")
    if memory is not None:
        scope = resources.get("knowledge_scope", {})
        knowledge = retrieve_knowledge(spec, memory, scope)
        if knowledge:
            outputs["knowledge"] = {"outputs": {"artifact": knowledge[0]}}
    lineage = tuple(apply_bindings(spec, outputs, module_inputs))
    context = {
        "spec": spec,
        "inputs": _freeze(module_inputs),
        "objective": objective,
        "llm_roles": spec["llm_roles"],
        "engine": spec["engine"]["name"],
    }
    fit_callable = resources.get("fit")
    if fit and fit_callable is not None:
        if not callable(fit_callable):
            raise TypeError("Trace engine fit resource must be callable")
        fit_callable(module, access, context)
    evaluator = resources.get("evaluator")
    if evaluator is None:
        evaluator = _evaluator_entry(spec["objective"]["evaluator_ref"])
    if not callable(evaluator):
        raise TypeError("engine evaluator resource must be callable")
    dataset = access.read("holdout", phase="final_evaluation")
    if not dataset:
        dataset = access.read("validation", phase="final_evaluation")
    if not dataset:
        dataset = access.read("train", phase="final_evaluation")
    evaluation = normalize_evaluation_result(evaluator(module, dataset, context))
    if not satisfies_hard_constraints(evaluation, objective["hard_constraints"]):
        evaluation = EvaluationResult(
            valid=False,
            status="constraint_failed",
            metrics=evaluation.metrics,
            feedback=evaluation.feedback,
            trace=evaluation.trace,
            usage=evaluation.usage,
            artifacts=evaluation.artifacts,
            error="hard constraints not satisfied",
        )
    artifact = snapshot_module(spec, module)
    status = "success" if evaluation.valid else "invalid"
    return RunResult(
        unit_id=unit.unit_id,
        plan_fingerprint="",
        spec_fingerprint=spec["fingerprint"],
        engine=spec["engine"]["name"],
        module_ref=spec["module"]["ref"],
        status=status,
        valid=evaluation.valid,
        evaluation=evaluation,
        artifact=_freeze(artifact),
        lineage=lineage,
        usage=_freeze(evaluation.usage),
        budget=_account_budget(
            spec, evaluation, candidates=1, evaluation_runs=1,
            started_at=started_at,
        ),
        metadata=_freeze({
            "engine_capabilities": sorted(engine.capabilities),
            "module_capabilities": sorted(_module_entry(spec["module"]["ref"]).capabilities),
            "objective_mode": objective["config"].mode,
        }),
        error=evaluation.error,
    )


def _default_module_evaluator(
    module: Module, dataset: Any, context: Mapping[str, Any]
) -> EvaluationResult:
    """Evaluate a module whose output already follows the canonical result shape."""
    item = dataset[0] if isinstance(dataset, (list, tuple)) and dataset else context["inputs"]
    return normalize_evaluation_result(module(item))


def _reasoning_evaluator(
    module: Module, dataset: Any, context: Mapping[str, Any]
) -> EvaluationResult:
    """Score a named reasoning component against deterministic expected output."""
    item = dataset[0] if isinstance(dataset, (list, tuple)) and dataset else None
    if not isinstance(item, Mapping):
        raise TypeError("reasoning evaluator requires a mapping dataset item")
    if tuple(context["spec"]["objective"]["metrics"]) != ("accuracy",):
        raise ValueError("reasoning evaluator supports only the accuracy metric")
    component = item.get("component")
    if not isinstance(component, str) or not component:
        raise ValueError("reasoning evaluator dataset requires a component name")
    if "expected" not in item:
        raise ValueError("reasoning evaluator dataset requires expected output")
    output = module(item.get("input", {}))
    components = output.get("components") if isinstance(output, Mapping) else None
    if not isinstance(components, Mapping) or component not in components:
        raise ValueError(f"reasoning module output is missing component {component!r}")
    actual = components[component]
    expected = item["expected"]
    score = 1.0 if actual == expected else 0.0
    return EvaluationResult(
        valid=True,
        status="ok",
        metrics={"accuracy": score},
        feedback=f"component {component!r}: expected {expected!r}, got {actual!r}",
        trace={"component": component},
    )


def _safe_error(error: Exception) -> str:
    """Return one-line, secret-redacted execution error text without traceback paths."""
    message = str(error).splitlines()[0] if str(error) else "execution failed"
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-<redacted>", message)
    return f"{type(error).__name__}: {message}"


def _build_component_module(
    spec: Mapping[str, Any], _resources: Mapping[str, Any]
) -> Module:
    """Build the registered reasoning workflow from named components."""
    config = spec["module"]["config"]
    components = config.get("components") if isinstance(config, Mapping) else None
    if not isinstance(components, Mapping):
        raise TypeError("reasoning workflow requires module.config.components mapping")
    if not all(isinstance(name, str) and name for name in components):
        raise ValueError("component names must be non-empty strings")
    return _ComponentModule(components)


def _build_legacy_levels_module(
    spec: Mapping[str, Any], _resources: Mapping[str, Any]
) -> Module:
    """Build the thin compatibility view over migrated legacy levels."""
    return _LegacyLevelsModule(spec["surface"]["levels"])


def _build_graph_module(
    spec: Mapping[str, Any], resources: Mapping[str, Any]
) -> Module:
    """Build a graph module from an explicitly supplied executor resource."""
    from opto.features.graph import GraphAdapter, GraphExecutor

    config = spec["module"]["config"]
    allowed = {"executor_ref", "input_key", "output_key", "input_codec", "output_codec"}
    if not isinstance(config, Mapping) or set(config) - allowed:
        raise ValueError(f"graph module config keys must be a subset of {sorted(allowed)}")
    executor_ref = config.get("executor_ref")
    if not isinstance(executor_ref, str) or not _VERSIONED_REF.fullmatch(executor_ref):
        raise ValueError("graph module requires an exact versioned executor_ref")
    executors = resources.get("graph_executors")
    if not isinstance(executors, Mapping) or executor_ref not in executors:
        raise ValueError(f"graph executor resource {executor_ref!r} is unavailable")
    executor = executors[executor_ref]
    if not isinstance(executor, GraphExecutor):
        raise TypeError(f"graph executor resource {executor_ref!r} must implement GraphExecutor")
    adapter = GraphAdapter(
        executor,
        input_key=config.get("input_key", "query"),
        output_key=config.get("output_key"),
        input_codec=config.get("input_codec", "graph.codec.state@1"),
        output_codec=config.get("output_codec", "graph.codec.output_key@1"),
    )
    return adapter.as_module()


def _snapshot_components(module: Module) -> Dict[str, Any]:
    """Snapshot named component parameters from a component module."""
    if not isinstance(module, _ComponentModule):
        raise TypeError("component artifact requires a registered component module")
    return {"components": {name: value.data for name, value in module.components.items()}}


def _restore_components(module: Module, artifact: Mapping[str, Any]) -> None:
    """Restore named component values without changing module topology."""
    if not isinstance(module, _ComponentModule):
        raise TypeError("component artifact requires a registered component module")
    components = artifact["components"]
    expected = set(module.components)
    if set(components) != expected:
        raise ValueError(
            f"artifact component keys {sorted(components)} do not match {sorted(expected)}"
        )
    for name, value in components.items():
        module.components[name]._set(value)


def _validate_component_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the portable component-dict artifact contract."""
    if not isinstance(artifact, Mapping) or set(artifact) != {"components"}:
        raise ValueError("component artifact must contain only 'components'")
    components = artifact["components"]
    if not isinstance(components, Mapping) or not components:
        raise ValueError("artifact.components must be a non-empty mapping")
    if not all(isinstance(name, str) and name for name in components):
        raise ValueError("artifact component keys must be non-empty strings")


def _snapshot_graph(module: Module) -> Dict[str, Any]:
    """Snapshot a registered graph module through its explicit adapter contract."""
    from opto.features.graph import GraphModule

    if not isinstance(module, GraphModule):
        raise TypeError("graph artifact requires a registered GraphModule")
    return module.snapshot()


def _restore_graph(module: Module, artifact: Mapping[str, Any]) -> None:
    """Restore a registered graph module without replacing its executor."""
    from opto.features.graph import GraphModule

    if not isinstance(module, GraphModule):
        raise TypeError("graph artifact requires a registered GraphModule")
    module.restore(artifact)


def _validate_graph_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the portable graph artifact without importing optional LangGraph."""
    from opto.features.graph import GraphAdapter

    GraphAdapter.validate_artifact(artifact)


def _snapshot_legacy(module: Module) -> Dict[str, Any]:
    """Snapshot a migrated legacy pipeline without executable objects."""
    if not isinstance(module, _LegacyLevelsModule):
        raise TypeError("legacy artifact requires a legacy levels module")
    return {"levels": _thaw(module.levels)}


def _restore_legacy(module: Module, artifact: Mapping[str, Any]) -> None:
    """Restore declarative legacy level metadata."""
    if not isinstance(module, _LegacyLevelsModule):
        raise TypeError("legacy artifact requires a legacy levels module")
    module.levels = tuple(_freeze(_thaw(level)) for level in artifact["levels"])


def _validate_legacy_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the migrated legacy-level artifact shape."""
    if not isinstance(artifact, Mapping) or set(artifact) != {"levels"}:
        raise ValueError("legacy artifact must contain only 'levels'")
    if not isinstance(artifact["levels"], (list, tuple)) or not artifact["levels"]:
        raise ValueError("legacy artifact levels must be non-empty")


def _expand_execution_units(spec: Mapping[str, Any]) -> List[_ExecutionUnit]:
    """Expand deterministic arm/seed/matrix products from a normalized spec."""
    experiment = spec["experiment"]
    seeds = list(experiment["seeds"]) or [None]
    if any(seed is not None and not isinstance(seed, int) for seed in seeds):
        raise TypeError("experiment.seeds must contain integers")
    arms = list(experiment["arms"]) or [{"id": "default"}]
    matrix = experiment["matrix"]
    matrix_paths = sorted(matrix)
    matrix_rows = list(product(*(matrix[path] for path in matrix_paths))) if matrix_paths else [()]
    units: List[_ExecutionUnit] = []
    for arm_index, raw_arm in enumerate(arms):
        if not isinstance(raw_arm, Mapping):
            raise TypeError(f"experiment.arms[{arm_index}] must be a mapping")
        _reject_unknown_keys(raw_arm, {"id", "engine", "overrides"}, f"experiment.arms[{arm_index}]")
        arm_id = raw_arm.get("id", f"arm-{arm_index}")
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError(f"experiment.arms[{arm_index}].id must be non-empty")
        for seed in seeds:
            for matrix_index, values in enumerate(matrix_rows):
                unit_raw = _thaw(spec)
                unit_raw.pop("fingerprint", None)
                unit_raw["experiment"] = {"seeds": [], "arms": [], "matrix": {}}
                unit_raw["runtime"]["seed"] = seed
                _apply_arm(unit_raw, raw_arm)
                selected = dict(zip(matrix_paths, values))
                for path, value in selected.items():
                    _set_dotted_path(unit_raw, path, value)
                unit_spec = normalize_spec(unit_raw)
                seed_label = "none" if seed is None else str(seed)
                unit_id = f"{arm_id}:seed-{seed_label}:matrix-{matrix_index}"
                units.append(_ExecutionUnit(unit_id, arm_id, seed, _freeze(selected), unit_spec))
    return units


def _apply_arm(spec: Dict[str, Any], arm: Mapping[str, Any]) -> None:
    """Materialize one arm's engine and dotted overrides into a unit spec."""
    engine = arm.get("engine")
    if isinstance(engine, str):
        spec["engine"]["name"] = engine
    elif isinstance(engine, Mapping):
        _reject_unknown_keys(engine, _BLOCK_KEYS["engine"], "experiment arm engine")
        spec["engine"].update(_thaw(engine))
    elif engine is not None:
        raise TypeError("experiment arm engine must be a name or mapping")
    overrides = arm.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise TypeError("experiment arm overrides must be a mapping")
    for path, value in overrides.items():
        _set_dotted_path(spec, path, value)


def _set_dotted_path(spec: Dict[str, Any], path: str, value: Any) -> None:
    """Set an existing canonical dotted path, rejecting hidden new controls."""
    if not isinstance(path, str) or not path or path.startswith(("fingerprint", "schema_version", "kind")):
        raise ValueError(f"invalid experiment override path {path!r}")
    parts = path.split(".")
    current: Dict[str, Any] = spec
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            raise ValueError(f"experiment override path {path!r} does not resolve to a mapping")
        current = next_value
    if parts[-1] not in current:
        raise ValueError(f"experiment override path {path!r} does not name an existing field")
    current[parts[-1]] = _thaw(value)


register_module(
    "recursive_opt.module.reasoning_workflow@1",
    ModuleRegistryEntry(
        build=_build_component_module,
        snapshot=_snapshot_components,
        restore=_restore_components,
        validate_artifact=_validate_component_artifact,
        capabilities=frozenset({"multi_component", "json_snapshot", "trace_module"}),
    ),
)
register_module(
    "recursive_opt.module.graph@1",
    ModuleRegistryEntry(
        build=_build_graph_module,
        snapshot=_snapshot_graph,
        restore=_restore_graph,
        validate_artifact=_validate_graph_artifact,
        capabilities=frozenset({
            "graph_executor", "json_snapshot", "trace_module", "input_output_codecs",
        }),
    ),
)
_EVALUATOR_REGISTRY.update({
    "recursive_opt.evaluator.module_output@1": _default_module_evaluator,
    "recursive_opt.evaluator.reasoning@1": _reasoning_evaluator,
})
register_codec(
    "recursive_opt.codec.artifact_to_prior@1",
    _artifact_to_prior,
    input_type=(ArtifactRecord, Mapping),
    output_type=dict,
    input_description="a mapping artifact with content",
)
register_codec(
    "recursive_opt.codec.component_dict@1",
    _component_dict,
    input_type=Mapping,
    output_type=dict,
    input_description="a string-keyed component mapping",
)
register_engine(
    "fixed",
    EngineRegistryEntry(
        run=_run_fixed_engine,
        capabilities=frozenset({
            "scalar", "weighted", "pareto", "trace_module", "rich_trace",
        }),
    ),
)
register_engine(
    "trace",
    EngineRegistryEntry(
        run=_run_trace_engine,
        capabilities=frozenset({
            "scalar", "weighted", "pareto", "trace_module",
            "heterogeneous_parameters", "rich_trace",
        }),
    ),
)
register_engine(
    "gepa_optimize_anything",
    EngineRegistryEntry(
        run=_run_gepa_engine,
        capabilities=frozenset({
            "scalar", "weighted", "trace_module", "multi_component", "rich_trace",
        }),
    ),
)
register_module(
    "recursive_opt.module.legacy_levels@1",
    ModuleRegistryEntry(
        build=_build_legacy_levels_module,
        snapshot=_snapshot_legacy,
        restore=_restore_legacy,
        validate_artifact=_validate_legacy_artifact,
        capabilities=frozenset({"legacy", "json_snapshot", "trace_module"}),
    ),
)


def migrate_legacy_spec(raw_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Migrate a supported legacy recursive spec into the single v2alpha shape."""
    if not isinstance(raw_spec, Mapping):
        raise TypeError("spec must be a mapping")
    spec = _thaw(raw_spec)
    if "schema_version" in spec or "kind" in spec:
        if spec.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SCHEMA_VERSION!r}, got {spec.get('schema_version')!r}"
            )
        if spec.get("kind") != SPEC_KIND:
            raise ValueError(f"kind must be {SPEC_KIND!r}, got {spec.get('kind')!r}")
        return spec

    unknown = set(spec) - _LEGACY_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown legacy spec keys: {sorted(unknown)}")
    levels = spec.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("legacy spec['levels'] must be a non-empty list")

    runtime = {
        "memory_root": spec.get("memory_root", "./trace_memory"),
        "reuse_priors": bool(spec.get("reuse_priors", False)),
    }
    for key in ("tracebench", "scoring", "prior_promotion", "trainer_kwargs", "run_id"):
        if key in spec:
            runtime[key] = spec[key]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "surface": {"kind": "recursive_levels", "levels": levels},
        "module": {
            "ref": "recursive_opt.module.legacy_levels@1",
            "config": {"families": spec.get("families", {})},
        },
        "engine": {"name": "trace"},
        "runtime": runtime,
        "budget": spec.get("budget", {}),
        "extensions": {
            **spec.get("extensions", {}),
            "recursive_opt.migration": {"source_schema": "legacy"},
        },
    }


def normalize_spec(raw_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a validated, immutable, secret-free canonical v2alpha spec."""
    migrated = migrate_legacy_spec(raw_spec)
    supplied_fingerprint = migrated.pop("fingerprint", None)
    _validate_no_callables_or_secrets(migrated)
    _validate_v2_structure(migrated)

    defaults: Dict[str, Any] = {
        "surface": {"kind": "module", "targets": [], "levels": []},
        "module": {"ref": None, "config": {}, "artifact": None, "inputs": {}},
        "engine": {"name": "trace", "config": {}},
        "runtime": {
            "strict_refs": True,
            "reproducible": True,
            "offline": False,
            "resume": False,
            "memory_root": "./trace_memory",
            "reuse_priors": False,
            "tracebench": {},
            "scoring": {},
            "prior_promotion": {},
            "trainer_kwargs": {},
            "run_id": None,
            "seed": None,
        },
        "objective": {
            "evaluator_ref": "recursive_opt.evaluator.module_output@1",
            "intent": "Maximize score.",
            "metrics": ["score"],
            "directions": {"score": "maximize"},
            "selection": {"mode": "scalar", "score_key": "score"},
            "hard_constraints": [],
            "aggregation": {"mode": "mean"},
            "feedback_channels": ["natural_language"],
        },
        "llm_profiles": {},
        "llm_roles": {role: None for role in sorted(_ROLE_KEYS)},
        "datasets": {"train": [], "validation": [], "holdout": []},
        "knowledge": {
            "store": "recursive_opt.knowledge.memory_lite@1",
            "retrieval": "best",
            "statuses": ["promoted"],
            "scope_fields": ["family", "level", "kind"],
            "top_k": 5,
            "injection_codec": "recursive_opt.codec.artifact_to_prior@1",
            "promotion_rule": {"min_support": 3},
            "rollback_rule": {"on_negative_transfer": True},
        },
        "bindings": [],
        "outputs": {"directory": None, "format": "json", "save_artifacts": True},
        "budget": {
            "optimizer_llm_calls": None,
            "eval_llm_calls": None,
            "candidates": None,
            "wall_time_s": None,
            "on_exceed": "return_best",
        },
        "experiment": {"seeds": [], "arms": [], "matrix": {}},
        "extensions": {},
    }
    normalized: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
    }
    for block in CANONICAL_SPEC_BLOCKS:
        value = migrated.get(block, defaults[block])
        if isinstance(defaults[block], dict):
            normalized[block] = _merge_defaults(defaults[block], value, block)
        else:
            normalized[block] = _thaw(value)
    normalized["extensions"] = _thaw(migrated.get("extensions", {}))
    _normalize_llm_profiles(normalized)
    normalized["bindings"] = [
        {"ordering_only": False, **_thaw(binding)}
        for binding in normalized["bindings"]
    ]
    _validate_v2_semantics(normalized)
    _validate_no_callables_or_secrets(normalized)
    fingerprint = hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()
    if supplied_fingerprint is not None and supplied_fingerprint != fingerprint:
        raise ValueError("spec fingerprint does not match normalized content")
    normalized["fingerprint"] = fingerprint
    return _freeze(normalized)


def explain_spec(raw_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serializable explanation of defaults and execution intent."""
    normalized = normalize_spec(raw_spec)
    experiment = normalized["experiment"]
    arm_count = max(1, len(experiment["arms"]))
    seed_count = max(1, len(experiment["seeds"]))
    matrix_count = 1
    for values in experiment["matrix"].values():
        matrix_count *= len(values)
    return {
        "schema_version": normalized["schema_version"],
        "kind": normalized["kind"],
        "fingerprint": normalized["fingerprint"],
        "engine": normalized["engine"]["name"],
        "module_ref": normalized["module"]["ref"],
        "objective_mode": normalized["objective"]["selection"]["mode"],
        "execution_units": arm_count * seed_count * matrix_count,
        "portable": True,
    }


def _merge_defaults(defaults: Dict[str, Any], value: Any, block: str) -> Dict[str, Any]:
    """Merge one validated control-plane block over its materialized defaults."""
    if not isinstance(value, Mapping):
        raise TypeError(f"spec[{block!r}] must be a mapping")
    return {**_thaw(defaults), **_thaw(value)}


def _thaw(value: Any) -> Any:
    """Copy mappings and sequences into mutable JSON-compatible containers."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    """Recursively freeze a JSON-compatible value without breaking json.dumps."""
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical_json(value: Any) -> str:
    """Serialize canonical spec content for stable SHA-256 fingerprints."""
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    """Reject structural typos at one control-plane path."""
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {path} keys: {sorted(unknown)}")


def _validate_v2_structure(spec: Dict[str, Any]) -> None:
    """Validate v2alpha container types and reject unknown structural keys."""
    _reject_unknown_keys(spec, _TOP_LEVEL_KEYS, "spec")
    if spec.get("schema_version") != SCHEMA_VERSION or spec.get("kind") != SPEC_KIND:
        raise ValueError(f"spec must declare {SCHEMA_VERSION!r} and kind {SPEC_KIND!r}")
    for block, allowed in _BLOCK_KEYS.items():
        value = spec.get(block, {})
        if not isinstance(value, Mapping):
            raise TypeError(f"spec[{block!r}] must be a mapping")
        _reject_unknown_keys(value, allowed, block)
    profiles = spec.get("llm_profiles", {})
    if not isinstance(profiles, Mapping):
        raise TypeError("spec['llm_profiles'] must be a mapping")
    for name, profile in profiles.items():
        if not isinstance(name, str) or not name:
            raise ValueError("llm profile names must be non-empty strings")
        if not isinstance(profile, Mapping):
            raise TypeError(f"llm profile {name!r} must be a mapping")
        _reject_unknown_keys(profile, _PROFILE_KEYS, f"llm_profiles.{name}")
    roles = spec.get("llm_roles", {})
    if not isinstance(roles, Mapping):
        raise TypeError("spec['llm_roles'] must be a mapping")
    _reject_unknown_keys(roles, _ROLE_KEYS, "llm_roles")
    for role, value in roles.items():
        if isinstance(value, Mapping):
            _reject_unknown_keys(value, _ROLE_OVERRIDE_KEYS, f"llm_roles.{role}")
        elif value is not None and not isinstance(value, str):
            raise TypeError(f"llm role {role!r} must be a profile name, mapping, or null")
    bindings = spec.get("bindings", [])
    if not isinstance(bindings, list):
        raise TypeError("spec['bindings'] must be a list")
    for index, binding in enumerate(bindings):
        if not isinstance(binding, Mapping):
            raise TypeError(f"bindings[{index}] must be a mapping")
        _reject_unknown_keys(binding, _BINDING_KEYS, f"bindings[{index}]")
    extensions = spec.get("extensions", {})
    if not isinstance(extensions, Mapping):
        raise TypeError("spec['extensions'] must be a mapping")
    for namespace in extensions:
        if not isinstance(namespace, str) or "." not in namespace:
            raise ValueError("extension keys must contain a namespace, for example 'acme.audit'")


def _normalize_llm_profiles(spec: Dict[str, Any]) -> None:
    """Materialize exact provider models, secret refs, fallbacks, and all roles."""
    profiles: Dict[str, Dict[str, Any]] = {}
    for name, raw_profile in spec["llm_profiles"].items():
        profile = {
            "provider": None,
            "model": None,
            "resolved_model": None,
            "api_key_ref": None,
            "fallbacks": [],
            "temperature": None,
            "max_tokens": None,
            "base_url": None,
            **_thaw(raw_profile),
        }
        if profile["provider"] == "openrouter":
            profile["model"] = profile["model"] or "deepseek/deepseek-v4-flash-0731"
            profile["resolved_model"] = f"openrouter/{profile['model']}"
            profile["api_key_ref"] = profile["api_key_ref"] or "env:OPENROUTER_API_KEY"
        elif profile["model"] is not None:
            profile["resolved_model"] = profile["resolved_model"] or profile["model"]
        profiles[name] = profile
    spec["llm_profiles"] = profiles
    roles = {role: None for role in sorted(_ROLE_KEYS)}
    roles.update(_thaw(spec["llm_roles"]))
    spec["llm_roles"] = {
        role: _materialize_role(value, profiles, role)
        for role, value in roles.items()
    }


def _materialize_role(
    value: Any, profiles: Mapping[str, Mapping[str, Any]], role: str
) -> Optional[Dict[str, Any]]:
    """Resolve one role name/override into a fully explicit profile mapping."""
    if value is None:
        return None
    if isinstance(value, str):
        profile_name = value
        overrides: Dict[str, Any] = {}
    elif isinstance(value, Mapping):
        profile_name = value.get("profile")
        overrides = {key: _thaw(item) for key, item in value.items() if key != "profile"}
    else:
        raise TypeError(f"llm role {role!r} must be a profile name, mapping, or null")
    if not isinstance(profile_name, str) or profile_name not in profiles:
        raise ValueError(f"llm role {role!r} references unknown profile {profile_name!r}")
    profile = {**_thaw(profiles[profile_name]), **overrides, "profile": profile_name}
    if profile["provider"] == "openrouter":
        profile["resolved_model"] = f"openrouter/{profile['model']}"
    elif profile.get("model") is not None:
        profile["resolved_model"] = profile.get("resolved_model") or profile["model"]
    return profile


def _validate_v2_semantics(spec: Dict[str, Any]) -> None:
    """Validate references, objective semantics, profiles, and experiment axes."""
    module_ref = spec["module"].get("ref")
    if not isinstance(module_ref, str) or not _VERSIONED_REF.fullmatch(module_ref):
        raise ValueError("module.ref must be a versioned registry ref such as 'namespace.name@1'")
    for path, ref in (
        ("knowledge.store", spec["knowledge"]["store"]),
        ("knowledge.injection_codec", spec["knowledge"]["injection_codec"]),
    ):
        if not isinstance(ref, str) or not _VERSIONED_REF.fullmatch(ref):
            raise ValueError(f"{path} must be a versioned registry ref")

    knowledge = spec["knowledge"]
    if knowledge["retrieval"] not in {"best", "recent"}:
        raise ValueError("knowledge.retrieval must be 'best' or 'recent'")
    if not isinstance(knowledge["statuses"], list) or not knowledge["statuses"]:
        raise ValueError("knowledge.statuses must be a non-empty list")
    if not set(knowledge["statuses"]) <= {
        "candidate", "promoted", "rejected", "rolled_back", "superseded"
    }:
        raise ValueError("knowledge.statuses contains an unknown status")
    if not isinstance(knowledge["scope_fields"], list) or not all(
        isinstance(field, str) and field for field in knowledge["scope_fields"]
    ):
        raise ValueError("knowledge.scope_fields must contain field names")
    if not isinstance(knowledge["top_k"], int) or knowledge["top_k"] <= 0:
        raise ValueError("knowledge.top_k must be a positive integer")

    for index, binding in enumerate(spec["bindings"]):
        if binding["ordering_only"]:
            continue
        for field in ("from", "to", "codec"):
            if not isinstance(binding.get(field), str) or not binding[field]:
                raise ValueError(f"bindings[{index}].{field} must be a non-empty string")
        if not _VERSIONED_REF.fullmatch(binding["codec"]):
            raise ValueError(f"bindings[{index}].codec must be a versioned registry ref")
        if not binding["to"].startswith("module.inputs."):
            raise ValueError(f"bindings[{index}].to must be below module.inputs")

    levels = spec["surface"]["levels"]
    if not isinstance(levels, list):
        raise TypeError("surface.levels must be a list")
    seen_levels: set[str] = set()
    for index, level in enumerate(levels):
        if not isinstance(level, Mapping):
            raise TypeError(f"surface.levels[{index}] must be a mapping")
        level_id = level.get("id")
        if not isinstance(level_id, str) or not level_id or level_id in seen_levels:
            raise ValueError("surface levels require unique non-empty ids")
        dependencies = level.get("depends_on", [])
        if not isinstance(dependencies, list) or any(dep not in seen_levels for dep in dependencies):
            raise ValueError(f"level {level_id!r} dependencies must reference earlier levels")
        if dependencies and not level.get("ordering_only", False):
            for dependency in dependencies:
                if not any(
                    binding.get("from", "").startswith(f"{dependency}.outputs.")
                    for binding in spec["bindings"]
                    if not binding["ordering_only"]
                ):
                    raise ValueError(
                        f"decorative dependency {dependency!r} -> {level_id!r} requires a binding"
                    )
        seen_levels.add(level_id)

    objective = spec["objective"]
    evaluator_ref = objective["evaluator_ref"]
    if not isinstance(evaluator_ref, str) or not _VERSIONED_REF.fullmatch(evaluator_ref):
        raise ValueError("objective.evaluator_ref must be a versioned registry ref")
    metrics = objective["metrics"]
    if not isinstance(metrics, list) or not metrics or not all(isinstance(item, str) and item for item in metrics):
        raise ValueError("objective.metrics must be a non-empty list of metric names")
    if len(set(metrics)) != len(metrics):
        raise ValueError("objective.metrics must be unique")
    directions = objective["directions"]
    if not isinstance(directions, Mapping) or set(directions) != set(metrics):
        raise ValueError("objective.directions must define every declared metric exactly once")
    if any(direction not in {"maximize", "minimize"} for direction in directions.values()):
        raise ValueError("objective directions must be 'maximize' or 'minimize'")
    selection = objective["selection"]
    if not isinstance(selection, Mapping):
        raise TypeError("objective.selection must be a mapping")
    _reject_unknown_keys(
        selection,
        {
            "mode", "weights", "score_key", "tie_break", "pareto_metrics",
            "seed", "scalarize_dict",
        },
        "objective.selection",
    )
    if selection.get("mode") not in {"scalar", "weighted", "pareto"}:
        raise ValueError("objective.selection.mode must be scalar, weighted, or pareto")
    weights = selection.get("weights", {})
    if not isinstance(weights, Mapping) or any(
        metric not in metrics or not isinstance(weight, (int, float)) or weight < 0
        for metric, weight in weights.items()
    ):
        raise ValueError("objective.selection.weights must be non-negative declared metrics")
    pareto_metrics = selection.get("pareto_metrics")
    if pareto_metrics is not None and (
        not isinstance(pareto_metrics, list)
        or not pareto_metrics
        or any(metric not in metrics for metric in pareto_metrics)
    ):
        raise ValueError("objective.selection.pareto_metrics must name declared metrics")
    for key in ("hard_constraints", "feedback_channels"):
        if not isinstance(objective[key], list):
            raise TypeError(f"objective.{key} must be a list")
    for index, constraint in enumerate(objective["hard_constraints"]):
        if not isinstance(constraint, Mapping):
            raise TypeError(f"objective.hard_constraints[{index}] must be a mapping")
        _reject_unknown_keys(
            constraint, {"metric", "op", "value"}, f"objective.hard_constraints[{index}]"
        )
        if constraint.get("metric") not in metrics:
            raise ValueError(f"hard constraint {index} must name a declared metric")
        if constraint.get("op") not in {"<", "<=", "==", "!=", ">=", ">"}:
            raise ValueError(f"hard constraint {index} has unsupported operator")
        if not isinstance(constraint.get("value"), (int, float)):
            raise TypeError(f"hard constraint {index} value must be numeric")
    if not isinstance(objective["aggregation"], Mapping):
        raise TypeError("objective.aggregation must be a mapping")
    _reject_unknown_keys(objective["aggregation"], {"mode", "weights"}, "objective.aggregation")

    for name, profile in spec["llm_profiles"].items():
        if not profile["provider"] or not profile["model"]:
            raise ValueError(f"llm profile {name!r} requires provider and exact model")
        if str(profile["model"]).endswith("latest") and spec["runtime"]["reproducible"]:
            raise ValueError(f"llm profile {name!r} cannot use a latest alias in reproducible mode")
        key_ref = profile["api_key_ref"]
        if key_ref is not None and (not isinstance(key_ref, str) or not key_ref.startswith("env:")):
            raise ValueError(f"llm profile {name!r} api_key_ref must use env:NAME")
        if not isinstance(profile["fallbacks"], list):
            raise TypeError(f"llm profile {name!r} fallbacks must be a list")
    for role, value in spec["llm_roles"].items():
        profile_name = value.get("profile") if isinstance(value, Mapping) else value
        if profile_name is not None and profile_name not in spec["llm_profiles"]:
            raise ValueError(f"llm role {role!r} references unknown profile {profile_name!r}")
    for key in ("seeds", "arms"):
        if not isinstance(spec["experiment"][key], list):
            raise TypeError(f"experiment.{key} must be a list")
    matrix = spec["experiment"]["matrix"]
    if not isinstance(matrix, Mapping) or any(not isinstance(values, list) or not values for values in matrix.values()):
        raise ValueError("experiment.matrix must map paths to non-empty value lists")


def _validate_no_callables_or_secrets(value: Any, path: str = "spec") -> None:
    """Reject callables, non-string mapping keys, secrets, and non-JSON values."""
    if callable(value):
        raise TypeError(f"{path} contains a callable; use a versioned registry ref")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            if key.lower() in _SECRET_KEYS and item is not None:
                raise ValueError(f"{path}.{key} contains a secret value; use an env reference")
            _validate_no_callables_or_secrets(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_no_callables_or_secrets(item, f"{path}[{index}]")
        return
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


def make_level_spec(*, id: str, surface: str, targets: Optional[List[str]] = None,
                    fixed: Optional[Dict[str, Any]] = None,
                    constraints: Optional[Dict[str, List[str]]] = None,
                    **kwargs: Any) -> Dict[str, Any]:
    """Safe level-spec builder (DRY for examples/notebooks).

    Raw dict literals silently drop duplicate keys (Python keeps the last one);
    that bug shipped in example E, losing the starting_artifact menu. Keyword
    arguments cannot be duplicated, so building specs through this helper makes
    that whole bug class impossible.
    """
    level: Dict[str, Any] = {"id": id, "surface": surface}
    if targets:
        level["targets"] = list(targets)
    if fixed:
        level["fixed"] = dict(fixed)
    if constraints:
        level["constraints"] = {k: list(v) for k, v in constraints.items()}
    level.update(kwargs)
    return level
# Default trainable fields for the family_policy (O2) and prior (O3) surfaces.
# These MUST be causally active or the surface "optimizes" fields the adapter
# ignores (the root cause of the prior surface underperforming). starting_artifact
# is always active; trace_type is feedback-active; batch_design is active when
# inner_steps>0. memory_policy was removed from the default because it has no
# consumer yet (it would be optimizing nothing). Override per level via `targets`.
_DEFAULT_POLICY_FIELDS = ("starting_artifact", "trace_type", "batch_design")


# --------------------------------------------------------------------------- #
# Step 1: validation + compilation                                            #
# --------------------------------------------------------------------------- #
def validate_spec(spec: dict) -> dict:
    """Validate a RecursiveSpec dict (structure, ids, surfaces, field values).

    Registers any per-level ``constraints`` and validates ``fixed`` config values
    against the existing registry, so illegal values fail *before* optimization.
    Returns the (unchanged) spec for chaining.
    """
    if not isinstance(spec, dict):
        raise TypeError("spec must be a dict")
    if "schema_version" in spec or "kind" in spec:
        return normalize_spec(spec)
    families = spec.get("families", {})
    if not isinstance(families, dict):
        raise TypeError("spec['families'] must be {name: [task_ids]}")
    levels = spec.get("levels")
    if not levels or not isinstance(levels, list):
        raise ValueError("spec['levels'] must be a non-empty list (its order is the depth)")
    if not isinstance(spec.get("budget", {}), dict):
        raise TypeError("spec['budget'] must be a dict")
    _validate_tracebench_config(spec.get("tracebench", {}))
    _validate_scoring_config(spec.get("scoring", {}))
    _validate_prior_promotion_config(spec.get("prior_promotion", {}))

    seen: set = set()
    for ls in levels:
        if not isinstance(ls, dict):
            raise TypeError("each level must be a dict")
        lid = ls.get("id")
        if not lid or lid in seen:
            raise ValueError(f"each level needs a unique 'id' (got {lid!r})")
        seen.add(lid)
        surface = ls.get("surface")
        if surface not in SURFACES:
            raise ValueError(f"level {lid}: surface must be one of {SURFACES}, got {surface!r}")

        # register declared constraints, then validate the seed/fixed values
        for field, allowed in (ls.get("constraints") or {}).items():
            register_config_values(field, allowed)
        if surface in ("config", "code", "family_policy", "prior"):
            fixed = ls.get("fixed") or {}
            cfg = LevelConfig(**fixed)
            fields = tuple(ls.get("targets") or ()) + tuple(fixed.keys())
            validate_level_config(cfg, fields)

        if surface in ("config", "family_policy", "prior"):
            _check_plumbing(ls)
        deps = ls.get("depends_on") or []
        earlier = seen - {lid}
        unknown = [d for d in deps if d not in earlier]
        if unknown:
            raise ValueError(
                f"level {lid}: depends_on {unknown} must reference EARLIER level ids "
                f"(seen so far: {sorted(earlier)}). depends_on is enforced, not decorative."
            )
        tasks = ls.get("tasks")
        if tasks is not None:
            if not isinstance(tasks, list) or not tasks or not all(str(t).strip() for t in tasks):
                raise ValueError(f"level {lid}: tasks must be a non-empty list of task ids")
        if surface == "config" and ls.get("family") not in families and not ls.get("task") and not tasks:
            raise ValueError(f"level {lid}: config needs a known 'family', explicit 'task', or non-empty 'tasks'")
        if surface == "code" and not isinstance(ls.get("component"), dict):
            raise ValueError(f"level {lid}: code surface needs a 'component' dict")
        if surface == "capability" and not callable(ls.get("evaluator")):
            raise TypeError(f"level {ls.get('id')}: capability surface requires a callable 'evaluator'")
        if surface == "custom" and not callable(ls.get("builder")):
            raise ValueError(f"level {lid}: custom surface needs a callable 'builder'")
        if surface in ("family_policy", "prior"):
            if not _resolve_families(ls, families):
                raise ValueError(f"level {lid}: {surface} needs at least one family")
    return spec


def compile_level(
    level_spec: dict,
    memory: MemoryLite,
    families: Dict[str, List[str]],
    scoring: Optional[dict] = None,
):
    """Compile one level dict into the matching existing level object."""
    surface = level_spec["surface"]
    score_config = level_spec.get("scoring", scoring)
    clip = _clip_bounds(score_config)
    # Invalid candidates score the worst LEGAL value. When an explicit clip is
    # configured, use its floor; otherwise fall back to a BOUNDED default (-1.0)
    # rather than the raw -1e9 sentinel. A single invalid config must sort last
    # WITHOUT destroying reported means/normalisation baselines (root cause of
    # the -1e9 / -333M leaks in the use-case tables when no scoring.clip is set).
    floor = clip[0] if clip else DEFAULT_INVALID_FLOOR

    if surface == "custom":
        return level_spec["builder"](level_spec, memory)

    if surface == "config":
        task_ids = _config_task_ids(level_spec, families)
        cfg = LevelConfig(**(level_spec.get("fixed") or {}))
        kwargs: Dict[str, Any] = {"memory": memory}
        if level_spec.get("targets"):
            kwargs["trainable_fields"] = tuple(level_spec["targets"])
        inner_runner = (
            _make_inner_runner(task_ids[0], score_config)
            if len(task_ids) == 1
            else _make_task_set_inner_runner(task_ids, score_config)
        )
        return MetaLevel(cfg=cfg, inner_runner=inner_runner,
                         invalid_floor=floor, **kwargs)

    if surface == "code":
        c = level_spec["component"]
        comp = ComponentSpec(
            name=c["name"], baseline=c["baseline"],
            evaluate=c["evaluate"], objective=c.get("objective", ""),
        )
        return CodeArtifactLevel(comp, memory=memory)

    if surface == "capability":
        return CapabilityArtifact(level_spec.get("seed", ""),
                                  evaluator=level_spec["evaluator"], memory=memory)

    if surface == "family_policy":
        fams = _resolve_families(level_spec, families)
        return FamilyPolicyLevel(
            fams, run_task=make_scored_task_runner(score_config), invalid_floor=floor,
            policy_fields=tuple(level_spec.get("targets") or _DEFAULT_POLICY_FIELDS),
            memory=memory,
        )

    if surface == "prior":
        fams = _resolve_families(level_spec, families)
        names = list(fams)
        train = {names[0]: fams[names[0]]}
        holdout = {n: fams[n] for n in names[1:]} or {names[0]: fams[names[0]]}
        return PriorInductionLevel(
            train, holdout, run_task=make_scored_task_runner(score_config), invalid_floor=floor,
            fields=tuple(level_spec.get("targets") or _DEFAULT_POLICY_FIELDS),
            memory=memory,
        )

    raise ValueError(f"unknown surface {surface!r}")


# --------------------------------------------------------------------------- #
# Step 2: prior/tool reuse, budget, and the runner                            #
# --------------------------------------------------------------------------- #
def reuse_priors(memory: MemoryLite, level, level_spec: dict) -> dict:
    """Warm-start a level from (family, level) memory and load reusable tools.

    Config levels warm-start from the promoted family prior (existing hook).
    For any surface, previously learned tools (stored as ``kind='tool'`` artifacts
    for the family) are loaded and returned so an agentic optimizer can reuse
    them. This is the transfer-learning entry point for a new project that points
    at the same family.
    """
    surface = level_spec["surface"]
    family = str(level_spec.get("family") or "*")
    used_prior = False

    if surface == "config" and hasattr(level, "warm_start_from_memory"):
        before = best_config_from(level)
        level.warm_start_from_memory(family)
        used_prior = (best_config_from(level) != before) or (memory.family_prior(family) is not None)
    else:
        # best-effort seed from a previously saved artifact of the same surface
        previous = memory.retrieve(family, kind=surface, topk=1)["artifacts"]
        prev = previous[0] if previous else None
        if prev is not None and hasattr(level, "propose"):
            try:
                _seed_from_text(level, surface, prev.content)
                used_prior = True
            except Exception:
                used_prior = False

    tools = [a.content for a in memory.retrieve(family, kind="tool")["artifacts"]]
    return {"used_prior": used_prior, "tools": tools}


def save_priors(memory: MemoryLite, level, level_spec: dict, score: float,
                metrics: Optional[dict] = None):
    """Persist the learned artifact (+ declared tools) tagged by family and level.

    Tagging by the spec's family makes the artifact retrievable for transfer to a
    new project via :func:`reuse_priors`. Episode/family-prior promotion already
    happens inside each level's ``forward`` during ``optimize``.
    """
    surface = level_spec["surface"]
    family = str(level_spec.get("family") or "*")
    rec = memory.record_artifact(
        level=surface, family=family, kind=surface,
        content=_artifact_text(level, surface), score=float(score), metrics=metrics,
    )
    for tool in (level_spec.get("tools") or []):
        memory.record_artifact(level=surface, family=family, kind="tool",
                               content=str(tool), score=float(score))
    return rec


def run_spec(spec: dict, *, optimizer=None, trainer: Optional[str] = None,
             budget: "RecursiveOptBudget | dict | None" = None,
             seeds: Optional[Iterable[int]] = None,
             resources: Optional[Mapping[str, Any]] = None) -> Any:
    """Compile and run every level in order (the ordering is the recursion depth).

    ``optimizer``/``trainer`` override the per-level choice (used for offline,
    no-LLM testing). ``budget`` (dict or RecursiveOptBudget) overrides
    ``spec["budget"]`` without mutating the spec. ``seeds`` (an iterable) switches
    to repeated multi-seed execution and returns a ``RepeatedResult`` per level
    instead of a single result (see :func:`run_spec_repeated`). Returns
    ``{"results", "levels", "memory", "progress"}``; the built level objects are
    returned so the compiler stays transparent and debuggable.
    """
    if isinstance(spec, dict) and ("schema_version" in spec or "kind" in spec):
        raw = _thaw(spec)
        raw.pop("fingerprint", None)
        if budget is not None:
            from .budget import budget_to_spec_dict

            raw["budget"] = (
                _thaw(budget)
                if isinstance(budget, dict)
                else budget_to_spec_dict(budget)
            )
        if seeds is not None:
            raw.setdefault("experiment", {})["seeds"] = list(seeds)
        runtime_resources = dict(resources or {})
        if optimizer is not None:
            runtime_resources.setdefault("optimizer", optimizer)
        if trainer is not None:
            runtime_resources.setdefault("trainer", trainer)
        results = execute_plan(compile_plan(raw), runtime_resources)
        return results[0] if len(results) == 1 else results
    if resources is not None:
        raise ValueError("resources are supported only by v2alpha specs")
    if seeds is not None:
        from .experiments import run_spec_repeated
        return run_spec_repeated(spec, seeds=seeds, optimizer=optimizer,
                                 trainer=trainer, budget=budget)
    configured_tracebench = False
    if isinstance(spec, dict) and "tracebench" in spec and isinstance(spec.get("tracebench"), dict):
        # The causal-effect validator depends on adapter run mode (notably
        # inner_steps). Configure spec-local Trace-Bench bounds before
        # validate_spec() checks active fields, otherwise validation can read a
        # stale adapter from a previous run.
        TB.configure_tracebench_adapter(spec.get("tracebench") or {}, require=True)
        configured_tracebench = True
    spec = validate_spec(spec)
    if "tracebench" in spec and not configured_tracebench:
        TB.configure_tracebench_adapter(spec.get("tracebench") or {}, require=True)
    families = spec.get("families", {})
    memory = _memory_from_spec(spec)
    _configure_budget(spec, override=budget)
    do_reuse = bool(spec.get("reuse_priors", False))

    run_id = str(spec.get("run_id") or f"recursive_opt:{int(time.time() * 1000)}")
    results: Dict[str, Any] = {}
    levels: Dict[str, Any] = {}
    progress_summary: Dict[str, Any] = {"run_id": run_id, "levels": {}}
    global_step = 0
    for level_index, ls in enumerate(spec["levels"]):
        lid = ls["id"]
        level = compile_level(ls, memory, families, spec.get("scoring"))
        levels[lid] = level
        reused = reuse_priors(memory, level, ls) if do_reuse else {"used_prior": False, "tools": []}

        iterations = int(ls.get("iterations") or current_iterations())
        # Generic trainer controls (e.g. {"num_threads": 1} for deterministic
        # unit tests) can live at spec or level scope without adding another
        # bespoke argument to run_spec.
        opt_kwargs: Dict[str, Any] = dict(spec.get("trainer_kwargs") or {})
        opt_kwargs.update(ls.get("trainer_kwargs") or {})
        oc = ls.get("objective_config")
        if oc:
            opt_kwargs["objective_config"] = _objective_config(oc)

        level_optimizer = ls.get("optimizer")
        agentic_factory = agentic_optimizer_factory(ls, memory, reused["tools"])
        if agentic_factory is not None and optimizer is None:
            level_optimizer = agentic_factory
        if ls.get("timed_guide"):
            # wall-time as a first-class objective: pair with
            # "objective_config": {"mode": "pareto", "minimize": ["wall_time"]}
            opt_kwargs["guide"] = TimedGuide(RecursiveGuide())
        objective_mode = str((oc or {}).get("mode", "scalar"))
        selected_by = "pareto" if objective_mode == "pareto" else "objective"
        task_ids = _level_task_ids(ls, families)
        progress_logger = RecursiveOptProgressLogger(
            memory=memory,
            run_id=run_id,
            level_id=lid,
            level_index=level_index,
            task_ids=task_ids,
            global_step_offset=global_step,
            echo=True,
        )
        # Keep progress persistence in recursive_opt, not in Trace core.
        opt_kwargs["logger"] = progress_logger
        memory.record_progress(
            run_id=run_id,
            level_id=lid,
            level_index=level_index,
            event="level_start",
            level_step=0,
            global_step=global_step,
            metrics={
                "planned_steps": iterations,
                "surface": ls["surface"],
                "objective_mode": objective_mode,
            },
            task_ids=task_ids,
            budget=budget_snapshot(),
            selected_by=selected_by,
        )
        _t0 = time.time()
        trainer_result = optimize(
            level, _dataset_for(ls, families, iterations),
            optimizer=(optimizer if optimizer is not None else level_optimizer),
            trainer=(trainer if trainer is not None else ls.get("trainer")),
            iterations=iterations, **opt_kwargs,
        )
        wall_s = round(time.time() - _t0, 3)

        selected_candidate = None
        try:
            score, data = _final_eval(level, ls, families)
            score = _clamp(score, _clip_bounds(spec.get("scoring")))  # belt: sentinel can never leak raw
            selected_candidate = _select_best_saved_candidate(memory, ls, families, score)
        except Exception as exc:
            selected_candidate = _select_best_saved_candidate(
                memory, ls, families, DEFAULT_INVALID_FLOOR
            )
            if selected_candidate is None:
                raise
            score = _clamp(float(selected_candidate.score), _clip_bounds(spec.get("scoring")))
            data = {
                "score": float(score),
                "feedback": "final evaluation failed; selected best saved candidate",
                "final_eval_error": f"{type(exc).__name__}: {exc}",
            }
        if selected_candidate is not None:
            _seed_from_text(level, ls["surface"], selected_candidate.content)
            score = _clamp(float(selected_candidate.score), _clip_bounds(spec.get("scoring")))
            if isinstance(data, dict):
                data = {
                    **data,
                    "selected_saved_candidate": selected_candidate.artifact_id,
                    "selected_saved_candidate_score": float(selected_candidate.score),
                }
        executed_steps = max(
            progress_logger.executed_steps,
            int(getattr(trainer_result, "n_iters", 0) or 0),
        )
        level_progress = progress_logger.build_summary(
            planned_steps=iterations,
            final_score=float(score),
            selected_by=selected_by,
            objective_mode=objective_mode,
        )
        level_progress["executed_steps"] = executed_steps
        artifact_metrics = dict(data) if isinstance(data, dict) else {}
        artifact_metrics["scores"] = dict(level_progress["scores"])
        artifact_metrics["progress"] = dict(level_progress)
        rec = save_priors(memory, level, ls, score,
                          metrics=artifact_metrics)
        level_progress["artifact_id"] = getattr(rec, "artifact_id", None)
        progress_summary["levels"][lid] = level_progress
        memory.record_progress(
            run_id=run_id,
            level_id=lid,
            level_index=level_index,
            event="level_end",
            level_step=max(0, executed_steps - 1) if executed_steps else None,
            global_step=global_step + executed_steps,
            artifact_id=getattr(rec, "artifact_id", None),
            problem_score=float(score),
            objective_score=float(score),
            metrics={"summary": level_progress, "wall_s": wall_s},
            task_ids=task_ids,
            budget=budget_snapshot(),
            selected_by=selected_by,
        )
        results[lid] = {
            "surface": ls["surface"],
            "score": score,
            "wall_s": wall_s,
            "artifact": _artifact_text(level, ls["surface"]),
            "reused_prior": reused["used_prior"],
            "tools": reused["tools"],
            "artifact_id": getattr(rec, "artifact_id", None),
            "depends_on": list(ls.get("depends_on") or []),  # recorded dependency edges
            "progress": level_progress,
        }
        global_step += executed_steps
    memory.write_run_summary(progress_summary)
    return {"results": results, "levels": levels, "memory": memory,
            "progress": progress_summary}


# --------------------------------------------------------------------------- #
# helpers (all over existing objects)                                         #
# --------------------------------------------------------------------------- #
def _resolve_families(level_spec: dict, families: Dict[str, List[str]]) -> Dict[str, List[str]]:
    sel = level_spec.get("families")
    fam = level_spec.get("family")
    if sel in (None, "*") and fam in (None, "*"):
        return dict(families)
    if isinstance(sel, list):
        return {k: families[k] for k in sel}
    if isinstance(fam, list):
        return {k: families[k] for k in fam}
    if isinstance(fam, str) and fam in families:
        return {fam: families[fam]}
    return dict(families)


def _config_task_ids(level_spec: dict, families: Dict[str, List[str]]) -> List[str]:
    """Return concrete task ids for a config level."""
    tasks = level_spec.get("tasks")
    if tasks is not None:
        return [str(task) for task in tasks]
    task = level_spec.get("task")
    if task:
        return [str(task)]
    return [str(families[level_spec["family"]][0])]


def _level_task_ids(level_spec: dict, families: Dict[str, List[str]]) -> List[str]:
    """Return task ids associated with any level for progress records."""
    if level_spec.get("surface") == "config" and (
        level_spec.get("tasks") or level_spec.get("task") or level_spec.get("family") in families
    ):
        return _config_task_ids(level_spec, families)
    selected = _resolve_families(level_spec, families)
    out: List[str] = []
    for tasks in selected.values():
        out.extend(str(task) for task in tasks)
    return out


def _dataset_for(level_spec: dict, families: Dict[str, List[str]], iterations: int) -> dict:
    fam = level_spec.get("family")
    task = level_spec.get("task")
    tasks = level_spec.get("tasks")
    # family_policy/prior iterate internally; custom/capability levels may be
    # label-free; all of these train on a None-input dataset.
    if level_spec["surface"] in ("family_policy", "prior") or not (fam or task or tasks):
        return {"inputs": [None] * iterations, "infos": [None] * iterations}
    if tasks:
        return TB.make_dataset([f"task_set:{level_spec.get('id', 'config')}"], repeats=iterations)
    return TB.make_dataset([fam or task], repeats=iterations)


def _artifact_text(level, surface: str) -> str:
    if surface == "config":
        return best_config_from(level)
    if surface == "code":
        return level.current_code()
    if surface == "capability":
        return str(level.impl.data)
    if surface == "family_policy":
        return str(getattr(level, "_policy_node").data)
    if surface == "prior":
        return str(getattr(level, "_prior_node").data)
    out = level.forward(None)
    data = out.data if hasattr(out, "data") else out
    return str(data)


def _candidate_artifact_kind(surface: str) -> Optional[str]:
    """Return the kind used for validated candidate artifacts, if any."""
    return {
        "config": "config_candidate",
        "family_policy": "policy",
        "prior": "prior",
    }.get(surface)


def _candidate_artifact_families(level_spec: dict, families: Dict[str, List[str]]) -> List[str]:
    """Return candidate artifact family labels for one level spec."""
    surface = level_spec["surface"]
    if surface == "config":
        task_ids = _config_task_ids(level_spec, families)
        labels = []
        if level_spec.get("tasks"):
            labels.append(f"task_set:{level_spec.get('id', 'config')}")
        for value in (level_spec.get("family"), level_spec.get("task")):
            if value:
                labels.append(str(value))
        labels.extend(task_ids)
        return list(dict.fromkeys(labels))
    if surface == "family_policy":
        return ["<multi>"]
    if surface == "prior":
        return ["<holdout>"]
    return []


def _select_best_saved_candidate(
    memory: MemoryLite,
    level_spec: dict,
    families: Dict[str, List[str]],
    final_score: float,
):
    """Return the best validated candidate when it beats the final state.

    Some optimizers can validate a good candidate and then leave the live
    trainable node in a worse or invalid state. Reporting the best saved
    candidate preserves the "keep best validated" contract without touching
    Trace core or making the notebook special-case a surface.
    """
    kind = _candidate_artifact_kind(level_spec["surface"])
    if not kind:
        return None
    candidates = []
    for family in _candidate_artifact_families(level_spec, families):
        candidates.extend(memory.artifact_history(family=family, kind=kind))
    if not candidates:
        return None
    best = max(candidates, key=lambda artifact: float(artifact.score))
    return best if float(best.score) > float(final_score) else None


def _seed_from_text(level, surface: str, text: str) -> None:
    if surface == "config":
        getattr(level, "_cfg_node")._data = text
    elif surface == "capability":
        level.impl._data = text
    elif surface == "family_policy":
        level.propose(text)
    elif surface == "prior":
        getattr(level, "_prior_node")._data = text
    elif surface == "code":
        getattr(level, "_impl")._data = text


def _final_eval(level, level_spec: dict, families: Dict[str, List[str]]):
    surface = level_spec["surface"]
    if surface == "config":
        label = level_spec.get("family") or level_spec.get("task")
        if label is None and level_spec.get("tasks"):
            label = f"task_set:{level_spec.get('id', 'config')}"
        if label is None:
            label = families[level_spec["family"]][0]
        out = level.forward(label)
    elif surface == "code":
        fam = level_spec.get("family")
        task = level_spec.get("task") or (families.get(fam, [None])[0] if fam else None)
        out = level.forward(task)
    else:
        out = level.forward(None)
    data = out.data if hasattr(out, "data") else out
    score = float(data.get("score", 0.0)) if isinstance(data, dict) else 0.0
    return score, data


def _check_plumbing(level_spec: dict) -> None:
    """Validate targets through the adapter's CAUSAL-EFFECT contract.

    Not binary plumbed/unplumbed: a field is valid when it has an ACTIVE causal
    path (artifact / optimization / feedback / trace / memory / budget / search /
    score) under the adapter's current run mode. Configurable per level:
      - ``allow_inactive: true`` (alias: legacy ``allow_unplumbed``) -> report,
        don't raise (deliberate diagnostic search);
      - ``effect_policy: {"required_effects": ["memory", ...]}`` -> only those
        effect kinds count as relevant for this experiment.
    Raises ``InactiveFieldError`` naming each dead field WITH its activating
    condition, so the error is the documentation.
    """
    from .effects import check_field_effects
    adapter = TB._TASK_ADAPTER
    if adapter is None:
        return
    policy = level_spec.get("effect_policy") or {}
    check_field_effects(
        adapter, level_spec.get("targets") or [],
        required_effects=policy.get("required_effects"),
        allow_inactive=bool(level_spec.get("allow_inactive")
                            or level_spec.get("allow_unplumbed")
                            or policy.get("allow_inactive")),
    )


def score_spread(task_id: str, probes: Optional[List[dict]] = None,
                 scoring: Optional[dict] = None) -> dict:
    """Pre-flight diagnostic: prove the config->score surface is non-flat.

    Evaluates a few probe configs (defaults exercise the artifact path, the one
    field guaranteed plumbed even at inner_steps=0) and reports the spread. Gate
    experiments on ``valid_spread > 0`` and ``catastrophic is False``: a flat
    result means optimization on this task with these probes cannot show gains,
    while catastrophic invalid probes mean the probe set is incompatible with
    the surface and should not be interpreted as signal.
    """
    probes = probes or [
        {},  # adapter/bundle default artifact
        {"starting_artifact": "Answer directly."},
        {"starting_artifact": "Plan step by step, then verify the answer before replying."},
    ]
    runner = make_scored_task_runner(scoring)
    rows = []
    for p in probes:
        try:
            score, _ = runner(LevelConfig(**p), task_id)
            value = float(score)
            if not math.isfinite(value):
                raise ValueError(f"non-finite score {value!r}")
            rows.append({"probe": p, "score": value})
        except Exception as exc:
            # Some plumbed probes are only valid for prompt-like tasks. Keep the
            # gate diagnostic alive and expose the incompatible arm explicitly.
            rows.append({
                "probe": p,
                "score": None,
                "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            })
    def is_invalid_probe(row: dict) -> bool:
        """Return whether a probe produced no usable score signal."""
        if row.get("score") is None:
            return True
        score = float(row["score"])
        return (not math.isfinite(score)) or score <= -999_999.0

    valid_scores = [
        float(row["score"])
        for row in rows
        if not is_invalid_probe(row)
    ]
    invalid_probes = sum(1 for row in rows if is_invalid_probe(row))
    valid_spread = max(valid_scores) - min(valid_scores) if valid_scores else 0.0
    return {
        "task": task_id,
        "rows": rows,
        "spread": valid_spread,          # backward-compatible alias
        "valid_spread": valid_spread,
        "flat": valid_spread < 1e-9,
        "failed_probes": invalid_probes, # backward-compatible alias
        "invalid_probes": invalid_probes,
        "catastrophic": invalid_probes > 0,
    }


def agentic_optimizer_factory(level_spec: dict, memory: MemoryLite,
                              reused_tools: Optional[List[str]] = None):
    """Build an AgenticOptimizer factory wiring (declared + reused) tools.

    Tool *names* select callables from ``default_optimizer_tools`` (memory-backed
    trace_search etc.), so tools learned/saved for a family are re-armed on reuse.
    Returns an optimizer class usable by Trace's existing ``load_optimizer`` API,
    or None when the level is not agentic.
    """
    agentic = level_spec.get("agentic")
    if not agentic:
        return None
    from .capabilities import (
        AgenticOptimizer,
        default_optimizer_tools,
        select_optimizer_tools,
    )

    cfg = agentic if isinstance(agentic, dict) else {}
    family = level_spec.get("family")
    available = default_optimizer_tools(
        memory=memory, family=family if isinstance(family, str) and family != "*" else None,
    )
    default_names = list(dict.fromkeys((level_spec.get("tools") or []) + list(reused_tools or [])))
    policy = cfg.get("tool_policy", level_spec.get("tool_policy"))
    if policy is None:
        tools = {n: available[n] for n in default_names if n in available} or available
    else:
        tools = select_optimizer_tools(
            available,
            policy,
            default_tools=default_names,
            max_tools=int(cfg.get("tool_budget", 3)),
        ) or ({n: available[n] for n in default_names if n in available} or available)
    configured_kwargs = {"tools": tools, "tool_budget": int(cfg.get("tool_budget", 3))}
    if cfg.get("base_optimizer_cls") is not None:
        configured_kwargs["base_optimizer_cls"] = cfg["base_optimizer_cls"]

    class ConfiguredAgenticOptimizer(AgenticOptimizer):
        """Agentic optimizer class configured from a declarative level spec."""

        keywords = configured_kwargs

        def __init__(self, parameters: list, **optimizer_kwargs: Any) -> None:
            super().__init__(parameters, **{**configured_kwargs, **optimizer_kwargs})

    return ConfiguredAgenticOptimizer


def _memory_from_spec(spec: dict) -> MemoryLite:
    """Create MemoryLite from spec-level prior-promotion controls."""
    promotion = spec.get("prior_promotion") or {}
    return MemoryLite(
        root=spec.get("memory_root", "./trace_memory"),
        promotion_min_support=int(promotion.get("min_support", 3)),
        promote_priors=bool(promotion.get("enabled", True)),
        promotion_min_score=promotion.get("min_score"),
    )


def make_scored_task_runner(
    scoring: Optional[dict] = None,
    *,
    raw_runner: Optional[Callable[[LevelConfig, str], Tuple[float, str]]] = None,
) -> Callable[[LevelConfig, str], Tuple[float, str]]:
    """Wrap a task runner with optional spec-level score normalization.

    Raw Trace-Bench scores remain the default. ``relative_delta`` converts each
    task score into improvement over a baseline config, which keeps cross-family
    policy learning from being dominated by incompatible raw score scales.
    """
    cfg = scoring or {}
    _validate_scoring_config(cfg)
    runner = raw_runner or TB.make_task_runner()
    mode = cfg.get("mode", "raw")
    clip = _clip_bounds(cfg)
    report_raw = bool(cfg.get("report_raw", mode != "raw"))
    baseline_cache: Dict[str, float] = {}
    baseline_cfg = _baseline_config(cfg)

    def run(level_cfg: LevelConfig, task_id: str) -> Tuple[float, str]:
        raw_score, feedback = runner(level_cfg, task_id)
        score = float(raw_score)
        meta: Dict[str, Any] = {"mode": mode, "raw_score": score}
        if mode == "relative_delta":
            key = str(task_id)
            if key not in baseline_cache:
                baseline_cache[key] = float(runner(baseline_cfg, task_id)[0])
            meta["baseline_score"] = baseline_cache[key]
            score = score - baseline_cache[key]
        if clip is not None:
            lo, hi = clip
            score = min(max(score, lo), hi)
            meta["clip"] = [lo, hi]
        meta["score"] = score
        if report_raw:
            feedback = (
                f"{feedback} SCORE_NORMALIZATION_JSON="
                f"{json.dumps(meta, sort_keys=True)}"
            )
        return float(score), feedback

    return run


def _make_inner_runner(
    task_id: str,
    scoring: Optional[dict],
) -> Callable[[LevelConfig, Any], Tuple[float, str]]:
    """Bind a possibly normalized task runner to one Trace-Bench task id."""
    run_task = make_scored_task_runner(scoring)

    def inner_runner(cfg: LevelConfig, _family: Any) -> Tuple[float, str]:
        return run_task(cfg, task_id)

    return inner_runner


def _make_task_set_inner_runner(
    task_ids: List[str],
    scoring: Optional[dict],
) -> Callable[[LevelConfig, Any], Tuple[float, str]]:
    """Bind a normalized task runner to a fixed multi-task evaluation set."""
    if not task_ids:
        raise ValueError("task_ids must be non-empty")
    run_task = make_scored_task_runner(scoring)

    def inner_runner(cfg: LevelConfig, _family: Any) -> Tuple[float, str]:
        scores: List[float] = []
        feedbacks: List[str] = []
        for task_id in task_ids:
            score, feedback = run_task(cfg, task_id)
            scores.append(float(score))
            feedbacks.append(f"{task_id}: {feedback}")
        mean_score = sum(scores) / len(scores)
        return (
            mean_score,
            "[task_set] mean="
            f"{mean_score:.3f} over {len(task_ids)} task(s). "
            + " || ".join(feedbacks),
        )

    return inner_runner


def _baseline_config(scoring: dict) -> LevelConfig:
    """Return the baseline config used by relative score normalization."""
    baseline = scoring.get("baseline", "default_config")
    if baseline in (None, "default_config"):
        return LevelConfig()
    if isinstance(baseline, dict):
        return LevelConfig(**baseline)
    raise ValueError("scoring.baseline must be 'default_config' or a LevelConfig dict")


def _clamp(value: float, clip: Optional[Tuple[float, float]]) -> float:
    """Bound a reported score to the configured clip range (no-op without one)."""
    if clip is None:
        return float(value)
    lo, hi = clip
    return float(min(hi, max(lo, value)))


def _clip_bounds(scoring: Optional[dict]) -> Optional[Tuple[float, float]]:
    """Return optional score clipping bounds from a scoring config."""
    if not scoring:
        return None
    raw = scoring.get("clip")
    if raw is None and scoring.get("mode") == "clip":
        raw = [scoring.get("min", float("-inf")), scoring.get("max", float("inf"))]
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("scoring.clip must be [min, max]")
    lo, hi = float(raw[0]), float(raw[1])
    if lo > hi:
        raise ValueError("scoring.clip minimum cannot exceed maximum")
    return lo, hi


def _validate_tracebench_config(config: dict) -> None:
    """Validate optional top-level Trace-Bench adapter config."""
    if not config:
        return
    if not isinstance(config, dict):
        raise TypeError("spec['tracebench'] must be a dict")
    TB.TraceBenchTaskAdapter.from_config(config)


def _validate_scoring_config(config: dict) -> None:
    """Validate optional score-normalization config."""
    if not config:
        return
    if not isinstance(config, dict):
        raise TypeError("spec['scoring'] must be a dict")
    mode = config.get("mode", "raw")
    if mode not in {"raw", "clip", "relative_delta"}:
        raise ValueError("scoring.mode must be one of raw, clip, relative_delta")
    _clip_bounds(config)
    if mode == "relative_delta":
        _baseline_config(config)


def _validate_prior_promotion_config(config: dict) -> None:
    """Validate optional prior-promotion config."""
    if not config:
        return
    if not isinstance(config, dict):
        raise TypeError("spec['prior_promotion'] must be a dict")
    min_support = int(config.get("min_support", 3))
    if min_support <= 0:
        raise ValueError("prior_promotion.min_support must be positive")
    min_score = config.get("min_score")
    if min_score is not None and not isinstance(min_score, (int, float)):
        raise TypeError("prior_promotion.min_score must be a number")


def _configure_budget(spec: dict, override: "RecursiveOptBudget | dict | None" = None):
    """Install the global budget from the spec dict (or an explicit override).

    Delegates the dict->budget mapping to the canonical ``make_budget`` so there
    is exactly one place that knows the spec budget keys. ``override`` (a dict or
    a RecursiveOptBudget) takes precedence over ``spec["budget"]`` without
    mutating the spec — handy for sweeps that reuse one spec at several budgets.
    """
    source = override if override is not None else spec.get("budget")
    budget = make_budget(source)
    if budget is None:
        return None
    reset_budget(budget)
    return budget


def _objective_config(oc):
    if isinstance(oc, dict):
        from opto.trainer.objectives import ObjectiveConfig
        return ObjectiveConfig(mode=oc.get("mode", "pareto"),
                               minimize=set(oc.get("minimize", [])))
    return oc
