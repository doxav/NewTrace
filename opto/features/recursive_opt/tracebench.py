"""
recursive_opt.tracebench  —  Trace-Bench glue  (learn A/B/C on real problems, D)
================================================================================

Turns a Trace-Bench task id into the two callables the recursive stack needs:

    agent_fn(artifact_node, x) -> output           # for O0 ArtifactLevel
    inner_runner(cfg, family)  -> (score, feedback) # for O1+ MetaLevel

Real Trace-Bench task ids (verified in the repo):
    internal:code_param  internal:numeric_param  internal:multi_param
    internal:multiobjective_bbeh   internal:multiobjective_gsm8k
    llm4ad:online_bin_packing_local   llm4ad:circle_packing
    llm4ad:optimization_admissible_set   llm4ad:machine_learning_moon_lander
    llm4ad:optimization_job_shop_scheduling
    hf:GSM8K  hf:BBEH  hf:HotpotQA
    veribench:<name>   kernelbench:<level/prob>

Loading uses Trace-Bench's own registry through an explicitly registered adapter.
If no adapter is registered, task scoring raises; no synthetic benchmark fallback
is provided.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from opto.trace.nodes import ParameterNode

from .budget import BudgetExceeded
from .levels import INVALID_CONFIG_SCORE, LevelConfig, validate_level_config

try:
    from trace_bench.registry import load_task_module, discover_tasks  # noqa: F401

    HAVE_TB = True
except Exception:
    HAVE_TB = False

try:
    from trace_bench.integrations.opto import evaluate_bundle as _evaluate_trace_bench_bundle
except Exception:
    _evaluate_trace_bench_bundle = None

# Trace-Bench's PUBLIC surface is a CLI/UI/config benchmarking framework, NOT a
# uniform ``load_task_module(task_id) -> {program, evaluate}`` API. So we do NOT
# assume that shape. Instead, real integration goes through an explicitly
# registered adapter; if none is registered task scoring raises. This removes
# the "optimistic real-mode that silently breaks or silently stubs" trap.
_TASK_ADAPTER = None

_TASK_ID_ALIASES = {
    "hf:gsm8k": "internal:multiobjective_gsm8k",
    "hf:bbeh": "hf:bbeh/boolean_expressions",
}


def register_task_adapter(adapter: Optional[object]) -> None:
    """Register the real Trace-Bench bridge.

    ``adapter`` must provide:
        adapter.run_task(cfg, task_id) -> (score: float, feedback: str)
    and optionally:
        adapter.agent_fn(task_id)      -> callable(artifact, x) -> output
    Once registered, ``make_inner_runner`` / ``make_task_runner`` / ``make_agent_fn``
    use it for task scoring. This is the supported path to real benchmarks.
    """
    global _TASK_ADAPTER
    _TASK_ADAPTER = adapter


def current_task_adapter() -> Optional[object]:
    """Return the currently registered Trace-Bench adapter, if any."""
    return _TASK_ADAPTER


def using_real_tasks() -> bool:
    return _TASK_ADAPTER is not None


def real_mode_status() -> str:
    if _TASK_ADAPTER is not None:
        status = getattr(_TASK_ADAPTER, "status", "registered Trace-Bench adapter")
        return f"REAL ({status})"
    if HAVE_TB:
        return ("trace_bench is importable but NO adapter is registered; "
                "task scoring will RAISE until register_task_adapter(...) is called.")
    return "NO adapter registered (task scoring will raise; no stub fallback)."


def normalize_task_id(task_id: str) -> str:
    """Return a Trace-Bench task id accepted by the installed registry."""
    if not str(task_id).strip():
        raise ValueError("task_id must be a non-empty string")
    raw = str(task_id).strip()
    return _TASK_ID_ALIASES.get(raw.lower(), raw)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _json_env(name: str) -> Dict[str, Any]:
    raw = os.environ.get(name)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return value


def _csv_env(name: str) -> Optional[Tuple[str, ...]]:
    """Read an optional comma-separated environment allowlist."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one comma-separated value")
    return values


def _positive_int_config(config: Dict[str, Any], key: str, default: int) -> int:
    """Read a positive integer from a spec/config dict."""
    raw = config.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tracebench.{key} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"tracebench.{key} must be positive, got {value}")
    return value


def _nonnegative_int_config(config: Dict[str, Any], key: str, default: int) -> int:
    """Read a non-negative integer from a spec/config dict."""
    raw = config.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tracebench.{key} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"tracebench.{key} must be non-negative, got {value}")
    return value


def default_tasks_root() -> Path:
    """Locate the default LLM4AD task root from env or Trace-Bench metadata."""
    override = os.environ.get("RECURSIVE_OPT_TRACEBENCH_TASKS_ROOT")
    if override:
        return Path(override)
    try:
        from trace_bench._paths import REPO_ROOT

        return Path(REPO_ROOT) / "benchmarks" / "LLM4AD" / "benchmark_tasks"
    except Exception:
        return Path("benchmarks") / "LLM4AD" / "benchmark_tasks"


def list_tasks(suite: str = None) -> List[str]:
    if not HAVE_TB:
        return [
            "internal:code_param",
            "internal:numeric_param",
            "llm4ad:online_bin_packing_local",
            "llm4ad:circle_packing",
            "hf:GSM8K",
            "hf:BBEH",
        ]
    return [spec.id for spec in discover_tasks(default_tasks_root(), bench=suite)]


def _dataset_infos(dataset: Dict[str, Any]) -> List[Any]:
    return list(dataset.get("infos") or dataset.get("info") or [])


def _evaluation_dataset(bundle: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    for name in ("validate_dataset", "validation_dataset", "train_dataset"):
        dataset = bundle.get(name)
        if not isinstance(dataset, dict):
            continue
        if dataset.get("inputs") and _dataset_infos(dataset):
            return name, dataset
    return "train_dataset", bundle.get("train_dataset", {})


def _extract_response(param: Any, task_input: Any) -> Any:
    if isinstance(param, ParameterNode):
        return param.data
    if callable(param):
        output = param(task_input)
        return output.data if hasattr(output, "data") else output
    return getattr(param, "data", param)


def _format_bundle_feedback(
    dataset_name: str,
    count: int,
    feedbacks: List[str],
    credit_horizon: str = "episode",
) -> str:
    """Format per-example guide feedback according to the credit horizon."""
    horizon = str(credit_horizon or "episode")
    if horizon == "full":
        selected = list(enumerate(feedbacks))
        label = "full feedback"
    elif horizon == "step":
        selected = list(enumerate(feedbacks[:5]))
        label = "step feedback"
    elif horizon == "truncated":
        selected = list(enumerate(feedbacks[:1]))
        label = "truncated feedback"
    else:
        selected = list(enumerate(feedbacks[:3]))
        label = "episode feedback"

    if horizon == "episode":
        details = " | ".join(fb for _, fb in selected)
    else:
        details = " | ".join(f"example[{i}]: {fb}" for i, fb in selected)
    if not details:
        details = "no guide feedback returned"
    return (
        f"{dataset_name}: mean over {count} real example(s). "
        f"credit_horizon={horizon}; {label}: {details}"
    )


def _score_bundle_local(
    bundle: Dict[str, Any],
    max_examples: int,
    credit_horizon: str = "episode",
) -> Tuple[float, str]:
    dataset_name, dataset = _evaluation_dataset(bundle)
    inputs = list(dataset.get("inputs") or [])
    infos = _dataset_infos(dataset)
    limit = min(len(inputs), len(infos), max_examples or len(inputs))
    if limit <= 0:
        raise ValueError(f"{dataset_name} is empty")

    guide = bundle["guide"]
    objective_config = bundle.get("objective_config")
    scores: List[float] = []
    feedbacks: List[str] = []
    for i in range(limit):
        response = _extract_response(bundle["param"], inputs[i])
        reward, feedback = guide(inputs[i], response, infos[i])
        score = float(reward)
        if hasattr(guide, "get_score_dict"):
            try:
                score = _scalarize_score_dict(
                    guide.get_score_dict(inputs[i], response, infos[i]),
                    objective_config,
                )
            except Exception as exc:
                from .runmode import _redact_secrets

                message = _redact_secrets(str(exc).splitlines()[0] or type(exc).__name__)
                raise RuntimeError(
                    "Trace-Bench multi-objective scoring failed for "
                    f"{dataset_name}[{i}]: {type(exc).__name__}: {message}"
                ) from exc
        scores.append(score)
        feedbacks.append(str(feedback))
    mean_score = sum(scores) / len(scores)
    return mean_score, _format_bundle_feedback(
        dataset_name,
        len(scores),
        feedbacks,
        credit_horizon,
    )


def _score_bundle(
    bundle: Dict[str, Any],
    max_examples: int,
    credit_horizon: str = "episode",
) -> Tuple[float, str]:
    """Score a Trace-Bench bundle, preferring Trace-Bench's public evaluator."""
    if _evaluate_trace_bench_bundle is None:
        return _score_bundle_local(bundle, max_examples, credit_horizon)

    dataset_name, _ = _evaluation_dataset(bundle)
    mean_reward, evals = _evaluate_trace_bench_bundle(
        bundle,
        max_examples=max_examples,
        strict_score_dict=True,
    )
    if not evals:
        return mean_reward, f"{dataset_name}: mean over 0 real example(s)."

    objective_config = bundle.get("objective_config")
    guide_has_score_dict = hasattr(bundle.get("guide"), "get_score_dict")
    scores: List[float] = []
    for i, ev in enumerate(evals):
        score = float(ev.reward)
        if ev.score_dict is not None:
            try:
                score = _scalarize_score_dict(ev.score_dict, objective_config)
            except Exception as exc:
                from .runmode import _redact_secrets

                message = _redact_secrets(str(exc).splitlines()[0] or type(exc).__name__)
                raise RuntimeError(
                    "Trace-Bench multi-objective scoring failed for "
                    f"{dataset_name}[{i}]: {type(exc).__name__}: {message}"
                ) from exc
        elif guide_has_score_dict:
            raise RuntimeError(
                "Trace-Bench multi-objective scoring failed for "
                f"{dataset_name}[{i}]: get_score_dict returned None"
            )
        scores.append(score)

    mean_score = sum(scores) / len(scores)
    feedbacks = [str(ev.feedback) for ev in evals]
    return mean_score, _format_bundle_feedback(
        dataset_name,
        len(scores),
        feedbacks,
        credit_horizon,
    )


def _score_dict_to_objectives(score_dict: Dict[str, Any], reward: float) -> Dict[str, float]:
    """Map Trace-Bench task metrics into recursive-opt's accuracy/cost view."""
    metrics = {str(k): float(v) for k, v in score_dict.items()}
    if "accuracy" in metrics:
        accuracy = metrics["accuracy"]
    elif "error" in metrics:
        accuracy = 1.0 - metrics["error"]
    else:
        accuracy = reward if reward >= 0 else 1.0 + reward
    token_cost = metrics.get("tokens_in", 0.0) + metrics.get("tokens_out", 0.0)
    if token_cost:
        cost = min(token_cost / 1000.0, 1.0)
    elif "execution_time_s" in metrics:
        cost = min(metrics["execution_time_s"], 1.0)
    else:
        cost = 0.0
    return {"accuracy": max(0.0, min(accuracy, 1.0)), "cost": max(0.0, min(cost, 1.0))}


def _scalarize_score_dict(score_dict: Dict[str, Any], objective_config: Any) -> float:
    """Convert Trace-Bench multiobjective metrics into a higher-is-better score."""
    metrics = {str(k): float(v) for k, v in score_dict.items()}
    if objective_config is None:
        if "score" in metrics:
            return metrics["score"]
        if "accuracy" in metrics:
            return metrics["accuracy"]
        if "error" in metrics:
            return -metrics["error"]
        return sum(metrics.values()) / len(metrics)

    from opto.trainer.objectives import apply_minimize, weighted_scalarize

    normalized = apply_minimize(
        metrics,
        getattr(objective_config, "minimize", frozenset()),
    )
    weights = getattr(objective_config, "weights", {}) or {}
    missing = getattr(objective_config, "missing_value", float("-inf"))
    if weights:
        return weighted_scalarize(normalized, weights, missing)
    score_key = getattr(objective_config, "score_key", "score")
    if score_key in normalized:
        return normalized[score_key]
    return weighted_scalarize(normalized, weights, missing)


class TraceBenchTaskAdapter:
    """Bridge recursive_opt task scoring to installed Trace-Bench task bundles."""

    def __init__(
        self,
        *,
        tasks_root: Optional[str | Path] = None,
        eval_kwargs: Optional[Dict[str, Any]] = None,
        max_examples: int = 1,
        inner_steps: int = 0,
        inner_candidates: int = 1,
        allowed_inner_trainers: Optional[Tuple[str, ...]] = None,
        mode: str = "real",
    ) -> None:
        self.tasks_root = Path(tasks_root) if tasks_root is not None else default_tasks_root()
        self.eval_kwargs = dict(eval_kwargs or {})
        self.max_examples = max_examples
        self.inner_steps = inner_steps
        self.inner_candidates = inner_candidates
        self.allowed_inner_trainers = allowed_inner_trainers
        self.mode = str(mode or "real")
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.status = (
            f"Trace-Bench bundle adapter; tasks_root={self.tasks_root}; "
            f"max_examples={self.max_examples}; inner_steps={self.inner_steps}; "
            f"inner_candidates={self.inner_candidates}; mode={self.mode}"
        )
        if self.allowed_inner_trainers is not None:
            self.status += (
                f"; allowed_inner_trainers={list(self.allowed_inner_trainers)}"
            )
        if self.max_examples <= 0:
            raise ValueError("max_examples must be positive")
        if self.inner_steps < 0:
            raise ValueError("inner_steps must be non-negative")
        if self.inner_candidates <= 0:
            raise ValueError("inner_candidates must be positive")
        if self.mode not in {"real", "stub"}:
            raise ValueError("mode must be either 'real' or 'stub'")

    @classmethod
    def from_env(cls) -> "TraceBenchTaskAdapter":
        """Build an adapter from recursive-opt environment settings."""
        eval_kwargs = _json_env("RECURSIVE_OPT_TRACEBENCH_EVAL_KWARGS")
        timeout = os.environ.get("RECURSIVE_OPT_TRACEBENCH_TIMEOUT_SECONDS")
        if timeout is not None:
            eval_kwargs.setdefault("timeout_seconds", int(timeout))
        max_examples = _int_env("RECURSIVE_OPT_TRACEBENCH_MAX_EXAMPLES", 10)
        inner_steps = _int_env("RECURSIVE_OPT_TRACEBENCH_INNER_STEPS", 1)
        return cls(
            tasks_root=os.environ.get("RECURSIVE_OPT_TRACEBENCH_TASKS_ROOT"),
            eval_kwargs=eval_kwargs,
            max_examples=max_examples,
            inner_steps=inner_steps,
            inner_candidates=_int_env("RECURSIVE_OPT_TRACEBENCH_INNER_CANDIDATES", 1) or 1,
            allowed_inner_trainers=_csv_env("RECURSIVE_OPT_TRACEBENCH_INNER_TRAINERS"),
            mode=os.environ.get("RECURSIVE_OPT_TRACEBENCH_MODE", "real"),
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TraceBenchTaskAdapter":
        """Build an adapter from a declarative RecursiveSpec ``tracebench`` dict."""
        if not isinstance(config, dict):
            raise TypeError("tracebench config must be a dict")
        eval_kwargs = dict(config.get("eval_kwargs") or {})
        timeout = config.get("timeout_seconds")
        if timeout is not None:
            eval_kwargs.setdefault("timeout_seconds", int(timeout))
        allowed = config.get("allowed_inner_trainers")
        if allowed is not None:
            if not isinstance(allowed, (list, tuple)):
                raise TypeError("tracebench.allowed_inner_trainers must be a list")
            allowed = tuple(str(value) for value in allowed if str(value).strip())
            if not allowed:
                raise ValueError("tracebench.allowed_inner_trainers cannot be empty")
        return cls(
            tasks_root=config.get("tasks_root"),
            eval_kwargs=eval_kwargs,
            max_examples=_positive_int_config(config, "max_examples", 10),
            inner_steps=_nonnegative_int_config(config, "inner_steps", 1),
            inner_candidates=_positive_int_config(config, "inner_candidates", 1),
            allowed_inner_trainers=allowed,
            mode=str(config.get("mode", "real")),
        )

    def _trainer_budget_feedback(self, cfg: LevelConfig, task_id: str) -> Optional[Tuple[float, str]]:
        """Return a budget failure when a nested trainer is not allowed."""
        if self.inner_steps <= 0 or self.allowed_inner_trainers is None:
            return None
        if cfg.trainer in self.allowed_inner_trainers:
            return None
        return (
            INVALID_CONFIG_SCORE,
            "[real_trace_bench:"
            f"{task_id}] trainer={cfg.trainer!r} is outside the live budget "
            f"allowlist {list(self.allowed_inner_trainers)}. Choose an allowed "
            "trainer or unset RECURSIVE_OPT_TRACEBENCH_INNER_TRAINERS for a "
            "full, potentially expensive nested benchmark run.",
        )

    def _trainer_hint(self, cfg: LevelConfig) -> str:
        """Return compact feedback about the nested trainer budget."""
        hints: List[str] = []
        if self.allowed_inner_trainers is not None:
            hints.append(f"allowed_inner_trainers={list(self.allowed_inner_trainers)}.")
        if self.inner_steps > 0 and cfg.trainer == "MinibatchAlgorithm":
            hints.append(
                "MinibatchAlgorithm is a cheap sampling/evaluation baseline; "
                "if the score is flat, try PrioritySearch for actual inner "
                "candidate updates under the bounded live budget."
            )
        return " ".join(hints)

    def _eval_kwargs_for_task(self, normalized_task_id: str) -> Dict[str, Any]:
        """Return task-specific eval kwargs without leaking unsupported kwargs."""
        eval_kwargs = dict(self.eval_kwargs)
        model_name = os.environ.get("RECURSIVE_OPT_MODEL") or os.environ.get("TRACE_LITELLM_MODEL")
        if model_name and normalized_task_id == "internal:multiobjective_gsm8k":
            eval_kwargs.setdefault("model", model_name)
        return eval_kwargs

    def _expanded_task_ids(self, normalized_task_id: str) -> Tuple[str, ...]:
        """Expand runner-style family task ids for direct adapter scoring."""
        try:
            from trace_bench.config import TaskConfig
            from trace_bench.registry import expand_special_tasks

            expanded = expand_special_tasks(
                [TaskConfig(id=normalized_task_id, eval_kwargs=self.eval_kwargs)],
                self.tasks_root,
            )
        except Exception:
            return (normalized_task_id,)
        ids = tuple(task.id for task in expanded)
        return ids or (normalized_task_id,)

    def _apply_runtime_mode(self, bundle: Dict[str, Any]) -> None:
        """Apply adapter runtime mode to a freshly loaded bundle."""
        if self.mode != "stub":
            return
        from trace_bench.runner import _stub_bundle

        _stub_bundle(bundle, "stub")

    def _load_bundle(self, task_id: str, *, fresh: bool = False) -> Dict[str, Any]:
        from trace_bench.registry import load_task_bundle

        normalized = normalize_task_id(task_id)
        eval_kwargs = self._eval_kwargs_for_task(normalized)
        cache_key = json.dumps(
            {"task_id": normalized, "eval_kwargs": eval_kwargs},
            sort_keys=True,
            default=str,
        )
        if not fresh and cache_key in self._cache:
            return self._cache[cache_key]
        bundle = load_task_bundle(normalized, self.tasks_root, eval_kwargs=eval_kwargs)
        self._apply_runtime_mode(bundle)
        if not fresh:
            self._cache[cache_key] = bundle
        return bundle

    #: Config fields this adapter actually plumbs into the scored run. Anything
    #: else is a NO-OP for run_task; searching it yields a flat score surface.
    #: ``trace_type`` changes the trace feedback collected around the real
    #: benchmark run; it does not add a synthetic score bonus or penalty.
    def field_effects(self) -> Dict[str, Any]:
        """Causal-effect contract for this adapter under its CURRENT run mode.

        Conditional fields report active=False (with the activating condition)
        instead of being hidden; "inactive" is information, not absence.
        """
        from .effects import Effect, FieldEffect
        training = self.inner_steps > 0
        return {
            "starting_artifact": FieldEffect("starting_artifact",
                (Effect.ARTIFACT, Effect.SCORE),
                condition="bundle param must expose system_prompt/parameters()/_data",
                probe_values=("", "Answer directly.",
                              "Plan step by step, then verify the answer before replying.")),
            "initial_knowledge": FieldEffect("initial_knowledge",
                (Effect.ARTIFACT, Effect.SCORE),
                condition="concatenated into the starting artifact"),
            "trainer": FieldEffect("trainer",
                (Effect.OPTIMIZATION, Effect.ARTIFACT, Effect.SCORE),
                active=training, condition="active only when inner_steps > 0"),
            "optimizer": FieldEffect("optimizer",
                (Effect.OPTIMIZATION, Effect.ARTIFACT, Effect.SCORE),
                active=training, condition="active only when inner_steps > 0"),
            "batch_size": FieldEffect("batch_size", (Effect.OPTIMIZATION,),
                active=training, probe_values=(1, 4, 8),
                condition="active only when inner_steps > 0 (trainer samples batches)"),
            "num_threads": FieldEffect("num_threads", (Effect.BUDGET, Effect.SEARCH),
                active=training, condition="active only when inner training runs in parallel"),
            "trace_type": FieldEffect("trace_type", (Effect.TRACE, Effect.FEEDBACK),
                probe_values=("internal", "otel", "hybrid"),
                condition="feedback-plumbed (changes optimizer-visible evidence), NOT score-plumbed",
                notes="a score effect can only appear later, via better update proposals"),
            "batch_design": FieldEffect("batch_design", (Effect.OPTIMIZATION,),
                active=False,
                condition="inactive until the inner trainer consumes a batch_design-controlled sampler"),
            "memory_policy": FieldEffect("memory_policy",
                (Effect.MEMORY, Effect.FEEDBACK, Effect.OPTIMIZATION), active=False,
                condition="inactive until retrieval/promotion feeds optimizer input or warm-start"),
            "credit_horizon": FieldEffect("credit_horizon", (Effect.FEEDBACK, Effect.TRACE),
                condition="controls how per-example guide feedback is summarized for the meta optimizer",
                probe_values=("step", "episode", "truncated", "full")),
        }

    PLUMBED_FIELDS = ("starting_artifact", "initial_knowledge", "trainer",
                      "optimizer", "batch_size", "num_threads", "trace_type",
                      "credit_horizon")

    def _trace_types_for_config(self, cfg: LevelConfig) -> Tuple[str, ...]:
        """Map the public LevelConfig trace label to concrete trace backends."""
        trace_type = str(getattr(cfg, "trace_type", "internal") or "internal")
        if trace_type == "internal":
            return ("internal",)
        if trace_type == "otel":
            return ("otel",)
        if trace_type == "sysmon":
            return ("sysmon",)
        if trace_type == "hybrid":
            return ("otel", "sysmon")
        raise ValueError(
            f"Unsupported trace_type {trace_type!r}; expected internal, otel, "
            "sysmon, or hybrid."
        )

    def _trace_backend_failure(self, cfg: LevelConfig) -> Optional[Tuple[float, str]]:
        """Return a clear failure when a requested trace backend is unavailable."""
        trace_types = self._trace_types_for_config(cfg)
        needs_external_backend = any(t in {"otel", "sysmon"} for t in trace_types)
        if not needs_external_backend:
            return None
        from . import traces

        if traces.HAVE_TRACE_IO:
            return None
        return (
            INVALID_CONFIG_SCORE,
            "[real_trace_bench] trace_type="
            f"{cfg.trace_type!r} requires graph/telemetry backends, but "
            "opto.features.graph/opto.trace.io are not importable.",
        )

    def _trace_feedback_note(self, cfg: LevelConfig, session: Any) -> str:
        """Summarize the collected trace feedback without changing task score."""
        try:
            tgj = session.to_tgj()
        except Exception as exc:
            return f"trace_type={cfg.trace_type}; trace feedback unavailable: {type(exc).__name__}: {exc}"
        sources = ",".join(str(src) for src in tgj.get("sources", []))
        documents = len(tgj.get("documents", []))
        nodes = len(tgj.get("nodes", []))
        edges = len(tgj.get("edges", []))
        return (
            f"trace_type={cfg.trace_type}; trace_sources={sources}; "
            f"trace_documents={documents}; trace_nodes={nodes}; trace_edges={edges}; "
            "task score remains the real benchmark score"
        )

    def _apply_starting_artifact(self, bundle: Dict[str, Any], cfg: LevelConfig) -> bool:
        """Seed the bundle's trainable param from cfg before scoring.

        This is the config->score connection that exists even at inner_steps=0:
        the artifact text itself is evaluated by the real benchmark.
        """
        text = str(getattr(cfg, "starting_artifact", "") or "").strip()
        knowledge = str(getattr(cfg, "initial_knowledge", "") or "").strip()
        if knowledge:
            text = f"{text}\n{knowledge}".strip()
        if not text:
            return False
        param = bundle.get("param")
        if hasattr(param, "system_prompt"):
            param.system_prompt._data = text
            return True
        params = getattr(param, "parameters", None)
        if callable(params):
            plist = params()
            if plist:
                plist[0]._data = text
                return True
        if hasattr(param, "_data"):
            param._data = text
            return True
        return False

    def _train_bundle(self, bundle: Dict[str, Any], cfg: LevelConfig) -> None:
        if self.inner_steps <= 0:
            return
        from opto import trainer as opto_trainer

        train_dataset = bundle["train_dataset"]
        inputs = list(train_dataset.get("inputs") or [])[: self.max_examples]
        infos = _dataset_infos(train_dataset)[: self.max_examples]
        optimizer_kwargs = dict(bundle.get("optimizer_kwargs") or {})
        model_name = os.environ.get("RECURSIVE_OPT_MODEL") or os.environ.get("TRACE_LITELLM_MODEL")
        if model_name and "llm" not in optimizer_kwargs:
            from .runmode import make_live_llm

            optimizer_kwargs["llm"] = make_live_llm(model_name)
        opto_trainer.train(
            model=bundle["param"],
            train_dataset={"inputs": inputs, "infos": infos},
            algorithm=cfg.trainer,
            optimizer=cfg.optimizer,
            guide=bundle["guide"],
            optimizer_kwargs=optimizer_kwargs,
            num_epochs=0,
            num_steps=self.inner_steps,
            batch_size=max(1, min(cfg.batch_size, len(inputs) or 1)),
            num_candidates=self.inner_candidates,
            num_threads=max(1, cfg.num_threads),
        )

    def _run_single_task(self, cfg: LevelConfig, normalized: str) -> Tuple[float, str]:
        """Evaluate one concrete Trace-Bench task id."""
        budget_failure = self._trainer_budget_feedback(cfg, normalized)
        if budget_failure is not None:
            return budget_failure
        trace_failure = self._trace_backend_failure(cfg)
        if trace_failure is not None:
            return trace_failure
        bundle = self._load_bundle(normalized, fresh=True)
        seeded = self._apply_starting_artifact(bundle, cfg)
        self._train_bundle(bundle, cfg)
        from . import traces

        trace_meta = {
            "semantic_names": [
                "_score_bundle",
                "_extract_response",
                "_scalarize_score_dict",
                "_format_bundle_feedback",
            ]
        }
        with traces.collect_traces(
            list(self._trace_types_for_config(cfg)),
            meta=trace_meta,
        ) as trace_session:
            import inspect

            if "credit_horizon" in inspect.signature(_score_bundle).parameters:
                score, feedback = _score_bundle(
                    bundle,
                    self.max_examples,
                    credit_horizon=cfg.credit_horizon,
                )
            else:
                score, feedback = _score_bundle(bundle, self.max_examples)
        trace_note = self._trace_feedback_note(cfg, trace_session)
        seed_note = "starting_artifact seeded; " if seeded else ""
        train_note = (
            f"{seed_note}inner_steps={self.inner_steps}; cfg applied through Trace trainer"
            if self.inner_steps
            else f"{seed_note}inner_steps=0; only plumbed fields "
                 f"{list(self.PLUMBED_FIELDS)} can affect this score"
        )
        hint = self._trainer_hint(cfg)
        suffix = f" {hint}" if hint else ""
        return score, f"[real_trace_bench:{normalized}] {train_note}. {trace_note}. {feedback}{suffix}"

    def run_task(self, cfg: LevelConfig, task_id: str) -> Tuple[float, str]:
        """Evaluate a recursive-opt config on a real Trace-Bench task bundle."""
        validate_level_config(
            cfg,
            (
                "batch_size",
                "batch_design",
                "trace_type",
                "memory_policy",
                "optimizer",
                "guide",
                "trainer",
                "credit_horizon",
            ),
        )
        normalized = normalize_task_id(task_id)
        task_ids = self._expanded_task_ids(normalized)
        if len(task_ids) == 1:
            return self._run_single_task(cfg, task_ids[0])

        scores: List[float] = []
        feedbacks: List[str] = []
        for subtask_id in task_ids:
            score, feedback = self._run_single_task(cfg, subtask_id)
            scores.append(score)
            feedbacks.append(f"{subtask_id}: {feedback}")
        mean_score = sum(scores) / len(scores)
        family_feedback = _format_bundle_feedback(
            normalized,
            len(scores),
            feedbacks,
            cfg.credit_horizon,
        )
        return (
            mean_score,
            f"[real_trace_bench:{normalized}] expanded into "
            f"{len(task_ids)} concrete task(s). {family_feedback}",
        )

    def agent_fn(self, task_id: str) -> Callable:
        """Return an O0 agent function backed by the Trace-Bench bundle param.

        The artifact is now genuinely INJECTED (DRY: same surface-aware path as
        ``starting_artifact``). Previously it was ignored (`_artifact`), so O0
        artifact optimization could look connected while training nothing.
        """
        normalized = normalize_task_id(task_id)

        def agent_fn(artifact: Any, x: Any) -> Any:
            bundle = self._load_bundle(normalized, fresh=True)
            text = artifact.data if hasattr(artifact, "data") else str(artifact or "")
            if text.strip():
                applied = self._apply_starting_artifact(
                    bundle, LevelConfig(starting_artifact=text))
                if not applied:
                    raise RuntimeError(
                        f"O0 artifact is inactive for task {normalized!r}: it does not fit "
                        "the task surface (prose on a code param?) or the bundle param "
                        "exposes none of system_prompt/parameters()/_data."
                    )
            return _extract_response(bundle["param"], x)

        return agent_fn


def resolve_trainable_fields(
    requested: Iterable[str],
    adapter: Optional[Any] = None,
    *,
    allow_inactive: bool = False,
    required_effects: Optional[Iterable[Any]] = None,
) -> Tuple[str, ...]:
    """Validate requested trainable fields against the registered adapter's
    causal-effect contract and return them (DRY entry for examples/notebooks)."""
    from .effects import check_field_effects
    adapter = adapter if adapter is not None else _TASK_ADAPTER
    report = check_field_effects(adapter, requested,
                                 required_effects=required_effects,
                                 allow_inactive=allow_inactive)
    if allow_inactive and not report.ok():
        print(f"[effects] proceeding with inactive fields: {report.inactive} "
              f"undeclared: {report.undeclared}")
    return tuple(requested)


def ensure_default_task_adapter(*, require: bool = False) -> bool:
    """Register the default Trace-Bench adapter when Trace-Bench is installed."""
    if _TASK_ADAPTER is not None:
        return True
    if not HAVE_TB:
        if require:
            raise RuntimeError("Trace-Bench is not importable; cannot run real task scoring.")
        return False
    try:
        register_task_adapter(TraceBenchTaskAdapter.from_env())
        return True
    except Exception as exc:
        if require:
            raise RuntimeError(f"Could not initialize Trace-Bench adapter: {exc}") from exc
        return False


def configure_tracebench_adapter(config: Dict[str, Any], *, require: bool = True) -> bool:
    """Register a Trace-Bench adapter from a declarative spec config.

    This is the spec-level replacement for notebook/example environment hacks:
    the benchmark bounds live beside the recursive optimization budget while
    still using the same adapter contract as ``ensure_default_task_adapter``.
    """
    if not config:
        return ensure_default_task_adapter(require=require)
    if config.get("enabled", True) is False:
        register_task_adapter(None)
        return False
    try:
        adapter = TraceBenchTaskAdapter.from_config(config)
    except Exception as exc:
        if require:
            raise RuntimeError(
                "Trace-Bench adapter could not be constructed from spec['tracebench'] "
                f"({type(exc).__name__}: {exc}). Install trace_bench or register an "
                "adapter explicitly via register_task_adapter(...)."
            ) from exc
        return False
    register_task_adapter(adapter)
    return True


def ensure_eval_only_task_adapter(
    *,
    require: bool = False,
    max_examples: int = 1,
    timeout_seconds: int = 1,
    eval_kwargs: Optional[Dict[str, Any]] = None,
) -> bool:
    """Register a bounded real Trace-Bench adapter for non-live demos.

    This is not a synthetic fallback: it loads real Trace-Bench bundles, scores a
    small number of examples, and runs no nested trainer (`inner_steps=0`). Use
    it when examples should be runnable without an optimizer LLM but still need
    benchmark-backed task scoring.
    """
    if _TASK_ADAPTER is not None:
        return True
    if not HAVE_TB:
        if require:
            raise RuntimeError("Trace-Bench is not importable; cannot run real eval-only scoring.")
        return False
    try:
        kwargs = dict(eval_kwargs or {})
        kwargs.setdefault("n_train", 1)
        kwargs.setdefault("n_val", 0)
        kwargs.setdefault("timeout_seconds", timeout_seconds)
        register_task_adapter(
            TraceBenchTaskAdapter(
                eval_kwargs=kwargs,
                max_examples=max_examples,
                inner_steps=0,
            )
        )
        return True
    except Exception as exc:
        if require:
            raise RuntimeError(f"Could not initialize eval-only Trace-Bench adapter: {exc}") from exc
        return False


def _text_cost(text: str, cap: int = 600) -> float:
    """Length-based token/compute proxy in [0,1]; longer policy => higher cost.

    Shared by every objective so 'minimize cost' consistently penalises verbosity
    (the capability text is the trainable artifact, so its length is a faithful,
    always-available proxy when token usage isn't reported by the backend).
    """
    return min(1.0, len(str(text)) / float(cap))


def _require_adapter(what: str):
    """Return the registered Trace-Bench adapter or raise (no stub fallback)."""
    if _TASK_ADAPTER is None:
        raise RuntimeError(
            f"{what} requires a registered Trace-Bench adapter; none is registered. "
            "Call register_task_adapter(<adapter>) first. No synthetic stub scoring "
            "is provided; results must come from a real benchmark."
        )
    return _TASK_ADAPTER


def make_agent_fn(task_id: str) -> Callable:
    """O0 agent: consume the trainable artifact, produce an answer for input x."""
    adapter = _require_adapter("make_agent_fn")
    if hasattr(adapter, "agent_fn"):
        return adapter.agent_fn(task_id)
    raise RuntimeError("The registered adapter does not implement agent_fn(task_id).")


def make_task_runner() -> Callable:
    """Return ``run(cfg, task_id) -> (score, feedback)`` used by O1/O2/O3.

    Requires a registered Trace-Bench adapter (``adapter.run_task``); there is no
    synthetic fallback. One contract for every recursion level (DRY).
    """
    def run(cfg, task_id: str) -> Tuple[float, str]:
        return _require_adapter("make_task_runner").run_task(cfg, task_id)

    return run


def make_inner_runner(task_id: str, n_tasks: int = 6) -> Callable:
    """O1 inner runner bound to one task: ``inner_runner(cfg, family) -> (score, fb)``.

    Delegates to the shared registered adapter, so real-mode is opt-in via
    ``register_task_adapter`` and never silently assumes a wrong public API.
    """
    run = make_task_runner()

    def inner_runner(cfg, family):
        return run(cfg, task_id)

    return inner_runner


def make_dataset(families: List[str], repeats: int = 4) -> dict:
    """Trainer dataset: inputs=families to optimize over, infos unused for meta."""
    inputs = [f for f in families for _ in range(repeats)]
    return {"inputs": inputs, "infos": [None] * len(inputs)}


# =========================================================================== #
# Evaluators for the CODE-improvement and CAPABILITY-synthesis examples.
# These component-level evaluators are deterministic local validators. They do
# not replace task scoring for A/C/D/E; they make code rewrites in example B
# climbable without pretending to be a full benchmark adapter.
# =========================================================================== #
def validate_batch_design_indices(raw_indices: Any, *, n: int, k: int) -> Tuple[float, str, List[int]]:
    """Score a batch-design output against a transparent hard-item validation pool."""
    try:
        indices = list(raw_indices)
    except TypeError:
        return 0.0, "returned a non-iterable value; expected k integer indices", []

    valid = [i for i in indices if isinstance(i, int) and 0 <= i < n]
    selected = valid[:k]
    hard_targets = [i for i in range(n) if i % 3 == 0][:k]
    if len(selected) != k:
        return (
            0.1 * (len(selected) / max(k, 1)),
            f"returned {len(selected)}/{k} valid indices; expected exactly k ints in [0,{n})",
            selected,
        )
    unique = len(set(selected))
    hard = sum(1 for i in selected if i in hard_targets)
    hard_ratio = hard / k
    diversity = unique / k
    score = 0.40 + 0.40 * hard_ratio + 0.20 * diversity
    if unique < k:
        score -= 0.20 * ((k - unique) / k)
    missing = [i for i in hard_targets if i not in selected]
    feedback = (
        f"validation_pool n={n}, k={k}; hard/failing indices are {hard_targets} "
        "(defined by idx % 3 == 0); picked "
        f"{selected}; hard_items={hard}/{k}; diversity={diversity:.2f}."
    )
    if missing:
        feedback += f" Missing hard indices {missing}; pick hard/failing items before easy items."
    else:
        feedback += " Good: all validation hard/failing items are selected."
    return max(0.0, min(score, 1.0)), feedback, selected


def make_code_evaluator(task_id: str, component: str):
    """Return evaluate(component_callable, family) -> (score, feedback) for
    improving the CODE of a library component (example_B).

    component in {"batch_design", "trace_summarizer"}.
    Real mode: run the candidate inside Trace-Bench's training loop on `task_id`
    and read held-out score. Stub mode: reward implementations that exhibit the
    properties the PDF identified as helpful, so code edits are climbable.
    """

    def evaluate(component_callable, family):
        if component == "batch_design":
            # Probe the candidate sampler on a synthetic pool with known "hard"
            # (failing) items at indices divisible by 3.
            try:
                idx = list(component_callable(n=12, k=4))
            except Exception as e:
                return 0.0, f"[{component}] candidate raised {type(e).__name__}: {e}"
            score, detail, _selected = validate_batch_design_indices(idx, n=12, k=4)
            return score, f"[{component}@{task_id}] {detail}"

        if component == "trace_summarizer":
            sample = (
                "ERROR: AssertionError line 42 expected 7 got 5\n"
                "INFO: started\nDEBUG: x=1\nWARN: slow\nINFO: done\n" * 3
            )
            try:
                summary = str(component_callable(trace_text=sample))
            except Exception as e:
                return 0.0, f"[{component}] candidate raised {type(e).__name__}: {e}"
            preserved = ("AssertionError" in summary) + ("expected 7 got 5" in summary)
            concise = max(0.0, 1.0 - len(summary) / max(len(sample), 1))
            score = 0.3 + 0.45 * (preserved / 2) + 0.25 * concise
            fb = (
                f"[{component}@{task_id}] len={len(summary)} (src {len(sample)}); "
                f"error_evidence_preserved={preserved}/2; conciseness={concise:.2f}. "
                f"{'good: keeps the failing assertion' if preserved==2 else 'tip: KEEP the AssertionError + expected/got, DROP INFO/DEBUG noise'}."
            )
            return min(score, 1.0), fb

        return 0.0, f"[{component}] unknown component"

    return evaluate


def make_tracebench_artifact_evaluator(
    task_id: str,
    *,
    max_examples: Optional[int] = None,
    credit_horizon: str = "episode",
) -> Callable[[str, Any], Tuple[float, str]]:
    """Score a raw artifact text through the registered Trace-Bench bundle.

    This is the generic bridge for code/PAL prompt-like artifacts that are not
    naturally represented as a Python callable. It reuses the same
    ``TraceBenchTaskAdapter._apply_starting_artifact`` and ``_score_bundle`` path
    as config/O0 scoring, so notebook experiments do not duplicate benchmark
    scoring logic or accidentally evaluate a different surface.
    """

    def evaluate_artifact(artifact_text: str, family: Any = None) -> Tuple[float, str]:
        adapter = _require_adapter("make_tracebench_artifact_evaluator")
        if not hasattr(adapter, "_load_bundle") or not hasattr(adapter, "_apply_starting_artifact"):
            raise RuntimeError(
                "make_tracebench_artifact_evaluator requires a TraceBenchTaskAdapter-like "
                "adapter with _load_bundle(...) and _apply_starting_artifact(...)."
            )
        target = normalize_task_id(str(family or task_id))
        bundle = adapter._load_bundle(target, fresh=True)
        text = str(artifact_text or "")
        applied = adapter._apply_starting_artifact(bundle, LevelConfig(starting_artifact=text))
        if text.strip() and not applied:
            raise RuntimeError(
                f"Artifact text is inactive for task {target!r}: the bundle param exposes "
                "none of system_prompt/parameters()/_data."
            )
        limit = int(max_examples or getattr(adapter, "max_examples", 1))
        score, feedback = _score_bundle(bundle, limit, credit_horizon=credit_horizon)
        mode = "seeded" if text.strip() else "bundle-default"
        return score, f"[artifact:{target}] mode={mode}; chars={len(text)}. {feedback}"

    return evaluate_artifact


def make_artifact_emitter_evaluator(
    task_id: str,
    *,
    max_examples: Optional[int] = None,
    credit_horizon: str = "episode",
) -> Callable[[Callable, Any], Tuple[float, str]]:
    """Adapt a code component that emits artifact text into a CodeArtifact evaluator.

    ``CodeArtifactLevel`` requires the evaluator to invoke the trainable callable
    so gradients/feedback reach the code parameter. The emitted string is then
    scored by :func:`make_tracebench_artifact_evaluator`, which keeps the actual
    Trace-Bench scoring path DRY.
    """
    evaluate_artifact = make_tracebench_artifact_evaluator(
        task_id,
        max_examples=max_examples,
        credit_horizon=credit_horizon,
    )

    def evaluate(component_callable: Callable, family: Any) -> Tuple[float, str]:
        try:
            artifact_text = component_callable()
        except TypeError:
            artifact_text = component_callable(task_id=family or task_id)
        score, feedback = evaluate_artifact(str(artifact_text), family or task_id)
        return score, f"[artifact_emitter:{task_id}] {feedback}"

    return evaluate


def make_tracebench_direct_answer_evaluator(
    task_id: str,
    *,
    max_examples: Optional[int] = None,
    target_getter: Optional[Callable[[Any], Any]] = None,
    normalizer: Optional[Callable[[Any], str]] = None,
) -> Callable[[Callable, Any], Tuple[float, str]]:
    """Score a code component that answers Trace-Bench examples directly.

    Use this when the trainable surface is real agent code
    ``candidate(question) -> answer`` and the bundle dataset's ``infos`` carry the
    reference answers. The official bundle still supplies the examples; this helper
    only replaces a task-specific guide that expects a different artifact shape
    (for example PAL code instead of direct answers).
    """
    normalize = normalizer or (lambda value: str(value).strip().lower())

    def evaluate(component_callable: Callable, family: Any) -> Tuple[float, str]:
        target = normalize_task_id(str(family or task_id))
        examples = load_tracebench_direct_answer_examples(
            target,
            max_examples=max_examples,
            target_getter=target_getter,
        )
        scores: List[float] = []
        feedbacks: List[str] = []
        for i, (question, expected) in enumerate(examples):
            try:
                try:
                    answer = component_callable(question=question)
                except TypeError:
                    answer = component_callable(question)
            except Exception as exc:
                scores.append(0.0)
                feedbacks.append(f"example[{i}]: ERR {type(exc).__name__}: {exc}")
                continue
            ok = normalize(answer) == normalize(expected)
            scores.append(1.0 if ok else 0.0)
            if ok:
                feedbacks.append(f"example[{i}]: CORRECT {answer!r}")
            else:
                feedbacks.append(
                    f"example[{i}]: WRONG got {answer!r}; expected {expected!r}"
                )
        score = sum(scores) / len(scores)
        return (
            score,
            f"[direct_answer:{target}] accuracy={score:.3f} "
            f"over {len(scores)} real example(s). " + " | ".join(feedbacks[:5]),
        )

    return evaluate


def load_tracebench_direct_answer_examples(
    task_id: str,
    *,
    max_examples: Optional[int] = None,
    target_getter: Optional[Callable[[Any], Any]] = None,
) -> List[Tuple[Any, Any]]:
    """Load ``(question, expected_answer)`` pairs from a registered Trace-Bench bundle.

    This helper is intentionally narrow: it supports direct-answer code/graph
    experiments that use Trace-Bench datasets but do not use a task's default
    artifact runner. Keeping dataset extraction here avoids each experiment
    reaching into adapter internals differently.
    """
    adapter = _require_adapter("load_tracebench_direct_answer_examples")
    if not hasattr(adapter, "_load_bundle"):
        raise RuntimeError(
            "load_tracebench_direct_answer_examples requires a TraceBenchTaskAdapter-like "
            "adapter with _load_bundle(...)."
        )
    get_target = target_getter or (
        lambda info: info.get("answer", info.get("target")) if isinstance(info, dict) else info
    )
    target = normalize_task_id(task_id)
    bundle = adapter._load_bundle(target, fresh=True)
    dataset_name, dataset = _evaluation_dataset(bundle)
    inputs = list(dataset.get("inputs") or [])
    infos = _dataset_infos(dataset)
    limit = min(
        len(inputs),
        len(infos),
        int(max_examples or getattr(adapter, "max_examples", 1)),
    )
    if limit <= 0:
        raise ValueError(f"{dataset_name} is empty")
    return [(question, get_target(info)) for question, info in zip(inputs[:limit], infos[:limit])]


def make_multiobjective_evaluator(task_ids, objectives, n_tasks: int = 4,
                                  required_terms: Optional[Tuple[str, ...]] = None):
    """Return evaluate(capability_callable, family) -> (score_dict, feedback)
    for learning a NEW CAPABILITY under multiple objectives (example_C).

    objectives: dict like {"accuracy": "max", "cost": "min"}.
    The capability_callable is the (trainable) capability implementation; it is
    run on `task_ids` and produces an answer + a token/cost estimate. We return a
    per-objective score dict (consumed by opto.trainer.objectives / OptoPrimeMulti)
    plus a scalarized score and a directional feedback string.
    """

    task_limit = int(n_tasks)
    if task_limit <= 0:
        raise ValueError("n_tasks must be positive")

    def _evaluate_real(capability_callable: Callable, family: Any):
        adapter = _TASK_ADAPTER
        if adapter is None or not hasattr(adapter, "_load_bundle"):
            raise RuntimeError("real Trace-Bench adapter is not registered")
        raw_capability = capability_callable(task=family)
        capability_text = (
            str(raw_capability.get("answer", raw_capability))
            if isinstance(raw_capability, dict)
            else str(raw_capability)
        )
        max_examples = min(
            getattr(adapter, "max_examples", 1),
            _int_env("RECURSIVE_OPT_CAPABILITY_MAX_EXAMPLES", 3),
            task_limit,
        )
        objective_rows: List[Dict[str, float]] = []
        notes: List[str] = []
        for tid in task_ids:
            normalized = normalize_task_id(tid)
            bundle = adapter._load_bundle(normalized)
            dataset_name, dataset = _evaluation_dataset(bundle)
            inputs = list(dataset.get("inputs") or [])
            infos = _dataset_infos(dataset)
            limit = min(len(inputs), len(infos), max_examples)
            if limit <= 0:
                notes.append(f"{normalized}: no examples")
                objective_rows.append({"accuracy": 0.0, "cost": 1.0})
                continue
            guide = bundle["guide"]
            param = bundle["param"]
            old_prompt = None
            mode = "raw_artifact"
            if hasattr(param, "system_prompt"):
                mode = "system_prompt"
                old_prompt = param.system_prompt.data
                param.system_prompt._data = capability_text
            rows: List[Dict[str, float]] = []
            feedbacks: List[str] = []
            try:
                for i in range(limit):
                    if hasattr(param, "forward") and hasattr(param, "system_prompt"):
                        from .budget import current_budget

                        current_budget().charge("eval_llm_calls")
                        response = param.forward(inputs[i])
                    else:
                        response = capability_text
                    reward, feedback = guide(inputs[i], response, infos[i])
                    if hasattr(guide, "get_score_dict"):
                        score_dict = guide.get_score_dict(inputs[i], response, infos[i])
                    else:
                        score_dict = {}
                    rows.append(_score_dict_to_objectives(score_dict, float(reward)))
                    feedbacks.append(str(feedback))
            finally:
                if old_prompt is not None:
                    param.system_prompt._data = old_prompt
            accuracy = sum(row["accuracy"] for row in rows) / len(rows)
            # Cost is a token/length proxy on the trainable capability text so that
            # "minimize cost" actually penalises verbosity (benchmark score_dicts
            # do not carry a comparable compute cost). Consistent across objectives.
            cost = _text_cost(capability_text)
            row = {"accuracy": accuracy, "cost": cost}
            if required_terms:
                lowered = capability_text.lower()
                row["compliance"] = sum(
                    1.0 for t in required_terms if t.lower() in lowered
                ) / len(required_terms)
            objective_rows.append(row)
            notes.append(
                f"{normalized}:{dataset_name} n={limit} "
                f"mode={mode} accuracy={accuracy:.2f},cost={cost:.2f}; "
                + " | ".join(feedbacks[:2])
            )
        score = {
            "accuracy": sum(row["accuracy"] for row in objective_rows) / len(objective_rows),
            "cost": sum(row["cost"] for row in objective_rows) / len(objective_rows),
        }
        if required_terms:
            score["compliance"] = sum(
                row.get("compliance", 0.0) for row in objective_rows
            ) / len(objective_rows)
        scalar = score["accuracy"] - 0.5 * score["cost"]
        if required_terms:
            # compliance guards the intended capability spec: a terse but
            # non-compliant artifact ("Answer directly.") can no longer dominate.
            scalar += 0.5 * score["compliance"]
        feedback = (
            "[real_trace_bench_multiobjective] "
            + "; ".join(notes)
            + f". aggregate accuracy={score['accuracy']:.2f} cost={score['cost']:.2f}. "
            "Prompt-like capabilities are applied as a learner system prompt when "
            "the Trace-Bench bundle exposes one; raw/code bundles evaluate the "
            "same artifact directly."
        )
        return score, feedback, scalar

    def evaluate(capability_callable, family):
        _require_adapter("make_multiobjective_evaluator")
        try:
            return _evaluate_real(capability_callable, family)
        except BudgetExceeded:
            raise
        except Exception as exc:
            return (
                {"accuracy": 0.0, "cost": 1.0},
                f"[real_trace_bench_multiobjective] raised {type(exc).__name__}: {exc}",
                -0.5,
            )

    return evaluate
