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
from typing import Any, Callable, Dict, List, Optional, Tuple

from opto.trace.nodes import ParameterNode

from .budget import BudgetExceeded
from .levels import INVALID_CONFIG_SCORE, LevelConfig, validate_level_config

try:
    from trace_bench.registry import load_task_module, discover_tasks  # noqa: F401

    HAVE_TB = True
except Exception:
    HAVE_TB = False

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


def _score_bundle(bundle: Dict[str, Any], max_examples: int) -> Tuple[float, str]:
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
    return mean_score, f"{dataset_name}: mean over {len(scores)} real example(s). " + " | ".join(feedbacks[:3])


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
    ) -> None:
        self.tasks_root = Path(tasks_root) if tasks_root is not None else default_tasks_root()
        self.eval_kwargs = dict(eval_kwargs or {})
        self.max_examples = max_examples
        self.inner_steps = inner_steps
        self.inner_candidates = inner_candidates
        self.allowed_inner_trainers = allowed_inner_trainers
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.status = (
            f"Trace-Bench bundle adapter; tasks_root={self.tasks_root}; "
            f"max_examples={self.max_examples}; inner_steps={self.inner_steps}; "
            f"inner_candidates={self.inner_candidates}"
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
        if not fresh:
            self._cache[cache_key] = bundle
        return bundle

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

    def run_task(self, cfg: LevelConfig, task_id: str) -> Tuple[float, str]:
        """Evaluate a recursive-opt config on a real Trace-Bench task bundle."""
        validate_level_config(
            cfg,
            ("batch_size", "batch_design", "trace_type", "memory_policy", "optimizer", "guide", "trainer"),
        )
        normalized = normalize_task_id(task_id)
        budget_failure = self._trainer_budget_feedback(cfg, normalized)
        if budget_failure is not None:
            return budget_failure
        bundle = self._load_bundle(normalized, fresh=self.inner_steps > 0)
        self._train_bundle(bundle, cfg)
        score, feedback = _score_bundle(bundle, self.max_examples)
        train_note = (
            f"inner_steps={self.inner_steps}; cfg applied through Trace trainer"
            if self.inner_steps
            else "inner_steps=0; real benchmark evaluation only, meta-config not inner-trained"
        )
        hint = self._trainer_hint(cfg)
        suffix = f" {hint}" if hint else ""
        return score, f"[real_trace_bench:{normalized}] {train_note}. {feedback}{suffix}"

    def agent_fn(self, task_id: str) -> Callable:
        """Return a simple O0 agent function backed by the Trace-Bench bundle param."""
        bundle = self._load_bundle(task_id)

        def agent_fn(_artifact: Any, x: Any) -> Any:
            return _extract_response(bundle["param"], x)

        return agent_fn


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
    if not HAVE_TB and require:
        raise RuntimeError("Trace-Bench is not importable; cannot run real task scoring.")
    register_task_adapter(TraceBenchTaskAdapter.from_config(config))
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
            "is provided — results must come from a real benchmark."
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


def make_multiobjective_evaluator(task_ids, objectives, n_tasks: int = 4):
    """Return evaluate(capability_callable, family) -> (score_dict, feedback)
    for learning a NEW CAPABILITY under multiple objectives (example_C).

    objectives: dict like {"accuracy": "max", "cost": "min"}.
    The capability_callable is the (trainable) capability implementation; it is
    run on `task_ids` and produces an answer + a token/cost estimate. We return a
    per-objective score dict (consumed by opto.trainer.objectives / OptoPrimeMulti)
    plus a scalarized score and a directional feedback string.
    """

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
            objective_rows.append({"accuracy": accuracy, "cost": cost})
            notes.append(
                f"{normalized}:{dataset_name} n={limit} "
                f"mode={mode} accuracy={accuracy:.2f},cost={cost:.2f}; "
                + " | ".join(feedbacks[:2])
            )
        score = {
            "accuracy": sum(row["accuracy"] for row in objective_rows) / len(objective_rows),
            "cost": sum(row["cost"] for row in objective_rows) / len(objective_rows),
        }
        scalar = score["accuracy"] - 0.5 * score["cost"]
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
