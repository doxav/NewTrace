"""Small live probes for the remaining recursive_opt A/B/C questions.

The main use-case notebook is intentionally broad. This script is narrower: it
tests whether harder Trace-Bench tasks, code surfaces, graph/tool topology, and
waste-aware objectives give stronger optimization signal under a tiny live
budget. Results are saved as JSON under examples/notebook_outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langgraph.graph import END, START, StateGraph

from opto.trainer.guide import Guide
from opto.features.graph import LangGraphAdapter
from opto.features.recursive_opt import (
    CodeArtifactLevel,
    ComponentSpec,
    MemoryLite,
    RecursiveGuide,
    optimize,
    reset_budget,
    score_spread,
)
from opto.features.recursive_opt.runmode import have_key, make_live_llm, preflight_model
from opto.features.recursive_opt.tracebench import (
    ensure_default_task_adapter,
    load_tracebench_direct_answer_examples,
    make_dataset,
    make_tracebench_direct_answer_evaluator,
)
TASK_BBEH = "internal:multiobjective_bbeh"
DEFAULT_OUTPUT_ROOT = Path("examples/notebook_outputs/recursive_opt_abc_probe")


def _require_experimental_graph_optimizer() -> Tuple[Any, Any]:
    """Load the historical graph optimizer only for legacy live probe paths."""
    try:
        from opto.trace.io.optimization import EvalResult, optimize_graph
    except ImportError as exc:
        raise RuntimeError(
            "The legacy graph probe requires the experimental opto.trace.io optimizer; "
            "use recursive_opt.module.graph@1 for supported control-plane runs."
        ) from exc
    return EvalResult, optimize_graph


def _norm_answer(value: Any) -> str:
    """Normalize short boolean answers for exact-match scoring."""
    seen: set[int] = set()
    while hasattr(value, "data") and id(value) not in seen:
        seen.add(id(value))
        value = getattr(value, "data")
    return str(value).strip().lower().replace(".", "").replace(" ", "")


def _mean(values: Iterable[float]) -> float:
    """Return a float mean, or NaN when no values are available."""
    values = list(values)
    return statistics.mean(values) if values else float("nan")


def _finite(value: Any) -> Optional[float]:
    """Return a finite float or None for missing/NaN scores."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _artifact_ref(root: Path, artifact_id: Optional[str]) -> Optional[str]:
    """Return a compact file reference to the best saved artifact."""
    path = root / "artifacts.jsonl"
    if not path.exists():
        return None
    return f"{path}#{artifact_id}" if artifact_id else str(path)


def _score_recursive_level(level: Any, task_id: str, guide: Guide) -> float:
    """Evaluate a recursive level once through its guide."""
    score, _feedback = guide(task_id, level.forward(task_id), None)
    return float(score)


def _score_graph_module(module: Any, examples: List[Tuple[Any, Any]], guide: Guide) -> float:
    """Evaluate a graph module over direct-answer examples."""
    scores: List[float] = []
    for question, expected in examples:
        score, _feedback = guide.get_feedback(question, module.forward(question), expected)
        scores.append(float(score))
    return _mean(scores)


def _score_suboptimizer_module(module: Any, centers: List[float], tolerance: float = 1e-4) -> float:
    """Evaluate a graph module on convex minimization centers."""
    scores: List[float] = []
    for center in centers:
        out = module.forward({"center": center})
        x_value = _float_value(out)
        scores.append(1.0 if abs(x_value - center) <= tolerance else 0.0)
    return _mean(scores)


def _score_conditional_suboptimizer_module(
    module: Any,
    centers: List[float],
    *,
    tolerance: float = 0.25,
    tool_cost: float = 0.25,
) -> float:
    """Score routing quality with an explicit penalty for unnecessary tool use."""
    scores: List[float] = []
    for center in centers:
        out = module.forward({"center": center})
        data = _raw_value(out)
        if isinstance(data, dict):
            x_value = _float_value(data.get("x"))
            used_tool = bool(data.get("used_tool"))
        else:
            x_value = _float_value(data)
            used_tool = False
        accuracy = 1.0 if abs(x_value - center) <= tolerance else 0.0
        scores.append(max(0.0, accuracy - (tool_cost if used_tool else 0.0)))
    return _mean(scores)


def weak_bbeh_direct_solver(self: Any, question: str) -> str:
    """Weak direct-answer baseline with real headroom on BBEH boolean examples."""
    return "True"


def wasteful_bool_solver(self: Any, question: str) -> str:
    """Correct but deliberately wasteful boolean solver used for C probes."""
    for _ in range(400_000):
        pass
    expr = str(question).strip()
    if expr.endswith(" is"):
        expr = expr[:-3].strip()
    try:
        value = eval(expr, {"__builtins__": {}}, {"True": True, "False": False})
    except Exception:
        value = False
    return "True" if bool(value) else "False"


def graph_draft_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Weak first graph agent: produces a cheap but biased draft."""
    return {"draft": "True"}


def graph_tool_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic downstream tool node for boolean-expression evaluation."""
    raw_question = state.get("question", "")
    expr = str(getattr(raw_question, "data", raw_question)).strip()
    if expr.endswith(" is"):
        expr = expr[:-3].strip()
    try:
        value = eval(expr, {"__builtins__": {}}, {"True": True, "False": False})
    except Exception:
        value = False
    return {"tool_answer": "True" if bool(value) else "False"}


def graph_merge_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Merge graph state; `route_policy=tool` uses the downstream tool answer."""
    raw_policy = state.get("route_policy", "draft")
    policy = str(getattr(raw_policy, "data", raw_policy)).strip().lower()
    if policy in {"tool", "use_tool", "tools", "tool_answer"}:
        return {"answer": state.get("tool_answer", "False")}
    return {"answer": state.get("draft", "False")}


def build_bool_tool_graph(
    graph_draft_agent: Any = graph_draft_agent,
    graph_tool_agent: Any = graph_tool_agent,
    graph_merge_agent: Any = graph_merge_agent,
    route_policy: str = "draft",
) -> Any:
    """Build a small 3-node graph: draft -> exact tool -> merge/router."""
    _ = route_policy  # read by LangGraphAdapter as a graph knob and by merge state.
    graph = StateGraph(dict)
    graph.add_node("graph_draft_agent", graph_draft_agent)
    graph.add_node("graph_tool_agent", graph_tool_agent)
    graph.add_node("graph_merge_agent", graph_merge_agent)
    graph.add_edge(START, "graph_draft_agent")
    graph.add_edge("graph_draft_agent", "graph_tool_agent")
    graph.add_edge("graph_tool_agent", "graph_merge_agent")
    graph.add_edge("graph_merge_agent", END)
    return graph


def _raw_value(value: Any) -> Any:
    """Unwrap Trace nodes and common container scalars for runtime graph helpers."""
    seen: set[int] = set()
    while hasattr(value, "data") and id(value) not in seen:
        seen.add(id(value))
        value = getattr(value, "data")
    return value


def _float_value(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion for graph state values."""
    try:
        return float(_raw_value(value))
    except (TypeError, ValueError):
        return float(default)


def subopt_draft_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Cheap but poor numeric optimizer baseline."""
    return {"draft_x": 0.0, "draft_method": "zero"}


def scipy_suboptimizer_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Use SciPy as a downstream local optimizer tool for a convex objective."""
    center = _float_value(state.get("center", 0.0))
    try:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            lambda x: (float(x) - center) ** 2,
            bounds=(-10.0, 10.0),
            method="bounded",
            options={"xatol": 1e-8},
        )
        x_value = float(result.x)
        method = "scipy"
    except Exception:
        # Deterministic fallback keeps the demo runnable when SciPy is absent.
        x_value = center
        method = "closed_form_fallback"
    return {"tool_x": x_value, "tool_method": method}


def subopt_merge_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Route between a cheap draft and the downstream sub-optimizer."""
    policy = str(_raw_value(state.get("route_policy", "draft"))).strip().lower()
    if policy in {"scipy", "tool", "suboptimizer", "sub_optimizer"}:
        return {"x": _float_value(state.get("tool_x")), "method": state.get("tool_method", "tool")}
    return {"x": _float_value(state.get("draft_x")), "method": state.get("draft_method", "draft")}


def conditional_subopt_merge_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Route to SciPy only when the cheap draft is unlikely to be accurate."""
    policy = str(_raw_value(state.get("route_policy", "draft"))).strip().lower()
    center = abs(_float_value(state.get("center", 0.0)))
    use_tool = policy in {"scipy", "tool", "suboptimizer", "sub_optimizer"}
    if policy in {"conditional", "threshold", "adaptive"}:
        use_tool = center > 0.25
    if use_tool:
        return {
            "result": {
                "x": _float_value(state.get("tool_x")),
                "method": _raw_value(state.get("tool_method", "tool")),
                "used_tool": True,
            }
        }
    return {
        "result": {
            "x": _float_value(state.get("draft_x")),
            "method": _raw_value(state.get("draft_method", "draft")),
            "used_tool": False,
        }
    }


def build_suboptimizer_graph(
    subopt_draft_agent: Any = subopt_draft_agent,
    scipy_suboptimizer_agent: Any = scipy_suboptimizer_agent,
    subopt_merge_agent: Any = subopt_merge_agent,
    route_policy: str = "draft",
) -> Any:
    """Build a graph that can route work to a SciPy sub-optimizer."""
    _ = route_policy
    graph = StateGraph(dict)
    graph.add_node("subopt_draft_agent", subopt_draft_agent)
    graph.add_node("scipy_suboptimizer_agent", scipy_suboptimizer_agent)
    graph.add_node("subopt_merge_agent", subopt_merge_agent)
    graph.add_edge(START, "subopt_draft_agent")
    graph.add_edge("subopt_draft_agent", "scipy_suboptimizer_agent")
    graph.add_edge("scipy_suboptimizer_agent", "subopt_merge_agent")
    graph.add_edge("subopt_merge_agent", END)
    return graph


def build_conditional_suboptimizer_graph(
    subopt_draft_agent: Any = subopt_draft_agent,
    scipy_suboptimizer_agent: Any = scipy_suboptimizer_agent,
    conditional_subopt_merge_agent: Any = conditional_subopt_merge_agent,
    route_policy: str = "draft",
) -> Any:
    """Build a graph for cost-aware conditional routing to a sub-optimizer."""
    _ = route_policy
    graph = StateGraph(dict)
    graph.add_node("subopt_draft_agent", subopt_draft_agent)
    graph.add_node("scipy_suboptimizer_agent", scipy_suboptimizer_agent)
    graph.add_node("conditional_subopt_merge_agent", conditional_subopt_merge_agent)
    graph.add_edge(START, "subopt_draft_agent")
    graph.add_edge("subopt_draft_agent", "scipy_suboptimizer_agent")
    graph.add_edge("scipy_suboptimizer_agent", "conditional_subopt_merge_agent")
    graph.add_edge("conditional_subopt_merge_agent", END)
    return graph


class BooleanAnswerGuide(Guide):
    """Exact-match guide with graph-specific feedback for route/tool learning."""

    def get_feedback(
        self,
        query: str,
        response: Any,
        reference: Any = None,
        **_: Any,
    ) -> Tuple[float, str]:
        data = response.data if hasattr(response, "data") else response
        score = 1.0 if _norm_answer(data) == _norm_answer(reference) else 0.0
        feedback = (
            f"question={query!r}; expected={reference!r}; got={data!r}. "
            "For BBEH boolean expressions, the graph has a deterministic tool node; "
            "set route_policy to 'tool' when the draft is unreliable."
        )
        return score, feedback

    def get_score_dict(
        self,
        query: str,
        response: Any,
        reference: Any = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        score, _feedback = self.get_feedback(query, response, reference, **kwargs)
        return {"score": float(score)}


def make_waste_evaluator(repeats: int = 5) -> Any:
    """Return a waste-aware evaluator using wall time as the cost signal."""
    examples = [
        ("True is", "True"),
        ("False is", "False"),
        ("not False is", "True"),
        ("not True is", "False"),
    ]

    def evaluate(component_callable: Any, _family: Any) -> Tuple[Dict[str, float], str, float]:
        scores: List[float] = []
        elapsed: List[float] = []
        failures: List[str] = []
        for _ in range(repeats):
            for question, expected in examples:
                start = time.perf_counter()
                try:
                    answer = component_callable(question=question)
                except TypeError:
                    answer = component_callable(question)
                elapsed.append(time.perf_counter() - start)
                ok = _norm_answer(answer) == _norm_answer(expected)
                scores.append(1.0 if ok else 0.0)
                if not ok:
                    failures.append(f"{question!r}: got {answer!r}, expected {expected!r}")
        accuracy = _mean(scores)
        mean_ms = 1_000.0 * _mean(elapsed)
        cost = min(1.0, mean_ms / 8.0)
        scalar = max(0.0, accuracy - 0.35 * cost)
        feedback = (
            f"accuracy={accuracy:.3f}; avg_wall_ms={mean_ms:.3f}; "
            f"cost_penalty={cost:.3f}; scalar={scalar:.3f}. "
            "Remove unnecessary loops/work while preserving exact True/False answers."
        )
        if failures:
            feedback += " Failures: " + " | ".join(failures[:3])
        return {"accuracy": accuracy, "wall_ms": mean_ms, "cost": cost}, feedback, scalar

    return evaluate


def configure_live(args: argparse.Namespace) -> None:
    """Configure model and Trace-Bench adapter for a live probe run."""
    if args.live and args.model:
        os.environ["RECURSIVE_OPT_MODEL"] = args.model
        os.environ["TRACE_LITELLM_MODEL"] = args.model
    if args.live and not have_key():
        raise SystemExit("--live requires OPENAI_API_KEY or OPENROUTER_API_KEY")
    os.environ["RECURSIVE_OPT_TRACEBENCH_MAX_EXAMPLES"] = str(args.max_examples)
    os.environ["RECURSIVE_OPT_TRACEBENCH_INNER_STEPS"] = "0"
    os.environ.setdefault("RECURSIVE_OPT_TRACEBENCH_TIMEOUT_SECONDS", str(args.timeout_seconds))
    if args.live and not args.skip_preflight:
        preflight_model(os.environ.get("TRACE_LITELLM_MODEL"))
    ensure_default_task_adapter(require=True)


def run_hard_task_probe(tasks: List[str]) -> List[Dict[str, Any]]:
    """Probe task score spread to detect saturation and invalid scoring paths."""
    probes = [
        {"starting_artifact": ""},
        {"starting_artifact": "Answer directly. Keep the answer short."},
        {"starting_artifact": "Think step by step, verify, then answer."},
        {"starting_artifact": "Extract all relevant facts before answering."},
    ]
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        try:
            spread = score_spread(task, probes=probes)
            raw_scores = [row.get("score") for row in spread.get("rows", [])]
            scores = [_finite(score) for score in raw_scores]
            valid = [score for score in scores if score is not None]
            rows.append(
                {
                    "task": task,
                    "spread": _finite(spread.get("spread")),
                    "valid_scores": valid,
                    "raw_scores": raw_scores,
                    "status": "ok" if len(valid) == len(raw_scores) else "partial_or_nan",
                }
            )
        except Exception as exc:
            rows.append({"task": task, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return rows


def run_code_bbeh(output_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Run direct code optimization on real BBEH examples."""
    root = output_root / "mem_code_bbeh"
    memory = MemoryLite(root=str(root))
    level = CodeArtifactLevel(
        ComponentSpec(
            name="weak_bbeh_direct_solver",
            baseline=weak_bbeh_direct_solver,
            evaluate=make_tracebench_direct_answer_evaluator(
                TASK_BBEH,
                max_examples=args.max_examples,
                normalizer=_norm_answer,
            ),
            objective="Parse BBEH boolean expressions and return exactly True or False.",
        ),
        memory=memory,
    )
    guide = RecursiveGuide()
    initial = _score_recursive_level(level, TASK_BBEH, guide)
    reset_budget()
    start = time.time()
    optimize(
        level,
        make_dataset([TASK_BBEH], repeats=args.max_examples),
        guide=guide,
        iterations=args.iterations,
        num_candidates=args.candidates,
    )
    best = memory.best_artifact(TASK_BBEH, "code")
    if best is not None and level.parameters():
        level.parameters()[0]._data = best.content
    final = _score_recursive_level(level, TASK_BBEH, guide)
    if best is not None and float(best.score) >= final:
        final = float(best.score)
    return {
        "surface": "code",
        "task": TASK_BBEH,
        "initial": initial,
        "final": final,
        "delta": final - initial,
        "wall_s": round(time.time() - start, 3),
        "artifact_file": _artifact_ref(root, best.artifact_id if best else None),
        "code": level.current_code(),
    }


def run_graph_bbeh(output_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Run graph/tool topology optimization on the same BBEH examples."""
    EvalResult, optimize_graph = _require_experimental_graph_optimizer()
    examples = load_tracebench_direct_answer_examples(TASK_BBEH, max_examples=args.max_examples)
    adapter = LangGraphAdapter(
        graph_factory=build_bool_tool_graph,
        function_targets={
            "graph_draft_agent": graph_draft_agent,
            "graph_tool_agent": graph_tool_agent,
            "graph_merge_agent": graph_merge_agent,
        },
        graph_knobs={"route_policy": "draft"},
        input_key="question",
        output_key="answer",
        train_graph_agents_functions=False,
    )
    module = adapter.as_module()
    guide = BooleanAnswerGuide()
    initial = _score_graph_module(module, examples, guide)
    adapter.graph_knobs["route_policy"]._set("tool")
    oracle_tool = _score_graph_module(module, examples, guide)
    adapter.graph_knobs["route_policy"]._set("draft")
    reset_budget()
    start = time.time()
    graph = adapter.instrument(backend="trace")
    expected_by_question = {question: expected for question, expected in examples}

    def eval_fn(payload: Dict[str, Any]) -> EvalResult:
        question = payload["query"]
        expected = expected_by_question[question]
        answer = payload["answer"]
        score = 1.0 if _norm_answer(answer) == _norm_answer(expected) else 0.0
        return EvalResult(
            score=score,
            feedback=(
                f"score={score:.1f}; expected={expected!r}; got={answer!r}. "
                "Only graph topology/design knob route_policy is trainable; "
                "set route_policy to 'tool' to use graph_tool_agent output."
            ),
        )

    result = optimize_graph(
        graph,
        queries=[question for question, _expected in examples],
        iterations=args.iterations,
        eval_fn=eval_fn,
        optimizer_kwargs={"llm": make_live_llm(args.model)},
        output_key="answer",
    )
    final = _score_graph_module(module, examples, guide)
    root = output_root / "mem_graph_bbeh"
    root.mkdir(parents=True, exist_ok=True)
    params = {binding: value.get() for binding, value in adapter.bindings.items()}
    artifact_path = root / "artifacts.jsonl"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_id": "graph:bool_tool:latest",
                "kind": "graph",
                "family": TASK_BBEH,
                "score": final,
                "content": params,
                "metrics": {
                    "score_history": result.score_history,
                    "best_score": result.best_score,
                    "best_iteration": result.best_iteration,
                },
            },
            default=str,
        )
        + "\n"
    )
    return {
        "surface": "graph_tool_topology",
        "task": TASK_BBEH,
        "initial": initial,
        "oracle_tool_score": oracle_tool,
        "final": final,
        "delta": final - initial,
        "wall_s": round(time.time() - start, 3),
        "artifact_file": f"{artifact_path}#graph:bool_tool:latest",
        "params": params,
        "score_history": result.score_history,
        "best_score": result.best_score,
        "best_iteration": result.best_iteration,
    }


def run_suboptimizer_graph(output_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Run graph routing that learns to use SciPy as a downstream sub-optimizer."""
    EvalResult, optimize_graph = _require_experimental_graph_optimizer()
    centers = [-3.5, 1.75, 4.25, 7.0]
    adapter = LangGraphAdapter(
        graph_factory=build_suboptimizer_graph,
        function_targets={
            "subopt_draft_agent": subopt_draft_agent,
            "scipy_suboptimizer_agent": scipy_suboptimizer_agent,
            "subopt_merge_agent": subopt_merge_agent,
        },
        graph_knobs={"route_policy": "draft"},
        input_key="center",
        output_key="x",
        train_graph_agents_functions=False,
    )
    module = adapter.as_module()
    initial = _score_suboptimizer_module(module, centers)
    adapter.graph_knobs["route_policy"]._set("scipy")
    oracle_tool = _score_suboptimizer_module(module, centers)
    adapter.graph_knobs["route_policy"]._set("draft")
    reset_budget()
    start = time.time()
    graph = adapter.instrument(backend="trace")

    def eval_fn(payload: Dict[str, Any]) -> EvalResult:
        query = payload["query"]
        center = float(query["center"] if isinstance(query, dict) else query)
        x_value = _float_value(payload["answer"])
        score = 1.0 if abs(x_value - center) <= 1e-4 else 0.0
        return EvalResult(
            score=score,
            feedback=(
                f"score={score:.1f}; center={center:.3f}; x={x_value:.6f}. "
                "Only graph topology/design knob route_policy is trainable; "
                "set route_policy to 'scipy' to use scipy_suboptimizer_agent."
            ),
        )

    result = optimize_graph(
        graph,
        queries=[{"center": center} for center in centers],
        iterations=args.iterations,
        eval_fn=eval_fn,
        optimizer_kwargs={"llm": make_live_llm(args.model)},
        output_key="x",
    )
    final = _score_suboptimizer_module(module, centers)
    root = output_root / "mem_suboptimizer_graph"
    root.mkdir(parents=True, exist_ok=True)
    params = {binding: value.get() for binding, value in adapter.bindings.items()}
    spec = {
        "surface": "graph_suboptimizer_tool",
        "task": "internal:convex_suboptimizer",
        "centers": centers,
        "tolerance": 1e-4,
        "trainable": ["route_policy"],
        "candidate_routes": ["draft", "scipy"],
    }
    spec_path = root / "graph_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    artifact_path = root / "artifacts.jsonl"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_id": "graph:suboptimizer:latest",
                "kind": "graph",
                "family": "internal:convex_suboptimizer",
                "score": final,
                "content": params,
                "metrics": {
                    "score_history": result.score_history,
                    "best_score": result.best_score,
                    "best_iteration": result.best_iteration,
                    "centers": centers,
                },
            },
            default=str,
        )
        + "\n"
    )
    return {
        "surface": "graph_suboptimizer_tool",
        "task": "internal:convex_suboptimizer",
        "initial": initial,
        "oracle_tool_score": oracle_tool,
        "final": final,
        "delta": final - initial,
        "wall_s": round(time.time() - start, 3),
        "artifact_file": f"{artifact_path}#graph:suboptimizer:latest",
        "spec_file": str(spec_path),
        "params": params,
        "score_history": result.score_history,
        "best_score": result.best_score,
        "best_iteration": result.best_iteration,
    }


def run_conditional_suboptimizer_graph(output_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Run cost-aware graph routing that should prefer a conditional policy."""
    EvalResult, optimize_graph = _require_experimental_graph_optimizer()
    centers = [-3.5, -0.1, 0.0, 0.2, 1.75, 4.25]
    adapter = LangGraphAdapter(
        graph_factory=build_conditional_suboptimizer_graph,
        function_targets={
            "subopt_draft_agent": subopt_draft_agent,
            "scipy_suboptimizer_agent": scipy_suboptimizer_agent,
            "conditional_subopt_merge_agent": conditional_subopt_merge_agent,
        },
        graph_knobs={"route_policy": "draft"},
        input_key="center",
        output_key="result",
        train_graph_agents_functions=False,
    )
    module = adapter.as_module()
    initial = _score_conditional_suboptimizer_module(module, centers)
    adapter.graph_knobs["route_policy"]._set("scipy")
    always_tool = _score_conditional_suboptimizer_module(module, centers)
    adapter.graph_knobs["route_policy"]._set("conditional")
    oracle_conditional = _score_conditional_suboptimizer_module(module, centers)
    adapter.graph_knobs["route_policy"]._set("draft")
    reset_budget()
    start = time.time()
    graph = adapter.instrument(backend="trace")

    def eval_fn(payload: Dict[str, Any]) -> EvalResult:
        query = payload["query"]
        center = float(query["center"] if isinstance(query, dict) else query)
        data = _raw_value(payload["answer"])
        result_data = data if isinstance(data, dict) else {"x": data, "used_tool": False}
        x_value = _float_value(result_data.get("x"))
        used_tool = bool(result_data.get("used_tool"))
        accuracy = 1.0 if abs(x_value - center) <= 0.25 else 0.0
        score = max(0.0, accuracy - (0.25 if used_tool else 0.0))
        return EvalResult(
            score=score,
            feedback=(
                f"score={score:.2f}; center={center:.3f}; x={x_value:.6f}; "
                f"used_tool={used_tool}. The dataset mixes near-zero easy cases "
                "where draft is enough and far cases where SciPy is needed; set "
                "route_policy to 'conditional' for the best accuracy/cost tradeoff."
            ),
        )

    result = optimize_graph(
        graph,
        queries=[{"center": center} for center in centers],
        iterations=args.iterations,
        eval_fn=eval_fn,
        optimizer_kwargs={"llm": make_live_llm(args.model)},
        output_key="result",
    )
    final = _score_conditional_suboptimizer_module(module, centers)
    root = output_root / "mem_conditional_suboptimizer_graph"
    root.mkdir(parents=True, exist_ok=True)
    params = {binding: value.get() for binding, value in adapter.bindings.items()}
    spec = {
        "surface": "graph_conditional_suboptimizer_tool",
        "task": "internal:cost_aware_convex_suboptimizer",
        "centers": centers,
        "tool_cost": 0.25,
        "tolerance": 0.25,
        "trainable": ["route_policy"],
        "candidate_routes": ["draft", "scipy", "conditional"],
    }
    (root / "graph_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    artifact_path = root / "artifacts.jsonl"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_id": "graph:conditional_suboptimizer:latest",
                "kind": "graph",
                "family": "internal:cost_aware_convex_suboptimizer",
                "score": final,
                "content": params,
                "metrics": {
                    "score_history": result.score_history,
                    "best_score": result.best_score,
                    "best_iteration": result.best_iteration,
                    "initial": initial,
                    "always_tool_score": always_tool,
                    "oracle_conditional_score": oracle_conditional,
                    "centers": centers,
                },
            },
            default=str,
        )
        + "\n"
    )
    return {
        "surface": "graph_conditional_suboptimizer_tool",
        "task": "internal:cost_aware_convex_suboptimizer",
        "initial": initial,
        "always_tool_score": always_tool,
        "oracle_tool_score": oracle_conditional,
        "final": final,
        "delta": final - initial,
        "wall_s": round(time.time() - start, 3),
        "artifact_file": f"{artifact_path}#graph:conditional_suboptimizer:latest",
        "spec_file": str(root / "graph_spec.json"),
        "params": params,
        "score_history": result.score_history,
        "best_score": result.best_score,
        "best_iteration": result.best_iteration,
    }


def run_waste_code(output_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Run waste-aware code optimization with measured wall-time penalty."""
    root = output_root / "mem_waste_code"
    memory = MemoryLite(root=str(root))
    level = CodeArtifactLevel(
        ComponentSpec(
            name="wasteful_bool_solver",
            baseline=wasteful_bool_solver,
            evaluate=make_waste_evaluator(),
            objective="Keep exact answers while removing unnecessary compute and loops.",
        ),
        memory=memory,
    )
    guide = RecursiveGuide()
    initial = _score_recursive_level(level, "internal:waste_bool", guide)
    reset_budget()
    start = time.time()
    optimize(
        level,
        make_dataset(["internal:waste_bool"], repeats=4),
        guide=guide,
        iterations=args.iterations,
        num_candidates=args.candidates,
    )
    best = memory.best_artifact("internal:waste_bool", "code")
    if best is not None and level.parameters():
        level.parameters()[0]._data = best.content
    final = _score_recursive_level(level, "internal:waste_bool", guide)
    if best is not None and float(best.score) >= final:
        final = float(best.score)
    return {
        "surface": "waste_aware_code",
        "task": "internal:waste_bool",
        "initial": initial,
        "final": final,
        "delta": final - initial,
        "wall_s": round(time.time() - start, 3),
        "artifact_file": _artifact_ref(root, best.artifact_id if best else None),
        "code": level.current_code(),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a bounded probe run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Require a live LLM/key for optimization.")
    parser.add_argument("--model", default=os.environ.get("RECURSIVE_OPT_MODEL", "gpt-5.4-nano"))
    parser.add_argument("--run-id", default=time.strftime("abc_probe_%Y%m%d_%H%M%S"))
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=35)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--sections",
        default="spread,code,graph,waste,subopt,conditional_subopt",
        help="Comma-separated subset: spread,code,graph,waste,subopt,conditional_subopt",
    )
    return parser.parse_args()


def main() -> None:
    """Run selected probes and save a machine-readable summary."""
    args = parse_args()
    sections = {section.strip() for section in args.sections.split(",") if section.strip()}
    live_sections = {"code", "graph", "waste", "subopt", "conditional_subopt"}
    if not args.live and live_sections & sections:
        raise SystemExit(
            "Sections code/graph/waste/subopt/conditional_subopt require --live; "
            "use --sections spread for eval-only probing."
        )
    configure_live(args)
    output_root = DEFAULT_OUTPUT_ROOT / args.run_id
    output_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "run_id": args.run_id,
        "model": args.model,
        "max_examples": args.max_examples,
        "iterations": args.iterations,
        "candidates": args.candidates,
        "output_root": str(output_root),
        "results": {},
    }
    if "spread" in sections:
        summary["results"]["spread"] = run_hard_task_probe(
            [
                "internal:multiobjective_gsm8k",
                "internal:multiobjective_bbeh",
                "hf:drop",
                "hf:qasper",
                "hf:hotpot_qa",
                "hf:strategy_qa",
                "hf:aqua_rat",
            ]
        )
    if "code" in sections:
        summary["results"]["code"] = run_code_bbeh(output_root, args)
    if "graph" in sections:
        summary["results"]["graph"] = run_graph_bbeh(output_root, args)
    if "waste" in sections:
        summary["results"]["waste"] = run_waste_code(output_root, args)
    if "subopt" in sections:
        summary["results"]["subopt"] = run_suboptimizer_graph(output_root, args)
    if "conditional_subopt" in sections:
        summary["results"]["conditional_subopt"] = run_conditional_suboptimizer_graph(output_root, args)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
