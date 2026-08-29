"""Non-generative optimizers for recursive_opt levels (Item 2).

Generative (LLM) optimizers are weak on low-dimensional numeric/categorical
search (the verified-weak UC2/UC4/UC6 config families).
SciPy least-squares (continuous) and Optuna/TPE (mixed categorical+int, sample-efficient) are
purpose-built for exactly those fields.
Both are injectable with NO core-Trace change:
``load_optimizer`` accepts an ``Optimizer`` instance, and ``run_spec`` reads a per-level ``optimizer``.

The friction these solve: a Trace trainer drives an optimizer via *graph
feedback*, but a numeric optimizer needs *scores over a search space* (an
ask/tell loop). ``_AskTellOptimizer`` resolves that once — it ignores graph
feedback and instead calls an ``evaluate(params)->score`` callback (which wraps
``level.forward``), so it is recursion-compatible at any level/depth.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from opto.optimizers.optimizer import Optimizer


# --------------------------------------------------------------------------- #
# Search-space description (derived from the targeted LevelConfig fields)
# --------------------------------------------------------------------------- #
# Enum domains for categorical config fields (kept in sync with levels.py).
_CATEGORICAL_DOMAINS: Dict[str, Tuple[str, ...]] = {
    "batch_design": ("random", "failure_balanced", "curriculum", "diversity"),
    "trace_type": ("internal", "otel", "hybrid"),
    "credit_horizon": ("episode", "step", "truncated", "full"),
    "trainer": ("PrioritySearch", "MinibatchAlgorithm"),
    "optimizer": ("OptoPrimeV2", "OPROv2"),
}
# Integer domains for numeric config fields (low, high inclusive).
_INTEGER_DOMAINS: Dict[str, Tuple[int, int]] = {
    "batch_size": (1, 8),
    "num_epochs": (1, 4),
    "num_threads": (1, 4),
}


def field_search_space(fields: List[str]) -> Dict[str, Any]:
    """Return {field: ('cat', domain) | ('int', (lo,hi))} for routable fields.

    Free-text fields (starting_artifact, initial_knowledge) are intentionally
    omitted: only a generative optimizer can search them.
    """
    space: Dict[str, Any] = {}
    for f in fields:
        if f in _CATEGORICAL_DOMAINS:
            space[f] = ("cat", _CATEGORICAL_DOMAINS[f])
        elif f in _INTEGER_DOMAINS:
            space[f] = ("int", _INTEGER_DOMAINS[f])
    return space


def is_numeric_field(field: str) -> bool:
    """True if a field has a numeric/categorical domain (numeric-optimizer-routable)."""
    return field in _CATEGORICAL_DOMAINS or field in _INTEGER_DOMAINS


# --------------------------------------------------------------------------- #
# Ask/tell base
# --------------------------------------------------------------------------- #
class _AskTellOptimizer(Optimizer):
    """Base for numeric/black-box optimizers driven by score, not graph feedback.

    ``evaluate(assignment: dict) -> float`` scores one candidate assignment of
    the targeted fields; it is supplied by the caller and typically wraps
    ``level.forward``. ``backward`` is a no-op (no LLM gradient), and ``step``
    runs one or more ask/tell rounds, writing the best assignment back onto the
    trainable parameters.
    """

    def __init__(self, parameters, *, evaluate: Callable[[Dict[str, Any]], float],
                 space: Dict[str, Any], max_trials: int = 16, seed: int = 0, **kwargs):
        super().__init__(parameters)
        self._evaluate = evaluate
        self._space = space
        self._max_trials = int(max_trials)
        # Explicit, reproducible search. Pass a different `seed` for a multi-seed sweep;
        # leaving it to the ambient global RNG makes a result depend on execution order.
        self.seed = int(seed)
        self._best: Optional[Tuple[float, Dict[str, Any]]] = None
        self._history: List[Tuple[Dict[str, Any], float]] = []

    # numeric optimizers do not read the computation graph
    def backward(self, *args, **kwargs):
        return None

    def zero_feedback(self):
        return None

    @property
    def best_assignment(self) -> Optional[Dict[str, Any]]:
        return None if self._best is None else dict(self._best[1])

    @property
    def history(self) -> List[Tuple[Dict[str, Any], float]]:
        """(assignment, score) per trial — used to show learning progress."""
        return list(self._history)

    def _record(self, assignment: Dict[str, Any], score: float) -> None:
        self._history.append((dict(assignment), float(score)))
        if self._best is None or score > self._best[0]:
            self._best = (float(score), dict(assignment))

    def step(self, *args, **kwargs):
        """Run the optimizer's trials; return the best assignment as an update."""
        self._search()
        return dict(self._best[1]) if self._best else {}

    def _search(self) -> None:  # implemented by subclasses
        raise NotImplementedError


class OptunaOptimizer(_AskTellOptimizer):
    """TPE search over a mixed categorical+integer space.

    ``optuna`` is an optional accelerator. When it is not installed, the class
    falls back to a deterministic bounded ask/tell sweep over the same space so
    examples and tests remain runnable without adding a hard production
    dependency.
    """

    def _search(self) -> None:
        try:
            import optuna
        except ModuleNotFoundError:
            self._deterministic_search()
            return

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: "optuna.Trial") -> float:
            assignment: Dict[str, Any] = {}
            for field, (kind, dom) in self._space.items():
                if kind == "cat":
                    assignment[field] = trial.suggest_categorical(field, list(dom))
                else:  # int
                    lo, hi = dom
                    assignment[field] = trial.suggest_int(field, lo, hi)
            score = self._evaluate(assignment)
            self._record(assignment, score)
            return score

        # Seed the sampler. An unseeded TPE study makes every result depend on the
        # ambient global RNG state, so the SAME search can succeed or fail purely
        # because of what ran before it -- which is indefensible in a package whose
        # purpose is reliable measurement, and which showed up as an order-dependent
        # test failure. `seed` is settable for deliberate multi-seed sweeps.
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(objective, n_trials=self._max_trials, show_progress_bar=False)

    def _deterministic_search(self) -> None:
        """Run a dependency-free bounded sweep over the declared search space."""
        if not self._space or self._max_trials <= 0:
            return
        candidates = list(_candidate_grid(self._space))
        if not candidates:
            return
        for i in range(self._max_trials):
            assignment = candidates[i % len(candidates)]
            score = self._evaluate(assignment)
            self._record(assignment, score)


def _candidate_grid(space: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield deterministic assignments, prioritizing useful integer endpoints."""
    fields: List[str] = list(space)
    domains: List[List[Any]] = []
    for field in fields:
        kind, dom = space[field]
        if kind == "cat":
            domains.append(list(dom))
        elif kind == "int":
            lo, hi = dom
            domains.append(list(range(int(hi), int(lo) - 1, -1)))
        else:
            raise ValueError(f"unsupported search-space kind for {field}: {kind!r}")
    for values in product(*domains):
        yield dict(zip(fields, values))


class LeastSquaresOptimizer(_AskTellOptimizer):
    """SciPy least-squares over CONTINUOUS fields only (residual = target - score).

    Use for numeric refinement (e.g. integer config fields treated as continuous
    then rounded). Categorical fields in the space are ignored — route those to
    Optuna. ``target`` is the score we drive residuals toward (default 1.0).
    """

    def __init__(self, parameters, *, target: float = 1.0, **kwargs):
        super().__init__(parameters, **kwargs)
        self._target = float(target)

    def _search(self) -> None:
        import numpy as np
        from scipy.optimize import least_squares

        int_fields = [(f, dom) for f, (kind, dom) in self._space.items() if kind == "int"]
        if not int_fields:
            return
        x0 = np.array([(lo + hi) / 2.0 for _, (lo, hi) in int_fields])
        lo_b = np.array([lo for _, (lo, _) in int_fields])
        hi_b = np.array([hi for _, (_, hi) in int_fields])

        def residuals(x: "np.ndarray") -> "np.ndarray":
            assignment = {f: int(round(val)) for (f, _), val in zip(int_fields, x)}
            score = self._evaluate(assignment)
            self._record(assignment, score)
            return np.array([self._target - score])

        least_squares(residuals, x0, bounds=(lo_b, hi_b),
                      max_nfev=self._max_trials, diff_step=0.5)


# --------------------------------------------------------------------------- #
# Routing policy: which optimizer for which fields, and in what order
# --------------------------------------------------------------------------- #
_DEFAULT_POLICY = {
    "numeric_optimizer": "optuna",      # "optuna" | "least_squares"
    "text_optimizer": "OptoPrimeV2",    # generative for free-text fields
    "order": "numeric_then_text",       # | "text_then_numeric" | "numeric_only" | "text_only"
}


def route_optimizers(targets: List[str], policy: Optional[dict] = None) -> Dict[str, Any]:
    """Split targets into numeric vs text groups and pick optimizers + order.

    Returns {"numeric_fields", "text_fields", "numeric_optimizer",
    "text_optimizer", "order"}. This is the declarative lesson — "seed the
    generative optimizer with a numeric one, or polish after" — turned into a
    resolved plan the runner can execute.
    """
    pol = {**_DEFAULT_POLICY, **(policy or {})}
    numeric = [t for t in targets if is_numeric_field(t)]
    text = [t for t in targets if not is_numeric_field(t)]
    order = pol["order"]
    if not numeric:
        order = "text_only"
    elif not text:
        order = "numeric_only"
    return {
        "numeric_fields": numeric,
        "text_fields": text,
        "numeric_optimizer": pol["numeric_optimizer"],
        "text_optimizer": pol["text_optimizer"],
        "order": order,
    }
