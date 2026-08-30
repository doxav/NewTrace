"""Measurement certification for recursive-opt experiments.

An optimization result is only interpretable if the instrument that produced it can
resolve the effect being claimed. Nothing in this package checked that, and the
consequences were concrete: a task whose objective reduced to a token counter, a
benchmark that returned ``accuracy 0.0`` in six microseconds without calling a model,
and a headline result that was the offset between two different metrics.

This module makes measurability an explicit, checkable precondition:

    certificate = certify_task("internal:multiobjective_gsm8k")
    if not certificate.usable:
        ...                      # do not spend an experiment on it

It measures four things a task must satisfy before an experiment runs on it:

1. **liveness**    - the evaluator actually runs (a task that returns instantly with a
                     constant score is broken, not hard);
2. **headroom**    - the quality metric is not saturated at its ceiling or floor;
3. **stability**   - the run-to-run noise of *repeating the identical measurement*;
4. **resolution**  - the smallest effect that noise permits detecting at a given n.

Design notes
------------
* Every worker builds its **own** bundle. ``_apply_starting_artifact`` mutates
  ``param.system_prompt`` in place and ``_load_bundle`` caches by default, so sharing a
  bundle across threads lets one arm's prompt score another arm's examples with no error.
* Failed evaluations are **counted and excluded**, never imputed to a score. Converting a
  crash into a number is the defect that made two earlier experiments meaningless.
* Rates are reported with Wilson intervals. A rate estimated from one event in twelve is
  not an effect size, and the certificate says so rather than implying precision.
"""
from __future__ import annotations

import math
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .levels import LevelConfig

DEFAULT_MAX_WORKERS = 8
#: Providers throttle under load; beyond this the failure rate rises faster than throughput.
MAX_SAFE_WORKERS = 16


#: Evaluation sampling temperature. Measured on internal:multiobjective_gsm8k, pinning
#: this cut the noise floor 3.2x (sd 0.0415 -> 0.0131). Combined with the request bounds
#: below it improves the resolvable effect at n=5 from 0.239 to 0.016 - a 14.6x gain, and
#: the difference between "needs 115 seeds" and "needs 1" for a 0.05 effect.
DEFAULT_EVAL_TEMPERATURE = 0.2
#: Greedy decoding (temperature 0.0) sent this model into unbounded degenerate repetition
#: and stalled a run indefinitely. An unbounded request is not a measurement.
DEFAULT_MAX_TOKENS = 512
DEFAULT_REQUEST_TIMEOUT = 60


class BoundedEvalLLM:
    """Wrap a bundle LLM so every evaluation call is explicit and terminating.

    ``opto.features.predefined_agents.learner.call_llm`` passes no ``temperature``,
    ``max_tokens`` or ``timeout``, so evaluation inherits provider defaults - typically
    temperature ~1.0 and no cap. That makes each measurement a draw from a wide
    distribution that can also fail to terminate.

    Applied to *evaluation* only. Optimizer proposals deliberately keep their own
    sampling, because diversity is wanted when generating candidates and unwanted when
    measuring them.

    ``max_tokens`` is **not neutral**: capping it changed the observed hf:qasper score
    from ~0.09 to ~0.045 by truncating answers. It must therefore be held identical
    across every arm of a comparison, which is why it is part of the certified triple
    rather than a free parameter.
    """

    def __init__(
        self,
        inner: Any,
        temperature: Optional[float] = DEFAULT_EVAL_TEMPERATURE,
        max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
        timeout: Optional[float] = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self._inner = inner
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._temperature is not None:
            kwargs.setdefault("temperature", self._temperature)
        if self._max_tokens is not None:
            kwargs.setdefault("max_tokens", self._max_tokens)
        if self._timeout is not None:
            kwargs.setdefault("timeout", self._timeout)
        return self._inner(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


#: Backwards-compatible alias for the temperature-only wrapper.
FixedTemperatureLLM = BoundedEvalLLM


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Return the Wilson score interval for a binomial rate.

    Preferred over the normal approximation because the rates here are small and the
    samples tiny, exactly where the normal interval misleads (it can cover negative
    rates, or collapse to zero width when no event is observed).
    """
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    d = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / d
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def required_n(sd: float, delta: float, *, alpha: float = 0.05, power: float = 0.8) -> Optional[int]:
    """Return the paired sample size needed to detect ``delta`` given noise ``sd``."""
    if delta <= 0:
        raise ValueError("delta must be positive")
    if sd <= 0:
        return 1
    z_alpha, z_beta = 1.959963985, 0.841621234  # two-sided 0.05, power 0.8
    return max(1, math.ceil((z_alpha + z_beta) ** 2 * (sd / delta) ** 2))


def resolvable_delta(sd: float, n: int, *, alpha: float = 0.05, power: float = 0.8) -> float:
    """Return the smallest effect detectable with ``n`` paired samples at noise ``sd``."""
    if n <= 0:
        raise ValueError("n must be positive")
    z_alpha, z_beta = 1.959963985, 0.841621234
    return (z_alpha + z_beta) * sd / math.sqrt(n)


@dataclass
class TaskCertificate:
    """Whether a (task, model, budget) triple can measure anything."""

    task_id: str
    model: Optional[str]
    max_examples: int
    repeats: int
    temperature: Optional[float]
    max_tokens: Optional[int] = None
    request_timeout: Optional[float] = None
    surface: Optional[str] = None
    calls_llm: Optional[bool] = None
    concurrency: Optional[int] = None
    live: Optional[bool] = None
    liveness_spread: Optional[float] = None
    scores: List[float] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    metric_keys: Tuple[str, ...] = ()
    mean_score: Optional[float] = None
    noise_sd: Optional[float] = None
    mean_eval_seconds: Optional[float] = None
    quality_rate: Optional[float] = None
    quality_ci: Optional[Tuple[float, float]] = None
    quality_metric: Optional[str] = None
    verdict: str = "unknown"
    reasons: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Whether an experiment may be run against this triple."""
        return self.verdict == "certified"

    def resolvable_at(self, n: int) -> Optional[float]:
        """Smallest effect this triple can resolve with ``n`` paired seeds."""
        return None if self.noise_sd is None else resolvable_delta(self.noise_sd, n)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["usable"] = self.usable
        data["resolvable_at_5"] = self.resolvable_at(5)
        data["resolvable_at_10"] = self.resolvable_at(10)
        return data


#: A wall-clock evaluation faster than this did not call a model.
_INSTANT_EVAL_SECONDS = 0.05
#: Exact failure sentinels used across the ecosystem. A magnitude *threshold* is wrong
#: here: llm4ad:online_bin_packing legitimately scores -2091.8 (a real bin-packing cost)
#: while returning -1e6 when the candidate program fails to run. Only the specific
#: sentinel magnitudes count, never "large".
_SENTINEL_MAGNITUDES = (1e6, 1e9, 1e12)
_SENTINEL_TOLERANCE = 1e-6
#: Quality metrics whose direction is "higher is better" / "lower is better".
_QUALITY_HIGHER = ("accuracy", "score", "pass_rate")
_QUALITY_LOWER = ("error",)

_CODE_MARKERS = ("def ", "class ", "import ", "from ", "return", "lambda", "--", "//", "#include")


@dataclass(frozen=True)
class TaskSurface:
    """What a task's trainable parameter actually is, and whether scoring calls a model.

    Certification injected a PROSE probe ("Answer directly.") into every task. For a task
    whose trainable parameter is Python source, a float, or Lean 4, that is not a probe -
    it is corruption, and it made five healthy tasks look broken. Worse, the "no model was
    called" heuristic flagged LLM-free evaluation as a defect when it is the correct and
    desirable behaviour: a deterministic evaluator has no sampling noise at all, which
    makes it the *best* available measurement surface, not the worst.
    """

    kind: str                 # 'prose' | 'code' | 'numeric' | 'unknown'
    calls_llm: bool
    param_name: Optional[str] = None
    sample: str = ""

    @property
    def accepts_prose_probe(self) -> bool:
        """Whether a prose artifact can legitimately be injected into this task."""
        return self.kind == "prose"


def detect_surface(bundle: Mapping[str, Any]) -> TaskSurface:
    """Classify a bundle's trainable surface and whether its evaluation calls a model."""
    param = bundle.get("param")
    if param is None:
        return TaskSurface("unknown", False)
    calls_llm = hasattr(param, "llm") or hasattr(param, "system_prompt")
    if hasattr(param, "system_prompt"):
        return TaskSurface("prose", calls_llm, "system_prompt",
                           str(getattr(param.system_prompt, "data", ""))[:120])

    node = None
    getter = getattr(param, "parameters", None)
    if callable(getter):
        nodes = list(getter() or [])
        node = nodes[0] if nodes else None
    if node is None and hasattr(param, "data"):
        node = param
    if node is None:
        return TaskSurface("unknown", calls_llm)

    text = str(getattr(node, "data", ""))
    name = getattr(node, "name", None)
    stripped = text.strip()
    try:
        float(stripped)
        return TaskSurface("numeric", calls_llm, name, text[:120])
    except (TypeError, ValueError):
        pass
    if any(marker in text for marker in _CODE_MARKERS):
        return TaskSurface("code", calls_llm, name, text[:120])
    if stripped:
        return TaskSurface("prose", calls_llm, name, text[:120])
    return TaskSurface("unknown", calls_llm, name, text[:120])


def _quality_from_metrics(metrics: Dict[str, float]) -> Tuple[Optional[str], Optional[float]]:
    """Return the quality metric name and its higher-is-better value, if identifiable."""
    for key in _QUALITY_HIGHER:
        if key in metrics:
            return key, float(metrics[key])
    for key in _QUALITY_LOWER:
        if key in metrics:
            return key, 1.0 - float(metrics[key])
    return None, None


def evaluate_once(
    task_id: str,
    *,
    artifact: Optional[str] = None,
    max_examples: int = 8,
    temperature: Optional[float] = DEFAULT_EVAL_TEMPERATURE,
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
    request_timeout: Optional[float] = DEFAULT_REQUEST_TIMEOUT,
    timeout_seconds: int = 90,
    eval_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one complete evaluation on a private bundle and return raw observations.

    Builds its own adapter and bundle so it is safe to call concurrently: nothing here
    touches the ``tracebench`` module-level adapter or a shared cached bundle.
    Exceptions are returned as ``valid=False`` with the message — never as a score.
    """
    from . import tracebench as TB

    started = time.time()
    kwargs = {"timeout_seconds": timeout_seconds, **(eval_kwargs or {})}
    try:
        adapter = TB.TraceBenchTaskAdapter(
            max_examples=max_examples, inner_steps=0, eval_kwargs=kwargs
        )
        bundle = adapter._load_bundle(task_id, fresh=True)
        surface = detect_surface(bundle)
        # Injecting prose into a code/numeric/Lean parameter is corruption, not a probe.
        # `artifact=None` (the default) scores the bundle exactly as shipped, which is the
        # only probe that is valid for every surface.
        if artifact:
            if not surface.accepts_prose_probe:
                return {"valid": False, "task_id": task_id, "surface": surface.kind,
                        "error": (f"refused to inject a prose artifact into a "
                                  f"{surface.kind} parameter ({surface.param_name!r})"),
                        "seconds": time.time() - started}
            adapter._apply_starting_artifact(bundle, LevelConfig(starting_artifact=artifact))
        param = bundle.get("param")
        if hasattr(param, "llm"):
            param.llm = BoundedEvalLLM(param.llm, temperature, max_tokens, request_timeout)

        name, dataset = TB._evaluation_dataset(bundle)
        inputs = list(dataset.get("inputs") or [])
        infos = TB._dataset_infos(dataset)
        limit = min(len(inputs), len(infos), max_examples)
        if limit <= 0:
            return {"valid": False, "error": f"{name} is empty", "task_id": task_id,
                    "seconds": time.time() - started}

        guide = bundle["guide"]
        per_example: List[Dict[str, float]] = []
        for i in range(limit):
            response = TB._extract_response(param, inputs[i])
            reward, _feedback = guide(inputs[i], response, infos[i])
            score_dict = (guide.get_score_dict(inputs[i], response, infos[i])
                          if hasattr(guide, "get_score_dict") else {}) or {}
            row = {k: float(v) for k, v in score_dict.items()}
            row.setdefault("reward", float(reward))
            per_example.append(row)

        score, _fb = TB._score_bundle(bundle, max_examples)
        return {"valid": True, "task_id": task_id, "dataset": name, "n_examples": limit,
                "score": float(score), "per_example": per_example,
                "surface": surface.kind, "calls_llm": surface.calls_llm,
                "bounds": {"temperature": temperature, "max_tokens": max_tokens,
                           "request_timeout": request_timeout},
                "seconds": time.time() - started}
    except Exception as exc:  # a failed measurement is not a measurement of zero
        return {"valid": False, "task_id": task_id,
                "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}",
                "seconds": time.time() - started}


#: Surface-appropriate perturbations used to prove an evaluator actually responds.
_PERTURBATIONS = {
    "numeric": ("0.0", "7.5"),
    "code": ("def _probe(*a, **k):\n    return None\n",),
    "prose": ("Answer directly.", "Plan step by step, then verify before replying."),
}


def probe_liveness(
    task_id: str,
    *,
    max_examples: int = 4,
    temperature: Optional[float] = DEFAULT_EVAL_TEMPERATURE,
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
    request_timeout: Optional[float] = DEFAULT_REQUEST_TIMEOUT,
    timeout_seconds: int = 90,
) -> Dict[str, Any]:
    """Does a surface-appropriate change to the trainable parameter move the score?

    A constant score is ambiguous on its own: a deterministic evaluator whose shipped
    baseline solves nothing scores a constant 0.0, and so does an evaluator that is
    simply broken. The difference is whether the score *responds*. This perturbs the
    parameter in a way that is valid for its own surface — a number for a float, code
    for code — and reports whether the evaluator noticed.
    """
    from . import tracebench as TB

    baseline = evaluate_once(task_id, artifact=None, max_examples=max_examples,
                             temperature=temperature, max_tokens=max_tokens,
                             request_timeout=request_timeout, timeout_seconds=timeout_seconds)
    if not baseline.get("valid"):
        return {"live": False, "reason": f"baseline evaluation failed: {baseline.get('error')}",
                "baseline": baseline}

    adapter = TB.TraceBenchTaskAdapter(max_examples=max_examples, inner_steps=0,
                                       eval_kwargs={"timeout_seconds": timeout_seconds})
    observed: List[Dict[str, Any]] = []
    for variant in _PERTURBATIONS.get(baseline.get("surface", ""), ()):
        try:
            bundle = adapter._load_bundle(task_id, fresh=True)
            param = bundle.get("param")
            if temperature is not None and hasattr(param, "llm"):
                param.llm = BoundedEvalLLM(param.llm, temperature, max_tokens, request_timeout)
            node = None
            getter = getattr(param, "parameters", None)
            if callable(getter):
                nodes = list(getter() or [])
                node = nodes[0] if nodes else None
            if node is None and hasattr(param, "data"):
                node = param
            if node is None:
                continue
            node._data = variant
            score, _fb = TB._score_bundle(bundle, max_examples)
            observed.append({"variant": variant[:60], "score": float(score)})
        except Exception as exc:
            observed.append({"variant": variant[:60], "score": None,
                             "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"})

    values = [baseline["score"]] + [o["score"] for o in observed if o.get("score") is not None]
    spread = (max(values) - min(values)) if len(values) > 1 else 0.0
    return {"live": spread > 1e-9, "spread": spread, "baseline_score": baseline["score"],
            "surface": baseline.get("surface"), "variants": observed,
            "reason": ("the score responds to a change in the trainable parameter"
                       if spread > 1e-9 else
                       "the score did not move for any valid change to the trainable parameter")}


def certify_task(
    task_id: str,
    *,
    model: Optional[str] = None,
    max_examples: int = 8,
    repeats: int = 3,
    temperature: Optional[float] = DEFAULT_EVAL_TEMPERATURE,
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
    request_timeout: Optional[float] = DEFAULT_REQUEST_TIMEOUT,
    artifact: Optional[str] = None,
    check_liveness: bool = True,
    target_delta: float = 0.05,
    target_delta_relative: float = 0.01,
    target_n: int = 5,
    max_workers: int = DEFAULT_MAX_WORKERS,
    concurrency: int = DEFAULT_MAX_WORKERS,
    timeout_seconds: int = 90,
) -> TaskCertificate:
    """Measure whether a (task, model, budget) triple can resolve ``target_delta``.

    Repeats the **identical** measurement ``repeats`` times; the spread across those
    repeats is the noise floor an experiment has to beat. The artifact is held fixed on
    purpose: this measures the instrument, not the optimizer.

    ``concurrency`` must match how the experiment will actually run. This is not a
    performance knob — it changes the answer. Measured on
    ``llm4ad:optimization/online_bin_packing``, whose evaluator runs the candidate
    heuristic under a time budget:

    ===============  ==========  ===========
    condition        noise sd    score range
    ===============  ==========  ===========
    sequential       0.0000       0.00
    8 concurrent     3.1487      10.80
    ===============  ==========  ===========

    Certifying it sequentially reported a *zero* noise floor and led directly to
    reporting a +4.8 "improvement" that was entirely inside the concurrency noise. A
    noise floor measured under conditions the experiment will not reproduce is not a
    noise floor.
    """
    if repeats < 2:
        raise ValueError("repeats must be at least 2 to estimate a noise floor")
    workers = max(1, min(int(max_workers), int(concurrency), MAX_SAFE_WORKERS, repeats))

    cert = TaskCertificate(task_id=task_id, model=model, max_examples=max_examples,
                           repeats=repeats, temperature=temperature,
                           max_tokens=max_tokens, request_timeout=request_timeout,
                           concurrency=workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        observations = list(pool.map(
            lambda _i: evaluate_once(task_id, artifact=artifact, max_examples=max_examples,
                                     temperature=temperature, max_tokens=max_tokens,
                                     request_timeout=request_timeout,
                                     timeout_seconds=timeout_seconds),
            range(repeats),
        ))

    ok = [o for o in observations if o.get("valid")]
    cert.failures = [o["error"] for o in observations if not o.get("valid")]
    cert.mean_eval_seconds = st.mean([o["seconds"] for o in observations]) if observations else None

    if not ok:
        cert.verdict = "broken"
        cert.reasons.append(f"every evaluation failed: {cert.failures[:2]}")
        return cert

    cert.surface = ok[0].get("surface")
    cert.calls_llm = ok[0].get("calls_llm")
    cert.scores = [o["score"] for o in ok]
    cert.mean_score = st.mean(cert.scores)
    cert.noise_sd = st.pstdev(cert.scores) if len(cert.scores) > 1 else 0.0
    rows = [row for o in ok for row in o["per_example"]]
    cert.metric_keys = tuple(sorted({k for row in rows for k in row}))

    name, values = None, []
    for row in rows:
        key, value = _quality_from_metrics(row)
        if key is not None:
            name, _ = key, None
            values.append(value)
    if values:
        cert.quality_metric = name
        cert.quality_rate = st.mean(values)
        successes = sum(1 for v in values if v >= 0.999)
        cert.quality_ci = wilson_interval(successes, len(values))

    # A constant score is ambiguous. Before calling it degenerate, ask whether the
    # evaluator responds AT ALL to a valid change in its own trainable parameter.
    if check_liveness and cert.noise_sd == 0.0:
        probe = probe_liveness(task_id, max_examples=max_examples, temperature=temperature,
                               max_tokens=max_tokens, request_timeout=request_timeout,
                               timeout_seconds=timeout_seconds)
        cert.live = bool(probe.get("live"))
        cert.liveness_spread = probe.get("spread")

    _apply_verdict(cert, target_delta=target_delta, target_n=target_n,
                   target_delta_relative=target_delta_relative)
    return cert


def _is_sentinel_score(value: float) -> bool:
    """Whether a score is a known failure sentinel rather than a measurement."""
    if not math.isfinite(value):
        return True
    magnitude = abs(value)
    return any(abs(magnitude - s) <= _SENTINEL_TOLERANCE * s for s in _SENTINEL_MAGNITUDES)


def effective_target_delta(mean_score: Optional[float], target_delta: float,
                           target_delta_relative: float) -> float:
    """Scale the target effect to the task's own units.

    An absolute target is meaningless across objectives with different scales. A
    bin-packing cost of -2092.6 with a noise floor of 0.8 is one of the quietest
    surfaces available (0.04% relative), yet an absolute target of 0.05 rejects it as
    "too noisy" purely because of units. A 1% relative floor keeps the absolute target
    for scores near 1 and adapts it for everything else.
    """
    if mean_score is None:
        return target_delta
    return max(target_delta, target_delta_relative * abs(float(mean_score)))


def _apply_verdict(cert: TaskCertificate, *, target_delta: float, target_n: int,
                   target_delta_relative: float = 0.01) -> None:
    """Classify a certificate as certified / broken / degenerate / saturated / too_noisy."""
    # A sentinel is checked FIRST and independently of variance. A constant failure code
    # has zero noise, and reading that as "quiet therefore certified" is precisely the
    # mistake this module exists to prevent.
    sentinels = [s for s in cert.scores if _is_sentinel_score(s)]
    if sentinels:
        cert.verdict = "broken"
        cert.reasons.append(
            f"evaluation returned the failure sentinel {sentinels[0]:g} "
            f"({len(sentinels)}/{len(cert.scores)} repeats) — the candidate never ran"
        )
        return
    if (cert.calls_llm and cert.mean_eval_seconds is not None
            and cert.mean_eval_seconds < _INSTANT_EVAL_SECONDS):
        cert.verdict = "broken"
        cert.reasons.append(
            f"evaluation returned in {cert.mean_eval_seconds:.4f}s but this task's scoring "
            "calls a model — no model was called"
        )
        return
    if cert.live is False:
        cert.verdict = "degenerate"
        cert.reasons.append(
            "the score did not move for any valid change to the trainable parameter — "
            "the evaluator is not responding, so nothing can be optimized against it"
        )
        return
    if (cert.noise_sd == 0.0 and len(set(cert.scores)) == 1 and cert.scores[0] == 0.0
            and not cert.live):
        cert.verdict = "degenerate"
        cert.reasons.append(
            f"every repeat scored exactly 0.0 after {cert.mean_eval_seconds:.1f}s of real "
            f"compute; suspect the evaluation bounds (max_tokens={cert.max_tokens}) are "
            "truncating answers, or the artifact does not fit this task's surface"
        )
        return
    if cert.live and cert.noise_sd == 0.0:
        spread = ("" if cert.liveness_spread is None
                  else f" (spread {cert.liveness_spread:.4g})")
        if (cert.concurrency or 1) <= 1:
            cert.reasons.append(
                f"noise 0.0 measured SEQUENTIALLY{spread}; this is not a usable floor unless "
                "the experiment also runs sequentially — re-certify at the concurrency you "
                "intend to run at"
            )
        else:
            cert.reasons.append(
                f"deterministic evaluator (noise 0.0 at concurrency {cert.concurrency})"
                f"{spread} — resolves any effect size at n=1"
            )
    if cert.quality_rate is not None:
        lo, hi = cert.quality_ci or (0.0, 1.0)
        if cert.quality_rate >= 0.999:
            cert.verdict = "saturated"
            cert.reasons.append(
                f"{cert.quality_metric} is already at the ceiling ({cert.quality_rate:.3f}, "
                f"CI {lo:.2f}-{hi:.2f}): the baseline solves it, so there is no headroom "
                "for any optimizer to demonstrate an improvement"
            )
            return
        if cert.quality_rate <= 0.001:
            # A floor is only a defect when the evaluator does not respond. A LIVE
            # evaluator sitting at its floor is the ideal starting point: maximum
            # headroom, and every gain is unambiguous.
            if cert.live is False:
                cert.verdict = "degenerate"
                cert.reasons.append(
                    f"{cert.quality_metric} is at floor and the score does not respond to "
                    "its parameter: nothing is being measured"
                )
                return
            cert.reasons.append(
                f"{cert.quality_metric} is at its floor ({cert.quality_rate:.3f}) but the "
                "evaluator responds — maximum headroom, every gain is unambiguous"
            )
    if cert.noise_sd is not None:
        detectable = resolvable_delta(cert.noise_sd, target_n)
        target = effective_target_delta(cert.mean_score, target_delta, target_delta_relative)
        scale = "" if target == target_delta else f" ({target_delta_relative:.0%} of |score|)"
        if detectable > target:
            cert.verdict = "too_noisy"
            cert.reasons.append(
                f"noise sd={cert.noise_sd:.4g} resolves only {detectable:.4g} at n={target_n}; "
                f"target delta is {target:.4g}{scale} "
                f"(needs n={required_n(cert.noise_sd, target)})"
            )
            return
        cert.reasons.append(
            f"noise sd={cert.noise_sd:.4g} resolves {detectable:.4g} at n={target_n}, "
            f"target {target:.4g}{scale}"
        )
    cert.verdict = "certified"


def certify_pool(
    task_ids: Sequence[str],
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    concurrency: int = DEFAULT_MAX_WORKERS,
    **kwargs: Any,
) -> Dict[str, TaskCertificate]:
    """Certify several tasks; each task owns its adapters and bundles.

    Tasks are certified one at a time so that each task's own ``concurrency`` is the only
    load in play. Fanning tasks out in parallel while measuring each sequentially reports
    a noise floor no experiment will ever see.
    """
    certs = [certify_task(t, max_workers=max_workers, concurrency=concurrency, **kwargs)
             for t in task_ids]
    return {c.task_id: c for c in certs}


def format_certificates(certs: Dict[str, TaskCertificate], *, target_n: int = 5) -> str:
    """Return a compact human-readable certification table."""
    header = (f"{'task':38s} {'verdict':11s} {'score':>8s} {'noise':>8s} "
              f"{'res@' + str(target_n):>8s} {'quality':>16s}  notes")
    lines = [header, "-" * len(header)]
    for task, c in sorted(certs.items()):
        res = c.resolvable_at(target_n)
        quality = "-" if c.quality_rate is None else (
            f"{c.quality_metric[:8]}={c.quality_rate:.2f}")
        lines.append(
            f"{task[:38]:38s} {c.verdict:11s} "
            f"{'-' if c.mean_score is None else f'{c.mean_score:+.4f}':>8s} "
            f"{'-' if c.noise_sd is None else f'{c.noise_sd:.4f}':>8s} "
            f"{'-' if res is None else f'{res:.4f}':>8s} {quality:>16s}  "
            f"{(c.reasons[0] if c.reasons else '')[:60]}"
        )
    return "\n".join(lines)


#: Provider failures that are worth retrying. Deliberately narrow: a programming error
#: (TypeError, KeyError, AttributeError) must surface immediately, because retrying it
#: three times only delays the report and hides the stack that explains it.
_TRANSIENT_MESSAGE_MARKERS = (
    "connection reset", "connection aborted", "connection error", "timeout", "timed out",
    "rate limit", "ratelimit", "too many requests", "overloaded", "service unavailable",
    "bad gateway", "temporarily unavailable", "internal server error",
    "502", "503", "504", "429",
)


def is_transient_provider_error(exc: BaseException) -> bool:
    """Whether an exception looks like a retryable provider failure."""
    from opto.optimizers.utils import LLMEmptyResponseError

    if isinstance(exc, LLMEmptyResponseError):
        return True
    if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError, ValueError)):
        return False  # a bug in our code, not the provider
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MESSAGE_MARKERS)


def _finish_reason(response: Any) -> Optional[str]:
    """Return the first choice's finish_reason, if the response exposes one."""
    choices = getattr(response, "choices", None)
    return getattr(choices[0], "finish_reason", None) if choices else None


def _is_empty_completion(response: Any) -> bool:
    """Whether a *successful* call came back with no usable message content."""
    choices = getattr(response, "choices", None)
    if choices is None:
        return False  # not a chat completion shape; leave it to the caller
    if not choices:
        return True
    return getattr(getattr(choices[0], "message", None), "content", None) is None


class RetryingLLM:
    """Retry transient provider failures so one bad response costs a call, not a run.

    Two experiment runs lost seeds to a single provider hiccup that surfaced as
    ``TypeError: argument of type 'NoneType' is not iterable`` — an empty completion
    under load, several frames from where it was produced. Retrying that specific class
    of failure is correct: it is transient, and the alternative is discarding an entire
    optimization arm.

    Retries are **counted**, not hidden. A provider failing systematically must show up
    as a large ``retries`` count rather than as a slow, quietly degraded run.
    """

    def __init__(self, inner: Any, attempts: int = 3, backoff_s: float = 2.0,
                 token_escalation: float = 2.0, max_token_ceiling: int = 32768) -> None:
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        self._inner = inner
        self.attempts = int(attempts)
        self.backoff_s = float(backoff_s)
        # finish_reason='length' is NOT transient: the completion hit the token budget and
        # an identical retry hits it again. Escalating the budget is the only retry that
        # can succeed, and it is what the error message tells the caller to do.
        self.token_escalation = float(token_escalation)
        self.max_token_ceiling = int(max_token_ceiling)
        self.retries = 0
        self.escalations = 0
        self.failures: List[str] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        last: Optional[BaseException] = None
        for attempt in range(self.attempts):
            try:
                response = self._inner(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                if not is_transient_provider_error(exc):
                    raise
                last = exc
            else:
                # An empty completion arrives as a SUCCESSFUL call returning
                # content=None, so the provider's own HTTP retry never sees it and the
                # failure only surfaces frames later inside the optimizer. Detect it
                # here, where retrying is still possible.
                if not _is_empty_completion(response):
                    return response
                from opto.optimizers.utils import LLMEmptyResponseError

                reason = _finish_reason(response)
                last = LLMEmptyResponseError(
                    f"provider returned a completion with no content (finish_reason={reason!r})"
                )
                if reason == "length":
                    current = kwargs.get("max_tokens")
                    if isinstance(current, int) and current < self.max_token_ceiling:
                        kwargs["max_tokens"] = min(
                            self.max_token_ceiling, int(current * self.token_escalation))
                        self.escalations += 1
            self.retries += 1
            self.failures.append(f"{type(last).__name__}: {str(last).splitlines()[0][:160]}")
            if attempt + 1 < self.attempts:
                time.sleep(self.backoff_s * (2 ** attempt))
        assert last is not None
        raise last

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "RetryingLLM":
        """Trainers deep-copy optimizers; share the client and the retry ledger."""
        memo[id(self)] = self
        return self


@dataclass(frozen=True)
class ObjectiveSplit:
    """Quality and cost reported separately instead of scalarized into one number.

    `internal:multiobjective_gsm8k` ships
    ``weights={'error': 1.0, 'tokens_in': 0.001, 'tokens_out': 0.001}``, which blends a
    rare discrete quality term with a continuous length term. Two consequences were
    measured rather than theorised:

    * 57% of the improvement in the one optimization run that beat the instrument's
      resolution came from token reduction, and the ranking between the two finalists —
      both at error 0.000 — was decided purely on length;
    * a ~200-token knowledge card costs 0.20 on that objective while the entire measured
      prompt effect is 0.075, so any knowledge-injecting arm loses by arithmetic before
      its content is considered at all.

    Cost is therefore split into what the *policy* generates (``tokens_out``) and what the
    *harness* supplies (``tokens_in``). Charging a policy for context it was handed is the
    mechanism that would make a knowledge-transfer experiment fail by construction.
    """

    quality: float
    quality_metric: str
    tokens_out: float
    tokens_in: float
    n_examples: int

    @property
    def policy_cost(self) -> float:
        """Cost attributable to the policy: what it chose to generate."""
        return self.tokens_out

    @property
    def total_tokens(self) -> float:
        return self.tokens_in + self.tokens_out

    def to_dict(self) -> Dict[str, Any]:
        return {"quality": self.quality, "quality_metric": self.quality_metric,
                "tokens_out": self.tokens_out, "tokens_in": self.tokens_in,
                "policy_cost": self.policy_cost, "total_tokens": self.total_tokens,
                "n_examples": self.n_examples}


def split_objective(observation: Mapping[str, Any]) -> Optional[ObjectiveSplit]:
    """Split one `evaluate_once` observation into separate quality and cost terms."""
    rows = observation.get("per_example") or []
    if not rows:
        return None
    metric, values = None, []
    for row in rows:
        key, value = _quality_from_metrics(row)
        if key is not None:
            metric = key
            values.append(value)
    if not values:
        return None
    return ObjectiveSplit(
        quality=st.mean(values),
        quality_metric=metric or "unknown",
        tokens_out=st.mean(float(r.get("tokens_out", 0.0)) for r in rows),
        tokens_in=st.mean(float(r.get("tokens_in", 0.0)) for r in rows),
        n_examples=len(rows),
    )


def compare_arms(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare two observations on quality and cost separately.

    Returns an explicit verdict rather than a single delta, because the interesting
    outcomes here are not orderings but *ties on quality with a difference in cost* —
    precisely the case a scalarized objective hides.
    """
    a, b = split_objective(baseline), split_objective(candidate)
    if a is None or b is None:
        return {"comparable": False, "reason": "quality metric unavailable for one arm"}
    d_quality = b.quality - a.quality
    d_cost = b.policy_cost - a.policy_cost
    if abs(d_quality) < 1e-9:
        verdict = ("equal quality, cheaper" if d_cost < 0 else
                   "equal quality, more expensive" if d_cost > 0 else "identical")
    elif d_quality > 0:
        verdict = "better quality, cheaper" if d_cost <= 0 else "better quality, more expensive"
    else:
        verdict = "worse quality, cheaper" if d_cost < 0 else "worse quality, more expensive"
    return {"comparable": True, "baseline": a.to_dict(), "candidate": b.to_dict(),
            "delta_quality": d_quality, "delta_policy_cost": d_cost, "verdict": verdict}
