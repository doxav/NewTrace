"""Tests for measurement certification (no LLM backend required).

The point of this module is to stop experiments running on instruments that cannot
resolve the effect being claimed, so the tests concentrate on the ways a certificate
could wrongly say "certified".
"""
from __future__ import annotations

import pytest

from opto.features.recursive_opt import measurement as M


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def test_wilson_interval_matches_known_values() -> None:
    lo, hi = M.wilson_interval(1, 12)
    assert (round(lo, 4), round(hi, 4)) == (0.0149, 0.3539)


def test_wilson_interval_is_wide_when_no_event_is_observed() -> None:
    """Zero observed events must not imply a zero rate."""
    lo, hi = M.wilson_interval(0, 6)
    assert lo == 0.0
    assert hi > 0.35, "0/6 is compatible with a rate near 40%; a point estimate would lie"


@pytest.mark.parametrize("trials", [0, -1])
def test_wilson_interval_is_maximally_uncertain_without_trials(trials: int) -> None:
    assert M.wilson_interval(0, trials) == (0.0, 1.0)


def test_required_n_and_resolvable_delta_are_inverse() -> None:
    sd, delta = 0.02, 0.05
    n = M.required_n(sd, delta)
    assert M.resolvable_delta(sd, n) <= delta
    assert M.resolvable_delta(sd, n - 1) > delta if n > 1 else True


def test_required_n_grows_quadratically_with_noise() -> None:
    assert M.required_n(0.04, 0.05) >= 3 * M.required_n(0.02, 0.05)


def test_required_n_rejects_a_nonpositive_target() -> None:
    with pytest.raises(ValueError, match="delta must be positive"):
        M.required_n(0.1, 0.0)


def test_resolvable_delta_rejects_a_nonpositive_n() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        M.resolvable_delta(0.1, 0)


# --------------------------------------------------------------------------- #
# temperature wrapper
# --------------------------------------------------------------------------- #
def test_fixed_temperature_injects_but_does_not_override() -> None:
    seen = {}

    def inner(*_a, **kw):
        seen.update(kw)
        return "ok"

    assert M.BoundedEvalLLM(inner, 0.0)(messages=[]) == "ok"
    assert seen["temperature"] == 0.0

    seen.clear()
    M.BoundedEvalLLM(inner, 0.0)(messages=[], temperature=0.7)
    assert seen["temperature"] == 0.7, "an explicit temperature must win"


def test_fixed_temperature_is_transparent_and_optional() -> None:
    class Inner:
        model_name = "m"

        def __call__(self, **kw):
            return kw

    wrapped = M.BoundedEvalLLM(Inner(), None)
    assert wrapped.model_name == "m"
    assert "temperature" not in wrapped(messages=[]), "None must leave calls untouched"


# --------------------------------------------------------------------------- #
# verdicts — each is a way a bad instrument could pass as good
# --------------------------------------------------------------------------- #
def _cert(**kw) -> M.TaskCertificate:
    base = {"task_id": "t", "model": None, "max_examples": 4, "repeats": 3,
            "temperature": 0.0}
    base.update(kw)
    return M.TaskCertificate(**base)


def test_instant_evaluation_is_broken_only_when_scoring_should_call_a_model() -> None:
    """Speed is a defect only if the task's scoring is supposed to invoke a model."""
    c = _cert(scores=[0.5, 0.5, 0.5], noise_sd=0.0, mean_eval_seconds=6e-06, calls_llm=True)
    M._apply_verdict(c, target_delta=0.05, target_n=5)
    assert c.verdict == "broken"
    assert "no model was called" in c.reasons[0]


def test_instant_evaluation_is_fine_for_an_llm_free_task() -> None:
    """A deterministic evaluator SHOULD return instantly, and has no sampling noise.

    Flagging that as broken is what made internal:code_param, internal:multi_param and
    veribench look dead when they were the quietest surfaces available.
    """
    c = _cert(scores=[0.5, 0.5, 0.5], noise_sd=0.0, mean_eval_seconds=0.007,
              calls_llm=False, surface="code",
              quality_metric="accuracy", quality_rate=0.5, quality_ci=(0.3, 0.7))
    M._apply_verdict(c, target_delta=0.05, target_n=5)
    assert c.verdict == "certified"
    assert c.usable is True
    assert c.resolvable_at(5) == 0.0, "a deterministic evaluator resolves any effect"


# --------------------------------------------------------------------------- #
# surface detection — injecting prose into a code parameter is corruption
# --------------------------------------------------------------------------- #
class _Node:
    def __init__(self, data, name="p"):
        self.data, self.name = data, name


class _Param:
    def __init__(self, nodes, llm=False, system_prompt=None):
        self._nodes = nodes
        if llm:
            self.llm = object()
        if system_prompt is not None:
            self.system_prompt = _Node(system_prompt, "system_prompt")

    def parameters(self):
        return self._nodes


@pytest.mark.parametrize("data,expected", [
    ("def f(x): return x", "code"),
    ("class A:\n    pass", "code"),
    ("-- placeholder: Lean 4 translation pending", "code"),
    ("1.0", "numeric"),
    ("-3", "numeric"),
    ("Answer the question based on the context.", "prose"),
    ("", "unknown"),
])
def test_detect_surface_classifies_the_trainable_parameter(data: str, expected: str) -> None:
    assert M.detect_surface({"param": _Param([_Node(data)])}).kind == expected


def test_detect_surface_prefers_an_explicit_system_prompt() -> None:
    s = M.detect_surface({"param": _Param([_Node("def f(): pass")], llm=True,
                                          system_prompt="You are helpful.")})
    assert s.kind == "prose" and s.calls_llm is True


def test_detect_surface_reports_whether_scoring_calls_a_model() -> None:
    assert M.detect_surface({"param": _Param([_Node("1.0")])}).calls_llm is False
    assert M.detect_surface({"param": _Param([_Node("hi there")], llm=True)}).calls_llm is True


def test_detect_surface_handles_a_bare_parameter_node() -> None:
    """llm4ad and bbeh expose a ParameterNode directly, not a Module."""
    assert M.detect_surface({"param": _Node("def solve(): pass")}).kind == "code"


def test_detect_surface_survives_a_missing_param() -> None:
    assert M.detect_surface({}).kind == "unknown"


@pytest.mark.parametrize("kind,ok", [("prose", True), ("code", False),
                                     ("numeric", False), ("unknown", False)])
def test_only_a_prose_surface_accepts_a_prose_probe(kind: str, ok: bool) -> None:
    assert M.TaskSurface(kind, False).accepts_prose_probe is ok


def test_evaluate_once_refuses_to_corrupt_a_non_prose_parameter(monkeypatch) -> None:
    """The bug that made five healthy tasks look broken."""
    from opto.features.recursive_opt import tracebench as TB

    class _Adapter:
        def __init__(self, *a, **kw):
            pass

        def _load_bundle(self, task_id, fresh=False):
            return {"param": _Param([_Node("def f(x): return x")])}

        def _apply_starting_artifact(self, *a, **kw):  # pragma: no cover - must not run
            raise AssertionError("prose must never be injected into a code parameter")

    monkeypatch.setattr(TB, "TraceBenchTaskAdapter", _Adapter)
    out = M.evaluate_once("internal:code_param", artifact="Answer directly.")

    assert out["valid"] is False
    assert "refused to inject a prose artifact" in out["error"]
    assert out["surface"] == "code"


def test_constant_zero_after_real_compute_is_degenerate_not_broken() -> None:
    """hf:qasper scored 0.0 three times — but after 204s, so a model DID run.

    Blaming the task would be wrong: the likely cause is the evaluation bounds
    truncating answers. The verdict must be unusable but the reason must be honest
    about which side is at fault.
    """
    c = _cert(scores=[0.0, 0.0, 0.0], noise_sd=0.0, mean_eval_seconds=204.0, max_tokens=512)
    M._apply_verdict(c, target_delta=0.05, target_n=5)
    assert c.verdict == "degenerate"
    assert c.usable is False
    assert "max_tokens=512" in c.reasons[0]


def test_failure_sentinel_is_rejected_despite_zero_variance() -> None:
    """llm4ad returns -1e6 when the candidate program fails to run.

    A constant failure code has zero noise. Reading that as "quiet, therefore
    certified" is exactly the mistake this module exists to prevent — and it is the
    mistake the first version of this code actually made.
    """
    c = _cert(scores=[-1e6, -1e6, -1e6], noise_sd=0.0, mean_eval_seconds=0.16)
    M._apply_verdict(c, target_delta=0.05, target_n=5)
    assert c.verdict == "broken"
    assert c.usable is False
    assert "sentinel" in c.reasons[0]


def test_sentinel_is_detected_even_when_only_some_repeats_fail() -> None:
    c = _cert(scores=[0.5, -1e6, 0.5], noise_sd=0.47, mean_eval_seconds=5.0)
    M._apply_verdict(c, target_delta=0.05, target_n=5)
    assert c.verdict == "broken"
    assert "1/3 repeats" in c.reasons[0]


@pytest.mark.parametrize("score", [1e6, -1e6, float("1e9")])
def test_implausible_magnitudes_are_never_certified(score: float) -> None:
    c = _cert(scores=[score] * 3, noise_sd=0.0, mean_eval_seconds=5.0)
    M._apply_verdict(c, target_delta=0.05, target_n=5)
    assert c.usable is False


def test_ceiling_saturation_is_rejected() -> None:
    """A task the model already solves has no headroom to demonstrate improvement."""
    c = _cert(scores=[1.0, 1.0, 1.0], noise_sd=0.0, mean_eval_seconds=5.0,
              quality_metric="accuracy", quality_rate=1.0, quality_ci=(1.0, 1.0))
    M._apply_verdict(c, target_delta=0.05, target_n=5)
    assert c.verdict == "saturated"
    assert "ceiling" in c.reasons[0]


def test_floor_with_an_unresponsive_evaluator_is_degenerate() -> None:
    c = _cert(scores=[0.0, 0.0, 0.0], noise_sd=0.0, mean_eval_seconds=5.0, live=False,
              quality_metric="accuracy", quality_rate=0.0, quality_ci=(0.0, 0.0))
    M._apply_verdict(c, target_delta=0.05, target_n=5)
    assert c.verdict == "degenerate"


def test_floor_with_a_responsive_evaluator_is_the_ideal_starting_point() -> None:
    """veribench ships a Lean placeholder scoring 0.0 — that is headroom, not death."""
    c = _cert(scores=[0.0, 0.0, 0.0], noise_sd=0.0, mean_eval_seconds=0.8, live=True,
              liveness_spread=0.1, surface="code", calls_llm=False,
              quality_metric="score", quality_rate=0.0, quality_ci=(0.0, 0.4))
    M._apply_verdict(c, target_delta=0.05, target_n=5)
    assert c.verdict == "certified"
    assert any("maximum headroom" in r for r in c.reasons)


def test_a_solved_baseline_is_saturated_even_with_a_wide_interval() -> None:
    """internal:code_param and bbeh both score 1.0 out of the box: nothing to optimize."""
    c = _cert(scores=[1.0, 1.0, 1.0], noise_sd=0.0, mean_eval_seconds=0.01, live=True,
              calls_llm=False, quality_metric="score", quality_rate=1.0, quality_ci=(0.44, 1.0))
    M._apply_verdict(c, target_delta=0.05, target_n=5)
    assert c.verdict == "saturated"
    assert any("no headroom" in r for r in c.reasons)


def test_a_large_but_real_score_is_not_a_sentinel() -> None:
    """llm4ad:online_bin_packing legitimately scores -2091.8; -1e6 is its failure code."""
    assert M._is_sentinel_score(-1e6) is True
    assert M._is_sentinel_score(-1e9) is True
    assert M._is_sentinel_score(float("-inf")) is True
    assert M._is_sentinel_score(-2091.8) is False
    assert M._is_sentinel_score(0.5) is False

    c = _cert(scores=[-2091.8, -2091.8, -2091.8], noise_sd=0.0, mean_eval_seconds=0.16,
              live=True, liveness_spread=997908.2, surface="code", calls_llm=False,
              quality_metric="score", quality_rate=-2091.8)
    M._apply_verdict(c, target_delta=0.05, target_n=5)
    assert c.verdict == "certified", "a real cost objective must not be read as a sentinel"


def test_noise_above_the_target_effect_is_rejected_with_the_required_n() -> None:
    """This is the measured gsm8k situation: sd 0.19 cannot resolve a 0.05 effect."""
    c = _cert(scores=[-0.20, -0.23, -0.62], noise_sd=0.1909, mean_eval_seconds=17.0,
              quality_metric="accuracy", quality_rate=0.9, quality_ci=(0.6, 0.99))
    M._apply_verdict(c, target_delta=0.05, target_n=5)
    assert c.verdict == "too_noisy"
    # the message must state the sample size that WOULD work, not just refuse
    assert f"needs n={M.required_n(0.1909, 0.05)}" in c.reasons[0]
    assert M.required_n(0.1909, 0.05) > 100, "this noise needs a 3-digit n for a 0.05 effect"
    assert c.resolvable_at(5) > 0.2, "at n=5 only effects above 0.2 are detectable"


def test_a_quiet_unsaturated_task_is_certified() -> None:
    c = _cert(scores=[0.50, 0.51, 0.49], noise_sd=0.008, mean_eval_seconds=5.0,
              quality_metric="accuracy", quality_rate=0.5, quality_ci=(0.3, 0.7))
    M._apply_verdict(c, target_delta=0.05, target_n=5)
    assert c.verdict == "certified"
    assert c.usable is True
    assert c.resolvable_at(5) < 0.05


def test_usable_is_false_for_every_non_certified_verdict() -> None:
    for verdict in ("broken", "saturated", "too_noisy", "unknown"):
        assert _cert(verdict=verdict).usable is False


# --------------------------------------------------------------------------- #
# failures are never scores
# --------------------------------------------------------------------------- #
def test_evaluate_once_reports_failure_instead_of_scoring_zero(monkeypatch) -> None:
    from opto.features.recursive_opt import tracebench as TB

    class _Boom:
        def __init__(self, *a, **kw):
            raise RuntimeError("provider exploded")

    monkeypatch.setattr(TB, "TraceBenchTaskAdapter", _Boom)
    out = M.evaluate_once("some:task", max_examples=2)

    assert out["valid"] is False
    assert "provider exploded" in out["error"]
    assert "score" not in out, "a failed evaluation must not carry a score"


def test_certify_task_marks_a_wholly_failing_task_broken(monkeypatch) -> None:
    monkeypatch.setattr(M, "evaluate_once",
                        lambda *a, **kw: {"valid": False, "error": "E: nope", "seconds": 0.5})
    cert = M.certify_task("some:task", repeats=2)

    assert cert.verdict == "broken"
    assert cert.usable is False
    assert cert.scores == []
    assert len(cert.failures) == 2


def test_certify_task_excludes_failures_from_the_mean(monkeypatch) -> None:
    """A failed repeat must shrink the sample, never be imputed as a value."""
    calls = {"n": 0}

    def fake(*_a, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"valid": False, "error": "E: transient", "seconds": 1.0}
        return {"valid": True, "score": 0.5, "per_example": [{"accuracy": 0.5}],
                "seconds": 5.0, "task_id": "t", "dataset": "d", "n_examples": 2}

    monkeypatch.setattr(M, "evaluate_once", fake)
    cert = M.certify_task("some:task", repeats=3)

    assert cert.scores == [0.5, 0.5]
    assert cert.mean_score == 0.5
    assert len(cert.failures) == 1


def test_certify_task_requires_enough_repeats_for_a_noise_estimate() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        M.certify_task("some:task", repeats=1)


def test_certify_task_never_exceeds_the_safe_worker_cap(monkeypatch) -> None:
    """Providers throttle under load; unbounded fan-out manufactures failures."""
    seen = {}
    real = M.ThreadPoolExecutor

    def spy(max_workers=None, **kw):
        seen["max_workers"] = max_workers
        return real(max_workers=max_workers, **kw)

    monkeypatch.setattr(M, "ThreadPoolExecutor", spy)
    monkeypatch.setattr(M, "evaluate_once",
                        lambda *a, **kw: {"valid": True, "score": 0.1, "per_example": [],
                                          "seconds": 1.0})
    M.certify_task("t", repeats=64, max_workers=1000)
    assert seen["max_workers"] <= M.MAX_SAFE_WORKERS


def test_format_certificates_renders_every_verdict() -> None:
    certs = {"a": _cert(task_id="a", verdict="certified", mean_score=0.5, noise_sd=0.01),
             "b": _cert(task_id="b", verdict="broken", reasons=["no model was called"])}
    text = M.format_certificates(certs)
    assert "certified" in text and "broken" in text and "no model was called" in text


# --------------------------------------------------------------------------- #
# request bounding — an unbounded request is not a measurement
# --------------------------------------------------------------------------- #
def test_bounded_eval_llm_injects_all_three_bounds_by_default() -> None:
    seen = {}

    def inner(**kw):
        seen.update(kw)
        return "ok"

    M.BoundedEvalLLM(inner)(messages=[])
    assert seen["temperature"] == M.DEFAULT_EVAL_TEMPERATURE
    assert seen["max_tokens"] == M.DEFAULT_MAX_TOKENS
    assert seen["timeout"] == M.DEFAULT_REQUEST_TIMEOUT


def test_bounded_eval_llm_lets_the_caller_win() -> None:
    seen = {}

    def inner(**kw):
        seen.update(kw)
        return "ok"

    M.BoundedEvalLLM(inner)(messages=[], temperature=0.9, max_tokens=8, timeout=1)
    assert (seen["temperature"], seen["max_tokens"], seen["timeout"]) == (0.9, 8, 1)


def test_bounded_eval_llm_can_be_fully_disabled() -> None:
    seen = {}

    def inner(**kw):
        seen.update(kw)
        return "ok"

    M.BoundedEvalLLM(inner, None, None, None)(messages=[])
    assert set(seen) == {"messages"}, "explicit None must leave the call untouched"


def test_default_temperature_is_the_measured_low_variance_setting() -> None:
    """0.0 stalled this model on degenerate repetition; 0.2 was measured at 3.2x quieter."""
    assert M.DEFAULT_EVAL_TEMPERATURE == 0.2
    assert M.DEFAULT_MAX_TOKENS is not None, "an uncapped evaluation can fail to terminate"
    assert M.DEFAULT_REQUEST_TIMEOUT is not None


def test_certificate_records_the_bounds_it_was_measured_under() -> None:
    """max_tokens changed the observed qasper score, so it is part of the triple."""
    c = _cert(max_tokens=512, request_timeout=60)
    assert c.to_dict()["max_tokens"] == 512
    assert c.to_dict()["request_timeout"] == 60


def test_certificate_reports_resolution_at_several_sample_sizes() -> None:
    c = _cert(noise_sd=0.0131)
    d = c.to_dict()
    assert d["resolvable_at_5"] > d["resolvable_at_10"], "more seeds resolve smaller effects"
    assert d["resolvable_at_5"] < 0.02, "the measured quiet setting resolves a 0.02 effect at n=5"


# --------------------------------------------------------------------------- #
# provider resilience — one bad response must cost a call, not a run
# --------------------------------------------------------------------------- #
def test_empty_completion_raises_a_diagnosable_error_not_a_typeerror() -> None:
    """The crash that cost seeds in two experiment runs.

    A provider returning message.content=None used to reach `if "TERMINATE" in response`
    as a bare None, producing `TypeError: argument of type 'NoneType' is not iterable`
    several frames from the cause.
    """
    from types import SimpleNamespace

    from opto.optimizers.utils import LLMEmptyResponseError, extract_response_content

    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=None), finish_reason="length")])
    with pytest.raises(LLMEmptyResponseError, match="no usable content"):
        extract_response_content(response, context="probe")

    # the finish_reason is the diagnostic that distinguishes the causes
    try:
        extract_response_content(response, context="probe")
    except LLMEmptyResponseError as exc:
        assert "length" in str(exc)
        assert "max_tokens" in str(exc)


def test_extract_response_content_rejects_a_response_with_no_choices() -> None:
    from types import SimpleNamespace

    from opto.optimizers.utils import LLMEmptyResponseError, extract_response_content

    with pytest.raises(LLMEmptyResponseError, match="no choices"):
        extract_response_content(SimpleNamespace(choices=[]), context="probe")


def test_extract_response_content_returns_real_content() -> None:
    from types import SimpleNamespace

    from opto.optimizers.utils import extract_response_content

    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="hello"), finish_reason="stop")])
    assert extract_response_content(response) == "hello"


@pytest.mark.parametrize("exc,transient", [
    (OSError("[Errno 104] Connection reset by peer"), True),
    (RuntimeError("429 Too Many Requests"), True),
    (RuntimeError("503 Service Unavailable"), True),
    (RuntimeError("request timed out"), True),
    (TypeError("argument of type 'NoneType' is not iterable"), False),
    (KeyError("missing"), False),
    (ValueError("bad config"), False),
])
def test_only_provider_failures_are_treated_as_transient(exc, transient) -> None:
    """Retrying a bug in our own code only delays the report and hides the stack."""
    assert M.is_transient_provider_error(exc) is transient


def test_empty_completion_is_transient() -> None:
    from opto.optimizers.utils import LLMEmptyResponseError

    assert M.is_transient_provider_error(LLMEmptyResponseError("no content")) is True


def test_retrying_llm_recovers_and_counts_the_retries() -> None:
    from opto.optimizers.utils import LLMEmptyResponseError

    calls = {"n": 0}

    def flaky(**_kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMEmptyResponseError("no content")
        return "ok"

    llm = M.RetryingLLM(flaky, attempts=3, backoff_s=0.0)
    assert llm(messages=[]) == "ok"
    assert llm.retries == 2, "retries must be counted, not hidden"
    assert len(llm.failures) == 2


def test_retrying_llm_reraises_after_exhausting_attempts() -> None:
    from opto.optimizers.utils import LLMEmptyResponseError

    def always_empty(**_kw):
        raise LLMEmptyResponseError("no content")

    llm = M.RetryingLLM(always_empty, attempts=2, backoff_s=0.0)
    with pytest.raises(LLMEmptyResponseError):
        llm(messages=[])
    assert llm.retries == 2


def test_retrying_llm_does_not_retry_a_programming_error() -> None:
    calls = {"n": 0}

    def buggy(**_kw):
        calls["n"] += 1
        raise TypeError("argument of type 'NoneType' is not iterable")

    llm = M.RetryingLLM(buggy, attempts=5, backoff_s=0.0)
    with pytest.raises(TypeError):
        llm(messages=[])
    assert calls["n"] == 1, "a bug must surface immediately, with its original stack"
    assert llm.retries == 0


def test_retrying_llm_is_transparent_and_deepcopy_shares_the_ledger() -> None:
    import copy

    class Inner:
        model_name = "m"

        def __call__(self, **kw):
            return "ok"

    llm = M.RetryingLLM(Inner())
    assert llm.model_name == "m"
    assert copy.deepcopy(llm) is llm, "trainers deep-copy optimizers; keep one ledger"


def test_retrying_llm_requires_at_least_one_attempt() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        M.RetryingLLM(lambda **kw: "x", attempts=0)


def test_truncated_completion_escalates_the_token_budget_instead_of_repeating() -> None:
    """finish_reason='length' is deterministic — an identical retry fails identically.

    Escalating the budget is the only retry that can succeed, and it is exactly what the
    error message instructs the caller to do. Probe F lost a seed to this.
    """
    from types import SimpleNamespace

    tried = []

    def truncating(**kw):
        tried.append(kw.get("max_tokens"))
        content = "done" if kw.get("max_tokens", 0) >= 2048 else None
        reason = "stop" if content else "length"
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=reason)])

    llm = M.RetryingLLM(truncating, attempts=4, backoff_s=0.0)
    out = llm(messages=[], max_tokens=512)

    assert tried == [512, 1024, 2048], "the budget must grow, not repeat"
    assert out.choices[0].message.content == "done"
    assert llm.escalations == 2


def test_token_escalation_respects_a_ceiling() -> None:
    from types import SimpleNamespace

    tried = []

    def always_truncated(**kw):
        tried.append(kw.get("max_tokens"))
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=None), finish_reason="length")])

    from opto.optimizers.utils import LLMEmptyResponseError

    llm = M.RetryingLLM(always_truncated, attempts=5, backoff_s=0.0, max_token_ceiling=2048)
    with pytest.raises(LLMEmptyResponseError):
        llm(messages=[], max_tokens=1024)
    assert max(t for t in tried if t) <= 2048


def test_a_non_length_empty_completion_is_retried_without_escalation() -> None:
    """A filtered or load-shed completion is transient; the budget was not the problem."""
    from types import SimpleNamespace

    calls = {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        content = "ok" if calls["n"] >= 2 else None
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason="content_filter")])

    llm = M.RetryingLLM(flaky, attempts=3, backoff_s=0.0)
    assert llm(messages=[], max_tokens=512).choices[0].message.content == "ok"
    assert llm.escalations == 0, "only a length truncation justifies raising the budget"
    assert llm.retries == 1


# --------------------------------------------------------------------------- #
# A2 — quality and cost must be reported separately
# --------------------------------------------------------------------------- #
def _obs(error, tokens_in, tokens_out, n=4):
    return {"per_example": [{"error": error, "tokens_in": tokens_in,
                             "tokens_out": tokens_out}] * n}


def test_split_objective_separates_quality_from_cost() -> None:
    s = M.split_objective(_obs(0.083, 120.0, 153.5))
    assert s.quality == pytest.approx(0.917), "error is reported as higher-is-better quality"
    assert s.tokens_out == 153.5 and s.tokens_in == 120.0
    assert s.total_tokens == 273.5


def test_policy_cost_excludes_context_supplied_by_the_harness() -> None:
    """Charging a policy for context it was handed makes knowledge injection self-defeating.

    A ~200-token card costs 0.20 on the shipped objective while the whole measured prompt
    effect is 0.075 — so every knowledge arm would lose by arithmetic (§11.4).
    """
    lean = M.split_objective(_obs(0.0, 100.0, 50.0))
    with_card = M.split_objective(_obs(0.0, 300.0, 50.0))  # 200 tokens of injected context

    assert with_card.policy_cost == lean.policy_cost, (
        "injected context must not change the policy's cost")
    assert with_card.total_tokens > lean.total_tokens, "but it is still reported"


def test_compare_arms_names_the_tie_a_scalarized_score_would_hide() -> None:
    """B's actual finding: both finalists at error 0.000, decided purely on length."""
    result = M.compare_arms(_obs(0.0, 130.0, 70.2), _obs(0.0, 130.0, 32.7))

    assert result["comparable"] is True
    assert result["delta_quality"] == pytest.approx(0.0)
    assert result["delta_policy_cost"] == pytest.approx(-37.5)
    assert result["verdict"] == "equal quality, cheaper"


@pytest.mark.parametrize("base,cand,verdict", [
    ((0.5, 100.0), (0.5, 50.0), "equal quality, cheaper"),
    ((0.5, 100.0), (0.5, 150.0), "equal quality, more expensive"),
    ((0.5, 100.0), (0.2, 50.0), "better quality, cheaper"),
    ((0.5, 100.0), (0.2, 150.0), "better quality, more expensive"),
    ((0.2, 100.0), (0.5, 50.0), "worse quality, cheaper"),
    ((0.2, 100.0), (0.5, 150.0), "worse quality, more expensive"),
])
def test_compare_arms_covers_every_quality_cost_quadrant(base, cand, verdict) -> None:
    result = M.compare_arms(_obs(base[0], 10.0, base[1]), _obs(cand[0], 10.0, cand[1]))
    assert result["verdict"] == verdict


def test_split_objective_returns_none_without_a_quality_metric() -> None:
    assert M.split_objective({"per_example": [{"tokens_out": 5.0}]}) is None
    assert M.split_objective({"per_example": []}) is None


def test_compare_arms_reports_incomparability_rather_than_guessing() -> None:
    result = M.compare_arms({"per_example": [{"tokens_out": 1.0}]}, _obs(0.0, 1.0, 1.0))
    assert result["comparable"] is False


# --------------------------------------------------------------------------- #
# effect size must be expressed in the task's own units
# --------------------------------------------------------------------------- #
def test_effective_target_keeps_the_absolute_floor_for_unit_scale_scores() -> None:
    assert M.effective_target_delta(-0.23, 0.05, 0.01) == 0.05


def test_effective_target_scales_with_a_large_objective() -> None:
    """A bin-packing cost of -2092.6 with noise 0.8 is 0.04% relative — very quiet.

    An absolute target of 0.05 rejected it as 'too noisy' purely because of units.
    """
    assert M.effective_target_delta(-2092.6, 0.05, 0.01) == pytest.approx(20.926)
    assert M.effective_target_delta(-6167594.0, 0.05, 0.01) == pytest.approx(61675.94)


def test_effective_target_falls_back_without_a_score() -> None:
    assert M.effective_target_delta(None, 0.05, 0.01) == 0.05


def test_a_quiet_large_scale_objective_is_certified_not_rejected() -> None:
    c = _cert(scores=[-2092.6, -2091.8, -2093.0], noise_sd=0.8, mean_score=-2092.6,
              mean_eval_seconds=1.8, live=True, surface="code", calls_llm=False)
    M._apply_verdict(c, target_delta=0.05, target_n=5, target_delta_relative=0.01)
    assert c.verdict == "certified"
    assert "% of |score|" in c.reasons[-1]


def test_a_genuinely_noisy_large_scale_objective_is_still_rejected() -> None:
    c = _cert(scores=[-2000.0, -3000.0, -1000.0], noise_sd=816.0, mean_score=-2000.0,
              mean_eval_seconds=1.8, live=True, surface="code", calls_llm=False)
    M._apply_verdict(c, target_delta=0.05, target_n=5, target_delta_relative=0.01)
    assert c.verdict == "too_noisy", "40% relative noise must still fail"


# --------------------------------------------------------------------------- #
# concurrency — a noise floor measured under the wrong conditions is not a floor
# --------------------------------------------------------------------------- #
def test_sequential_zero_noise_is_not_reported_as_a_usable_floor() -> None:
    """The flaw that produced a false positive.

    llm4ad:online_bin_packing measures 0.0 noise sequentially and 3.15 at concurrency 8,
    because its evaluator runs the candidate under a time budget. Certifying it
    sequentially reported a zero floor, and a +4.8 "improvement" was then read as real
    when it sat inside a 10.8 concurrency range.
    """
    c = _cert(scores=[1.0, 1.0], noise_sd=0.0, mean_eval_seconds=1.0, live=True,
              concurrency=1, calls_llm=False, surface="code")
    M._apply_verdict(c, target_delta=0.05, target_n=5)
    assert any("SEQUENTIALLY" in r for r in c.reasons)
    assert any("re-certify at the concurrency you intend" in r for r in c.reasons)


def test_zero_noise_under_real_concurrency_is_a_usable_floor() -> None:
    c = _cert(scores=[1.0, 1.0], noise_sd=0.0, mean_eval_seconds=1.0, live=True,
              concurrency=8, calls_llm=False, surface="code")
    M._apply_verdict(c, target_delta=0.05, target_n=5)
    assert any("concurrency 8" in r for r in c.reasons)
    assert c.verdict == "certified"


def test_certify_task_records_the_concurrency_it_measured_under(monkeypatch) -> None:
    monkeypatch.setattr(M, "evaluate_once",
                        lambda *a, **kw: {"valid": True, "score": 0.5, "per_example": [],
                                          "seconds": 1.0})
    cert = M.certify_task("t", repeats=4, concurrency=3, check_liveness=False)
    assert cert.concurrency == 3
    assert cert.to_dict()["concurrency"] == 3


def test_concurrency_is_bounded_by_the_safe_worker_cap(monkeypatch) -> None:
    seen = {}
    real = M.ThreadPoolExecutor

    def spy(max_workers=None, **kw):
        seen["max_workers"] = max_workers
        return real(max_workers=max_workers, **kw)

    monkeypatch.setattr(M, "ThreadPoolExecutor", spy)
    monkeypatch.setattr(M, "evaluate_once",
                        lambda *a, **kw: {"valid": True, "score": 0.1, "per_example": [],
                                          "seconds": 1.0})
    M.certify_task("t", repeats=64, concurrency=999, check_liveness=False)
    assert seen["max_workers"] <= M.MAX_SAFE_WORKERS
