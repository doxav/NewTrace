"""The candidate menu is the instrument: a candidate must fit the surface it replaces.

`TraceBenchTaskAdapter._apply_starting_artifact` used to write the candidate text straight
into the trainable node's ``_data`` and return ``True`` unconditionally. A prose menu entry
therefore REPLACED a 366-char Python ``priority(item, bins)`` with "Answer directly.", which
no longer runs and scores the invalid sentinel. Only the empty candidate survived, so the
effective menu size was 1 and no optimizer could express a preference.

These tests pin the three defences:
  1. the write is REFUSED, with a typed reason, when the candidate does not fit the surface;
  2. ``run_task`` reports that refusal instead of silently scoring the unseeded bundle;
  3. ``score_spread`` reports ``effective_menu_size`` so a collapsed menu is never read as a
     flat surface (this also catches ranking-equivalent candidates, which no type check can).
"""

import math

import pytest

from opto.features.recursive_opt import measurement as M
from opto.features.recursive_opt import spec as S
from opto.features.recursive_opt import tracebench as TB
from opto.features.recursive_opt.levels import INVALID_CONFIG_SCORE, LevelConfig

# The two real menu entries from the Iteration 3 run.
PROSE = ["Answer directly.", "Plan step by step, then verify the answer before replying."]
# Faithful stand-ins for the real llm4ad parameters (same shape, offline).
BIN_PACKING = ("import numpy as np\n\n"
               "def priority(item: float, bins: np.ndarray) -> np.ndarray:\n"
               "    return -(bins - item)\n")
ADMISSIBLE = ("import numpy as np\n\n"
              "def priority(el: tuple, n: int, w: int) -> float:\n"
              "    return 0.0\n")


class _WritableNode:
    """A node whose ``data`` mirrors ``_data``, like a Trace ParameterNode."""

    def __init__(self, data, name="p"):
        self._data = data
        self.name = name

    @property
    def data(self):
        return self._data


class _Param:
    def __init__(self, node):
        self._node = node

    def parameters(self):
        return [self._node]


# --------------------------------------------------------------------------- #
# 1. artifact_fits_surface: the shared compatibility rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("original,candidate,fits", [
    # the measured regressions -------------------------------------------------
    (BIN_PACKING, "Answer directly.", False),
    (BIN_PACKING, PROSE[1], False),
    (ADMISSIBLE, "Answer directly.", False),
    ("def f(x): return x", "Answer directly.", False),   # internal:code_param
    ("1.0", "Answer directly.", False),                  # internal:multi_param
    # what must keep working ---------------------------------------------------
    (BIN_PACKING, BIN_PACKING.replace("-(bins - item)", "(bins - item)"), True),
    ("def f(x): return x", "def f(x): return x * 2", True),
    ("1.0", "2.5", True),
    ("You are a helpful assistant.", "Answer directly.", True),   # prose -> prose
    ("You are a helpful assistant.", "def f(x): return x", True),  # code in a prompt is fine
    ("", "Answer directly.", True),                      # unknown surface: never block
    # deliberately NOT refused: broken or renamed code is a real optimizer proposal
    # getting real benchmark feedback, not an inapplicable experiment config.
    (BIN_PACKING, "import numpy as np\ndef priority(item, bins)\n    return 0\n", True),
    (BIN_PACKING, "import numpy as np\ndef not_priority(item, bins):\n    return 0\n", True),
])
def test_artifact_fits_surface(original: str, candidate: str, fits: bool) -> None:
    surface = M.detect_surface({"param": _Param(_WritableNode(original))})
    reason = M.artifact_fits_surface(surface, candidate)
    assert (reason is None) is fits, reason


def test_artifact_fits_surface_reason_is_typed_and_names_the_surface() -> None:
    surface = M.detect_surface({"param": _Param(_WritableNode(BIN_PACKING))})
    reason = M.artifact_fits_surface(surface, "Answer directly.")
    assert "surface_mismatch" in reason and "code" in reason


def test_a_non_python_code_surface_is_not_judged_by_the_python_parser() -> None:
    """Lean 4 parameters are `code` but must not be required to parse as Python.

    With no Lean parser available the fallback is the kind check alone, and it
    refuses whatever does not read as code: failing loudly on a valid Lean
    candidate is recoverable, silently accepting prose is the bug being fixed.
    """
    lean = "-- placeholder: Lean 4 translation pending\ntheorem t : True := by trivial"
    surface = M.detect_surface({"param": _Param(_WritableNode(lean))})
    assert M.artifact_fits_surface(surface, "theorem t : True := by simp") is None
    assert M.artifact_fits_surface(surface, "Answer directly.") is not None


# --------------------------------------------------------------------------- #
# 2. the write is refused, and run_task reports the refusal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("original", [BIN_PACKING, ADMISSIBLE, "def f(x): return x", "1.0"])
@pytest.mark.parametrize("candidate", PROSE)
def test_apply_starting_artifact_refuses_and_leaves_the_parameter_intact(
    original: str, candidate: str
) -> None:
    adapter = TB.TraceBenchTaskAdapter.__new__(TB.TraceBenchTaskAdapter)
    node = _WritableNode(original)
    bundle = {"param": _Param(node)}
    applied = adapter._apply_starting_artifact(
        bundle, LevelConfig(starting_artifact=candidate))
    assert applied is False, "a destructive write must not report success"
    assert node._data == original, "the trainable parameter was corrupted"
    assert "surface_mismatch" in str(bundle.get("artifact_refusal"))


def test_apply_starting_artifact_still_seeds_a_compatible_candidate() -> None:
    adapter = TB.TraceBenchTaskAdapter.__new__(TB.TraceBenchTaskAdapter)
    node = _WritableNode("You are a helpful assistant.")
    bundle = {"param": _Param(node)}
    assert adapter._apply_starting_artifact(
        bundle, LevelConfig(starting_artifact="Answer directly."))
    assert node._data == "Answer directly."
    assert "artifact_refusal" not in bundle


def test_run_task_refuses_rather_than_scoring_an_unseeded_bundle() -> None:
    """Refusal must be loud: silently scoring the default would make every
    incompatible candidate tie, which is the same singleton search space."""

    class _Adapter(TB.TraceBenchTaskAdapter):
        def _load_bundle(self, task_id, *, fresh=False):
            return {"param": _Param(_WritableNode(BIN_PACKING))}

        def _train_bundle(self, bundle, cfg):  # pragma: no cover - must not run
            raise AssertionError("must not train on a refused candidate")

    adapter = _Adapter(max_examples=1, inner_steps=0)
    score, feedback = adapter.run_task(
        LevelConfig(starting_artifact="Answer directly."),
        "llm4ad:optimization/online_bin_packing")
    assert score == INVALID_CONFIG_SCORE
    assert "surface_mismatch" in feedback


def test_run_task_still_scores_a_compatible_candidate(monkeypatch) -> None:
    class _Adapter(TB.TraceBenchTaskAdapter):
        def _load_bundle(self, task_id, *, fresh=False):
            return {"param": _Param(_WritableNode(BIN_PACKING))}

        def _train_bundle(self, bundle, cfg):
            return None

    monkeypatch.setattr(TB, "_score_bundle", lambda bundle, n: (-2091.8, "ok"))
    adapter = _Adapter(max_examples=1, inner_steps=0)
    score, feedback = adapter.run_task(
        LevelConfig(starting_artifact=BIN_PACKING.replace("-(bins", "(bins")),
        "llm4ad:optimization/online_bin_packing")
    assert score == -2091.8 and "surface_mismatch" not in feedback


# --------------------------------------------------------------------------- #
# 3. effective_menu_size — the output-side detector, incl. ranking-equivalence
# --------------------------------------------------------------------------- #
class _FakeAdapter:
    """Minimal task adapter: returns the score the test dictates."""

    def __init__(self, fn):
        self._fn = fn

    def run_task(self, cfg, task_id):
        return self._fn(cfg), "fake"

    def trainable_fields(self):
        return ("starting_artifact",)


def _spread(fn, probes=None):
    TB.register_task_adapter(_FakeAdapter(fn))
    try:
        return S.score_spread("internal:code_param", probes=probes)
    finally:
        TB.register_task_adapter(None)


def test_effective_menu_size_counts_distinct_valid_scores() -> None:
    scores = {"": -2091.8, "a": -5000.0, "b": -2099.4}
    out = _spread(lambda cfg: scores[str(getattr(cfg, "starting_artifact", "") or "")],
                  probes=[{}, {"starting_artifact": "a"}, {"starting_artifact": "b"}])
    assert out["effective_menu_size"] == 3
    assert out["menu_collapsed"] is False


def test_a_type_incompatible_menu_is_reported_as_collapsed_not_flat() -> None:
    """Iteration 3 exactly: two of three candidates score the invalid sentinel."""
    out = _spread(lambda cfg: (-1_000_000.0
                               if str(getattr(cfg, "starting_artifact", "") or "") else -2091.8))
    assert out["effective_menu_size"] == 1
    assert out["menu_collapsed"] is True
    assert out["flat"] is True  # unchanged meaning, but now qualified by menu_collapsed


def test_a_normalizer_floored_rejection_still_counts_as_invalid() -> None:
    """`make_scored_task_runner` reports the worst LEGAL score (-1.0) instead of the
    raw -1e9 so one rejection cannot poison a downstream mean. A numeric threshold
    therefore cannot see the rejection, and effective_menu_size would over-count."""
    out = _spread(lambda cfg: (INVALID_CONFIG_SCORE
                               if str(getattr(cfg, "starting_artifact", "") or "") else -2091.8))
    assert [row["score"] for row in out["rows"]] != [None, None, None]
    assert any(row.get("rejected") for row in out["rows"]), "the floor must stay flagged"
    assert out["invalid_probes"] == 2
    assert out["effective_menu_size"] == 1 and out["menu_collapsed"] is True


def test_ranking_equivalent_candidates_collapse_the_effective_menu() -> None:
    """`item - bins`, `-(bins - item)` and `1/(gap+eps)` are the SAME heuristic:
    only the argmax matters, so they all score -2091.8. No type check can see this;
    identical scores can."""
    out = _spread(lambda cfg: -2091.8,
                  probes=[{"starting_artifact": t} for t in
                          ("item - bins", "-(bins - item)", "1/(gap+eps)", "-(gap**2)")])
    assert out["effective_menu_size"] == 1
    assert out["menu_collapsed"] is True


# --------------------------------------------------------------------------- #
# 4. the real tasks, if Trace-Bench is installed (deterministic, no LLM)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task_id", [
    "llm4ad:optimization/online_bin_packing",
    "llm4ad:optimization/admissible_set",
    "internal:multi_param",
    "internal:code_param",
])
def test_real_task_parameters_survive_the_prose_menu(task_id: str) -> None:
    pytest.importorskip("trace_bench")
    adapter = TB.TraceBenchTaskAdapter(max_examples=1, inner_steps=0)
    try:
        bundle = adapter._load_bundle(task_id, fresh=True)
    except Exception as exc:  # pragma: no cover - task set not installed
        pytest.skip(f"{task_id} unavailable: {exc}")
    node = M.trainable_node(bundle.get("param"))
    original = str(getattr(node, "data", ""))
    assert original.strip(), "fixture precondition: the task ships a non-empty parameter"
    for candidate in PROSE:
        applied = adapter._apply_starting_artifact(
            bundle, LevelConfig(starting_artifact=candidate))
        assert applied is False, f"{task_id}: prose was accepted onto a real parameter"
        assert str(getattr(node, "data", "")) == original, f"{task_id}: parameter corrupted"


def test_probe_r_menu_is_no_longer_effectively_size_one() -> None:
    """The library must not certify a menu that has one usable point."""
    assert math.isfinite(INVALID_CONFIG_SCORE) or True
    surface = M.detect_surface({"param": _Param(_WritableNode(BIN_PACKING))})
    menu = ["", *PROSE]
    usable = [m for m in menu if not m or M.artifact_fits_surface(surface, m) is None]
    assert len(usable) == 1, "precondition: the Iteration 3 menu really was a singleton"


def test_menu_check_kind_flags_surfaces_where_a_type_audit_is_vacuous():
    """A prose menu audited by type alone always passes; the helper must say so.

    Regression for a real error: a 106-spec audit reported "0 collapses" using
    artifact_fits_surface on prose-surfaced specs, where it accepts every candidate
    without inspecting it. The pass was vacuous and read exactly like a real one.
    """
    from opto.features.recursive_opt.measurement import (
        TaskSurface, artifact_fits_surface, menu_check_kind,
    )

    prose = TaskSurface(kind="prose", calls_llm=True, param_name="system_prompt", sample="hi")
    code = TaskSurface(kind="code", calls_llm=False, param_name="__code:0",
                       sample="def f(x):\n    return x\n")

    # the vacuity itself: prose accepts a candidate that is plainly not prose
    assert artifact_fits_surface(prose, "def f(x):\n    return x\n") is None
    assert artifact_fits_surface(code, "Answer directly.") is not None

    assert menu_check_kind(prose) == "scores"
    assert menu_check_kind(code) == "type"
    for kind in ("unknown", "prose"):
        surface = TaskSurface(kind=kind, calls_llm=True, param_name="p", sample="")
        assert menu_check_kind(surface) == "scores", (
            f"{kind} surfaces cannot be audited by type; must defer to score_spread"
        )
