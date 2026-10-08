"""Declarative code patches (patches.py) and declared recursion (child_spec@1), offline."""

from __future__ import annotations

import copy
import threading
from typing import Any

import pytest

from opto.features.recursive_opt import child_spec as C
from opto.features.recursive_opt import patches as P
from opto.features.recursive_opt import spec as S
from opto.optimizers.optimizer import Optimizer
from opto.trainer.algorithms import variation_search as V
from opto.trainer.objectives import EvaluationResult

MODE = 'opto.trainer.algorithms.variation_search:VariationSearch._next_mode'
TEXTS = 'opto.trainer.algorithms.variation_search:VARIATION_INSTRUCTIONS'


class _NoLLMOptimizer(Optimizer):
    def __init__(self, parameters, **_kwargs):
        super().__init__(parameters)

    def step(self, *_a, **_k):
        return {}

    def zero_feedback(self):
        pass

    def backward(self, *_a, **_k):
        pass


def _diverge_text_evaluator(output: Any, example: Any, context: Any) -> EvaluationResult:
    """Score = length of the (possibly patched) DIVERGE instruction, read where the trainer reads it."""
    return EvaluationResult(valid=True, status='ok', metrics={'score': float(len(V.VARIATION_INSTRUCTIONS['diverge']))}, feedback='ok')


S.register_evaluator('tests.patches.diverge_len@1', _diverge_text_evaluator)


def _child(text: str | None = None) -> dict:
    patch = [{'target': TEXTS, 'value': {**V.VARIATION_INSTRUCTIONS, 'diverge': text}}] if text is not None else []
    return {'schema_version': S.SCHEMA_VERSION, 'kind': S.SPEC_KIND, 'runtime': {'offline': True, 'test_mode': True},
            'levels': [{'id': 'O0', 'surface': {'kind': 'module', 'targets': ['x']},
                        'module': {'ref': 'recursive_opt.module.reasoning_workflow@1', 'config': {'components': {'x': 'a'}}, 'inputs': {}},
                        'engine': {'name': 'trace', 'config': {'iterations': 1, 'num_candidates': 1, 'patches': patch}},
                        'objective': {'evaluator_ref': 'tests.patches.diverge_len@1', 'metrics': {'score': {'direction': 'maximize', 'source': 'evaluation.metrics.score'}},
                                      'selection': {'mode': 'scalar', 'score_key': 'score'}},
                        'datasets': {'train': [{'q': 1}], 'validation': [], 'holdout': [{'q': 2}]}}]}


def test_function_patch_applies_and_restores() -> None:
    original = V.VariationSearch._next_mode
    with P.applied([{'target': MODE, 'source': "def _next_mode(self):\n    return 'diverge'"}]):
        assert object.__new__(V.VariationSearch)._next_mode() == 'diverge'
    assert V.VariationSearch._next_mode is original


def test_default_source_round_trips_and_is_equivalent() -> None:
    source = P.default_source(MODE)
    assert source.startswith('def _next_mode(self)')
    P.validate([{'target': MODE, 'source': source}])


def test_value_patch_and_null_source_is_no_patch() -> None:
    with P.applied([{'target': TEXTS, 'value': {**V.VARIATION_INSTRUCTIONS, 'diverge': 'X'}}]):
        assert V.VARIATION_INSTRUCTIONS['diverge'] == 'X'
    assert V.VARIATION_INSTRUCTIONS['diverge'] != 'X'
    with P.applied([{'target': MODE, 'source': None}]):
        pass


@pytest.mark.parametrize('patch, message', [
    ({'target': 'opto.features.recursive_opt.spec:_evaluate_dataset', 'source': 'def _evaluate_dataset():\n    pass'}, 'outside'),
    ({'target': 'os:getcwd', 'value': 1}, 'under opto.'),
    ({'target': MODE, 'source': 'def _next_mode(self, extra):\n    return 1'}, 'parameters'),
    ({'target': MODE, 'source': 'def _next_mode(self):\n    import os\n    return os.name'}, 'import'),
    ({'target': MODE, 'source': 'def other(self):\n    return 1'}, 'exactly one function'),
    ({'target': MODE + 'X', 'value': 1}, 'does not exist'),
])
def test_invalid_patches_are_rejected(patch, message) -> None:
    with pytest.raises(ValueError, match=message):
        P.validate([patch])


def test_failing_patch_falls_back_and_is_counted() -> None:
    before = P.FAILURES.get(MODE, 0)
    with P.applied([{'target': MODE, 'source': 'def _next_mode(self):\n    return 1 / 0'}]):
        trainer = object.__new__(V.VariationSearch)
        trainer.variation_schedule = 'free'
        assert trainer._next_mode() == 'free'
    assert P.FAILURES[MODE] == before + 1


def test_two_patch_versions_in_one_process_are_refused_but_isolated_workers_run_both() -> None:
    entered, release = threading.Event(), threading.Event()

    def hold():
        with P.applied([{'target': TEXTS, 'value': {'diverge': 'A'}}]):
            entered.set()
            release.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    entered.wait(5)
    with pytest.raises(RuntimeError, match='separate worker processes'):
        with P.applied([{'target': TEXTS, 'value': {'diverge': 'B'}}]):
            pass
    release.set()
    thread.join()
    results = P.run_isolated([(_read_diverge_under, ('A' * n,), {}) for n in (1, 2, 3)], workers=3)
    assert results == [1, 2, 3] and V.VARIATION_INSTRUCTIONS['diverge'] not in ('A', 'AA', 'AAA')


def _read_diverge_under(text: str) -> int:
    with P.applied([{'target': TEXTS, 'value': {'diverge': text}}]):
        return len(V.VARIATION_INSTRUCTIONS['diverge'])


def test_engine_patches_are_validated_and_reach_the_run() -> None:
    bad = _child('x')
    bad['levels'][0]['engine']['config']['patches'] = [{'target': MODE, 'source': 'def nope(self):\n    pass'}]
    with pytest.raises(ValueError, match='exactly one function'):
        S.compile_plan(bad)
    (result,) = S.execute_plan(S.compile_plan(_child('12345')), {'optimizer': _NoLLMOptimizer})
    assert result.valid and result.evaluation.metrics['score'] == 5.0
    assert V.VARIATION_INSTRUCTIONS['diverge'] != '12345'


def _parent(train: list, slot_value: Any) -> dict:
    """O1 level over the child's patch value; the fixed engine just evaluates the declared artifact."""
    config = {'child': _child('seed text'), 'slots': {'diverge': 'levels.O0.engine.config.patches.0.value'}, 'example_paths': ['levels.O0.datasets']}
    return {'schema_version': S.SCHEMA_VERSION, 'kind': S.SPEC_KIND, 'runtime': {'offline': True, 'test_mode': True},
            'levels': [{'id': 'O1', 'surface': {'kind': 'module', 'targets': ['diverge']},
                        'module': {'ref': 'recursive_opt.module.child_spec@1', 'inputs': {}, 'config': config,
                                   'artifact': {'components': {'diverge': slot_value}}},
                        'engine': {'name': 'fixed'},
                        'objective': {'evaluator_ref': 'recursive_opt.evaluator.child_spec@1'},
                        'datasets': {'train': train, 'validation': [], 'holdout': train}}]}


def test_child_spec_runs_the_child_with_slot_and_episode_and_charges_usage(monkeypatch) -> None:
    monkeypatch.setattr(C, 'CHILD_RESOURCES', {'optimizer': _NoLLMOptimizer})
    episode = {'levels.O0.datasets': {'train': [{'q': 3}], 'validation': [], 'holdout': [{'q': 4}]}}
    value = {**V.VARIATION_INSTRUCTIONS, 'diverge': '1234567'}
    (result,) = S.execute_plan(S.compile_plan(_parent([episode], value)), {})
    assert result.valid and result.evaluation.metrics['score'] == 7.0
    assert 'child score 7' in str(result.evaluation.feedback)
    with pytest.raises(ValueError, match='outside example_paths'):
        C._evaluate({'components': {'diverge': value}}, {'runtime.seed': 1}, {'spec': S.normalize_spec(_parent([episode], value))['levels'][0]})


def test_child_spec_rejects_a_slot_that_is_not_a_field() -> None:
    raw = _parent([{}], {'diverge': 'x'})
    raw['levels'][0]['module']['config']['slots'] = {'diverge': 'levels.O0.engine.config.no_such_field'}
    with pytest.raises((KeyError, ValueError)):
        S.compile_plan(raw)


def test_invalid_slot_value_is_an_invalid_candidate_not_a_crash(monkeypatch) -> None:
    monkeypatch.setattr(C, 'CHILD_RESOURCES', {'optimizer': _NoLLMOptimizer})
    raw = _parent([{}], 'not a mapping')
    raw['levels'][0]['module']['config']['slots'] = {'diverge': 'levels.O0.engine.config.patches.0'}
    result = C._evaluate({'components': {'diverge': {'target': 'os:getcwd', 'value': 1}}}, {}, {'spec': S.normalize_spec(raw)['levels'][0]})
    assert not result.valid and 'under opto.' in result.error


def test_child_spec_runs_several_episodes_in_parallel_and_averages(monkeypatch) -> None:
    monkeypatch.setattr(C, 'CHILD_RESOURCES', {'optimizer': _NoLLMOptimizer})
    ep = lambda q: {'levels.O0.datasets': {'train': [{'q': q}], 'validation': [], 'holdout': [{'q': q}]}}  # noqa: E731
    raw = _parent([{'episodes': [ep(1), ep(2)]}], {**V.VARIATION_INSTRUCTIONS, 'diverge': '1234'})
    (result,) = S.execute_plan(S.compile_plan(raw), {})
    assert result.valid and result.evaluation.metrics['score'] == 4.0
    assert 'per unit [4.0, 4.0]' in str(result.evaluation.feedback)
