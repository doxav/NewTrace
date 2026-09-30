"""Offline tests for the EXP23 simulator, traced selection, meta arms and control plane."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opto.optimizers.optoprime_v2 import OptoPrimeV2
from opto.utils.llm import DummyLLM

from src import (
    meta,
    traced,
)
from src.control_plane import alternatives, register, run
from src.mock_llm import MockOptimizerLLM
from src.policies import (
    CODE_STOCK,
    REFERENCE,
    PolicyInvalid,
    compile_policy,
    dump_knobs,
    stock_text,
)
from src.search import paired_score, run_fixed, step
from src.world import Population, World


class WorldTests(unittest.TestCase):
    def test_common_random_numbers(self) -> None:
        world = World('prism', 'W1')
        a, b = Population.start(world.init), Population.start(world.init)
        self.assertEqual(world.generate(a.members[0], 3, 1), world.generate(b.members[0], 3, 1))

    def test_identical_policies_identical_runs(self) -> None:
        world, select = World('signal_processing', 'W0'), compile_policy('knobs', dump_knobs(REFERENCE['soft_0.3']))
        self.assertEqual(run_fixed(world, select, 5), run_fixed(world, select, 5))

    def test_wear_shrinks_only_positive_deltas(self) -> None:
        worn = World('prism', 'W2')
        population = Population.start(worn.init)
        parent = population.members[0]
        parent.uses = 10
        children = [worn.generate(parent, s, 1) for s in range(300)]
        gains = [c - parent.score for c in children if c is not None and c > parent.score]
        self.assertTrue(all(g < 0.01 for g in gains))


class PolicyTests(unittest.TestCase):
    def test_knob_bounds_rejected_not_clipped(self) -> None:
        bad = dict(REFERENCE['greedy'], temperature=0.0)
        with self.assertRaises(PolicyInvalid):
            compile_policy('knobs', json.dumps(bad))

    def test_code_validator(self) -> None:
        compile_policy('code', CODE_STOCK)
        for source in ('x = 1', 'def select_parent(members, rng):\n    return len(members)', 'def select_parent(members, rng):\n    import os\n    return 0'):
            with self.assertRaises(PolicyInvalid):
                compile_policy('code', source)

    def test_runtime_failure_falls_back_and_is_recorded(self) -> None:
        world = World('prism', 'W1')
        population = Population.start(world.init)
        record = step(world, population, lambda members, rng: 1 / 0, 1, 1)
        self.assertEqual(record['policy_error'], 'ZeroDivisionError')


class ScoreTests(unittest.TestCase):
    def test_paired_requires_both_roles(self) -> None:
        self.assertEqual(paired_score([], 'new_best', 1.0)['score'], 0.0)


class TracedTests(unittest.TestCase):
    def _prompt(self, arm_output, feedback: str, param) -> str:
        llm = MockOptimizerLLM(0)
        optimizer = OptoPrimeV2([param], llm=DummyLLM(llm), memory_size=5, log=False)
        optimizer.zero_feedback()
        optimizer.backward(arm_output, feedback)
        before = param.data
        optimizer.step()
        self.assertNotEqual(before, param.data, 'mock proposal must change the policy through the real parser')
        return llm.prompt_chars[-1], llm

    def test_graph_contains_executed_decisions_and_stays_small(self) -> None:
        world = World('prism', 'W1')
        population = Population.start(world.init)
        for i in range(1, 30):
            step(world, population, compile_policy('knobs', stock_text('knobs')), 1, i)
        module = traced.SelectionPolicy('knobs', dump_knobs(REFERENCE['top5_reuse']))
        arena = traced.Arena(world, population, 1, 29, 20, 'knobs', 'paired', compile_policy('knobs', stock_text('knobs')))
        output = module(arena)
        self.assertEqual(output.data['decisions'], 20)
        self.assertEqual({r['tag'] for r in arena.records}, {'challenger', 'incumbent'})
        chars, _ = self._prompt(output, traced.feedback_text(output.data, stock_text('knobs')), module.selection_policy)
        self.assertLess(chars, 20_000)

    def test_prompt_mentions_decision_lines(self) -> None:
        world = World('prism', 'W1')
        module = traced.SelectionPolicy('knobs', dump_knobs(REFERENCE['greedy']))
        arena = traced.Arena(world, Population.start(world.init), 2, 0, 6, 'knobs', 'solo')
        output = module(arena)
        captured = {}
        def llm(*args, **kwargs):
            captured['prompt'] = kwargs['messages'][-1]['content']
            return MockOptimizerLLM(0)(*args, **kwargs)
        optimizer = OptoPrimeV2([module.selection_policy], llm=DummyLLM(llm), log=False)
        optimizer.zero_feedback()
        optimizer.backward(output, 'score=0')
        optimizer.step()
        self.assertIn('run_selection_window', captured['prompt'])
        self.assertIn('parent_rank=0', captured['prompt'])


class BudgetTests(unittest.TestCase):
    def test_every_arm_spends_exactly_the_horizon(self) -> None:
        world = World('prism', 'W1')
        for arm in ('fixed:uniform',) + meta.ONLINE_ARMS:
            calls = []
            real = meta.step
            def counting(*args, calls=calls, real=real, **kwargs):
                calls.append(1)
                return real(*args, **kwargs)
            with mock.patch.object(meta, 'step', counting), mock.patch.object(traced, 'step', counting):
                meta.run_online(world, 11, arm, horizon=100)
            self.assertEqual(len(calls), 100, arm)


class ControlPlaneTests(unittest.TestCase):
    def test_all_alternatives_compile_and_stubs_refuse(self) -> None:
        from opto.features.recursive_opt import spec as S
        register()
        specs = alternatives(Path(self.enterContext(tempfile.TemporaryDirectory())))
        for raw in specs.values():
            S.compile_plan(raw)
        raw = json.loads(json.dumps(specs['A4/live_paired_coevolution']))
        raw['runtime'].update(offline=True, test_mode=True)
        raw['llm_profiles']['main'].pop('api_key_ref')
        (result,) = run(raw)
        self.assertEqual(result.status, 'error')
        self.assertIn('not implemented', result.error)

    def test_online_and_priority_search_execute(self) -> None:
        specs = alternatives(Path(self.enterContext(tempfile.TemporaryDirectory())))
        online = json.loads(json.dumps(specs['A1/trace_paired-stagnation']))
        online['levels'][0]['datasets']['holdout']['config']['count'] = 2
        online['budget']['candidates'] = None
        (result,) = run(online)
        self.assertEqual(result.status, 'success', result.error)
        search = json.loads(json.dumps(specs['A2/priority_search-knobs']))
        level = search['levels'][0]
        level['engine']['config']['iterations'] = 2
        for split in level['datasets']:
            level['datasets'][split]['config']['count'] = 2
        (result,) = run(search)
        self.assertEqual(result.status, 'success', result.error)
        compile_policy('knobs', result.artifact['selection_policy'])


if __name__ == '__main__':
    unittest.main()


class ScheduleTests(unittest.TestCase):
    def test_one_plus_two_by_four_budget(self) -> None:
        from src import schedule
        for arm in schedule.ARMS:
            calls, llm = [], MockOptimizerLLM(3)
            real = schedule.step
            def counting(*args, calls=calls, real=real, **kwargs):
                calls.append(1)
                return real(*args, **kwargs)
            with mock.patch.object(schedule, 'step', counting), mock.patch.object(traced, 'step', counting):
                result = schedule.run_schedule(World('prism', 'W1'), 5, arm, 'code', llm)
            self.assertEqual(len(calls), 12, arm)
            self.assertEqual(llm.calls, 2, arm)
            self.assertEqual(result['solution_steps'], 12, arm)
