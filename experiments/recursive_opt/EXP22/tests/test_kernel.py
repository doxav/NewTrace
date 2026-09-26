"""Exercise real Trace proposals and stock policy validation/migration offline."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from black import FileMode, format_str

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'worktrees/trace_cp_b'), str(ROOT)]
from skydiscover.optimize.search.default_discovery_controller import (
    DiscoveryControllerInput,
)
from skydiscover.optimize.search.registry import create_database, get_program
from src.control_plane import ENGINE, MODULE, register, specification
from src.evaluation import SKY, TASKS
from src.kernel import (
    POLICY,
    PolicyModule,
    TraceMetaCoEvolutionController,
    configuration,
    digest,
)
from src.transport import TraceOpenRouter

from opto.features.recursive_opt import spec as S


class KernelTests(unittest.TestCase):
    """Prove compiled artifact connectivity and preserve a real stock population."""

    def test_registration_compile_and_artifact(self) -> None:
        """The intended versioned engine and complete code artifact resolve."""
        register()
        plan = S.compile_plan(specification('prism', 'TRACE-RECURSIVE', 100, ROOT/'runs/fixture'))
        self.assertEqual(plan.explain()['engines'], [ENGINE])
        self.assertEqual(plan.explain()['module_refs'], [MODULE])
        self.assertEqual(plan.spec['budget']['candidates'], 100)
        module = S.build_module(plan.spec)
        self.assertEqual(module.policy_source.data, POLICY.read_text())
        self.assertEqual(len(module.parameters()), 1)

    def test_selected_cp_a_canonical_execution(self) -> None:
        """Execute the selected engine with explicit non-portable factory metadata."""
        register()
        result = {'horizon': 1, 'iterations_observed': 1, 'final_metrics': {'combined_score': 1.0}, 'gate_failure': None}
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temporary, patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture'}), patch('src.control_plane.run_kernel', new=AsyncMock(return_value=result)) as kernel:
            raw = specification('prism', 'TRACE-RECURSIVE', 1, Path(temporary))
            actual = S.run_spec(raw, resources={'llm_factory': TraceOpenRouter})
            self.assertTrue(actual.valid)
            self.assertFalse(actual.portable)
            self.assertFalse(actual.promotable)
            self.assertEqual(actual.metadata['kernel_result']['iterations_observed'], 1)
            self.assertEqual(kernel.await_count, 1)

    def test_trace_mock_policy_swap_and_feedback(self) -> None:
        """A real OptoPrimeV2 step consumes feedback and activates through stock migration."""
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temporary, patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture'}):
            directory = Path(temporary)
            config = configuration('prism', directory)
            database = create_database('evox', config.search.database)
            module = PolicyModule(POLICY.read_text())
            candidate = (POLICY.read_text() + '\n# EXP22 deterministic policy fixture.\n').strip()
            calls = []

            def llm(*args: Any, **kwargs: Any) -> Any:
                """Return a fixed valid policy using the real optimizer output grammar."""
                calls.append(kwargs)
                content = f'<reasoning>Fixture</reasoning><variable><name>{module.policy_source.name}</name><value>{candidate}</value></variable>'
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

            controller = TraceMetaCoEvolutionController(DiscoveryControllerInput(config, str(SKY/TASKS['prism']/'evaluator/evaluator.py'), database, output_dir=temporary), module, llm)
            self.addCleanup(controller.close)
            self.addCleanup(controller.search_controller.close)
            program = get_program(config, 'def fixture(): return 1\n', 'initial-fixture', {'combined_score': 21.0}, 0)
            database.add(program, iteration=0)
            database.initial_program_id = program.id
            database.initial_program_score = 21.0
            controller.window_observation = {'active_policy_hash': digest(POLICY.read_text()), 'window_metrics': {'combined_score': 0.125, 'search_window_start_score': 21.0, 'search_window_end_score': 22.0}, 'history': [21.0, 22.0]}
            result = asyncio.run(controller.search_controller.run_discovery(1, 1, False))
            self.assertIsNone(result.error)
            self.assertEqual(len(calls), 1)
            self.assertIn('0.125', json.dumps(calls))
            self.assertEqual(result.child_program_dict['metrics']['validity'], 1)
            proposal = json.loads((directory/'policy_proposals.jsonl').read_text())
            self.assertTrue(proposal['valid'])
            self.assertEqual((directory/proposal['source_path']).read_text(), result.child_program_dict['solution'])
            self.assertTrue(controller._switch_to_new_search_algorithm(result))
            self.assertEqual(controller.database.get(program.id).solution, program.solution)
            self.assertEqual(controller.database.get(program.id).metrics, program.metrics)
            self.assertEqual(digest(controller._active_search_algorithm_code), digest(format_str(candidate, mode=FileMode())))
            controller.window_observation['active_policy_hash'] = 'wrong-policy'
            with self.assertRaisesRegex(ValueError, 'disconnected'):
                asyncio.run(controller._trace_proposal(2, 1, False))
            self.assertEqual(len(calls), 1)
            with patch('src.kernel.CoEvolutionController._switch_to_new_search_algorithm', return_value=True), patch.object(controller.database, 'get', return_value=None), self.assertRaisesRegex(ValueError, 'lost the existing population'):
                controller._switch_to_new_search_algorithm(result)
            self.assertEqual(controller.gate_failure, 'Policy switch changed or lost the existing population')
            self.assertTrue(controller.shutdown_event.is_set())

    def test_checkpoints_and_structural_stops(self) -> None:
        """Flat valid search continues; sustained invalidity and budget drift stop."""
        records = []
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temporary, patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture'}), patch('src.kernel.HTTP_RECORDS', records):
            directory = Path(temporary)
            config = configuration('prism', directory)
            database = create_database('evox', config.search.database)
            controller = TraceMetaCoEvolutionController(DiscoveryControllerInput(config, str(SKY/TASKS['prism']/'evaluator/evaluator.py'), database, output_dir=temporary), PolicyModule(POLICY.read_text()), None)
            self.addCleanup(controller.close)
            self.addCleanup(controller.search_controller.close)
            program = get_program(config, 'pass\n', 'audit-fixture', {'combined_score': 21.0}, 0)
            database.add(program, iteration=0)
            controller.total_solution_iterations = 101
            controller._max_solution_iterations = 100
            for index in range(20):
                records.append({'role': 'solution', 'passed': True, 'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'cost': 0.01}})
                controller.pending_attempt_validity = [index < 10]
                controller.current_candidate_score = 21.0
                controller._record_search_window_step()
                if index < 19:
                    self.assertFalse(controller.shutdown_event.is_set())
            checkpoint = json.loads((directory/'checkpoint_010.json').read_text())
            self.assertEqual(checkpoint['valid_candidates'], 10)
            self.assertEqual(checkpoint['llm_usage']['total_calls'], 10)
            self.assertAlmostEqual(checkpoint['llm_usage']['reported_cost'], 0.1)
            self.assertEqual(controller.gate_failure, 'No valid solution over ten consecutive generation attempts')
            self.assertTrue(controller.shutdown_event.is_set())
            records.append({'role': 'solution', 'passed': True})
            with self.assertRaisesRegex(ValueError, 'budget diverged'):
                asyncio.run(controller._run_iteration(21))
