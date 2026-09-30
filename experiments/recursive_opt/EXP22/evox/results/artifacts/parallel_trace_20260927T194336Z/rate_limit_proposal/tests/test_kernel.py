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
from skydiscover.optimize.search.utils.discovery_utils import SerializableResult
from src.control_plane import ENGINE, MODULE, register, specification
from src.evaluation import SKY, TASKS
from src.kernel import (
    POLICY,
    AuditController,
    PolicyModule,
    TraceMetaCoEvolutionController,
    configuration,
    digest,
)
from src.transport import TraceOpenRouter

from opto.features.recursive_opt import spec as S


class KernelTests(unittest.TestCase):
    """Prove compiled artifact connectivity and preserve a real stock population."""

    def test_failed_http_attempts_do_not_restore_policy_or_escape_budget(self) -> None:
        """Transport and ordinary generation failures retain every consumed attempt."""
        for transport_passed, missing_prompt, fallback_active in ((False, True, True), (False, True, False), (True, True, True), (True, False, True)):
            with self.subTest(transport_passed=transport_passed, missing_prompt=missing_prompt, fallback_active=fallback_active), tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temporary, patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture'}), patch('src.kernel.HTTP_RECORDS', []) as records:
                config = configuration('prism', Path(temporary), fixed=True)
                database = create_database('evox', config.search.database)
                controller = AuditController(DiscoveryControllerInput(config, str(SKY/TASKS['prism']/'evaluator/evaluator.py'), database, output_dir=temporary))
                self.addCleanup(controller.close)
                self.addCleanup(controller.search_controller.close)
                database.add(get_program(config, 'pass\n', 'initial-fixture', {'combined_score': 21.0}, 0), iteration=0)
                fallback = object() if fallback_active else None
                controller._fallback_database = fallback
                policy = controller._active_search_algorithm_code
                result = SerializableResult(error='Fixture generation failure', attempts_used=3, prompt=None if missing_prompt else {'system': 'fixture', 'user': 'fixture'})

                async def fail_generation(iteration: int, retry_times: int = 3, transport_passed: bool = transport_passed, result: SerializableResult = result) -> SerializableResult:
                    """Consume three attempts, with the last optionally failing transport."""
                    records.extend({'role': 'solution', 'passed': index < 2 or transport_passed} for index in range(3))
                    return result

                with patch('src.kernel.CoEvolutionController._run_iteration', new=AsyncMock(side_effect=fail_generation)) as generate, patch.object(controller, '_check_meta_llm_availability', new=AsyncMock()), patch.object(controller, '_generate_variation_operators', new=AsyncMock()), patch.object(controller, '_restore_fallback_database') as restore:
                    asyncio.run(controller.run_discovery(0, 100 if not transport_passed else 3))
                self.assertEqual(generate.await_count, 1)
                restore.assert_not_called()
                self.assertIs(controller._fallback_database, fallback)
                self.assertEqual(controller._active_search_algorithm_code, policy)
                self.assertEqual(len(controller.curve), 3)
                self.assertEqual(controller.attempt_validity, [False] * 3)
                self.assertEqual(controller.shutdown_event.is_set(), not transport_passed)
                self.assertEqual(controller.gate_failure, None if transport_passed else 'HTTP transport or serving-identity validation failed')
                self.assertEqual(result.prompt, {} if missing_prompt else {'system': 'fixture', 'user': 'fixture'})

    def test_pre_http_database_failure_still_restores_fallback(self) -> None:
        """A database error before HTTP retries the same iteration without consuming budget."""
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temporary, patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture'}), patch('src.kernel.HTTP_RECORDS', []) as records:
            config = configuration('prism', Path(temporary), fixed=True)
            database = create_database('evox', config.search.database)
            controller = AuditController(DiscoveryControllerInput(config, str(SKY/TASKS['prism']/'evaluator/evaluator.py'), database, output_dir=temporary))
            self.addCleanup(controller.close)
            self.addCleanup(controller.search_controller.close)
            database.add(get_program(config, 'pass\n', 'initial-fixture', {'combined_score': 21.0}, 0), iteration=0)
            controller._fallback_database = object()
            database_failure = SerializableResult(error='Fixture database failure')
            calls = []

            async def fail_then_generate(iteration: int, retry_times: int = 3) -> SerializableResult:
                """Fail before HTTP once, then consume a normal failed generation attempt."""
                calls.append(iteration)
                if len(calls) == 1:
                    return database_failure
                records.append({'role': 'solution', 'passed': True})
                return SerializableResult(error='Fixture parse failure', prompt={'system': 'fixture', 'user': 'fixture'})

            def restore_fallback() -> None:
                """Model the stock fallback being consumed after a database error."""
                controller._fallback_database = None

            with patch('src.kernel.CoEvolutionController._run_iteration', new=AsyncMock(side_effect=fail_then_generate)), patch.object(controller, '_check_meta_llm_availability', new=AsyncMock()), patch.object(controller, '_generate_variation_operators', new=AsyncMock()), patch.object(controller, '_restore_fallback_database', side_effect=restore_fallback) as restore:
                asyncio.run(controller.run_discovery(0, 1))
            restore.assert_called_once_with()
            self.assertEqual(calls, [0, 0])
            self.assertIsNone(database_failure.prompt)
            self.assertEqual(len(records), 1)
            self.assertEqual(len(controller.curve), 1)
            self.assertFalse(controller.shutdown_event.is_set())
            self.assertIsNone(controller.gate_failure)

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
            raw_observation = json.loads((directory/'observation_001.json').read_text())
            self.assertEqual(raw_observation, controller.window_observation)
            feedback_text = (directory/'feedback_001.json').read_text().rstrip('\n')
            self.assertEqual(json.loads(feedback_text)['root'], raw_observation)
            prompt = '\n'.join(message['content'] for message in calls[0]['messages'])
            self.assertEqual(prompt.count(feedback_text), 1)
            self.assertIn(digest(feedback_text), prompt)
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
