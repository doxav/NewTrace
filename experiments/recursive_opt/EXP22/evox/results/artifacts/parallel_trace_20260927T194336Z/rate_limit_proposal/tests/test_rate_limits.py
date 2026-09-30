"""Offline review of the unactivated, narrowly bounded solution-429 proposal."""

import asyncio
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, call, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'worktrees/trace_cp_b'), str(ROOT)]
from scripts import run_stage
from skydiscover.optimize.search.default_discovery_controller import (
    DiscoveryControllerInput,
)
from skydiscover.optimize.search.registry import create_database, get_program
from skydiscover.optimize.search.utils.discovery_utils import SerializableResult
from src import kernel
from src.evaluation import SKY, TASKS
from src.kernel import AuditController, configuration
from src.transport import (
    EXTRA_BODY,
    MODEL,
    is_recoverable_rate_limit,
    transport_records_acceptable,
)


def refused_request() -> dict[str, Any]:
    """Represent the exact empty Novita shared-pool refusal captured in the live run."""
    return {
        'role': 'solution', 'http_status': 429, 'passed': False,
        'outbound_body': {'model': MODEL, **copy.deepcopy(EXTRA_BODY), 'temperature': 0.7, 'max_tokens': 32000},
        'error': {'code': 429, 'metadata': {'provider_name': 'Novita', 'limit_source': 'upstream_provider_shared_pool'}},
        'content_lengths': [], 'finish_reasons': [], 'returned_model': None,
        'serving_provider': None, 'generation_id': None,
    }


class RateLimitTests(unittest.TestCase):
    """Keep refused calls budgeted, bounded, auditable, and distinct from valid replies."""

    def test_exact_refusal_and_shared_decision(self) -> None:
        """Only the identified refusal qualifies; final and controller guards agree."""
        record = refused_request()
        original = copy.deepcopy(record)
        self.assertTrue(is_recoverable_rate_limit(record))
        self.assertTrue(transport_records_acceptable([record, copy.deepcopy(record)]))
        self.assertFalse(transport_records_acceptable([record] * 3))
        self.assertEqual(record, original)
        self.assertIs(kernel.transport_records_acceptable, run_stage.transport_records_acceptable)

    def test_reject_other_failures_content_or_changed_request(self) -> None:
        """Do not waive unknown transport, serving, routing, or model discrepancies."""
        changes = [
            ('role', 'meta'), ('http_status', 400), ('http_status', 200), ('http_status', None),
            ('error.metadata.provider_name', 'Other'), ('error.metadata.limit_source', 'account_limit'),
            ('error', None), ('error.metadata', None), ('error.code', 400),
            ('outbound_body.model', 'other-model'), ('outbound_body.provider', {'only': ['other']}),
            ('outbound_body.reasoning_effort', 'high'), ('outbound_body.session_id', 'other'),
            ('outbound_body.max_tokens', 16000), ('outbound_body.temperature', 1.0),
            ('outbound_body', None), ('content_lengths', [1]), ('content_lengths', None),
            ('finish_reasons', ['stop']), ('returned_model', MODEL), ('serving_provider', 'Novita'),
            ('generation_id', 'unexpected-generation'),
        ]
        for path, value in changes:
            with self.subTest(path=path, value=value):
                record = refused_request()
                target = record
                names = path.split('.')
                for name in names[:-1]:
                    target = target[name]
                target[names[-1]] = value
                self.assertFalse(is_recoverable_rate_limit(record))
                self.assertFalse(transport_records_acceptable([record]))

    def run_sequence(self, responses: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
        """Exercise the unchanged stock outer loop with mocked generation and sleep."""
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temporary, patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture'}), patch('src.kernel.HTTP_RECORDS', []) as records:
            config = configuration('prism', Path(temporary), fixed=True)
            database = create_database('evox', config.search.database)
            controller = AuditController(DiscoveryControllerInput(config, str(SKY/TASKS['prism']/'evaluator/evaluator.py'), database, output_dir=temporary))
            self.addCleanup(controller.close)
            self.addCleanup(controller.search_controller.close)
            database.add(get_program(config, 'pass\n', 'initial-fixture', {'combined_score': 21.0}, 0), iteration=0)
            fallback = object()
            controller._fallback_database = fallback
            policy = controller._active_search_algorithm_code

            async def generate(iteration: int, retry_times: int = 3) -> SerializableResult:
                """Consume precisely one planned HTTP record for this solution attempt."""
                row = copy.deepcopy(responses[len(records)])
                records.append(row)
                if not row['passed']:
                    return SerializableResult(iteration=iteration, error='Recorded provider refusal')
                child = get_program(config, 'pass\n# improved\n', f'child-{iteration}', {'combined_score': 22.0}, iteration)
                return SerializableResult(iteration=iteration, child_program_dict=child.to_dict())

            with patch('src.kernel.CoEvolutionController._run_iteration', new=AsyncMock(side_effect=generate)) as generation, patch.object(controller, '_check_meta_llm_availability', new=AsyncMock()), patch.object(controller, '_generate_variation_operators', new=AsyncMock()), patch.object(controller, '_restore_fallback_database') as restore, patch('src.kernel.asyncio.sleep', new=AsyncMock()) as sleep:
                asyncio.run(controller.run_discovery(0, horizon))
            restore.assert_not_called()
            self.assertIs(controller._fallback_database, fallback)
            self.assertEqual(controller._active_search_algorithm_code, policy)
            self.assertEqual(generation.await_count, len(records))
            self.assertEqual(len(controller.curve), len(records))
            return {'records': records, 'curve': controller.curve, 'validity': controller.attempt_validity, 'stopped': controller.shutdown_event.is_set(), 'gate_failure': controller.gate_failure, 'sleeps': sleep.await_args_list}

    def test_refusal_then_valid_solution_consumes_two_attempts(self) -> None:
        """The first refusal waits once, preserves fallback, then permits one valid call."""
        result = self.run_sequence([refused_request(), {'role': 'solution', 'passed': True, 'http_status': 200}], 2)
        self.assertEqual(result['validity'], [False, True])
        self.assertEqual([row['best_score'] for row in result['curve']], [21.0, 22.0])
        self.assertEqual(result['sleeps'], [call(30)])
        self.assertFalse(result['stopped'])
        self.assertIsNone(result['gate_failure'])
        self.assertFalse(result['records'][0]['passed'])
        self.assertTrue(run_stage.transport_records_acceptable(result['records']))

    def test_third_refusal_stops_after_all_three_are_accounted(self) -> None:
        """Two pauses are allowed; a third refusal stops before any fourth request."""
        result = self.run_sequence([refused_request() for _ in range(3)], 100)
        self.assertEqual(len(result['records']), 3)
        self.assertEqual(result['validity'], [False, False, False])
        self.assertEqual(result['sleeps'], [call(30), call(30)])
        self.assertTrue(result['stopped'])
        self.assertEqual(result['gate_failure'], 'HTTP transport or serving-identity validation failed')
        self.assertTrue(all(not row['passed'] for row in result['records']))
        self.assertFalse(run_stage.transport_records_acceptable(result['records']))

    def test_nonrecoverable_http_error_stops_without_backoff(self) -> None:
        """An ordinary provider error keeps the existing immediate-stop behavior."""
        record = refused_request()
        record['http_status'] = 400
        result = self.run_sequence([record], 100)
        self.assertEqual(len(result['curve']), 1)
        self.assertEqual(result['sleeps'], [])
        self.assertTrue(result['stopped'])
