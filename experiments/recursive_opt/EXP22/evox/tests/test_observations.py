"""Verify lossless feedback deduplication and real OptoPrime prompt assembly."""

import ast
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'worktrees/trace_cp_b'), str(ROOT)]
from skydiscover.optimize.search.base_database import Program
from skydiscover.optimize.search.evox.utils.coevolve_logging import (
    make_json_serializable,
)
from src.kernel import PolicyModule, digest
from src.observations import encode_observation

from opto.optimizers.optoprime_v2 import OptoPrimeV2


def decode_observation(encoded: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the stock-normalized evidence for independent round-trip checks."""
    def decode(value: Any) -> Any:
        """Expand references while preserving literal marker dictionaries."""
        if isinstance(value, dict):
            if set(value) == {'$ref'}:
                return decode(encoded['values'][value['$ref']])
            if set(value) == {'$literal'}:
                return {key: decode(child) for key, child in value['$literal'].items()}
            return {key: decode(child) for key, child in value.items()}
        if isinstance(value, list):
            return [decode(child) for child in value]
        return value

    return decode(encoded['root'])


class ObservationTests(unittest.TestCase):
    """Preserve scientific evidence while removing exact repetition only."""

    def test_live_programs_versions_order_and_stable_references(self) -> None:
        """Share identical records, preserving distinct versions and population order."""
        program = Program(id='fixture', solution='pass\n' * 200, language='python', metrics={'combined_score': 1.0})
        updated = copy.deepcopy(program)
        updated.metrics['combined_score'] = 2.0
        observation = {'active_policy_hash': 'fixture-policy', 'current': [program, updated, program], 'archive': [program.to_dict(), updated.to_dict()]}
        original = copy.deepcopy(observation)
        encoded = encode_observation(observation)
        expected = make_json_serializable(observation)
        self.assertEqual(decode_observation(encoded), expected)
        self.assertEqual(observation, original)
        self.assertEqual(encoded, encode_observation(dict(reversed(list(observation.items())))))
        refs = encoded['root']['current']
        self.assertEqual(refs[0], refs[2])
        self.assertNotEqual(refs[0], refs[1])
        for key, value in encoded['values'].items():
            restored = decode_observation({**encoded, 'root': value})
            canonical = json.dumps(restored, sort_keys=True, separators=(',', ':'), allow_nan=False)
            self.assertEqual(key, hashlib.sha256(canonical.encode()).hexdigest())
        sources = [value for value in encoded['values'].values() if value == program.solution]
        self.assertEqual(sources, [program.solution])

    def test_opaque_strings_marker_collisions_and_empty_evidence(self) -> None:
        """Keep repr strings, Unicode, marker-shaped user data, and empty values intact."""
        literal = {'$ref': 'opaque' * 100}
        observation = {'active_policy_hash': 'fixture', 'values': [None, [], {}, 'é\n', literal, literal, {'$literal': literal}, 'EvolvedProgram(solution="opaque")' * 20]}
        self.assertEqual(decode_observation(encode_observation(observation)), observation)
        minimal = {'active_policy_hash': 'fixture'}
        self.assertEqual(decode_observation(encode_observation(minimal)), minimal)

    def test_invalid_observation_and_nonfinite_measurements(self) -> None:
        """Reject missing provenance and nonfinite scientific values before prompting."""
        for observation in ({}, {'active_policy_hash': None}, {'active_policy_hash': 'fixture', 'score': float('nan')}, {'active_policy_hash': 'fixture', 'score': float('inf')}):
            with self.subTest(observation=observation), self.assertRaises((TypeError, ValueError)):
                encode_observation(observation)

    def test_recorded_context_failure_roundtrip_and_real_prompt_size(self) -> None:
        """Reassemble the failed fourth Signal proposal offline without discarding data."""
        run = ROOT / 'runs/strict_signal_processing_TRACE-RECURSIVE_20260927T194615.183737Z'
        requests = json.loads((run / 'http_requests.json').read_text())
        failed = next(row for row in requests if row['role'] == 'meta' and row['http_status'] == 400)
        user = failed['outbound_body']['messages'][-1]['content']
        observation, _ = json.JSONDecoder().raw_decode(user.split('# Feedback', 1)[1].lstrip())
        # Persisted reprs are opaque in production. Restore this fixture's original
        # live Program types using literal-only AST decoding, never eval or exec.
        opaque = encode_observation(observation)
        self.assertEqual(decode_observation(opaque), observation)
        programs = observation['search_stats']['db_stats']['previous_programs']
        for index, representation in enumerate(programs):
            node = ast.parse(representation, mode='eval').body
            self.assertIsInstance(node, ast.Call)
            self.assertIsInstance(node.func, ast.Name)
            self.assertEqual(node.func.id, 'EvolvedProgram')
            self.assertFalse(node.args)
            programs[index] = Program(**{keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords})
        encoded = encode_observation(observation)
        normalized = make_json_serializable(observation)
        self.assertEqual(decode_observation(encoded), normalized)
        self.assertLess(len(json.dumps(encoded)), len(json.dumps(normalized)) * 0.35)
        source = (run / 'policy_003.py').read_text()
        self.assertEqual(digest(source), observation['active_policy_hash'])
        module = PolicyModule(source)
        llm = Mock(side_effect=AssertionError('Offline prompt assembly must not call an API'))
        optimizer = OptoPrimeV2(module.parameters(), llm=llm, max_tokens=32000, log=False, initial_var_char_limit=100000, objective='Improve downstream solution search by rewriting the complete EvolvedProgramDatabase Python source. Preserve all interfaces and solution metrics. Use the measured window feedback; return exactly one policy proposal.')
        output = module(encoded)
        optimizer.zero_feedback()
        optimizer.backward(output, json.dumps(encoded))
        system, prompt = optimizer.construct_prompt(optimizer.summarize())
        accepted_sizes = [sum(len(message['content']) for message in row['outbound_body']['messages']) for row in requests if row['role'] == 'meta' and row['passed']]
        self.assertLess(len(system) + len(prompt), max(accepted_sizes))
        self.assertIn(json.dumps(encoded), prompt)
        llm.assert_not_called()
