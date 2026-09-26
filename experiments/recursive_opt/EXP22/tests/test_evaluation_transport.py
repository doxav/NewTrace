"""Check the stock evaluation adapter and direct smoke serialization."""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluator_preflight import equivalent, stable
from scripts.transport_smoke import MODEL, SESSION, payload, request_json
from src.evaluation import TraceEvaluatorAdapter, stock_evaluator


class EvaluationTests(unittest.TestCase):
    """Cover cascade boundaries, missing entry points and metric comparison."""

    def test_cascade_thresholds(self) -> None:
        """Stock threshold includes equality and treats absent metrics as failure."""
        evaluator = stock_evaluator("signal_processing")
        self.addCleanup(evaluator.close)
        self.assertTrue(evaluator._passes_threshold({"combined_score": 0.3}, 0.3))
        self.assertFalse(evaluator._passes_threshold({"combined_score": 0.299}, 0.3))
        self.assertFalse(evaluator._passes_threshold({}, 0.3))

    def test_missing_entry_point_parity(self) -> None:
        """Invalid source has the same metrics on both evaluation boundaries."""
        for task in ("prism", "signal_processing"):
            evaluator = stock_evaluator(task)
            adapter = TraceEvaluatorAdapter(task)
            self.addCleanup(evaluator.close)
            self.addCleanup(adapter.close)
            actual = asyncio.run(adapter.evaluate("pass\n"))
            expected = asyncio.run(evaluator.evaluate_program("pass\n")).metrics
            self.assertTrue(equivalent(stable(actual), stable(expected)))

    def test_comparison_rejects_changed_or_missing_quality(self) -> None:
        """Timing exclusion must not hide a quality drift or missing field."""
        self.assertEqual(stable({"combined_score": 1.0, "execution_time": 2.0}), {"combined_score": 1.0})
        self.assertFalse(equivalent({"score": 1.0}, {"score": 1.001}))
        self.assertFalse(equivalent({"score": 1.0}, {}))
        self.assertTrue(equivalent({"score": 1.0}, {"score": 1.0 + 1e-13}))


class TransportTests(unittest.TestCase):
    """Inspect the final urllib HTTP body, not just an intermediate config."""

    def test_outbound_request(self) -> None:
        """Exact required routing survives JSON serialization without a secret."""
        with patch("urllib.request.urlopen") as mocked:
            response = mocked.return_value.__enter__.return_value
            response.status = 200
            response.read.return_value = b'{"id":"fixture"}'
            request_json("/chat/completions", "fixture-credential", payload())
            request = mocked.call_args.args[0]
            body = json.loads(request.data)
            self.assertEqual(body["model"], MODEL)
            self.assertEqual(body["provider"], {"only": ["DeepInfra"]})
            self.assertEqual(body["session_id"], SESSION)
            self.assertNotIn("seed", body)
            self.assertNotIn("fixture-credential", request.data.decode())
            self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/chat/completions")
