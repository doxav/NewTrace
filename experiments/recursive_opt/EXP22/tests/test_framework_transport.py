"""Intercept real framework SDK boundaries, including portable Trace routing."""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'worktrees/trace_cp_b'))
from src.transport import (
    BASE_URL,
    EXTRA_BODY,
    MODEL,
    SESSION,
    SkyOpenRouter,
    TraceOpenRouter,
    model_config,
)

from opto.features.recursive_opt import spec as S


def normalized_profile() -> dict:
    """Build a routing profile through canonical normalization defaults."""
    value = {'llm_profiles': {'main': {'provider': 'openrouter', 'model': MODEL, 'openrouter_routing': {'only': ['DeepInfra']}, 'request_params': {'extra_body': {'session_id': SESSION}}, 'max_tokens': 32, 'temperature': 0.7, 'request_timeout_s': 600}}}
    S._normalize_llm_profiles(value)
    return value['llm_profiles']['main']


class FrameworkTransportTests(unittest.TestCase):
    """Assert identity guards and non-secret payloads on each framework path."""

    def test_sky_sdk_body_and_identity(self) -> None:
        """Stock OpenAILLM's final SDK call carries mandatory extra fields."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture'}), patch('openai.OpenAI') as constructor:
            create = constructor.return_value.chat.completions.create
            create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))])
            client = SkyOpenRouter(model_config())
            asyncio.run(client.generate('', [{'role': 'user', 'content': 'ping'}]))
            body = create.call_args.kwargs
            self.assertEqual(body['model'], MODEL)
            self.assertEqual(body['extra_body'], EXTRA_BODY)
            self.assertEqual(body['max_tokens'], 32000)
            self.assertNotIn('seed', body)
            self.assertEqual(constructor.call_args.kwargs['base_url'], BASE_URL)
            config = model_config()
            config.name = 'gpt-5'
            with self.assertRaises(ValueError):
                SkyOpenRouter(config)

    def test_cp_a_and_cp_b_final_payloads(self) -> None:
        """The behavioral factory and portable guarded client agree on routing."""
        profile = normalized_profile()
        S._validate_profile(profile, 'profile', {}, True, None)
        messages = [{'role': 'user', 'content': 'ping'}]
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))], usage={'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2})
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture'}), patch('openai.OpenAI') as constructor:
            constructor.return_value.chat.completions.create.return_value = response
            cp_a = S._make_guarded_role_client(profile, 'forward', TraceOpenRouter, {}, S._BudgetGuard({}))
            cp_a(messages=messages)
            actual_a = constructor.return_value.chat.completions.create.call_args.kwargs
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture'}), patch('litellm.completion', return_value=response) as completion:
            cp_b = S._make_guarded_role_client(profile, 'forward', None, {}, S._BudgetGuard({}))
            cp_b(messages=messages)
            actual_b = completion.call_args.kwargs
            model_b = completion.call_args.args[0]
        self.assertEqual(actual_a['extra_body'], EXTRA_BODY)
        self.assertEqual(actual_a['extra_body'], actual_b['extra_body'])
        self.assertEqual(model_b, 'openrouter/' + MODEL)
        self.assertNotIn('fixture', json.dumps(profile))

    def test_identity_and_routing_validation(self) -> None:
        """Dedicated routing never opens arbitrary provider/credential overrides."""
        for params in ({'provider': {'only': ['DeepInfra']}}, {'extra_body': {'provider': {}}}, {'model': 'other'}, {'api_key': 'fixture'}):
            with self.assertRaises(ValueError):
                S._validate_request_params(params, 'request_params')
        for routing in ({'only': []}, {'only': ['DeepInfra', 'DeepInfra']}, {'only': 'DeepInfra'}, {'sort': 'price'}):
            profile = normalized_profile()
            profile['openrouter_routing'] = routing
            with self.assertRaises(ValueError):
                S._validate_profile(profile, 'profile', {}, True, None)

    def test_portable_preflight_routes_its_own_call(self) -> None:
        """The implicit control-plane probe must also carry provider/session."""
        normalized = {'levels': [{'llm_roles': {'forward': normalized_profile()}}], 'llm_profiles': {}}
        with patch.object(S, 'normalize_spec', return_value=normalized), patch('opto.features.recursive_opt.runmode.make_live_llm') as factory:
            S.preflight_llm_profiles({})
            self.assertEqual(factory.return_value.call_args.kwargs['extra_body'], EXTRA_BODY)
