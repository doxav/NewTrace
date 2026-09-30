"""Verify real HTTP serialization and serving identities for both frameworks."""

import asyncio
import contextlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'worktrees/trace_cp_b'))
import httpx
import httpx2
import openai
from scripts.preflight import write_json
from src.control_plane import PRIMARY_VARIANT
from src.transport import (
    EXTRA_BODY,
    HTTP_ROLES,
    MODEL,
    PROVIDER,
    REASONING_EFFORT,
    SERVING_PROVIDER,
    SESSION,
    SkyOpenRouter,
    TraceOpenRouter,
    model_config,
)

from opto.features.recursive_opt import spec as S


def observer(original: Callable[..., Any], records: list[dict[str, Any]]) -> Callable[..., Any]:
    """Inspect the actual HTTP body and retain only sanitized response metadata."""
    def send(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        """Fail before sending any completion with a different routing identity."""
        if not str(request.url).endswith('/chat/completions'):
            return original(client, request, *args, **kwargs)
        body = json.loads(request.content)
        if body.get('model') != MODEL or any(body.get(k) != v for k, v in EXTRA_BODY.items()):
            raise ValueError('Actual HTTP request violates EXP22 routing')
        if any(row.get('http_status') == 200 and not row['passed'] for row in records):
            raise ValueError('An earlier serving identity violation prevents further requests')
        record: dict[str, Any] = {
            'role': HTTP_ROLES.get(id(client), 'trace_meta_or_preflight'),
            'outbound_body': body, 'http_status': None, 'passed': False,
            'usage': None, 'started_at_unix': time.time(),
        }
        records.append(record)
        try:
            response = original(client, request, *args, **kwargs)
            record['http_status'] = response.status_code
            response.read()
            value = response.json()
            if not isinstance(value, dict):
                raise TypeError('OpenRouter returned a non-object response')
            record.update({
                'returned_model': value.get('model'), 'serving_provider': value.get('provider'),
                'generation_id': value.get('id'), 'usage': value.get('usage'), 'error': value.get('error'),
                'finish_reasons': [choice.get('finish_reason') for choice in value.get('choices', [])],
                'content_lengths': [len((choice.get('message') or {}).get('content') or '') for choice in value.get('choices', [])],
            })
            record['passed'] = response.status_code == 200 and value.get('provider') == SERVING_PROVIDER and (value.get('model') == MODEL or (value.get('model') or '').startswith(MODEL + '-'))
        except (httpx.HTTPError, httpx2.HTTPError, ValueError, TypeError, AttributeError) as error:
            record['error'] = {'type': type(error).__name__}
            raise
        finally:
            record['finished_at_unix'] = time.time()
        return response
    return send


def main() -> int:
    """Run one sequential smoke per client and require matching serving evidence."""
    path = ROOT / 'artifacts/openrouter_transport_validation.json'
    evidence = json.loads(path.read_text())
    messages = [{'role': 'user', 'content': 'Reply with OK.'}]
    profiles = {'llm_profiles': {'main': {'provider': 'openrouter', 'model': MODEL, 'max_tokens': 32, 'temperature': 0.7, 'request_timeout_s': 600, 'transport_max_attempts': 1, 'openrouter_routing': {'only': [PROVIDER]}, 'request_params': {'extra_body': {'session_id': SESSION, 'reasoning_effort': REASONING_EFFORT}}}}}
    S._normalize_llm_profiles(profiles)
    for name in ('skydiscover', 'trace_cp_a', 'trace_cp_b'):
        if name == 'trace_cp_b' and PRIMARY_VARIANT != 'CP-B':
            evidence[name] = {'status': 'EXCLUDED_TOKEN_ESCALATION', 'requests': []}
            continue
        records: list[dict[str, Any]] = []
        try:
            with contextlib.ExitStack() as stack:
                for module in (httpx, httpx2):
                    stack.enter_context(patch.object(module.Client, 'send', observer(module.Client.send, records)))
                if name == 'skydiscover':
                    asyncio.run(SkyOpenRouter(model_config()).generate('', messages, max_tokens=32))
                else:
                    profile = dict(profiles['llm_profiles']['main'])
                    factory = None
                    if name == 'trace_cp_a':
                        profile.pop('openrouter_routing')
                        factory = TraceOpenRouter
                    client = S._make_guarded_role_client(profile, 'forward', factory, {}, S._BudgetGuard({}))
                    client(messages=messages)
            evidence[name] = {'passed': len(records) == 1 and records[0]['passed'], 'requests': records}
        except (openai.APIError, httpx.HTTPError, httpx2.HTTPError, ValueError, TypeError, AttributeError, RuntimeError) as error:
            evidence[name] = {'passed': False, 'requests': records, 'exception_type': type(error).__name__}
        write_json(path, evidence)
        print(json.dumps({'client': name, 'passed': evidence[name]['passed']}))
        if not evidence[name]['passed'] and (name != 'trace_cp_b' or PRIMARY_VARIANT == 'CP-B'):
            return 2
    evidence['trace'] = {'primary_variant': PRIMARY_VARIANT, 'portable': PRIMARY_VARIANT == 'CP-B'}
    active_trace = 'trace_cp_a' if PRIMARY_VARIANT == 'CP-A' else 'trace_cp_b'
    evidence['passed'] = evidence['direct']['passed'] and all(evidence[name]['passed'] for name in ('skydiscover', active_trace))
    write_json(path, evidence)
    return 0 if evidence['passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
