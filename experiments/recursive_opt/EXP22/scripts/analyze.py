"""Recompute the observed EXP22 provider stop; never infer unrun quality results."""

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preflight import git, source_hashes, write_json


def analyze() -> dict[str, Any]:
    """Collect immutable transport and smoke evidence for a diagnostic report."""
    transport = json.loads((ROOT/'artifacts/openrouter_transport_validation.json').read_text())
    completed = [transport['direct']] + [row for client in ('skydiscover', 'trace_cp_a', 'trace_cp_b') for row in transport[client]['requests']]
    attempts = []
    for directory in sorted((ROOT/'runs').glob('one_*')):
        result = json.loads((directory/'final_result.json').read_text())
        requests = json.loads((directory/'http_requests.json').read_text())
        attempts.append({'directory': directory.name, 'result': result, 'http_statuses': [row['http_status'] for row in requests], 'provider_error_codes': [(row.get('error') or {}).get('metadata', {}).get('provider_error_code') for row in requests]})
    if len(attempts) != 2 or any(row['result']['passed'] or row['result']['valid_candidates'] != 0 or row['http_statuses'] != [429, 429, 429] or row['provider_error_codes'] != ['engine_overloaded'] * 3 for row in attempts):
        raise ValueError('Observed evidence no longer matches this diagnostic stop; extend analysis explicitly')
    report = {
        'status': 'STOPPED_PROVIDER', 'successful_transport_calls': len(completed),
        'transport_tokens': sum(row['usage']['total_tokens'] for row in completed),
        'reported_transport_cost_usd': sum(row['usage']['cost'] for row in completed),
        'rejected_requests': 6, 'rejected_request_cost': None,
        'valid_generated_candidates': 0, 'strict_runs_completed': [], 'attempts': attempts,
        'gates': {'S0': True, 'S1': True, 'S2': True, 'S3': True, 'S4': False, 'S5': False},
    }
    write_json(ROOT/'artifacts/diagnostic_summary.json', report)
    write_json(ROOT/'artifacts/gates.json', report['gates'])
    write_json(ROOT/'artifacts/STOP.json', {'status': report['status'], 'reason': 'DeepInfra shared pool rejected both S4 attempts with HTTP 429 engine_overloaded', 'iteration': 0, 'last_valid_score': 21.891622105209393, 'last_valid_score_origin': 'stock initial program; no generated candidate', 'last_active_policy': 'stock initial_search_strategy.py', 'evidence': ['artifacts/diagnostic_summary.json', *[f"runs/{row['directory']}/http_requests.json" for row in attempts]], 'recommended_fix': 'Retry S4 only when the same GLM/DeepInfra route is available; no provider/model substitution. Complete S4/S5 before strict execution.'})
    manifest = json.loads((ROOT/'manifest.json').read_text())
    patched = ROOT/'worktrees/trace_cp_b'
    manifest.update({'status': report['status'], 'completed_runs': [], 'completion_requests': 10, 'successful_completions': 4, 'gates': report['gates'], 'primary_control_plane': 'CP-B (transport/runtime/mocked parity validated; full execution unvalidated)', 'trace_cp_b': {'head': git(patched, 'rev-parse', 'HEAD'), 'path': str(patched), 'patch': 'artifacts/trace_cp_b.patch', 'status': git(patched, 'status', '--porcelain')}})
    manifest['paid_calls'] = 4
    manifest['gates_not_run'] = ['S4 remaining task/framework candidates', 'S5 pilots', 'strict runs', 'advanced phase']
    write_json(ROOT/'manifest.json', manifest)
    hashes = json.loads((ROOT/'artifacts/source_hashes.json').read_text())
    hashes['trace_cp_b'] = source_hashes(patched, ['opto', 'pyproject.toml', 'setup.py'])
    write_json(ROOT/'artifacts/source_hashes.json', hashes)
    return report


if __name__ == '__main__':
    print(json.dumps(analyze(), indent=2))
