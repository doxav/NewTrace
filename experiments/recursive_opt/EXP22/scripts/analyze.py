"""Recompute the observed EXP22 provider stop; never infer unrun quality results."""

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preflight import git, source_hashes, write_json


def analyze_deepinfra() -> dict[str, Any]:
    """Collect immutable transport and smoke evidence for a diagnostic report."""
    transport = json.loads((ROOT/'artifacts/openrouter_transport_validation.json').read_text())
    if transport['direct']['outbound_body']['provider'] != {'only': ['DeepInfra']}:
        raise ValueError('Historical DeepInfra analyzer cannot overwrite a different provider series')
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


def analyze() -> dict[str, Any]:
    """Summarize only the current routing series without pooling historical runs."""
    transport = json.loads((ROOT/'artifacts/openrouter_transport_validation.json').read_text())
    routing = transport['direct']['outbound_body']['provider']
    reasoning_effort = transport['direct']['outbound_body'].get('reasoning_effort')
    if routing == {'only': ['DeepInfra']}:
        return analyze_deepinfra()
    records = [transport['direct']] + [row for client in ('skydiscover', 'trace_cp_a', 'trace_cp_b') for row in transport.get(client, {}).get('requests', [])]
    attempts = []
    for directory in sorted((ROOT/'runs').glob('*')):
        config_path = directory/'openrouter_config_sanitized.json'
        if not config_path.exists() or json.loads(config_path.read_text())['provider'] != routing:
            continue
        if json.loads(config_path.read_text()).get('reasoning_effort') != reasoning_effort:
            continue
        if not (directory/'final_result.json').exists():
            raise ValueError('Cannot seal analysis while a matching run is unfinished')
        result = json.loads((directory/'final_result.json').read_text())
        requests = json.loads((directory/'http_requests.json').read_text())
        records.extend(requests)
        history = [json.loads(line) for line in (directory/'candidate_history.jsonl').read_text().splitlines()]
        attempts.append({'directory': directory.name, 'config': json.loads((directory/'config.json').read_text()), 'result': result, 'http_passed': bool(requests) and all(row['passed'] for row in requests), 'candidate_errors': [row['error'] for row in history if row.get('error')]})
    latest = {}
    for row in attempts:
        config = row['config']
        latest[(config['stage'], config['task'], config['arm'])] = row
    failed = [row for row in latest.values() if not row['result']['passed']]
    empty_generation = any('LLM returned None response' in row['candidate_errors'] for row in failed)
    status = 'STOPPED_PROVIDER' if empty_generation or any(not row['http_passed'] for row in failed) or not transport.get('passed', True) else 'STOPPED_PRECHECK' if failed else 'PARTIAL'
    gates = json.loads((ROOT/'artifacts/gates.json').read_text())
    report = {
        'status': status, 'provider': routing, 'reasoning_effort': reasoning_effort, 'primary_control_plane': transport['trace'],
        'completion_requests': len(records), 'successful_http_responses': sum(row['http_status'] == 200 for row in records),
        'reported_tokens': sum((row.get('usage') or {}).get('total_tokens', 0) for row in records),
        'reported_cost_usd': sum((row.get('usage') or {}).get('cost', 0) or 0 for row in records),
        'calls_missing_cost': sum((row.get('usage') or {}).get('cost') is None for row in records),
        'valid_generated_candidates': sum(row['result'].get('valid_candidates', 0) for row in attempts),
        'strict_runs_completed': [row['directory'] for row in attempts if row['config']['stage'] == 'strict' and row['result']['passed']],
        'attempts': attempts, 'gates': gates,
    }
    write_json(ROOT/'artifacts/diagnostic_summary.json', report)
    if failed:
        last = failed[-1]
        write_json(ROOT/'artifacts/STOP.json', {
            'status': status, 'reason': 'Fixed-parameter generation returned no candidate' if empty_generation else 'Required smoke gate failed',
            'iteration': last['result'].get('iterations_observed'), 'last_valid_score': last['result'].get('final_best_score'),
            'last_active_policy': f"runs/{last['directory']}/best_policy.py",
            'evidence': ['artifacts/diagnostic_summary.json', *[f"runs/{row['directory']}/http_requests.json" for row in failed]],
            'recommended_fix': 'Establish usable generation under the frozen settings before S4/S5 or strict runs; any request-parameter change requires an explicit protocol amendment.',
        })
    manifest = json.loads((ROOT/'manifest.json').read_text())
    manifest.update({key: report[key] for key in ('status', 'provider', 'completion_requests', 'gates')})
    manifest.update({'successful_completions': report['successful_http_responses'], 'paid_calls': report['successful_http_responses'], 'completed_runs': report['strict_runs_completed']})
    write_json(ROOT/'manifest.json', manifest)
    return report


if __name__ == '__main__':
    print(json.dumps(analyze(), indent=2))
