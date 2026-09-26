"""Recompute EXP22 evidence, strict metrics and diagnostic stops from saved runs."""

import hashlib
import json
import math
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preflight import git, source_hashes, write_json
from src.accounting import usage_summary


def resolve_roles(requests: list[dict[str, Any]], guide_templates: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve unregistered stock summary pools by exact, source-verified prompts."""
    resolved, evidence = [], []
    for index, request in enumerate(requests):
        row = dict(request)
        if row['role'] == 'trace_meta_or_preflight':
            messages = row['outbound_body'].get('messages', [])
            system = [message['content'].strip() for message in messages if message['role'] == 'system']
            matches = [name for name, content in guide_templates.items() if system == [content.strip()]]
            if len(matches) != 1:
                raise ValueError('Unclassified strict HTTP role cannot be inferred from its exact stock prompt')
            row['role'] = 'guide'
            evidence.append({'request_index': index, 'recorded_role': request['role'], 'resolved_role': 'guide', 'source_template': matches[0]})
        resolved.append(row)
    return resolved, evidence


def stock_guide_templates() -> dict[str, str]:
    """Load only guide templates matching the preregistered stock source hashes."""
    sky = Path('/home/xav/code/evo-compare/repos/skydiscover')
    hashes = json.loads((ROOT/'artifacts/source_hashes.json').read_text())['skydiscover']
    templates = {}
    for name in ('stats_insight_system_message.txt', 'problem_context_summary_system_message.txt', 'batch_summary_prompt.txt'):
        relative = f'skydiscover/optimize/context_builder/evox/templates/{name}'
        path = sky/relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[relative]['working_sha256']:
            raise ValueError('Stock guide template changed since source preflight')
        content = path.read_text()
        templates[relative] = content.split('===SYSTEM===', 1)[1].split('===', 1)[0].strip() if name == 'batch_summary_prompt.txt' else content
    return templates


def events(path: Path) -> list[dict[str, Any]]:
    """Read optional event evidence without inventing missing observations."""
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run_result(directory: Path) -> dict[str, Any]:
    """Read raw results and explicitly verified recovery metadata after interruption."""
    path = directory/'final_result.json'
    result = json.loads(path.read_text())
    recovery_path = directory/'diagnostic_recovery.json'
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text())
        if result.get('passed') or recovery['result'].get('passed') is not False:
            raise ValueError('Recovery must not turn an interrupted run into a passed run')
        if hashlib.sha256(path.read_bytes()).hexdigest() != recovery['raw_final_result_sha256']:
            raise ValueError('Recovery metadata does not match the preserved raw result')
        result = {**result, **recovery['result'], 'diagnostic_recovery': 'diagnostic_recovery.json'}
    return result


def strict_metrics(directory: Path, result: dict[str, Any], requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure observed trajectories; reserve 100-attempt AUC for complete runs."""
    curve = events(directory/'solution_curve.jsonl')
    policies = events(directory/'policy_history.jsonl')
    proposals = events(directory/'policy_proposals.jsonl')
    windows = events(directory/'window_history.jsonl')
    history = events(directory/'candidate_history.jsonl')
    requests, role_evidence = resolve_roles(requests, stock_guide_templates() if any(row['role'] == 'trace_meta_or_preflight' for row in requests) else {})
    initial = result.get('initial_score')
    scores = [row['best_score'] for row in curve]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError('Non-finite score in strict evidence')
    if any(row['iteration'] != index for index, row in enumerate(curve, 1)):
        raise ValueError('Strict curve has missing or duplicate attempt indices')
    if any(right < left - 1e-12 for left, right in pairwise(scores)):
        raise ValueError('Best-so-far curve decreases')
    usage = usage_summary(requests)
    complete = result.get('passed', False) and len(curve) == 100 and usage['roles'].get('solution') == 100
    if result.get('passed') and not complete:
        raise ValueError('Passed strict run does not have 100 observed solution calls')
    curve = [{**row, 'llm_usage': usage_summary(requests, solution_limit=row['iteration'])} for row in curve]
    final = result.get('final_best_score', scores[-1] if scores else None)
    if scores and final is not None and abs(scores[-1] - final) > 1e-12:
        raise ValueError('Final score differs from the observed curve')
    gain = final - initial if final is not None and initial is not None else None
    window_rows = [{**row, 'gain': row['search_window_end_score'] - row['search_window_start_score']} for row in windows]
    activated = [row for row in policies if row['activated']]
    if any(row['population_after'] < row['population_before'] for row in activated):
        raise ValueError('Policy switch lost the persistent population')
    native = result.get('final_metrics', {})
    inverse_pressure = native.get('max_kvpr')
    return {
        'complete': complete, 'observed_attempts': len(curve), 'initial_score': initial,
        'unobserved_solution_attempts': usage['roles'].get('solution', 0) - len(curve),
        'final_best_score': final, 'absolute_gain': gain,
        'relative_gain': gain / abs(initial) if gain is not None and initial else None,
        'auc_100': sum(scores) if complete else None,
        'mean_best_score_100': sum(scores) / 100 if complete else None,
        'observed_auc': sum(scores),
        'first_improvement_iteration': next((row['iteration'] for row in curve if initial is not None and row['best_score'] > initial + 1e-12), None),
        'best_solution_iteration': 0 if gain is not None and gain <= 1e-12 else next((row['iteration'] for row in curve if final is not None and abs(row['best_score'] - final) <= 1e-12), None),
        'policy_switches': len(activated), 'policy_proposals': len(proposals),
        'valid_policy_proposals': sum(row['valid'] for row in proposals),
        'policy_validation_failure_rate': sum(not row['valid'] for row in proposals) / len(proposals) if proposals else None,
        'windows': window_rows,
        'fraction_windows_improving': sum(row['gain'] > 1e-12 for row in window_rows) / len(window_rows) if window_rows else None,
        'usage': usage, 'role_resolution_evidence': role_evidence, 'wall_time_seconds': result.get('wall_time'),
        'semantic_retry_attempts': sum(max(0, row['attempts_used'] - 1) for row in history),
        'transport_retry_attempts': 0,
        'evaluator_calls': None,
        'evaluator_calls_note': 'Stock evaluator reused; exact invocation count was not separately instrumented. Valid/invalid candidates are not a substitute for evaluator calls.',
        'reasoning_tokens': sum(((row.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens', 0) or 0 for row in requests),
        'native_metrics': native,
        'prism_mean_max_pressure_successful_cases': 1 / inverse_pressure if inverse_pressure and inverse_pressure > 0 else None,
        'curve': curve, 'policy_events': policies,
    }


def comparisons(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare completed strict arms only and apply the exploratory eligibility gate."""
    strict = {(row['config']['task'], row['config']['arm']): row for row in attempts if row['config']['stage'] == 'strict' and row.get('metrics', {}).get('complete')}
    pairs = [('SD-EVOX', 'SD-FIXED'), ('TRACE-RECURSIVE', 'TRACE-FIXED'), ('TRACE-RECURSIVE', 'SD-EVOX'), ('TRACE-FIXED', 'SD-FIXED')]
    contrasts = {}
    for task in ('prism', 'signal_processing'):
        contrasts[task] = {}
        for left, right in pairs:
            if (task, left) in strict and (task, right) in strict:
                a, b = strict[task, left]['metrics'], strict[task, right]['metrics']
                contrasts[task][f'{left} - {right}'] = {name: a[name] - b[name] for name in ('final_best_score', 'relative_gain', 'auc_100')}
    deployed = any(row['config']['arm'] == 'TRACE-RECURSIVE' and row['metrics']['policy_switches'] > 0 for row in strict.values())
    signals = {task: rows['TRACE-RECURSIVE - TRACE-FIXED']['relative_gain'] for task, rows in contrasts.items() if 'TRACE-RECURSIVE - TRACE-FIXED' in rows}
    positive = any(value > 1e-12 for value in signals.values())
    return {'contrasts': contrasts, 'all_eight_strict_complete': len(strict) == 8,
            'trace_policy_deployed': deployed, 'positive_trace_vs_fixed': positive,
            'advanced_numerically_eligible': len(strict) == 8 and deployed and positive,
            'advanced_task_if_eligible': max(signals, key=signals.get) if len(strict) == 8 and deployed and positive else None,
            'advanced_decision_note': 'Eligibility additionally requires no unresolved scientific confound; credible adaptation without a positive final contrast requires explicit evidence review.'}


def write_report(report: dict[str, Any]) -> None:
    """Publish observed strict evidence with explicit missing-comparison limits."""
    strict = [row for row in report['attempts'] if row['config']['stage'] == 'strict']
    lines = [f"STATUS: {report['status']}", '',
             'FACT: All low-effort S0–S5 gates passed before strict execution. Two independent SkyDiscover PRISM checks and one Trace check returned valid code with 886–996 completion tokens (51–74 reasoning tokens). The two earlier default-reasoning PRISM calls each exhausted 32,000 completion tokens without code. Their evidence is archived separately.', '',
             'FACT: The fixed route is `z-ai/glm-5.3-flash` through OpenRouter, provider `novita`, reasoning effort `low`, session `benchmark-PRIMS-SIGNAL-run-001`, temperature 0.7, maximum 32,000 tokens and timeout 600 seconds. Trace uses CP-A: exact transport with a nonportable, nonpromotable control-plane override. CP-B was excluded because its empty-response fallback changes the token ceiling.', '',
             'MEASURED RESULT: Eight five-attempt pilots passed. Trace PRISM deployed two generated policies while retaining its population. A Signal Trace S4 diff-format failure is preserved alongside its successful bounded retry. Pilots are excluded from strict quality comparisons.', '',
             'MEASURED RESULT: Strict runs below start from the stock initial solution and consume up to 100 solution HTTP attempts. Partial runs have no 100-attempt AUC and are excluded from contrasts.', '',
             '| Task | Arm | Attempts | Initial | Final best | Gain | Relative gain | AUC / 100 | First / best iteration | Status |',
             '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    if report.get('execution_stop'):
        lines[2:2] = [
            'FACT: Execution stopped because the stock evaluator reported a 360-second timeout while its candidate worker continued running. Candidate 21 completed evaluation while that earlier worker was still CPU-active, violating the required sequential execution. The owned process was cancelled; all HTTP evidence was flushed before its remaining worker was terminated. This is an execution-validity failure, not a negative result about Trace meta-optimization.', '',
            'MEASURED RESULT: PRISM SD-EVOX completed 100 attempts. PRISM TRACE-RECURSIVE recorded 21 outcomes and 22 solution HTTP calls; the last returned call was interrupted before evaluation. Its raw `final_result.json` is preserved, with explicitly hashed recovery metadata for the best source and metrics. The best source first appeared at attempt 12, before the timeout. No strict Trace policy proposal had yet occurred. The six remaining strict runs and the advanced phase did not start.', '',
            'FACT: A bounded unpaid reproduction in `artifacts/timeout_diagnostic.json` confirms that the stock outer timeout leaves its thread active and permits a subsequent evaluation; PRISM’s inner executor context also waits for its worker after its nominal timeout. Every reproduction worker was released and joined. No frozen runtime source or source repository was patched after strict execution started.', '',
            'LIMITATION: The full timed-out candidate was not retained: stock retry prompts truncate failed source. Its surviving excerpt is labelled accordingly. The interrupted Trace run has no completed canonical control-plane result or comparable normal kernel wall-time measurement. No missing artifact is represented as a successful result.', '',
        ]
    def number(value: Any) -> str:
        """Format missing measurements explicitly rather than displaying zero."""
        return 'unmeasured' if value is None else f'{value:.8g}'
    for row in strict:
        m, c = row['metrics'], row['config']
        lines.append(f"| {c['task']} | {c['arm']} | {m['observed_attempts']} | {number(m['initial_score'])} | {number(m['final_best_score'])} | {number(m['absolute_gain'])} | {number(m['relative_gain'])} | {number(m['mean_best_score_100'])} | {m['first_improvement_iteration']} / {m['best_solution_iteration']} | {'complete' if m['complete'] else 'diagnostic stop'} |")
    lines += ['', '| Task / arm | Valid / invalid | Policy switches | Valid proposals / total | Improving windows | Solution / meta / guide calls | Input / output / cached tokens | Cost USD | Wall seconds |',
              '|---|---|---:|---|---:|---|---|---:|---:|']
    for row in strict:
        m, c, r = row['metrics'], row['config'], row['result']
        u = m['usage']; roles = u['roles']
        lines.append(f"| {c['task']} / {c['arm']} | {r.get('valid_candidates')} / {r.get('invalid_attempts')} | {m['policy_switches']} | {m['valid_policy_proposals']} / {m['policy_proposals']} | {number(m['fraction_windows_improving'])} | {roles.get('solution', 0)} / {roles.get('meta', 0)} / {roles.get('guide', 0)} | {u['input_tokens']} / {u['output_tokens']} / {u['cached_tokens']} | {u['reported_cost']:.8f} | {number(m['wall_time_seconds'])} |")
    lines += ['', 'MEASURED RESULT: PRISM native components. The stock `max_kvpr` field is inverse mean maximum pressure over successful cases; lower implied pressure is better, but success rate must also be considered.', '',
              '| Arm | Combined score | Inverse pressure | Implied mean maximum pressure | Success rate |', '|---|---:|---:|---:|---:|']
    for row in strict:
        if row['config']['task'] == 'prism':
            m = row['metrics']; native = m['native_metrics']
            lines.append(f"| {row['config']['arm']} | {number(native.get('combined_score'))} | {number(native.get('max_kvpr'))} | {number(m['prism_mean_max_pressure_successful_cases'])} | {number(native.get('success_rate'))} |")
    if report.get('execution_stop'):
        lines += ['', 'INFERENCE: SD-EVOX’s higher native score came with placement success falling from 100% to 14%; it does not establish better placement reliability. The partial Trace best retained 100% success, but unequal budgets and the execution stop prevent an optimizer comparison. `artifacts/best_solution_rechecks.json` records separate sequential stock-evaluator checks of both saved best sources.']
    fields = ('combined_score', 'composite_score', 'correlation', 'noise_reduction', 'slope_changes', 'lag_error', 'success_rate')
    lines += ['', 'MEASURED RESULT: Signal Processing native components.', '', '| Arm | ' + ' | '.join(fields) + ' |', '|---|' + '---:|' * len(fields)]
    if not any(row['config']['task'] == 'signal_processing' for row in strict):
        lines += ['| No strict Signal run executed | ' + ' | '.join('unmeasured' for _ in fields) + ' |']
    for row in strict:
        if row['config']['task'] == 'signal_processing':
            lines.append('| ' + row['config']['arm'] + ' | ' + ' | '.join(number(row['metrics']['native_metrics'].get(field)) for field in fields) + ' |')
    lines += ['', 'MEASURED RESULT: Completed strict contrasts only. Positive score differences favor the first arm.', '',
              '| Task | Contrast | Final score difference | Relative-gain difference | AUC difference |', '|---|---|---:|---:|---:|']
    contrasts = report['comparison']['contrasts']
    for task, pairs in contrasts.items():
        for label, values in pairs.items():
            lines.append(f"| {task} | {label} | {number(values['final_best_score'])} | {number(values['relative_gain'])} | {number(values['auc_100'])} |")
    if not any(contrasts.values()):
        lines.append('| Unmeasured | No completed strict pair | unmeasured | unmeasured | unmeasured |')
    for task in ('prism', 'signal_processing'):
        if any(row['config']['task'] == task for row in strict):
            lines += ['', f'![{task} trajectories and compute](artifacts/{task}_strict_curves.png)']
    lines += ['', 'INFERENCE: ' + ('The strict matrix is complete. See the within-framework contrasts to assess policy evolution; cross-framework score differences alone do not identify the effect of recursion.' if report['comparison']['all_eight_strict_complete'] else 'The strict matrix is incomplete. The unrun within-framework contrasts cannot establish whether Trace or EvoX improves over its fixed policy, or whether Trace matches EvoX at equal budget.'), '',
              'INFERENCE: Advanced phase ' + ('is numerically eligible, pending review of scientific confounds; it has not yet run.' if report['comparison']['advanced_numerically_eligible'] else 'is not authorized by the current evidence gate. It has not run; no additional-freedom benefit has been measured.'), '',
              'LIMITATION: This is a controlled single-run benchmark, not a statistical replication. No p-values are computed. Equal solution attempts do not equal total compute. The shared session and sequential run order can affect cache, latency and cost. Costs are provider-reported; calls with missing cost are not treated as free. Exact evaluator invocation counts were not separately instrumented; candidate counts cannot recover evaluator retries or cascaded stage calls.', '',
              f"MEASURED RESULT: Entire current low-effort series, including diagnostics and pilots: {report['completion_requests']} completion requests, {report['reported_tokens']} reported tokens, ${report['reported_cost_usd']:.8f} reported cost; {report['calls_missing_cost']} calls have unknown cost. Historical DeepInfra and default-reasoning Novita evidence is excluded.", '',
              'FACT: Detailed curves, role accounting, semantic retries, policy validation and per-window gains are in `artifacts/diagnostic_summary.json`. Runtime source hashes are frozen in `artifacts/strict_source_hashes.json`. Each immutable run directory retains source, requests, results and logs.', '',
              'Validation commands: `experiments/recursive_opt/EXP22/.venv/bin/python -I -m unittest discover -s experiments/recursive_opt/EXP22/tests -v`; `ruff check experiments/recursive_opt/EXP22/src experiments/recursive_opt/EXP22/scripts experiments/recursive_opt/EXP22/tests`; `python3 experiments/recursive_opt/EXP22/scripts/analyze.py`; `python3 experiments/recursive_opt/EXP22/scripts/plot_results.py`. Existing targeted Trace tests: 162 passed, six unrelated integration cases deselected. Credential scan required before publication. Generated stock YAML/log whitespace is preserved as execution evidence.']
    if report['status'].startswith('STOPPED'):
        stop = json.loads((ROOT/'artifacts/STOP.json').read_text())
        lines += ['', f"Diagnostic stop: {stop['reason'].rstrip('.')}. Last recorded iteration: {stop['iteration']}; last valid score: {stop['last_valid_score']}. See `artifacts/STOP.json`."]
        lines += ['', 'Recommended next experiment: ' + stop['recommended_fix'], '',
                  'Reproduce the timeout diagnosis without paid calls: `experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/scripts/timeout_diagnostic.py`.']
    validation_path = ROOT/'artifacts/final_validation.json'
    if validation_path.exists() and json.loads(validation_path.read_text()).get('passed'):
        lines += ['', 'FACT: Final validation passed: 32 EXP22 tests, lint, bytecode compilation, unchanged source hashes and frozen runtime, credential scan, saved-source re-evaluation, and visual plot review. No owned benchmark process remains active. Evidence: `artifacts/final_validation.json`.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines) + '\n')


def analyze_deepinfra() -> dict[str, Any]:
    """Collect immutable transport and smoke evidence for a diagnostic report."""
    transport = json.loads((ROOT/'artifacts/openrouter_transport_validation.json').read_text())
    if transport['direct']['outbound_body']['provider'] != {'only': ['DeepInfra']}:
        raise ValueError('Historical DeepInfra analyzer cannot overwrite a different provider series')
    completed = [transport['direct']] + [row for client in ('skydiscover', 'trace_cp_a', 'trace_cp_b') for row in transport[client]['requests']]
    attempts = []
    for directory in sorted((ROOT/'runs').glob('one_*')):
        result = run_result(directory)
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
        result = run_result(directory)
        requests = json.loads((directory/'http_requests.json').read_text())
        records.extend(requests)
        history = [json.loads(line) for line in (directory/'candidate_history.jsonl').read_text().splitlines()]
        row = {'directory': directory.name, 'config': json.loads((directory/'config.json').read_text()), 'result': result, 'http_passed': bool(requests) and all(row['passed'] for row in requests), 'candidate_errors': [row['error'] for row in history if row.get('error')]}
        if row['config']['stage'] == 'strict':
            row['metrics'] = strict_metrics(directory, result, requests)
        attempts.append(row)
    latest = {}
    for row in attempts:
        config = row['config']
        latest[(config['stage'], config['task'], config['arm'])] = row
    failed = [row for row in latest.values() if not row['result']['passed']]
    empty_generation = any('LLM returned None response' in row['candidate_errors'] for row in failed)
    status = 'STOPPED_PROVIDER' if empty_generation or any(not row['http_passed'] for row in failed) or not transport.get('passed', True) else 'STOPPED_PRECHECK' if failed else 'PARTIAL'
    gates = json.loads((ROOT/'artifacts/gates.json').read_text())
    comparison = comparisons(attempts)
    if status == 'STOPPED_PRECHECK' and any(row['config']['arm'] == 'TRACE-RECURSIVE' and row['result'].get('gate_failure') for row in failed):
        status = 'STOPPED_TRACE_NO_PROGRESS'
    if not failed and comparison['all_eight_strict_complete'] and all(gates.get(f'S{i}') is True for i in range(6)):
        status = 'SUCCESS'
    execution_stop_path = ROOT/'artifacts/evaluator_execution_stop.json'
    execution_stop = json.loads(execution_stop_path.read_text()) if execution_stop_path.exists() else None
    if execution_stop and any(row['directory'] == execution_stop['directory'] for row in failed):
        status = execution_stop['status']
    report = {
        'status': status, 'provider': routing, 'reasoning_effort': reasoning_effort, 'primary_control_plane': transport['trace'],
        'completion_requests': len(records), 'successful_http_responses': sum(row['http_status'] == 200 for row in records),
        'reported_tokens': sum((row.get('usage') or {}).get('total_tokens', 0) for row in records),
        'reported_cost_usd': sum((row.get('usage') or {}).get('cost', 0) or 0 for row in records),
        'calls_missing_cost': sum((row.get('usage') or {}).get('cost') is None for row in records),
        'valid_generated_candidates': sum(row['result'].get('valid_candidates', 0) for row in attempts),
        'strict_runs_completed': [row['directory'] for row in attempts if row['config']['stage'] == 'strict' and row['result']['passed']],
        'attempts': attempts, 'gates': gates, 'comparison': comparison, 'execution_stop': execution_stop,
    }
    write_json(ROOT/'artifacts/diagnostic_summary.json', report)
    if failed:
        last = failed[-1]
        write_json(ROOT/'artifacts/STOP.json', {
            'status': status, 'reason': last['result'].get('gate_failure') or last['result'].get('error') or ('Fixed-parameter generation returned no candidate' if empty_generation else 'Required execution gate failed'),
            'iteration': last['result'].get('iterations_observed'), 'last_valid_score': last['result'].get('final_best_score'),
            'last_active_policy': f"runs/{last['directory']}/best_policy.py",
            'evidence': ['artifacts/diagnostic_summary.json', *(['artifacts/evaluator_execution_stop.json', 'artifacts/timeout_diagnostic.json', f"runs/{last['directory']}/candidate_history.jsonl", f"runs/{last['directory']}/diagnostic_recovery.json"] if execution_stop else []), *[f"runs/{row['directory']}/http_requests.json" for row in failed]],
            'recommended_fix': 'Run unchanged benchmark evaluators inside an owned process boundary with enforceable termination on timeout, prove parity and no surviving workers on both arms, then preregister and rerun strict comparisons from stock initial programs.' if execution_stop else 'Establish usable generation under the frozen settings before S4/S5 or strict runs; any request-parameter change requires an explicit protocol amendment.',
        })
    manifest = json.loads((ROOT/'manifest.json').read_text())
    manifest.update({key: report[key] for key in ('status', 'provider', 'completion_requests', 'gates')})
    manifest.update({'successful_completions': report['successful_http_responses'], 'paid_calls': report['successful_http_responses'], 'completed_runs': report['strict_runs_completed']})
    write_json(ROOT/'manifest.json', manifest)
    return report


if __name__ == '__main__':
    report = analyze()
    if 'comparison' in report:
        write_report(report)
    print(json.dumps(report, indent=2))
