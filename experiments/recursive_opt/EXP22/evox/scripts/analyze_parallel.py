"""Summarize the four fresh parallel Trace runs without mixing earlier series."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze import events, run_result, strict_metrics
from scripts.preflight import write_json

TASKS = ('prism', 'signal_processing')
ARMS = ('TRACE-RECURSIVE', 'TRACE-FIXED')
LIMITATION = ('One stochastic run per task and arm cannot establish statistical reliability. '
              'Workers run concurrently: shared provider load and local CPU contention can affect '
              'latency, cost and timeout outcomes. Equal solution attempts do not equal total compute. '
              'Incomplete pairs cannot establish whether meta-optimization helps.')


def worker_summary(worker: dict[str, Any]) -> dict[str, Any]:
    """Use sealed evidence when available and label live measurements as partial."""
    row = dict(worker)
    row.update({'complete': False, 'metrics': None, 'result': None})
    directory_name = worker.get('run_directory')
    returncode = worker.get('returncode')
    row['state'] = 'running' if returncode is None else 'failed'
    if directory_name is None:
        row['state'] = 'starting' if returncode is None else 'failed before run directory'
        return row
    directory = Path(directory_name)
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError('Worker run_directory must be an existing absolute directory')
    config = json.loads((directory/'config.json').read_text())
    if any(config.get(key) != worker[key] for key in ('task', 'arm')) or config.get('stage') != 'strict' or config.get('horizon') != 100:
        raise ValueError('Worker configuration does not match its strict 100-attempt manifest entry')
    if (directory/'final_result.json').exists() and (directory/'http_requests.json').exists():
        result = run_result(directory)
        metrics = strict_metrics(directory, result, json.loads((directory/'http_requests.json').read_text()))
        row.update({'result': result, 'metrics': metrics, 'complete': metrics['complete'] and returncode == 0,
                    'usage_note': 'Sealed HTTP request evidence; missing reported costs remain unknown.'})
        row['state'] = 'complete' if row['complete'] else ('awaiting worker exit' if returncode is None else 'incomplete')
        return row
    curve = events(directory/'solution_curve.jsonl')
    policies = events(directory/'policy_history.jsonl')
    last = curve[-1] if curve else {}
    row['result'] = {'valid_candidates': last.get('valid_candidates'), 'invalid_attempts': last.get('invalid_candidates')}
    row['metrics'] = {
        'complete': False, 'observed_attempts': len(curve), 'initial_score': None,
        'final_best_score': last.get('best_score'), 'absolute_gain': None, 'relative_gain': None,
        'auc_100': None, 'mean_best_score_100': None,
        'policy_switches': sum(event['activated'] for event in policies),
        'policy_proposals': len(events(directory/'policy_proposals.jsonl')),
        'usage': last.get('llm_usage'), 'native_metrics': {}, 'wall_time_seconds': None,
    }
    row['usage_note'] = ('Live curve snapshot: usage includes only recorded calls through the last '
                         'solution outcome; later meta, guide and in-flight calls may be absent. '
                         'Initial score, native metrics, gains and full-horizon AUC await sealed evidence.')
    return row


def summarize(manifest_path: Path) -> dict[str, Any]:
    """Validate exactly four distinct arms and compare only complete task pairs."""
    manifest = json.loads(manifest_path.read_text())
    workers = manifest.get('workers', [])
    expected = {(task, arm) for task in TASKS for arm in ARMS}
    if len(workers) != 4 or {(worker.get('task'), worker.get('arm')) for worker in workers} != expected:
        raise ValueError('Manifest must contain exactly the four distinct Trace task/arm combinations')
    attempts = [worker_summary(worker) for worker in workers]
    completed = {(row['task'], row['arm']): row['metrics'] for row in attempts if row['complete']}
    contrasts = {}
    for task in TASKS:
        if all((task, arm) in completed for arm in ARMS):
            recursive, fixed = (completed[task, arm] for arm in ARMS)
            contrasts[task] = {name: recursive[name] - fixed[name]
                               if recursive[name] is not None and fixed[name] is not None else None
                               for name in ('final_best_score', 'absolute_gain', 'relative_gain', 'auc_100')}
    return {'manifest': str(manifest_path.resolve()), 'status': manifest.get('status'),
            'started_at_utc': manifest.get('started_at_utc'),
            'snapshot_at_utc': datetime.now(timezone.utc).isoformat(),
            'all_four_complete': all(row['complete'] for row in attempts), 'attempts': attempts,
            'completed_contrasts_recursive_minus_fixed': contrasts, 'limitation': LIMITATION}


def number(value: Any) -> str:
    """Render absent measurements explicitly rather than presenting zero."""
    return 'unmeasured' if value is None else f'{value:.8g}'


def write_report(report: dict[str, Any], directory: Path) -> None:
    """Write a self-contained snapshot beside its manifest, preserving old reports."""
    lines = [f"Parallel Trace comparison — {report['status']}", '',
             f"Snapshot: {report['snapshot_at_utc']}. Each worker targets 100 solution HTTP attempts.", '',
             '| Task | Arm | State | Calls / outcomes | Best score | Gain | Relative gain | AUC / 100 | Valid / invalid | Policies deployed / proposed | Reported USD | Calls missing cost |',
             '|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|']
    for row in report['attempts']:
        metrics, result = row['metrics'] or {}, row['result'] or {}
        usage = metrics.get('usage') or {}
        calls = (usage.get('roles') or {}).get('solution')
        lines.append(f"| {row['task']} | {row['arm']} | {row['state']} | {number(calls)} / {number(metrics.get('observed_attempts'))} | "
                     f"{number(metrics.get('final_best_score'))} | {number(metrics.get('absolute_gain'))} | {number(metrics.get('relative_gain'))} | "
                     f"{number(metrics.get('mean_best_score_100'))} | {number(result.get('valid_candidates'))} / {number(result.get('invalid_attempts'))} | "
                     f"{number(metrics.get('policy_switches'))} / {number(metrics.get('policy_proposals'))} | "
                     f"{number(usage.get('reported_cost'))} | {number(usage.get('calls_missing_cost'))} |")
    lines += ['', ('AUC / 100 is the mean best-so-far score across all 100 attempts; unavailable for incomplete runs. '
              'Live usage is a lower bound through the latest recorded outcome. Costs are provider-reported; missing costs are unknown.'), '',
              '| Task | Completed contrast | Final score difference | Gain difference | Relative-gain difference | AUC difference |',
              '|---|---|---:|---:|---:|---:|']
    contrasts = report['completed_contrasts_recursive_minus_fixed']
    for task, values in contrasts.items():
        lines.append(f"| {task} | TRACE-RECURSIVE − TRACE-FIXED | " + ' | '.join(number(values[name]) for name in ('final_best_score', 'absolute_gain', 'relative_gain', 'auc_100')) + ' |')
    if not contrasts:
        lines.append('| Unmeasured | No completed matched pair | unmeasured | unmeasured | unmeasured | unmeasured |')
    lines += ['', 'Positive differences favor TRACE-RECURSIVE for these observed runs only.', '',
              '| Task | Arm | Native best-solution metrics |', '|---|---|---|']
    for row in report['attempts']:
        native = (row['metrics'] or {}).get('native_metrics') or {}
        lines.append(f"| {row['task']} | {row['arm']} | " + (json.dumps(native, sort_keys=True).replace('|', '\\|') if native else 'unmeasured') + ' |')
    lines += ['', 'PRISM’s max_kvpr is inverse mean maximum pressure over successful cases. Interpret it alongside success_rate.', '', LIMITATION, '',
              'Detailed trajectories, policy events, native metrics, tokens, costs and per-worker evidence locations are in results.json.']
    write_json(directory/'results.json', report)
    (directory/'results.md').write_text('\n'.join(lines) + '\n')


def main() -> int:
    """Publish a snapshot for an explicitly selected parallel-run manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    write_report(summarize(args.manifest), args.manifest.resolve().parent)
    return 0


if __name__ == '__main__':
    sys.exit(main())
