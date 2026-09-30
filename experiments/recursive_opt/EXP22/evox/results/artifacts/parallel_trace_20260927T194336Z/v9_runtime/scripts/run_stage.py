"""Run a gated, immutable EXP22 candidate smoke, pilot or strict arm."""

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'worktrees/trace_cp_b'), str(ROOT)]
import httpx
import httpx2
from scripts.framework_smoke import observer
from scripts.preflight import write_json
from src.accounting import usage_summary
from src.control_plane import register, specification
from src.evaluation import EVALUATOR_PROTOCOL, SKY, TASKS
from src.kernel import run_kernel
from src.transport import (
    EXTRA_BODY,
    HTTP_RECORDS,
    MODEL,
    TraceOpenRouter,
)

from opto.features.recursive_opt import spec as S


def redact_diagnostics(directory: Path) -> None:
    """Remove provider account identifiers before sealing diagnostic logs."""
    pattern = re.compile(r'\buser_[A-Za-z0-9]{20,}\b')
    for name in ('stdout.log', 'stderr.log'):
        path = directory/name
        if path.exists():
            path.write_text(pattern.sub('[REDACTED_USER_ID]', path.read_text()))
    history = directory/'candidate_history.jsonl'
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    for row in rows:
        if row.get('error'):
            row['error'] = pattern.sub('[REDACTED_USER_ID]', row['error'])
    history.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def main() -> int:
    """Enforce stage gates and persist a complete result even on diagnostic failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('prism', 'signal_processing'), required=True)
    parser.add_argument('--arm', choices=('SD-EVOX', 'SD-FIXED', 'TRACE-RECURSIVE', 'TRACE-FIXED'), required=True)
    parser.add_argument('--stage', choices=('one', 'pilot', 'strict'), required=True)
    args = parser.parse_args()
    horizon = {'one': 1, 'pilot': 5, 'strict': 100}[args.stage]
    transport = json.loads((ROOT/'artifacts/openrouter_transport_validation.json').read_text())
    if not json.loads((ROOT/'artifacts/evaluator_parity.json').read_text())['passed'] or not transport['passed']:
        raise ValueError('Evaluator and transport gates must pass first')
    if any(transport['direct']['outbound_body'].get(key) != value for key, value in EXTRA_BODY.items()):
        raise ValueError('Transport evidence belongs to a different routing identity')
    if json.loads((ROOT/'artifacts/evaluator_parity.json').read_text()).get('evaluator_protocol') != EVALUATOR_PROTOCOL:
        raise ValueError('Evaluator parity evidence belongs to a different execution protocol')
    if args.stage == 'strict':
        gates = json.loads((ROOT/'artifacts/gates.json').read_text())
        if not all(gates.get(name) is True for name in ('S0', 'S1', 'S2', 'S3', 'S4', 'S5')):
            raise ValueError('Full execution requires all six gates')
        expected = json.loads((ROOT/'artifacts/strict_source_hashes.json').read_text())
        paths = [*sorted((ROOT/'src').glob('*.py')), *[ROOT/'scripts'/name for name in ('run_stage.py', 'framework_smoke.py', 'preflight.py')]]
        actual = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        if actual != expected:
            raise ValueError('Strict runtime source differs from the validated source lock')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory = ROOT/'runs'/f'{args.stage}_{args.task}_{args.arm}_{stamp}'
    directory.mkdir(parents=True, exist_ok=False)
    for origin, name in ((ROOT/'manifest.json', 'source_manifest.json'), (ROOT/'artifacts/environment.json', 'environment.json')):
        shutil.copyfile(origin, directory/name)
    write_json(directory/'openrouter_config_sanitized.json', {'model': MODEL, **EXTRA_BODY, 'temperature': 0.7, 'max_tokens': 32000, 'timeout_seconds': 600})
    write_json(directory/'config.json', {'task': args.task, 'arm': args.arm, 'stage': args.stage, 'horizon': horizon, 'concurrency': 1, 'seed': 42, 'evaluator_protocol': EVALUATOR_PROTOCOL})
    for name in ('solution_curve.jsonl', 'candidate_history.jsonl', 'policy_history.jsonl'):
        (directory/name).touch()
    HTTP_RECORDS.clear()
    records = HTTP_RECORDS
    result: dict[str, Any] = {'passed': False}
    register()
    raw = specification(args.task, args.arm, horizon, directory) if args.arm.startswith('TRACE') else None
    if raw:
        plan = S.compile_plan(raw)
        for name, value in (('raw_spec', raw), ('normalized_spec', plan.spec), ('resolved_execution_plan', plan.explain())):
            write_json(directory/f'{name}.json', value)
    with (directory/'stdout.log').open('w') as stdout, (directory/'stderr.log').open('w') as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        logging.basicConfig(stream=stderr, level=logging.WARNING, force=True)
        try:
            with contextlib.ExitStack() as stack:
                for module in (httpx, httpx2):
                    stack.enter_context(patch.object(module.Client, 'send', observer(module.Client.send, records)))
                if raw:
                    canonical = S.run_spec(raw, resources={'llm_factory': TraceOpenRouter} if raw['runtime']['test_mode'] else None)
                    write_json(directory/'control_plane_result.json', canonical.to_dict())
                    if not canonical.valid:
                        raise ValueError('Canonical Trace execution returned an invalid result')
                    result = dict(canonical.metadata['kernel_result'])
                else:
                    with S._seed_scope(42):
                        result = asyncio.run(run_kernel(args.task, args.arm, horizon, directory))
                result['passed'] = result['iterations_observed'] == horizon and not result.get('gate_failure') and bool(records) and all(record['passed'] for record in records)
        except (ValueError, TypeError, RuntimeError, KeyError, AttributeError) as error:
            kernel_path = directory/'kernel_result.json'
            diagnostic = json.loads(kernel_path.read_text()) if kernel_path.exists() else {}
            result = {**diagnostic, 'passed': False, 'error_type': type(error).__name__, 'error': str(error)}
        finally:
            history = [json.loads(line) for line in (directory/'candidate_history.jsonl').read_text().splitlines()]
            candidates = [row['candidate'] for row in history if row.get('candidate') and not row.get('error')]
            result['valid_candidates'] = len(candidates)
            result['invalid_attempts'] = sum(row['attempts_used'] for row in history) - len(candidates)
            result['passed'] = result['passed'] and bool(candidates) and sum(row['role'] == 'solution' for row in records) == horizon
            if args.stage == 'one':
                initial_source = (SKY/TASKS[args.task]/'initial_program.py').read_text()
                result['passed'] = result['passed'] and len(candidates) == 1 and candidates[0]['solution'] != initial_source
            write_json(directory/'http_requests.json', records)
            write_json(directory/'llm_usage.json', usage_summary(records))
            write_json(directory/'final_result.json', result)
    redact_diagnostics(directory)
    print(json.dumps({'directory': str(directory), 'result': result}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
