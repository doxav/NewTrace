"""EXP25 benchmark arm: stock SkyDiscover EvoX (CoEvolutionController) on Signal Processing.

Uses EXP22's audited stock controller (via EXP23/livekernel/worker.py for the Novita-pinned GLM transport and
call recording). Stock stagnation trigger, stock evaluator with cascade, labels auto-generated. The controller caps
retries by the remaining budget, so a run consumes exactly `horizon` solution attempts. Transient transport errors
are retried for up to one hour so an outage does not consume budget.

Usage (EXP22 venv, OPENROUTER_API_KEY set): python run_evox_stock.py --seed 42 --out DIR [--horizon 100]
"""

import argparse
import asyncio
import importlib.util
import json
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = HERE.parents[1] / 'EXP23' / 'livekernel' / 'worker.py'
spec = importlib.util.spec_from_file_location('exp23_livekernel_worker', WORKER)
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)  # patches EXP22's kernel accounting and records SkyDiscover calls in W.CALLS
K, T = W.K, W.T

import openai  # noqa: E402

_recorded = T.SkyOpenRouter._call_api


async def _robust_call(self, params):
    """Retry transient provider errors for up to one hour (same policy as the native arms)."""
    first, attempt = time.time(), 0
    while True:
        try:
            return await _recorded(self, params)
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
            if time.time() - first > 3600:
                raise
            await asyncio.sleep(min(120, 10 * 2 ** min(attempt, 4)))
            attempt += 1


T.SkyOpenRouter._call_api = _robust_call


async def run(seed: int, horizon: int, output: Path) -> dict:
    config = K.configuration('signal_processing', output, fixed=False)  # stock EvoX: switch_interval = 10% of horizon
    config.search.database.random_seed = seed
    database = W.create_database('evox', config.search.database)
    benchmark = W.SKY / W.TASKS['signal_processing']
    controller = K.AuditController(K.DiscoveryControllerInput(config, str(benchmark / 'evaluator/evaluator.py'), database, output_dir=str(output)))
    source = (benchmark / 'initial_program.py').read_text()
    metrics = (await controller.evaluator.evaluate_program(source)).metrics
    initial = W.get_program(config, source, str(uuid.uuid4()), metrics, 0)
    database.add(initial, iteration=0)
    database.initial_program_id, database.initial_program_score = initial.id, W.get_score(metrics)
    started = time.monotonic()
    best = await controller.run_discovery(1, horizon)
    result = {'arm': 'evox_stock', 'seed': seed, 'initial_score': W.get_score(metrics), 'final_best_score': W.get_score(best.metrics), 'best_metrics': best.metrics,
              'solution_attempts': len(controller.curve), 'policy_switches': sum(e['activated'] for e in controller.policy_events),
              'meta_failures': controller._meta_evolution_failures, 'wall_s': round(time.monotonic() - started, 1), 'gate_failure': controller.gate_failure}
    (output / 'best_program.py').write_text(best.solution)
    (output / 'best_policy.py').write_text(controller._active_search_algorithm_code)
    controller.close()
    controller.search_controller.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'run_manifest.json').write_text(json.dumps({'experiment': 'EXP25', 'arm': 'evox_stock', 'seed': args.seed, 'horizon': args.horizon, 'model': T.MODEL,
                                                       'provider_routing': T.EXTRA_BODY.get('provider'), 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}, indent=1) + '\n')
    try:
        result = asyncio.run(run(args.seed, args.horizon, out))
        result['status'] = 'success' if not result['gate_failure'] else 'gate_failure'
    except Exception as error:  # noqa: BLE001 - recorded in the summary
        result = {'arm': 'evox_stock', 'seed': args.seed, 'status': 'error', 'error': f'{type(error).__name__}: {error}'}
    result['skydiscover_calls'] = len(W.CALLS)
    (out / 'summary.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    (out / 'calls_skydiscover.json').write_text(json.dumps(W.CALLS, indent=1) + '\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'final_best_score', 'solution_attempts', 'policy_switches', 'error')}))


if __name__ == '__main__':
    main()
