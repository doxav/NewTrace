"""EXP28 reference arm: stock SkyDiscover EvoX on PRISM, with EXP25's audited harness (imported, not copied):
Novita-pinned GLM transport, one-hour transient-retry policy, call recording, stock stagnation trigger, stock
evaluator, auto-generated labels. Only the task differs from EXP25's run().

Usage (EXP22 venv, OPENROUTER_API_KEY set): python run_evox_prism.py --seed 42 --out DIR [--horizon 100]
"""
import argparse
import asyncio
import importlib.util
import json
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp25_run_evox_stock', HERE.parents[1] / 'EXP25' / 'scripts' / 'run_evox_stock.py')
E25S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E25S)
W, K, T = E25S.W, E25S.K, E25S.T
TASK = 'prism'


async def run(seed: int, horizon: int, output: Path) -> dict:
    config = K.configuration(TASK, output, fixed=False)
    config.search.database.random_seed = seed
    database = W.create_database('evox', config.search.database)
    benchmark = W.SKY / W.TASKS[TASK]
    controller = K.AuditController(K.DiscoveryControllerInput(config, str(benchmark / 'evaluator/evaluator.py'), database, output_dir=str(output)))
    source = (benchmark / 'initial_program.py').read_text()
    metrics = (await controller.evaluator.evaluate_program(source)).metrics
    initial = W.get_program(config, source, str(uuid.uuid4()), metrics, 0)
    database.add(initial, iteration=0)
    database.initial_program_id, database.initial_program_score = initial.id, W.get_score(metrics)
    started = time.monotonic()
    best = await controller.run_discovery(1, horizon)
    result = {'arm': 'evox_stock', 'task': TASK, 'seed': seed, 'initial_score': W.get_score(metrics), 'final_best_score': W.get_score(best.metrics),
              'best_metrics': best.metrics, 'solution_attempts': len(controller.curve), 'policy_switches': sum(e['activated'] for e in controller.policy_events),
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
    (out / 'run_manifest.json').write_text(json.dumps({'experiment': 'EXP28', 'task': TASK, 'arm': 'evox_stock', 'seed': args.seed, 'horizon': args.horizon,
                                                       'model': T.MODEL, 'provider_routing': T.EXTRA_BODY.get('provider'), 'runner': 'EXP25 run_evox_stock harness, task=prism',
                                                       'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}, indent=1) + '\n')
    try:
        result = asyncio.run(run(args.seed, args.horizon, out))
        result['status'] = 'success' if not result['gate_failure'] else 'gate_failure'
    except Exception as error:  # noqa: BLE001 - recorded in the summary
        result = {'arm': 'evox_stock', 'task': TASK, 'seed': args.seed, 'status': 'error', 'error': f'{type(error).__name__}: {error}'}
    result['skydiscover_calls'] = len(W.CALLS)
    (out / 'summary.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    (out / 'calls_skydiscover.json').write_text(json.dumps(W.CALLS, indent=1) + '\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'final_best_score', 'solution_attempts', 'policy_switches', 'error')}))


if __name__ == '__main__':
    main()
