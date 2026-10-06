"""EXP26 reference arm: stock SkyDiscover EvoX on Signal Processing, exactly as EXP25, with its labels persisted.

Runs EXP25's run() unchanged; the only addition is a controller subclass that records the generated labels
(and whether they silently fell back to stock defaults) when run() closes the controller. EXP25 never saved them.
Usage (EXP22 venv; OPENROUTER_API_KEY): python run_evox_stock.py --seed 42 --out DIR [--horizon 100]
"""

import argparse
import asyncio
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` drops the script dir
from stock import E25S, K, T, W, labels_of

CAPTURED: dict = {}


class LabelRecordingController(K.AuditController):
    def close(self):
        CAPTURED.update(labels_of(self))
        return super().close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'run_manifest.json').write_text(json.dumps({'experiment': 'EXP26', 'arm': 'evox_stock', 'seed': args.seed, 'horizon': args.horizon,
                                                       'model': T.MODEL, 'provider_routing': T.EXTRA_BODY.get('provider'), 'runner': 'EXP25 run() unchanged',
                                                       'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}, indent=1) + '\n')
    K.AuditController = LabelRecordingController  # EXP25's run() resolves K.AuditController at call time
    try:
        result = asyncio.run(E25S.run(args.seed, args.horizon, out))
        result['status'] = 'success' if not result['gate_failure'] else 'gate_failure'
    except Exception as error:  # noqa: BLE001 - recorded in the summary
        result = {'arm': 'evox_stock', 'seed': args.seed, 'status': 'error', 'error': f'{type(error).__name__}: {error}'}
    result['labels_fallback'] = CAPTURED.get('fallback')
    result['skydiscover_calls'] = len(W.CALLS)
    (out / 'labels.json').write_text(json.dumps(CAPTURED, indent=1) + '\n')
    (out / 'summary.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    (out / 'calls_skydiscover.json').write_text(json.dumps(W.CALLS, indent=1) + '\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'final_best_score', 'solution_attempts', 'labels_fallback', 'error')}))


if __name__ == '__main__':
    main()
