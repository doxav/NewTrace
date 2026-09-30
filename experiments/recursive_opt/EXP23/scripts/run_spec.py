"""Execute one compiled EXP23 alternative through compile_plan -> execute_plan.

Offline specs receive the feedback-blind mock optimizer LLM; live specs need
OPENROUTER_API_KEY in the environment (never printed) and spend real calls.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.control_plane import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='alternative key, e.g. A1/trace_paired-stagnation')
    parser.add_argument('--holdout', type=int, help='override holdout seed count (smoke runs)')
    parser.add_argument('--train', type=int, help='override train seed count')
    parser.add_argument('--iterations', type=int, help='override trainer iterations (trace engine)')
    parser.add_argument('--live', action='store_true', help='required to execute a spec whose runtime.offline is false')
    args = parser.parse_args()
    raw = json.loads((ROOT / 'configs' / args.name / 'raw_spec.json').read_text())
    if not raw['runtime']['offline'] and not args.live:
        sys.exit('refusing to run a live spec without --live')
    level = raw['levels'][0]
    for split, count in (('holdout', args.holdout), ('train', args.train)):
        if count is not None and split in level['datasets']:
            level['datasets'][split]['config']['count'] = count
    if args.iterations is not None:
        level['engine']['config']['iterations'] = args.iterations
    if args.holdout is not None or args.train is not None or args.iterations is not None:
        raw['budget']['candidates'] = None
    (result,) = run(raw)
    summary = {'name': args.name, 'status': result.status, 'error': result.error, 'metrics': dict(result.evaluation.metrics), 'artifact': dict(result.artifact), 'usage': json.loads(json.dumps(result.usage, default=str))}
    out = ROOT / 'results' / 'runs' / (args.name.replace('/', '__') + '.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**summary, 'metadata': json.loads(json.dumps(result.metadata, default=str))}, indent=2) + '\n')
    print(json.dumps(summary, indent=2)[:3000])


if __name__ == '__main__':
    main()
