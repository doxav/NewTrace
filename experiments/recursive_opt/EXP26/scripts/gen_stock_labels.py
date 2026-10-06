"""Draw labels with stock EvoX's own startup sequence (probe, reset window, generate) for native_stocklabels.

A draw that is empty or equals stock's default templates is a silent fallback: it is kept in draws.jsonl but
rejected, and drawing continues up to --max-draws. The first accepted draw is written to labels_s<seed>.json.
Usage (EXP22 venv; OPENROUTER_API_KEY): python gen_stock_labels.py --seed 42 --out results/<campaign>/stock_labels
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from stock import controller_for, labels_of


async def draw(seed: int, workdir: Path) -> dict:
    controller = controller_for(seed, workdir)
    try:
        await controller._check_meta_llm_availability()
        controller._reset_search_window()
        await controller._generate_variation_operators()
        return {**labels_of(controller), 'guide_available': controller._guide_llm_available}
    finally:
        controller.close()
        controller.search_controller.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-draws', type=int, default=3)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for attempt in range(args.max_draws):
        labels = asyncio.run(draw(args.seed, out / f'work_s{args.seed}_{attempt}'))
        with (out / 'draws.jsonl').open('a') as stream:
            stream.write(json.dumps({'seed': args.seed, 'attempt': attempt, 'at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **labels}) + '\n')
        if not labels['fallback']:
            (out / f'labels_s{args.seed}.json').write_text(json.dumps(labels, indent=1) + '\n')
            print(json.dumps({'seed': args.seed, 'attempt': attempt, 'accepted': True, 'chars': [len(labels['diverge']), len(labels['refine'])]}))
            return
    raise SystemExit(f'seed {args.seed}: every draw fell back to stock defaults; see draws.jsonl')


if __name__ == '__main__':
    main()
