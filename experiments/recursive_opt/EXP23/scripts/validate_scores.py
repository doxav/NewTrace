"""Test whether a policy score ranks selection policies correctly (offline, no LLM).

Ground truth: mean final best after 100 solution calls with the policy held fixed,
over many seeds. Each scoring design then compares two policies at the SAME cost
(2 x window solution calls) starting from a stage reached with the stock policy:

  sequential_log_window  EvoX: A for one window, then B (random order), LogWindowScorer
  paired_<credit>        A and B interleaved on the live population in the same window
  forked_<credit>        A and B each run one window from copies of the same snapshot

Reports P(score orders the pair like ground truth) and the stage bias of each score.
"""

import argparse
import itertools
import json
import random
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.policies import REFERENCE, knob_selector
from src.search import (
    CREDITS,
    UNIFORM,
    credit,
    interleave,
    log_window,
    paired_score,
    run_fixed,
    step,
)
from src.world import TASKS, WORLDS, Population, World


def burn_in(world: World, seed: int, stage: int) -> Population:
    """Reach a search stage with the stock policy."""
    population = Population.start(world.init)
    for iteration in range(1, stage + 1):
        step(world, population, UNIFORM, seed, iteration)
    return population


def compare(world: World, a, b, seed: int, stage: int, window: int) -> dict:
    """All designs for one (A, B, seed, stage); positive means A judged better."""
    out = {}
    base = burn_in(world, seed, stage)
    # EvoX sequential windows, random order.
    population, order = base.copy(), [('A', a), ('B', b)] if random.Random(f'order:{seed}:{stage}').random() < 0.5 else [('B', b), ('A', a)]
    scores, it = {}, stage
    for name, select in order:
        start = population.best
        for _ in range(window):
            it += 1
            step(world, population, select, seed, it)
        scores[name] = log_window(start, population.best)
    out['sequential_log_window'] = scores['A'] - scores['B']
    # EvoX as deployed: the incumbent always runs first, the challenger second.
    for label, (first, second) in (('challenger_is_A', (b, a)), ('challenger_is_B', (a, b))):
        population, it, got = base.copy(), stage, []
        for select in (first, second):
            start = population.best
            for _ in range(window):
                it += 1
                step(world, population, select, seed, it)
            got.append(log_window(start, population.best))
        out[f'evox_incumbent_first|{label}'] = got[1] - got[0] if label == 'challenger_is_A' else got[0] - got[1]
    # Interleaved paired window on the live population.
    population, records = base.copy(), []
    for offset, tag in enumerate(interleave(2 * window, seed, stage)):
        records.append(step(world, population, a if tag == 'challenger' else b, seed, stage + offset + 1, tag))
    for kind in CREDITS:
        out[f'paired_{kind}'] = paired_score(records, kind, world.scale)['score']
    # Forked: each policy runs one window from the same snapshot, common random numbers.
    ends, branch = {}, {}
    for name, select in (('A', a), ('B', b)):
        population = base.copy()
        branch[name] = [step(world, population, select, seed, stage + k + 1) for k in range(window)]
        ends[name] = population.best
    out['forked_best'] = ends['A'] - ends['B']
    for kind in ('gain', 'pct'):
        out[f'forked_{kind}'] = mean(credit(r, kind, world.scale) for r in branch['A']) - mean(credit(r, kind, world.scale) for r in branch['B'])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth-seeds', type=int, default=1000)
    parser.add_argument('--trials', type=int, default=300)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'results' / 'score_validation.json'))
    args = parser.parse_args()
    report = {'window': args.window, 'cost_per_comparison': 2 * args.window, 'results': {}}
    for task, world_name in itertools.product(TASKS, WORLDS):
        world = World(task, world_name)
        truth = {}
        for name, knobs in REFERENCE.items():
            finals = [run_fixed(world, knob_selector(knobs), 10_000 + s)[-1] for s in range(args.truth_seeds)]
            truth[name] = {'mean': mean(finals), 'sd': stdev(finals)}
        pairs = [(x, y) for x, y in itertools.combinations(REFERENCE, 2) if abs(truth[x]['mean'] - truth[y]['mean']) > 0.1 * max(truth[x]['sd'], truth[y]['sd'])]
        correct: dict[str, list[float]] = {}
        correct_clear: dict[str, list[float]] = {}
        stage_bias: dict[str, dict[int, list[float]]] = {}
        for x, y in pairs:
            better, worse = (x, y) if truth[x]['mean'] > truth[y]['mean'] else (y, x)
            for trial in range(args.trials):
                stage = (0, 20, 40, 60)[trial % 4]
                result = compare(world, knob_selector(REFERENCE[better]), knob_selector(REFERENCE[worse]), 20_000 + trial, stage, args.window)
                clear = abs(truth[better]['mean'] - truth[worse]['mean']) > 0.5 * max(truth[better]['sd'], truth[worse]['sd'])
                for design, value in result.items():
                    hit = 1.0 if value > 0 else 0.5 if value == 0 else 0.0
                    correct.setdefault(design.split('|')[0], []).append(hit)
                    if clear:
                        correct_clear.setdefault(design.split('|')[0], []).append(hit)
        # Stage bias: the same policy against itself should score 0 at every stage.
        for trial in range(args.trials):
            stage = (0, 20, 40, 60)[trial % 4]
            select = knob_selector(REFERENCE['soft_0.3'])
            population = burn_in(world, 30_000 + trial, stage)
            start = population.best
            for k in range(args.window):
                step(world, population, select, 30_000 + trial, stage + k + 1)
            stage_bias.setdefault('log_window_absolute', {}).setdefault(stage, []).append(log_window(start, population.best))
            same = compare(world, select, select, 30_000 + trial, stage, args.window)
            for design, value in same.items():
                if design.endswith('|challenger_is_B'):
                    continue  # mirror of challenger_is_A for identical policies; keep challenger-minus-incumbent only
                design = design.split('|')[0]
                stage_bias.setdefault(design, {}).setdefault(stage, []).append(value)
        report['results'][f'{task}/{world_name}'] = {
            'truth_final_best': {k: {'mean': round(v['mean'], 4), 'sd': round(v['sd'], 4)} for k, v in truth.items()},
            'pairs_tested': len(pairs),
            'p_correct_order': {k: round(mean(v), 3) for k, v in correct.items()},
            'p_correct_order_clear_pairs': {k: round(mean(v), 3) for k, v in correct_clear.items()},
            'same_policy_mean_by_stage': {k: {s: round(mean(v), 4) for s, v in d.items()} for k, d in stage_bias.items()},
        }
        r = report['results'][f'{task}/{world_name}']
        print(f'\n## {task} / {world_name}  (pairs {len(pairs)}, trials/pair {args.trials})')
        print('  truth:', {k: v['mean'] for k, v in r['truth_final_best'].items()})
        for design, p in sorted(r['p_correct_order'].items(), key=lambda kv: -kv[1]):
            clear = r['p_correct_order_clear_pairs'].get(design, float('nan'))
            print(f'  {design:24s} P(correct)={p:.3f} clear-pairs={clear:.3f}  same-policy by stage {r["same_policy_mean_by_stage"].get(design)}')
        print('  log_window absolute, same policy, by stage:', r['same_policy_mean_by_stage']['log_window_absolute'])
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
