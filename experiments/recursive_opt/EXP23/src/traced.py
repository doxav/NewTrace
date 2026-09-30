"""Trace module whose graph contains the policy's actual selection decisions.

EXP22 traced ``policy_source -> identity(observation)``: the decisions the policy
made ran outside Trace, so the optimizer saw one opaque scalar in a 30-70k token
JSON dump. Here the solution calls themselves execute inside
``run_selection_window``; its output is a compact per-decision record, and
``summarize_decisions`` computes the score from those records. OptoPrimeV2
therefore sees #Code (policy -> decisions -> score), the decisions in #Others,
and bounded feedback. The generator (LLM + evaluator) remains a black box:
Trace structures evidence about the policy's choices; it cannot differentiate
through the benchmark.
"""

from opto import trace

from src.policies import compile_policy
from src.search import interleave, paired_score, step, summarize

OUTCOME = {None: 'invalid'}


class Arena:
    """Live search state handed to the traced window; printed compactly in prompts."""

    def __init__(self, world, population, seed: int, start: int, calls: int, surface: str, mode: str, incumbent=None, credit: str = 'new_best') -> None:
        if mode not in {'paired', 'solo'}:
            raise ValueError(mode)
        self.world, self.population, self.seed, self.start, self.calls = world, population, seed, start, calls
        self.surface, self.mode, self.incumbent, self.credit = surface, mode, incumbent, credit
        self.records: list[dict] = []

    def __repr__(self) -> str:
        return (f'Arena(task={self.world.task}, world={self.world.name}, first_iteration={self.start + 1}, solution_calls={self.calls}, '
                f'mode={self.mode}, credit={self.credit}, population={len(self.population.members)}, best={self.population.best:.4f})')


def _outcome(record: dict) -> str:
    if record['child'] is None:
        return 'invalid'
    if record['child'] > record['best_before']:
        return 'new_best'
    return 'beat_parent' if record['child'] > record['parent_score'] else 'worse'


@trace.bundle()
def run_selection_window(selection_policy, arena):
    """Deploy selection_policy on the live population for arena.solution_calls solution calls.

    In paired mode each call is assigned at random to the challenger (selection_policy)
    or to the incumbent policy, so both act in the same search stage. Returns one line per
    decision: 'role parent_rank=<0 is best> parent_uses=<prior reuse> outcome'.
    """
    select = compile_policy(arena.surface, selection_policy)
    tags = interleave(arena.calls, arena.seed, arena.start) if arena.mode == 'paired' else ['challenger'] * arena.calls
    lines = []
    for offset, tag in enumerate(tags):
        record = step(arena.world, arena.population, select if tag == 'challenger' else arena.incumbent, arena.seed, arena.start + offset + 1, tag)
        arena.records.append(record)
        lines.append(f"{tag} parent_rank={record['parent_rank']} parent_uses={record['parent_uses']} {_outcome(record)}")
    return lines


@trace.bundle()
def summarize_decisions(decisions, arena):
    """Aggregate the window's decisions by role, parent rank and reuse, and score the challenger.

    score = challenger minus incumbent mean credit (paired mode), or the challenger's
    mean credit (solo mode). Credit 'new_best' counts children that beat the global best.
    """
    records = arena.records
    if arena.mode == 'paired':
        score = paired_score(records, arena.credit, arena.world.scale)['score']
    else:
        from src.search import credit
        score = sum(credit(r, arena.credit, arena.world.scale) for r in records) / max(1, len(records))
    return {'score': score, 'best_after': arena.population.best, 'task': arena.world.task, 'world': arena.world.name, 'seed': arena.seed,
            'horizon': arena.start + arena.calls, **summarize(records, arena.world.scale)}


class SelectionPolicy(trace.Module):
    """One trainable string: the parent-selection policy on the chosen surface."""

    def __init__(self, surface: str, text: str) -> None:
        super().__init__()
        description = ('JSON knobs {temperature in [0.02,5], epsilon in [0,1], reuse_penalty in [0,3], elite_k in [1,64]} for rank-softmax parent selection.'
                       if surface == 'knobs' else 'Python source defining select_parent(members, rng) -> index; see its docstring.')
        self.surface = surface
        self.selection_policy = trace.node(text, trainable=True, name='selection_policy', description=description)

    def forward(self, arena: Arena):
        """Run the traced window on the arena and return the traced summary."""
        return summarize_decisions(run_selection_window(self.selection_policy, arena), arena)


def feedback_text(summary: dict, incumbent_text: str) -> str:
    """Bounded feedback: score semantics, per-bucket outcomes, and the incumbent it was compared with."""
    lines = [f"score={summary['score']:+.4f} (challenger minus incumbent new-best rate; >0 means the proposal beat the incumbent in the same stage).",
             f"decisions={summary['decisions']} policy_errors={summary['policy_errors']} best_after={summary['best_after']:.4f}",
             'outcomes by role|parent rank|reuse:']
    for key, row in summary['by_tag_parent_rank_reuse'].items():
        lines.append(f'  {key}: {row}')
    lines.append(f'incumbent policy: {incumbent_text}')
    return '\n'.join(lines)
