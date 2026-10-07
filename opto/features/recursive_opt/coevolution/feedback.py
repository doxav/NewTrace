"""Meta-level feedback: what a policy proposer sees about the running search.

Mirrors EvoX's meta prompt (search window context, population state, per-decision
execution trace, parent/context policies with their window statistics, problem context)
with optional LLM summaries produced through the ``feedback`` role. Summaries are made
exactly where EvoX makes them: population insight on every prompt build, problem context
once per run (cached), and a batch summary of context policies when any exist.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .scheduling import ArchiveEntry
from .state import filter_statistics

LLMText = Callable[[str, str], str]

POLICY_CONTRACT = """class Policy:
    def __init__(self, labels): ...            # labels: {name: instruction text}, e.g. 'diverge', 'refine'
    def observe(self, candidate): ...          # called once per new candidate (and for the existing population on deployment)
    def sample(self, population, rng, num_context):
        return parent, contexts, label_name    # candidates from population.members; label_name in labels or ''
Candidate fields: id, content, metrics (dict, main key 'combined_score'), iteration, parent_id, context_ids, label, artifacts.
Population: .members (insertion order), .score(candidate), .best(), .get(id), .statistics(). Use rng for every random choice.
Never modify candidates."""

# Design brief condensed from SkyDiscover's evox_search_sys_prompt.txt (label, context and diversity rules), restated
# for POLICY_CONTRACT. Used as the TraceProposer instruction when CoevolutionConfig.meta_brief is set.
EVOX_POLICY_BRIEF = """Rewrite the selection policy so that it improves how fast the best solution improves.
The policy decides (1) which solution to mutate next (the parent), (2) which other solutions to show as context,
and (3) which variation instruction (label) to attach.

LABEL RULES:
- The label MUST be '' (no instruction) by default; let parent and context selection do the work.
- Use labels only when progress is stagnating, and choose them from the search state, not a fixed rule:
  'diverge' when the current approaches look fundamentally limited (a new direction is needed);
  'refine' when a promising, recently found candidate needs a few iterations of refinement to reach its potential.
- Do not overuse any label or any program as the target of a label.
- With 'diverge' or 'refine' you may return an empty context list, for a targeted variation of the parent alone.

SELECTION PRINCIPLES:
- Avoid deterministic selections that always pick the same parent or the same contexts.
- Context should give complementary perspectives: different approaches, contrasting examples, different score ranges.
- Exploit or explore? Is the population diverse or converging? Diversity itself can be a selection signal.
- When progress stalls, ask whether parent/context selection is the issue or the reasonable variations are exhausted.
- Count improvements as meaningful only above 1% relative or 0.01 absolute. Keep the policy simple.
Change the policy only as far as the evidence supports, and keep the same interface."""

META_SYSTEM = """You are an expert coder evolving a search algorithm for program optimization.
The search algorithm is a selection policy: it decides which previously evaluated solution (the parent) an LLM mutates next,
which other solutions to show as context, and which variation instruction (label) to attach, e.g. a DIVERGE or REFINE instruction.
Your policy runs on the downstream problem for a window of iterations and is scored by how much the best solution improves:
    policy score = ((best_score - start_score) * (1 + log(1 + start_score))) / sqrt(horizon)
Implement the contract below. Return the COMPLETE policy source in one ```python block.

""" + POLICY_CONTRACT

STATS_SYSTEM = """Summarize the population state with NUMBER-BACKED observations: state (one sentence), 3-4 key numbers,
2-3 factual patterns (parent selection, context selection, outcomes of parent/context/child scores, overused programs, label usage).
Every statement must cite a number from the stats. No recommendations."""

PROBLEM_SYSTEM = """Summarize the downstream problem for a search algorithm designer in under 100 words:
**Task:** what the solution must optimize. **Scoring:** combined_score = <formula from the evaluator's final return>. **Goal:** Maximize `combined_score`."""

BATCH_SYSTEM = """You are summarizing previous search algorithm attempts. For EACH program give, in under 60 words:
SIGNALS OBSERVED, WHAT SEEMED TO HELP, POTENTIAL KEY INSIGHT. Ground statements in the statistics. Use [PROGRAM N] markers."""


def _fmt(value: Any) -> str:
    return f'{value:.4f}' if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def population_state(statistics: Mapping[str, Any]) -> str:
    """Readable population statistics (subset of SkyDiscover format_population_state)."""
    if not statistics:
        return ''
    summary = statistics.get('solution_score_summary') or {}
    lines = [f"- population_size: {statistics.get('population_size')}"]
    if summary.get('best') is not None:
        lines.append('- score_distribution: ' + ', '.join(f'{k}={_fmt(summary.get(k))}' for k in ('best', 'q75', 'q50', 'q25', 'worst')))
    if statistics.get('top_solution_scores'):
        lines.append('- top_scores: ' + ', '.join(_fmt(s) for s in statistics['top_solution_scores']))
    recent = statistics.get('recent_solution_stats') or {}
    if recent.get('iterations_without_improvement'):
        lines.append(f"- No improvement by more than {_fmt(recent.get('improvement_threshold'))} for {recent['iterations_without_improvement']} iterations")
    for key in ('most_reused_parent_ratio', 'most_reused_context_ratio'):
        if recent.get(key):
            lines.append(f'- {key}: {_fmt(recent[key])}')
    trace = recent.get('execution_trace') or []
    if trace:
        lines.append('- execution trace (iteration: child score <- parent[label] score | context scores):')
        for entry in trace:
            parent = entry.get('parent')
            parent_text = f"{_fmt(parent[2])}[{parent[0] or '-'}]" if parent else 'none'
            contexts = ', '.join(_fmt(c[2]) for c in entry.get('context') or [])
            lines.append(f"  {entry['iteration']}: {_fmt(entry['program'][1])} <- {parent_text} | {contexts or '-'}")
    return '\n'.join(lines)


class FeedbackComposer:
    """Builds the (system, user) meta prompt for one policy proposal attempt."""

    def __init__(self, summarizer: Optional[LLMText] = None, system_message: str = META_SYSTEM, problem_description: str = '', evaluator_context: str = '') -> None:
        self.summarizer, self.system_message = summarizer, system_message
        self.problem_description, self.evaluator_context = problem_description, evaluator_context
        self._problem_cache: Dict[str, str] = {}
        self.calls: Dict[str, int] = {'stats_insight': 0, 'problem_context': 0, 'batch_summary': 0}

    def _summarize(self, kind: str, system: str, user: str) -> str:
        self.calls[kind] += 1
        try:
            return self.summarizer(system, user) or ''
        except Exception:  # noqa: BLE001 - EvoX drops failed summaries silently
            return ''

    def problem_context(self) -> str:
        template = f'## Problem Description\n{self.problem_description}\n\n## Problem Evaluator\n{self.evaluator_context}'
        if self.summarizer is None or not self.problem_description.strip() or not self.evaluator_context.strip():
            return template
        key = hashlib.sha256(template.encode()).hexdigest()
        if key not in self._problem_cache:
            self._problem_cache[key] = self._summarize('problem_context', PROBLEM_SYSTEM, template)
        return self._problem_cache[key] or template

    def compose(self, parent: ArchiveEntry, context: Sequence[ArchiveEntry], previous: Optional[ArchiveEntry], window: Mapping[str, Any],
                statistics: Mapping[str, Any], errors: Sequence[str] = ()) -> Tuple[str, str]:
        horizon = int(parent.metrics.get('search_horizon') or window.get('horizon') or 0)
        statistics = filter_statistics(statistics, horizon) if horizon > 0 else dict(statistics)
        state = population_state(statistics)
        if self.summarizer is not None and state:
            state = self._summarize('stats_insight', STATS_SYSTEM, f'Population Statistics:\n\n{state}')
        problem = self.problem_context()
        summaries: Dict[str, str] = {}
        documented = [c for c in context if c.metadata.get('start_stats') and c.metadata.get('end_stats')]
        if self.summarizer is not None and documented:
            batch = '\n'.join(f"=== PROGRAM {i} (score={_fmt(c.score)}) ===\nCODE:\n```python\n{c.source}\n```\nSTATS AT START:\n{population_state(c.metadata['start_stats'])}\nSTATS AT END:\n{population_state(c.metadata['end_stats'])}"
                              for i, c in enumerate(documented, 1))
            summaries['all'] = self._summarize('batch_summary', BATCH_SYSTEM, f'Below are {len(documented)} PREVIOUS SEARCH ALGORITHM ATTEMPTS.\n\n{batch}')
        trend = '- Focus on improving the search algorithm score (combined_score)'
        if previous is not None and previous.score is not None and parent.score is not None:
            word = 'improved' if parent.score > previous.score else 'declined' if parent.score < previous.score else 'unchanged'
            trend = f'- Search algorithm score {word}: {_fmt(previous.score)} -> {_fmt(parent.score)}'
        start, total, horizon_w, threshold = window.get('window_start_iteration', 0), window.get('total_iterations', 0), window.get('horizon', 0), window.get('improvement_threshold', 0.0)
        window_text = (f'- Your newly designed search algorithm will start at iteration {start} out of {total}. It will run for at least {horizon_w} iterations '
                       f'(potentially more if improving), but will be cut to just {horizon_w} iterations if it fails to improve the solution score by more than {_fmt(threshold)}.\n'
                       f'- Goal: design a better search strategy (how to select and manage solutions) to improve the downstream solution score.\n'
                       f'- NOTE: exactly one solution is generated per iteration.')
        parts = ['# DOWNSTREAM PROBLEM CONTEXT', problem, '---------------------------', "# Search Algorithm Information\n## Your Algorithm's Search Window", window_text,
                 '## Solution Population Statistics', state, '# What You Are Writing', f'# Current Program (score {_fmt(parent.score)})\n{trend}\n```python\n{parent.source}\n```']
        for index, entry in enumerate(context, 1):
            parts.append(f'## Other Search Algorithm {index} (score {_fmt(entry.score)})\n```python\n{entry.source}\n```')
        if summaries:
            parts.append('## Summaries of previous search algorithms\n' + summaries['all'])
        if errors:
            parts.append('## Previous attempts failed validation\n' + '\n'.join(f'- {e}' for e in errors))
        parts.append('---------------------------\n# Task\nRewrite the program to improve the search algorithm score based on the population state. '
                     'Keep things SIMPLE and keep the same interface.\n```python\n# Your rewritten search algorithm here\n```')
        return self.system_message, '\n\n'.join(p for p in parts if p)


def digest(statistics: Mapping[str, Any]) -> str:
    """Compact JSON digest (for traced evidence)."""
    return json.dumps(statistics, default=str)[:20000]
