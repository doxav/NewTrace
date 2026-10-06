"""Population-conditioned O0 operator: one LLM mutation of a policy-selected parent.

The prompt is assembled from what the selection policy returned: the parent (code,
metrics, evaluator artifacts), context candidates, a variation label (e.g. diverge /
refine instruction text), a few previous attempts, and failed attempts on retry.
The LLM answers with SEARCH/REPLACE diffs or a full rewrite. Parsing and failure rules
follow SkyDiscover's DiscoveryController so the operator can reproduce EvoX's O0.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .policy import Selection
from .projections import ProjectionError
from .state import Candidate, Population

LLMText = Callable[[str, str], str]
Evaluate = Callable[[str], Tuple[Dict[str, Any], Dict[str, Any]]]

DIFF_PATTERN = re.compile(r'<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE', re.DOTALL)

# SkyDiscover DEFAULT_DIVERGE_TEMPLATE / DEFAULT_REFINE_TEMPLATE (general, task-agnostic).
DEFAULT_LABELS: Dict[str, str] = {
    'diverge': """## IMPORTANT: YOU MUST FOLLOW THE FOLLOWING IN YOUR GENERATION, OTHERWISE THE SOLUTION WILL BE REJECTED.

### Goal
Produce a **fundamentally different** solution than the current one.
This must be a real **strategy shift**, not minor edits or small tweaks.

Allowed: new structure, new strategy, different generation plan, different style/format, new components.
DO NOT do: superficial rewrites, polishing, or the same idea with only small changes.

### Constraints
- Keep the output **valid** for the same task and constraints.
- Prefer changes that introduce something **not present** in the following current solution.
- If reliable tools or shortcuts exist, use them instead of manual re-creation.

### Output
1. APPROACH: One short paragraph describing what is different and why.
2. OUTPUT: The full generated solution.
""",
    'refine': """## IMPORTANT: YOU MUST FOLLOW THE FOLLOWING IN YOUR GENERATION, OTHERWISE THE SOLUTION WILL BE REJECTED.

### Goal
Improve the current solution **within the same core structure**.
Do **not** change the fundamental structure of the following solution; instead make it stronger, cleaner, and more reliable.

Allowed: rewriting for clarity, fixing weaknesses, improving completeness, better edge-case handling, and added polish.
DO NOT switch to a fundamentally different approach.

### Constraints
- Keep the output **valid** for the same task and constraints.
- Spend effort on higher-quality execution and fewer mistakes.

### Output
1. REFINEMENT: One short paragraph describing what you improved and how.
2. OUTPUT: The full refined solution.
""",
}

LABEL_GENERATION_SYSTEM = """You write two reusable instruction blocks for an LLM that mutates candidate solutions.
DIVERGE: demand a fundamentally different approach (different algorithm, library, or formulation) that stays valid.
REFINE: demand keeping the approach and squeezing performance (tuning, polish, edge cases).
Use the problem description and evaluator to name concrete approaches and libraries.
Answer exactly in this format:
=== DIVERGE ===
<instruction block>
=== REFINE ===
<instruction block>"""


def generate_labels(llm: LLMText, task: str, evaluator_context: str = '') -> Dict[str, str]:
    """One LLM call producing task-specific diverge/refine instructions (EvoX variation operators)."""
    reply = llm(LABEL_GENERATION_SYSTEM, f'## Problem\n{task}\n\n## Evaluator\n{evaluator_context}')
    match = re.search(r'=== DIVERGE ===\s*(.*?)\s*=== REFINE ===\s*(.*)', reply or '', re.DOTALL)
    if not match or not match.group(1).strip() or not match.group(2).strip():
        return dict(DEFAULT_LABELS)
    return {'diverge': match.group(1).strip(), 'refine': match.group(2).strip()}


def extract_diffs(text: str) -> List[Tuple[str, str]]:
    return [(search.rstrip(), replace.rstrip()) for search, replace in DIFF_PATTERN.findall(text or '')]


def apply_search_replace(original: str, response: str) -> Tuple[Optional[str], Optional[str]]:
    """Apply SEARCH/REPLACE blocks (first exact line match each). Returns (solution, error)."""
    blocks = extract_diffs(response)
    if not blocks:
        return None, 'No valid diffs found in response'
    lines = original.split('\n')
    for search, replace in blocks:
        search_lines, replace_lines = search.split('\n'), replace.split('\n')
        for i in range(len(lines) - len(search_lines) + 1):
            if lines[i:i + len(search_lines)] == search_lines:
                lines[i:i + len(search_lines)] = replace_lines
                break
    result = '\n'.join(lines)
    if result == original:
        return None, 'Diff SEARCH blocks did not match parent solution - no changes applied'
    return result, None


def parse_full_rewrite(response: str, language: str = 'python') -> Optional[str]:
    """First fenced block in ``language``, else any fenced block, else the raw text."""
    for pattern in (r'```' + re.escape(language) + r'\n(.*?)```', r'```(.*?)```'):
        matches = re.findall(pattern, response or '', re.DOTALL)
        if matches:
            return matches[0].strip()
    return response


def evaluation_failed(metrics: Mapping[str, Any], artifacts: Mapping[str, Any]) -> Optional[str]:
    """SkyDiscover failure rule; returns the error message or None."""
    failed = (metrics.get('validity') in (0, -1)
              or (metrics.get('timeout') is True and metrics.get('validity') is None)
              or (metrics.get('combined_score') == 0 and (metrics.get('error') is not None or 'error' in artifacts)))
    if not failed:
        return None
    error = metrics.get('error') if isinstance(metrics.get('error'), str) else None
    return error or artifacts.get('error') or metrics.get('error_message') or 'Evaluation failed (validity=0)'


@dataclass
class OperatorResult:
    candidate: Optional[Candidate]
    attempts_used: int
    error: Optional[str] = None
    failed_attempts: List[Dict[str, Any]] = field(default_factory=list)
    prompts: List[Tuple[str, str]] = field(default_factory=list)


class PopulationOperator:
    """Generate, parse, evaluate and retry one child of the selected parent."""

    def __init__(self, llm: LLMText, evaluate: Evaluate, system_message: str, mode: str = 'diff', retries: int = 3,
                 labels: Optional[Mapping[str, str]] = None, num_previous_attempts: int = 3, max_solution_chars: int = 60000,
                 language: str = 'python', score_key: str = 'combined_score', timeout_s: Optional[float] = None,
                 projections: Optional[List[Callable[[str], Tuple[str, str]]]] = None) -> None:
        if mode not in {'diff', 'rewrite'}:
            raise ValueError("mode must be 'diff' or 'rewrite'")
        self.llm, self.evaluate, self.system_message, self.mode, self.retries = llm, evaluate, system_message, mode, retries
        self.labels = dict(labels or {})
        self.num_previous_attempts, self.max_solution_chars, self.language = num_previous_attempts, max_solution_chars, language
        self.score_key, self.timeout_s = score_key, timeout_s
        self.projections = list(projections or [])

    def project(self, source: str) -> Tuple[str, List[str]]:
        """Apply every projection in order; raises ProjectionError to reject before evaluation."""
        notes = []
        for projection in self.projections:
            source, note = projection(source)
            if note:
                notes.append(note)
        return source, notes

    # ---- prompt
    def _score(self, metrics: Mapping[str, Any]) -> str:
        value = metrics.get(self.score_key)
        return f'{value:.4f}' if isinstance(value, (int, float)) else str(value)

    def _program(self, candidate: Candidate, heading: str, label_text: str = '') -> str:
        lines = [heading]
        if label_text:
            lines.append(f'\n{label_text}\n')
        lines.append('\n## Program Information\n')
        lines.append(f'{self.score_key}: {self._score(candidate.metrics)}\n')
        if candidate.metrics.get('error'):
            lines.append(f"error: {candidate.metrics['error']}\n")
        others = {k: v for k, v in candidate.metrics.items() if k not in (self.score_key, 'error') and isinstance(v, (int, float, str, bool))}
        if others:
            lines.append('Score breakdown:' + ''.join(f'\n  - {k}: {v:.4f}' if isinstance(v, float) else f'\n  - {k}: {v}' for k, v in others.items()) + '\n')
        lines.append(f'\n```{self.language}\n{candidate.content}\n```\n')
        for key, value in (candidate.artifacts or {}).items():
            if value is None:
                continue
            text = str(value)[:2000]
            lines.append(f"\n## {'Evaluator Feedback' if key == 'feedback' else key}\n{text}\n")
        return ''.join(lines)

    def build_prompt(self, selection: Selection, population: Population, failed: List[Dict[str, Any]]) -> Tuple[str, str]:
        parent = selection.parent
        recent = population.statistics(improvement_threshold=0.01)['previous_programs']
        previous = recent[:self.num_previous_attempts]  # SkyDiscover takes the first N of the ascending recent list
        trend = 'Focus on improving the combined score'
        if recent:
            prev_score = recent[-1].metrics.get(self.score_key, 0.0)
            current = parent.metrics.get(self.score_key, 0.0)
            if isinstance(prev_score, (int, float)) and isinstance(current, (int, float)):
                trend = ('Combined score improved' if current > prev_score else 'Combined score declined' if current < prev_score else 'Combined score unchanged') + f': {prev_score:.4f} -> {current:.4f}'
        sections = [f'# Current Solution Information\n- Main Metrics: {self.score_key}={self._score(parent.metrics)}\n- Focus areas: - {trend}\n',
                    '# Program Generation History\n## Previous Attempts\n']
        for index, candidate in enumerate(previous, 1):
            sections.append(f'### Attempt {index}\n- Changes: {candidate.changes or "n/a"}\n- Metrics: {self.score_key}={self._score(candidate.metrics)}\n')
        if selection.contexts:
            sections.append('## Other Context Solutions\n' + '\n'.join(self._program(c, f'### Context {i}\n') for i, c in enumerate(selection.contexts, 1)))
        if failed:
            sections.append('## Failed Attempts (fix these problems)\n' + '\n'.join(
                f"### Attempt {f['attempt']}\nerror: {f['error']}\n```{self.language}\n{f.get('solution') or ''}\n```" for f in failed))
        sections.append(self._program(parent, '# Current Solution\n', self.labels.get(selection.label, '')))
        if self.mode == 'diff':
            sections.append('# Task\nSuggest improvements to the program that will improve its COMBINED_SCORE.\nUse the exact SEARCH/REPLACE format:\n'
                            '<<<<<<< SEARCH\n# Original code to find and replace (must match exactly)\n=======\n# New replacement code\n>>>>>>> REPLACE\n'
                            'Each SEARCH section must EXACTLY match code in "# Current Solution". If an "## IMPORTANT" instruction is given above, follow it.')
        else:
            sections.append(f'# Task\nRewrite the program to improve its COMBINED_SCORE. If an "## IMPORTANT" instruction is given above, follow it.\n```{self.language}\n# full new program\n```')
        if self.timeout_s:
            sections.append(f'- Time limit: Programs should complete execution within {self.timeout_s} seconds; otherwise, they will timeout.')
        return self.system_message, '\n'.join(sections)

    # ---- one iteration
    def run(self, selection: Selection, population: Population, iteration: int, retries: Optional[int] = None) -> OperatorResult:
        retries = retries or self.retries
        failed: List[Dict[str, Any]] = []
        prompts: List[Tuple[str, str]] = []
        for attempt in range(1, retries + 1):
            system, user = self.build_prompt(selection, population, failed)
            prompts.append((system, user))
            try:
                response = self.llm(system, user)
            except Exception as error:  # noqa: BLE001 - SkyDiscover returns immediately on generation failure
                return OperatorResult(None, attempt, f'LLM generation failed: {error}', failed, prompts)
            if response is None:
                return OperatorResult(None, attempt, 'LLM returned None response', failed, prompts)
            if self.mode == 'diff':
                solution, error = apply_search_replace(selection.parent.content, response)
                changes = f'{len(extract_diffs(response))} diff block(s)' if solution else None
            else:
                solution, error, changes = parse_full_rewrite(response, self.language), None, 'Full rewrite'
                if not solution:
                    error = 'No valid solution found in response'
            if solution and len(solution) > self.max_solution_chars:
                error, solution = f'Generated solution exceeds maximum length ({len(solution)} > {self.max_solution_chars})', None
            if error:
                failed.append({'attempt': attempt, 'error': error, 'solution': solution or ''})
                if attempt < retries:
                    continue
                return OperatorResult(None, retries, f'{error} (after {retries} attempts)', failed, prompts)
            try:
                deployable, notes = self.project(solution)
            except ProjectionError as projection_error:
                failed.append({'attempt': attempt, 'error': f'Projection rejected the candidate: {projection_error}', 'solution': solution})
                if attempt < retries:
                    continue
                return OperatorResult(None, retries, f'Projection rejected the candidate: {projection_error} (after {retries} attempts)', failed, prompts)
            metrics, artifacts = self.evaluate(deployable)
            artifacts = dict(artifacts or {})
            if notes:
                artifacts['projection'] = '; '.join(notes)
            eval_error = evaluation_failed(metrics, artifacts)
            if eval_error:
                failed.append({'attempt': attempt, 'error': eval_error, 'solution': solution})
                if attempt < retries:
                    continue
                return OperatorResult(None, retries, f'Evaluator failed after {retries} attempts: {eval_error}', failed, prompts)
            child = Candidate(id=str(uuid.uuid4()), content=solution, metrics=dict(metrics), iteration=iteration, parent_id=selection.parent.id,
                              context_ids=tuple(c.id for c in selection.contexts), label=selection.label,
                              context_labels=tuple('' for _ in selection.contexts), artifacts=artifacts, changes=changes,
                              metadata={'deployable': deployable} if deployable != solution else {})
            return OperatorResult(child, attempt, None, failed, prompts)
        raise AssertionError('unreachable')
