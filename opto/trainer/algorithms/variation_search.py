"""VariationSearch: PrioritySearch with an explicit, scheduled mutation intent (EvoX-style REFINE / DIVERGE / combine).

PrioritySearch decides *where* to search (which candidate to expand). The edit itself is left to the optimizer's generic
instruction, which neither asks for a local refinement nor for a fundamentally different approach. VariationSearch
adds the missing axis: at every step it chooses a variation mode and appends the matching instruction to the
optimizer's ``objective`` (the ``#Instruction`` section of OptoPrime-style prompts) for that step only.

Modes (instruction texts adapted from SkyDiscover's default DIVERGE / REFINE templates, worded for any trainable type):
    free     the optimizer's own instruction, unchanged
    refine   improve within the same core structure
    diverge  fundamentally different approach (code/text) or a different region / category (numeric / categorical)
    combine  diverge with `num_inspirations` other evaluated candidates (random, non-elite) shown in the prompt

Schedules (when an exploration step happens):
    'stagnation'  explore after `patience` steps without improvement (default);
                  refine for `refine_after_gain` steps after an exploration step that improved; free otherwise
    'periodic'    explore every `period`-th step; refine after such a step improved; free otherwise
    'free'        never changes the instruction (equivalent to PrioritySearch)

Inspirations (what an exploration step shows), independent of the schedule:
    inspiration_mode   'never' (default: plain diverge) | 'always' (every exploration step is a combine step) |
                       'alternate' (exploration steps alternate plain diverge, combine, diverge, ...)
    inspiration_style  'combine' (default: explicit instruction to synthesize from the inspirations) |
                       'context' (EvoX-like: plain diverge instruction, inspirations listed as context only)

Parameter types: only the instruction text changes, so numeric and categorical parameters keep working; the texts
state what diverge/refine mean for them. Optimizers without an ``objective`` attribute are left unchanged.
"""
import random
from typing import Any, Dict, List, Optional

from opto.trainer.algorithms.priority_search import PrioritySearch, ModuleCandidate

VARIATION_INSTRUCTIONS: Dict[str, str] = {
    'refine': """VARIATION MODE: REFINE. Improve the current values within the same core structure.
Do not switch to a fundamentally different approach. For code or text variables: fix weaknesses, improve robustness,
edge cases and parameters, keep the same overall algorithm. For numeric variables: make small, targeted adjustments.
For categorical variables: keep the current choice unless the feedback clearly shows it is wrong.""",
    'diverge': """VARIATION MODE: DIVERGE. Produce a fundamentally different solution than the current one: a real strategy
shift, not minor edits or small tweaks. For code or text variables: use a different algorithm, structure or formulation,
or a reliable library/tool that does the job, and introduce something not present in the current values. For numeric
variables: move to a substantially different region of the allowed range. For categorical variables: choose a
different category. The result must stay valid for the same task and constraints.""",
    'combine': """VARIATION MODE: DIVERGE + COMBINE. Produce a fundamentally different solution by combining the strongest ideas of
the current values with those of the other evaluated solutions listed below (inspirations). Do not copy one of them:
synthesize a new approach that is not present in any of them, and keep it valid for the same task and constraints.""",
}


class VariationSearch(PrioritySearch):
    """PrioritySearch plus an explicit mutation intent chosen per step (see module docstring)."""

    def train(self, guide, train_dataset, *,
              variation_schedule: str = 'stagnation',  # 'stagnation' | 'periodic' | 'free'
              patience: int = 5,  # stagnation: steps without improvement before diverging
              period: int = 3,  # periodic: diverge every `period`-th step
              refine_after_gain: int = 2,  # refine steps after a diverge/combine step that improved the best score
              inspiration_mode: str = 'never',  # 'never' | 'always' | 'alternate' (see module docstring)
              inspiration_style: str = 'combine',  # 'combine' | 'context'
              num_inspirations: int = 2,  # inspirations shown on a combine step (used only when inspiration_mode != 'never')
              inspiration_chars: int = 6000,  # truncation of each inspiration's parameter text
              improvement_threshold: float = 0.0,  # minimum gain of the best score that counts as an improvement
              variation_seed: Optional[int] = None,
              **kwargs):
        if variation_schedule not in ('stagnation', 'periodic', 'free'):
            raise ValueError("variation_schedule must be 'stagnation', 'periodic' or 'free'")
        if inspiration_mode not in ('never', 'always', 'alternate') or inspiration_style not in ('combine', 'context'):
            raise ValueError("inspiration_mode must be 'never', 'always' or 'alternate'; inspiration_style 'combine' or 'context'")
        if patience < 1 or period < 1 or refine_after_gain < 0 or num_inspirations < (0 if inspiration_mode == 'never' else 1):
            raise ValueError('patience and period must be >= 1; refine_after_gain >= 0; num_inspirations >= 1 unless inspiration_mode is never')
        self.variation_schedule, self.patience, self.period = variation_schedule, patience, period
        self.refine_after_gain, self.num_inspirations, self.inspiration_chars = refine_after_gain, num_inspirations, inspiration_chars
        self.inspiration_mode, self.inspiration_style, self._variation_explorations = inspiration_mode, inspiration_style, 0
        self.improvement_threshold = improvement_threshold
        self._variation_rng = random.Random(variation_seed)
        self._variation_best: Optional[float] = None
        self._variation_stall, self._variation_refine_left, self._variation_step = 0, 0, 0
        self._variation_pending: Optional[str] = None  # mode of the step whose outcome is not yet known
        self.variation_log: List[Dict[str, Any]] = []
        return super().train(guide=guide, train_dataset=train_dataset, **kwargs)

    # ---- mode choice
    def _next_mode(self) -> str:
        if self.variation_schedule == 'free':
            return 'free'
        if self._variation_refine_left > 0:
            self._variation_refine_left -= 1
            return 'refine'
        if self.variation_schedule == 'stagnation' and self._variation_stall >= self.patience:
            self._variation_stall = 0
            return self._explore_mode()
        if self.variation_schedule == 'periodic' and self._variation_step % self.period == self.period - 1:
            return self._explore_mode()
        return 'free'

    def _explore_mode(self) -> str:
        """'diverge' or 'combine' for this exploration step, following inspiration_mode."""
        self._variation_explorations += 1
        if self.inspiration_mode == 'always' or (self.inspiration_mode == 'alternate' and self._variation_explorations % 2 == 0):
            return 'combine'
        return 'diverge'

    def _record_outcome(self) -> None:
        """Update stagnation and refine counters from the best score after the last proposal step."""
        best = self._best_candidate_priority if getattr(self, '_best_candidate', None) is not None else None
        if best is None:
            return
        best = float(best)
        improved = self._variation_best is None or best > self._variation_best + self.improvement_threshold
        if self._variation_best is not None and improved and self._variation_pending in ('diverge', 'combine'):
            self._variation_refine_left = self.refine_after_gain
        self._variation_stall = 0 if improved else self._variation_stall + 1
        self._variation_best = best if self._variation_best is None else max(self._variation_best, best)
        if self.variation_log and self.variation_log[-1].get('best_after') is None:
            self.variation_log[-1].update(best_after=best, improved=improved)

    # ---- prompt
    @staticmethod
    def _parameter_text(candidate: ModuleCandidate, limit: int) -> str:
        values = candidate.update_dict or {p: p.data for p in candidate.base_module.parameters()}
        text = '\n'.join(f'{getattr(p, "py_name", str(p))} = {v}' for p, v in values.items())
        return text if len(text) <= limit else text[:limit] + '\n... (truncated)'

    def _inspirations(self, exclude: List[ModuleCandidate]) -> List[ModuleCandidate]:
        """Non-elite inspirations: random candidates from memory other than the ones being expanded."""
        excluded = {id(c) for c in exclude}
        pool = [c for _, c in self.memory if id(c) not in excluded and c.update_dict]
        self._variation_rng.shuffle(pool)
        return pool[:self.num_inspirations]

    def _instruction(self, mode: str) -> str:
        if mode == 'free':
            return ''
        if mode != 'combine':
            return VARIATION_INSTRUCTIONS[mode]
        shown = self._inspirations(self._exploration_candidates)
        if not shown:  # nothing to show yet: plain diverge
            return VARIATION_INSTRUCTIONS['diverge']
        context = self.inspiration_style == 'context'
        name = 'Other evaluated solution' if context else 'Inspiration'
        blocks = [f'{name} {i}' + (f' (mean score {c.mean_score():.4f})' if c.mean_score() is not None else '')
                  + f':\n{self._parameter_text(c, self.inspiration_chars)}' for i, c in enumerate(shown, 1)]
        head = VARIATION_INSTRUCTIONS['diverge'] + '\n\nOther evaluated solutions, for context:' if context else VARIATION_INSTRUCTIONS['combine']
        return head + '\n\n' + '\n\n'.join(blocks)

    def propose(self, samples, verbose: bool = False, **kwargs):
        self._record_outcome()
        mode = self._next_mode()
        instruction = self._instruction(mode)
        optimizers = [o for o in [self.optimizer] + [c.optimizer for c in self._exploration_candidates if c.optimizer is not None]
                      if hasattr(o, 'objective')]
        for o in optimizers:  # the base instruction is recorded once; deep copies of the optimizer inherit it
            if not hasattr(o, '_variation_base_objective'):
                o._variation_base_objective = o.objective
            o.objective = o._variation_base_objective + (f'\n\n{instruction}' if instruction else '')
        self.variation_log.append({'step': self._variation_step, 'mode': mode, 'instruction_chars': len(instruction), 'best_after': None})
        self._variation_pending = mode
        self._variation_step += 1
        candidates = []
        try:
            candidates = super().propose(samples, verbose=verbose, **kwargs)
        finally:
            # New candidates carry deep copies of the optimizer made while the instruction was attached: reset them too,
            # or the instruction would persist into later steps that expand those candidates.
            for o in optimizers + [c.optimizer for c in candidates if c.optimizer is not None]:
                if hasattr(o, '_variation_base_objective'):
                    o.objective = o._variation_base_objective
        return candidates
