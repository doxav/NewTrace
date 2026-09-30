# EVOLVE-BLOCK-START
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Search strategy database with recency- and diversity-aware sampling.

    Rationale from measured window feedback: the search plateaued (10
    iterations without improvement) while repeatedly sampling near-duplicate
    high scorers. The best new program came from a mid-tier parent with
    diverse context. This policy therefore:
      1. Draws parents with a recency-weighted random choice (recent
         programs are more likely, so improvements propagate faster and
         saturated old lineages are not over-exploited), while occasionally
         picking the global best for exploitation.
      2. Builds context from distinct solutions, mixing top scorers with
         mid-tier programs to expose the LLM to varied, non-redundant code.
    Scalar and multiobjective modes both follow this policy; the
    ProgramDatabase contract (add/sample signatures, initial_program,
    _save_program, _update_best_program) is preserved.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs
    ) -> str:
        """Add a program to the database."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def _score_of(self, program: EvolvedProgram) -> float:
        """Best available scalar score for a program (0.0 if missing)."""
        metrics = getattr(program, "metrics", None) or {}
        for key in ("combined_score", "overall_score", "composite_score"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    def _recency_weighted_choice(
        self, candidates: List[EvolvedProgram]
    ) -> EvolvedProgram:
        """Choose a program weighted by recency (exponential decay) and score.

        Recent programs get higher weight so new improvements are explored
        immediately; scores provide a mild exploitation bias. A small
        probability is reserved for the global best program.
        """
        if len(candidates) == 1:
            return candidates[0]

        # Occasionally exploit the global best directly.
        if self.rng.random() < 0.2:
            return max(candidates, key=self._score_of)

        max_iter = max(getattr(p, "iteration_found", 0) or 0 for p in candidates)
        weights = []
        for p in candidates:
            age = max_iter - (getattr(p, "iteration_found", 0) or 0)
            recency = 0.5**age  # exponential decay by iteration age
            score = max(self._score_of(p), 1e-6)
            weights.append(recency * (0.5 + score))
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(candidates)
        r = self.rng.random() * total
        acc = 0.0
        for p, w in zip(candidates, weights):
            acc += w
            if r <= acc:
                return p
        return candidates[-1]

    def _diverse_context(
        self, parent: EvolvedProgram, candidates: List[EvolvedProgram], k: int
    ) -> List[EvolvedProgram]:
        """Pick up to k context programs distinct from parent and each other.

        Strategy: sort by score, then greedily pick programs whose solution
        text differs from already-picked ones, alternating between top and
        mid-tier scorers to maximize diversity of ideas in context.
        """
        if k is None or k <= 0:
            return []

        pool = [p for p in candidates if p.id != parent.id]
        if not pool:
            return []

        pool_sorted = sorted(pool, key=self._score_of, reverse=True)
        picked: List[EvolvedProgram] = []
        seen_solutions = {parent.solution}

        # Alternate: one from the top of the list, one from the middle band,
        # to mix exploitation exemplars with diverse mid-tier variants.
        top_idx, mid_idx = 0, len(pool_sorted) // 2
        while len(picked) < k and (
            top_idx < len(pool_sorted) or mid_idx < len(pool_sorted)
        ):
            for idx in (top_idx, mid_idx):
                if idx >= len(pool_sorted):
                    continue
                cand = pool_sorted[idx]
                if cand.id in {p.id for p in picked}:
                    continue
                if cand.solution in seen_solutions and len(picked) < k:
                    # Allow duplicates only if we cannot fill otherwise.
                    continue
                picked.append(cand)
                seen_solutions.add(cand.solution)
                if len(picked) >= k:
                    break
            top_idx += 1
            mid_idx += 1

        # Fallback: fill remaining slots with any unused candidates.
        if len(picked) < k:
            for cand in pool_sorted:
                if len(picked) >= k:
                    break
                if cand.id not in {p.id for p in picked}:
                    picked.append(cand)

        return picked[:k]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Picks a parent and set of context programs.

        Parent: recency-weighted random choice (with occasional global-best
        exploitation). Context: diverse, distinct solutions spanning top and
        mid-tier scorers.
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        parent = self._recency_weighted_choice(candidates)
        examples = self._diverse_context(parent, candidates, num_context_programs or 0)

        parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END
