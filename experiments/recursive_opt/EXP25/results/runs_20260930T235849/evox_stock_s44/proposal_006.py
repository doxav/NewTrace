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
    """Search strategy for a plateaued, tightly-packed top tier.

    Strategy:
    - Default: REFINE the best (or a lightly-used top-3) program with the
      top scorers as context — this is what historically produced gains.
    - When stagnating (no meaningful improvement for several iterations),
      DIVERGE from a mid-tier parent with empty context to unlock new ideas.
    - Track usage to avoid over-refining a single parent.
    """

    STAGNATION_DIVERGE_THRESHOLD = 4

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score: float = float("-inf")
        self.iterations_since_improvement: int = 0
        self.parent_use_counts: Dict[str, int] = {}
        self.diverge_since_last: int = 0

    # ------------------------------------------------------------------
    def _score_of(self, program: Optional[EvolvedProgram]) -> float:
        if program is None or not isinstance(program.metrics, dict):
            return float("-inf")
        v = program.metrics.get("combined_score")
        if not isinstance(v, (int, float)):
            return float("-inf")
        return float(v)

    # ------------------------------------------------------------------
    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        # Track meaningful improvement (>1% relative or >0.01 absolute).
        score = self._score_of(program)
        if score > float("-inf"):
            if score > self.best_score * 1.01 + 0.01 or (
                self.best_score == float("-inf") and score > float("-inf")
            ):
                self.best_score = max(self.best_score, score)
                self.iterations_since_improvement = 0
            else:
                self.iterations_since_improvement += 1
            self.best_score = max(self.best_score, score)

        # Track parent usage for diversity.
        if program.parent_id:
            self.parent_use_counts[program.parent_id] = (
                self.parent_use_counts.get(program.parent_id, 0) + 1
            )

        self._update_best_program(program)
        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    # ------------------------------------------------------------------
    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = [p for p in self.programs.values() if self._score_of(p) > float("-inf")]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        candidates.sort(key=self._score_of, reverse=True)
        n_ctx = num_context_programs if num_context_programs else 0

        stagnating = self.iterations_since_improvement >= self.STAGNATION_DIVERGE_THRESHOLD

        if stagnating:
            # DIVERGE: pick a mid-tier parent, empty context for a fresh idea.
            mid_start = max(1, len(candidates) // 3)
            mid_end = max(mid_start + 1, int(len(candidates) * 0.75))
            mid_pool = candidates[mid_start:mid_end] or candidates
            parent = self.rng.choice(mid_pool)
            self.iterations_since_improvement = max(
                0, self.iterations_since_improvement - 2
            )
            self.diverge_since_last = 0
            return {self.DIVERGE_LABEL: parent}, {}

        # Default: REFINE the top tier. Rotate among top-3, preferring
        # less-used parents to avoid over-refining one program.
        top_k = candidates[: min(3, len(candidates))]
        top_k.sort(key=lambda p: self.parent_use_counts.get(p.id, 0))
        parent = top_k[0] if self.rng.random() < 0.7 else self.rng.choice(top_k)

        # Context: best scorers (excluding parent) plus one diverse mid-tier.
        context_ids = []
        examples: List[EvolvedProgram] = []
        for p in candidates:
            if p.id != parent.id and len(examples) < max(1, n_ctx - 1):
                examples.append(p)
                context_ids.append(p.id)
        # Add one mid/low-tier program for a contrasting perspective.
        if n_ctx > 1 and len(candidates) > 3:
            diverse_pool = candidates[len(candidates) // 2:]
            for p in diverse_pool:
                if p.id != parent.id and p.id not in context_ids:
                    examples.append(p)
                    break

        return {self.REFINE_LABEL: parent}, {"": examples[:n_ctx]}


# EVOLVE-BLOCK-END