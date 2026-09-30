# EVOLVE-BLOCK-START
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


def _score(program: EvolvedProgram) -> float:
    """Safely extract numeric combined_score."""
    value = program.metrics.get("combined_score", None)
    return float(value) if isinstance(value, (int, float)) else float("-inf")


def _is_meaningful(new: float, old: float) -> bool:
    return new > old * 1.01 or new > old + 0.01


class EvolvedProgramDatabase(ProgramDatabase):
    """Search database tuned for a short window on a plateaued population.

    Strategy:
    - Default: REFINE the best program with top-tier + diverse context
      (evidence: REFINE-on-best produced the largest observed gains).
    - On stagnation: DIVERGE from a *high-tier* (never low-tier) program
      that differs from the best and is least-used, with empty context,
      to inject a fresh strong direction without catastrophic regressions.
    - Track usage counts to avoid overusing any single program.
    """

    STAGNATION_WINDOW = 3

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score: float = float("-inf")
        self.iters_since_improvement: int = 0
        self.stagnation_counter: int = 0
        self.diverge_in_a_row: int = 0
        self.parent_use_count: Dict[str, int] = {}
        self.context_use_count: Dict[str, int] = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        score = _score(program)
        if score > float("-inf"):
            if self.best_score > float("-inf") and score > self.best_score:
                if _is_meaningful(score, self.best_score):
                    self.iters_since_improvement = 0
                    self.stagnation_counter = 0
                else:
                    self.iters_since_improvement = 0
                self.best_score = score
            else:
                self.iters_since_improvement += 1
                if self.iters_since_improvement >= self.STAGNATION_WINDOW:
                    self.stagnation_counter += 1
                    self.iters_since_improvement = 0

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = [p for p in self.programs.values() if _score(p) > float("-inf")]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = sorted(candidates, key=_score, reverse=True)
        best = scored[0]
        n = num_context_programs or 0
        top_third = scored[: max(1, len(scored) // 3)]

        stuck = self.stagnation_counter > 0

        # --- parent selection ---
        if stuck and self.diverge_in_a_row < 2:
            # Diverge from a high-tier program that is NOT the current best,
            # preferring least-used ones for a fresh strong direction.
            self.diverge_in_a_row += 1
            parent_label = self.DIVERGE_LABEL
            pool = [p for p in top_third if p.id != best.id] or top_third
            parent = min(pool, key=lambda p: (self.parent_use_count.get(p.id, 0),
                                              -_score(p)))
            context = []  # empty context focuses divergence on the parent
        else:
            self.diverge_in_a_row = 0
            parent_label = self.REFINE_LABEL if stuck else ""
            parent = best
            # Context: strongest programs other than parent, then diverse picks.
            context: List[EvolvedProgram] = []
            pool = [p for p in scored if p.id != parent.id]
            pool.sort(key=lambda p: (self.context_use_count.get(p.id, 0), -_score(p)))
            # 1-2 top scorers first
            context.extend(pool[: min(2, max(0, n))])
            # remaining slots: least-used across full population (diversity)
            if len(context) < n:
                rest = [p for p in pool
                        if all(p.id != c.id for c in context)]
                rest.sort(key=lambda p: (self.context_use_count.get(p.id, 0),
                                         self.rng.random()))
                context.extend(rest[: n - len(context)])

        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1
        for c in context:
            self.context_use_count[c.id] = self.context_use_count.get(c.id, 0) + 1

        return {parent_label: parent}, {"": context}


# EVOLVE-BLOCK-END