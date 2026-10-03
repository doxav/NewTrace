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
    """Plateau-focused search strategy.

    Given a converged population, this strategy exclusively exploits the
    productive pattern observed in the search history: refine top-tier
    programs with the best program provided as context. Parents rotate
    among distinct elite programs to avoid overuse, and REFINE labels are
    applied to the best program when stagnation deepens.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score_seen = None
        self.stall_count = 0          # iterations since last meaningful improvement
        self.parent_use: Dict[str, int] = {}  # parent_id -> times used as parent
        self.sample_count = 0

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        val = program.metrics.get("combined_score") if program.metrics else None
        if isinstance(val, (int, float)):
            return float(val)
        return float("-inf")

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add a program to the database and track progress state."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        # Track stagnation: meaningful improvement = >1% relative or >0.01 absolute
        score = self._score(program)
        if self.best_score_seen is None:
            self.best_score_seen = score
        elif score > self.best_score_seen + max(0.01, 0.01 * abs(self.best_score_seen)):
            self.best_score_seen = score
            self.stall_count = 0
        else:
            self.stall_count += 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Picks a parent and context programs.

        Strategy: parent is drawn from the top-score tier (with least-used
        preferred for rotation). Context is the best program plus a couple
        of other distinct top-tier programs. When stagnation is deep, apply
        REFINE to the single best program.
        """
        candidates = [p for p in self.programs.values() if self._score(p) > float("-inf")]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        candidates.sort(key=self._score, reverse=True)
        best = candidates[0]
        n_ctx = num_context_programs if num_context_programs is not None else 4

        self.sample_count += 1

        # Elite pool: top ~8 distinct programs (population is highly converged)
        elite = candidates[: min(8, len(candidates))]

        if self.stall_count >= 4:
            # Deep stagnation: refine the best program directly.
            parent = best
            label = self.REFINE_LABEL
        else:
            # Rotate among elites, preferring least-used parents.
            pool = sorted(elite, key=lambda p: self.parent_use.get(p.id, 0))
            min_use = self.parent_use.get(pool[0].id, 0)
            least_used = [p for p in pool if self.parent_use.get(p.id, 0) == min_use]
            parent = self.rng.choice(least_used)
            label = ""

        self.parent_use[parent.id] = self.parent_use.get(parent.id, 0) + 1

        # Context: best program + other distinct elites (never the parent).
        context: List[EvolvedProgram] = []
        if best.id != parent.id:
            context.append(best)
        for p in elite:
            if p.id != parent.id and all(c.id != p.id for c in context):
                context.append(p)
            if len(context) >= n_ctx:
                break
        # Fallback fill from broader population if needed.
        if len(context) < n_ctx:
            for p in candidates:
                if p.id != parent.id and all(c.id != p.id for c in context):
                    context.append(p)
                if len(context) >= n_ctx:
                    break

        parent_dict = {label: parent}
        context_programs_dict = {"": context}
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END