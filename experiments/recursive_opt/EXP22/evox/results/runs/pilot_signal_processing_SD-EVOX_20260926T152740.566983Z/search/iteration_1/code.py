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
    """Simple score-guided search database.

    Strategy:
    - Track best-score history in add() to measure stagnation.
    - sample(): usually mutate the best program (exploit), sometimes a
      diverse/under-explored program (explore). Context mixes the best
      program with a diverse, score-varied set of others.
    - When stagnating, alternate between DIVERGE (fresh direction from a
      mid-tier program) and REFINE (polish the best program).
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score: float = float("-inf")
        self.best_id: Optional[str] = None
        self.since_improvement: int = 0
        self.stagnation_counter: int = 0
        self.parent_use_count: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> float:
        val = program.metrics.get("combined_score")
        return float(val) if isinstance(val, (int, float)) else float("-inf")

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        score = self._score(program)
        if score > self.best_score * 1.01 + 0.01 or score > self.best_score + 0.01:
            self.best_score = max(self.best_score, score)
            self.best_id = program.id
            self.since_improvement = 0
        else:
            self.since_improvement += 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = [p for p in self.programs.values()]
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = sorted(candidates, key=self._score, reverse=True)
        best = scored[0] if self.best_id in self.programs else scored[0]
        n_ctx = num_context_programs or 0

        stagnating = self.since_improvement >= 2
        label = ""

        if stagnating:
            self.stagnation_counter += 1
            if self.stagnation_counter % 2 == 1:
                # Diverge: fresh direction from a mid-tier program, no context.
                mid = scored[len(scored) // 2] if len(scored) > 2 else scored[-1]
                parent_dict = {self.DIVERGE_LABEL: mid}
                return parent_dict, {}
            else:
                # Refine the best program toward its potential.
                label = self.REFINE_LABEL
                parent = best
        else:
            # Exploit best ~70% of the time; otherwise explore a less-used program.
            if self.rng.random() < 0.7:
                parent = best
            else:
                low_use = [p for p in scored[1:] if self.parent_use_count.get(p.id, 0) == 0]
                parent = self.rng.choice(low_use) if low_use else self.rng.choice(scored[1:] or scored)

        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1

        # Context: best (if not parent) + diverse score-varied others.
        context: List[EvolvedProgram] = []
        if best.id != parent.id:
            context.append(best)
        others = [p for p in scored if p.id not in (parent.id, best.id)]
        if others:
            # Take spread across score ranking for diversity.
            step = max(1, len(others) // max(1, n_ctx))
            picked = others[::step]
            for p in picked:
                if p.id not in {c.id for c in context}:
                    context.append(p)
                if len(context) >= n_ctx:
                    break
        # Ensure at least some context if available.
        if not context and others:
            context.append(self.rng.choice(others))

        parent_dict = {label: parent}
        context_programs_dict = {"": context[:n_ctx]}
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END