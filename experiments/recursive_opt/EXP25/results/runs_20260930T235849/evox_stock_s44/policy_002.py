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


def _score(p) -> float:
    v = p.metrics.get("combined_score", 0.0)
    return float(v) if isinstance(v, (int, float)) else 0.0


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive search: exploit best with strong context; escalate
    REFINE -> DIVERGE on stagnation; rotate parents after failures."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self._best_score = -1.0
        self._best_id = None
        self._stagnation = 0
        self._escalation = 0  # 0=normal, 1=refine-best, 2=diverge
        self._last_parent_id = None

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        s = _score(program)
        if s > self._best_score + 0.01:
            self._best_score = s
            self._best_id = program.id
            self._stagnation = 0
            self._escalation = 0
        else:
            self._stagnation += 1
            if self._stagnation >= 2:
                self._escalation = min(self._escalation + 1, 2)

        self._update_best_program(program)
        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        ranked = sorted(candidates, key=_score, reverse=True)
        best = self.get(self._best_id) if self._best_id else None
        if best is None or best.id not in self.programs:
            best = ranked[0]

        top = [p for p in ranked[:6] if p.id != best.id]

        # Escalation 2: DIVERGE from a strong non-best parent, empty context
        if self._escalation >= 2:
            pool = [p for p in ranked[:8]
                    if p.id != best.id and p.id != self._last_parent_id]
            parent = self.rng.choice(pool) if pool else top[0]
            self._escalation = 1  # after diverge, try refining best again
            self._last_parent_id = parent.id
            return {self.DIVERGE_LABEL: parent}, {}

        # Escalation 1 (or default): REFINE/best with top-score context
        parent = best
        context = top[: num_context_programs or 4]
        # occasionally mix in one diverse mid-tier program for fresh perspective
        if len(ranked) > 8 and context:
            diverse = self.rng.choice(ranked[8:])
            context = context[: max(1, (num_context_programs or 4) - 1)] + [diverse]

        label = self.REFINE_LABEL if self._escalation == 1 else ""
        if self._escalation == 1:
            self._escalation = 0
        self._last_parent_id = parent.id
        return {label: parent}, {"": context[: num_context_programs or 4]}


# EVOLVE-BLOCK-END