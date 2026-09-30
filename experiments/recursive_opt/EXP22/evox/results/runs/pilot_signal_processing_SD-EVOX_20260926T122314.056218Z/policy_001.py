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


def _score(p: EvolvedProgram) -> float:
    v = p.metrics.get("combined_score") if p.metrics else None
    return float(v) if isinstance(v, (int, float)) else float("-inf")


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive search database.

    Principles:
    1. Exploit the best program by default (refine what works), but
       occasionally mutate weaker/underexplored parents since a weak parent
       recently produced the best child.
    2. Track stagnation in add(); when stuck, escalate with REFINE on the
       best program, then DIVERGE from it to escape plateaus.
    """

    STAGNATION_REFINE = 2   # iterations without meaningful improvement
    STAGNATION_DIVERGE = 4

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score: float = float("-inf")
        self.best_id: Optional[str] = None
        self.stagnation: int = 0
        self.since_diverge: int = 0
        self.parent_use_count: Dict[str, int] = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        # Track progress state here (persists across checkpoints).
        s = _score(program)
        meaningful = s > self.best_score + max(0.01, 0.01 * abs(self.best_score if self.best_score == self.best_score else 0))
        if self.best_id is None or s > self.best_score:
            if meaningful or self.best_id is None:
                self.stagnation = 0
            else:
                self.stagnation += 1
            if s > self.best_score:
                self.best_score = s
                self.best_id = program.id
        else:
            self.stagnation += 1
        self.since_diverge += 1

        pid = program.parent_id
        if pid:
            self.parent_use_count[pid] = self.parent_use_count.get(pid, 0) + 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        label = ""
        best = self.get(self.best_id) if self.best_id else None
        if best is None:
            best = max(candidates, key=_score)
            self.best_id = best.id

        # Escalating anti-stagnation strategy.
        if self.stagnation >= self.STAGNATION_DIVERGE and self.since_diverge >= self.STAGNATION_DIVERGE:
            label = self.DIVERGE_LABEL
            parent = best
            self.since_diverge = 0
        elif self.stagnation >= self.STAGNATION_REFINE:
            label = self.REFINE_LABEL
            parent = best
        else:
            # Mostly exploit the best; ~30% pick a less-used / weaker parent
            # for diversity (weak parents can yield strong children).
            if self.rng.random() < 0.3 and len(candidates) > 1:
                others = [p for p in candidates if p.id != best.id]
                others.sort(key=lambda p: (self.parent_use_count.get(p.id, 0), -_score(p)))
                parent = others[0]
            else:
                parent = best

        # Context: best + diverse others (avoid repeating same context set).
        pool = [p for p in candidates if p.id != parent.id]
        self.rng.shuffle(pool)
        pool.sort(key=lambda p: -_score(p) * (0.7 + 0.6 * self.rng.random()))
        context = pool[: max(1, num_context_programs or 4)]

        return {label: parent}, {"": context}
# EVOLVE-BLOCK-END