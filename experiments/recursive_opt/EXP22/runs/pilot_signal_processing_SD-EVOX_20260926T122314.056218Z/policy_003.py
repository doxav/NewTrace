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
    """Search strategy: alternate best-parent refinement with
    weak-parent + strong-context pairing; REFINE the best when stuck."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_id: Optional[str] = None
        self.best_score: float = -1e18
        self.stagnation: int = 0
        self.since_weak_parent: int = 0  # iterations since last weak-parent pairing
        self._last_sampled_ids: List[str] = []

    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score") if program.metrics else None
        return float(v) if isinstance(v, (int, float)) else -1e18

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        s = self._score(program)
        if s > self.best_score:
            meaningful = (s - self.best_score) > 0.01 or (
                self.best_score > -1e17 and (s - self.best_score) > 0.01 * max(abs(self.best_score), 1e-9)
            )
            if self.best_id is not None and not meaningful:
                self.stagnation += 1
            else:
                self.stagnation = 0
            self.best_score = s
            self.best_id = program.id
        else:
            self.stagnation += 1

        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = [p for p in self.programs.values()]
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 0
        best = self.get(self.best_id) if self.best_id else None
        if best is None or best.id not in self.programs:
            best = max(candidates, key=self._score)

        parent = None
        label = ""
        context: List[EvolvedProgram] = []

        # Deep stagnation: REFINE the best with focused context.
        if self.stagnation >= 2 and self.since_weak_parent >= 1:
            parent = best
            label = self.REFINE_LABEL
            others = sorted((p for p in candidates if p.id != best.id), key=self._score, reverse=True)
            context = others[: max(1, n_ctx // 2)]
            self.stagnation = 0  # give this attempt room
        # Every other iteration: weak parent + strong context (observed winner).
        elif self.since_weak_parent >= 1 and len(candidates) >= 3:
            ranked = sorted(candidates, key=self._score)
            weak_pool = ranked[: max(1, len(ranked) // 3)]
            weak_pool = [p for p in weak_pool if p.id != best.id] or ranked[:1]
            parent = self.rng.choice(weak_pool)
            strong = sorted(
                (p for p in candidates if p.id != parent.id), key=self._score, reverse=True
            )
            context = strong[:n_ctx]
            self.since_weak_parent = 0
        # Default: mutate the best, with diverse context (best + random others).
        else:
            parent = best
            others = [p for p in candidates if p.id != best.id]
            self.rng.shuffle(others)
            context = ([best] if False else []) + others[:n_ctx]
            self.since_weak_parent += 1

        self._last_sampled_ids = [parent.id] + [c.id for c in context]
        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END