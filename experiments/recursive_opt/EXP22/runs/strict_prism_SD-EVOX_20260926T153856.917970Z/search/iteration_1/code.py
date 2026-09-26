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
    """Adaptive search database.

    Strategy:
    - Track best score history to detect stagnation.
    - Normally mutate the best program (exploit) with context drawn from
      diverse score tiers (best + mid + worst) for contrastive guidance.
    - On stagnation, alternate between DIVERGE (fresh direction from a
      diverse parent) and REFINE (deep polish of the best program).
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score = None
        self.best_id = None
        self.iterations_since_improvement = 0
        self.stall_actions: List[str] = []  # history of stagnation responses

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score") if program.metrics else None
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        s = self._score(program)
        if self.best_score is None or s > self.best_score + max(0.01, 0.01 * abs(self.best_score)):
            self.best_score = s
            self.best_id = program.id
            self.iterations_since_improvement = 0
        else:
            self.iterations_since_improvement += 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def _tiered_context(self, parent: EvolvedProgram, k: int) -> List[EvolvedProgram]:
        """Pick context spanning score tiers: best, a mid, a low scorer, plus random."""
        progs = [p for p in self.programs.values() if p.id != parent.id]
        if not progs:
            return []
        progs.sort(key=self._score, reverse=True)
        picked: List[EvolvedProgram] = []
        seen = {parent.id}
        # top scorer, a middle scorer, a low scorer
        for idx in [0, len(progs) // 2, len(progs) - 1]:
            p = progs[idx]
            if p.id not in seen:
                picked.append(p)
                seen.add(p.id)
        rest = [p for p in progs if p.id not in seen]
        self.rng.shuffle(rest)
        picked.extend(rest)
        return picked[:k]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")
        k = num_context_programs or 0

        best = self.get(self.best_id) if self.best_id else None
        if best is None or best.id not in self.programs:
            best = max(candidates, key=self._score)

        stalled = self.iterations_since_improvement >= 4

        if stalled:
            action = self.stall_actions[-1] if self.stall_actions else None
            if action != self.DIVERGE_LABEL:
                # Diverge: mutate a random non-best program for a fresh direction.
                others = [p for p in candidates if p.id != best.id]
                parent = self.rng.choice(others) if others else best
                label = self.DIVERGE_LABEL
            else:
                # Refine: polish the best program with focused context.
                parent = best
                label = self.REFINE_LABEL
            self.stall_actions.append(label)
            if label == self.REFINE_LABEL:
                # Focused context: top programs only.
                progs = sorted((p for p in candidates if p.id != parent.id),
                               key=self._score, reverse=True)
                context = progs[:k]
                return {label: parent}, {"": context}
            # Diverge: contrasting context (best + diverse tiers).
            context = self._tiered_context(parent, k)
            return {label: parent}, {"": context}

        # Normal mode: mostly exploit the best, sometimes a strong-but-varied parent.
        if self.rng.random() < 0.25:
            others = [p for p in candidates if p.id != best.id]
            parent = self.rng.choice(others) if others else best
        else:
            parent = best
        context = self._tiered_context(parent, k)
        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END