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

    Strategy: mostly refine the best program, keep diversity in context,
    and when progress stalls, alternate between refining the best
    (REFINE) and diverging to a fresh lineage (DIVERGE).
    """

    STAGNATION_THRESHOLD = 4
    DIVERGE_INTERVAL = 3  # every Nth stalled iteration, diverge

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score: Optional[float] = None
        self.best_id: Optional[str] = None
        self.stagnation: int = 0
        self.stall_counter: int = 0
        self.last_parent_id: Optional[str] = None
        self.parent_use_count: Dict[str, int] = {}

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

        # Track meaningful improvements (1% relative or 0.01 absolute)
        s = self._score(program)
        if s > float("-inf"):
            if self.best_score is None:
                self.best_score, self.best_id = s, program.id
            elif s > self.best_score + max(0.01, 0.01 * abs(self.best_score)):
                self.best_score, self.best_id = s, program.id
                self.stagnation = 0
            elif program.id == self.best_id:
                self.stagnation = 0
            else:
                self.stagnation += 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def _pick_diverse_context(
        self, exclude_ids: set, num: int
    ) -> List[EvolvedProgram]:
        progs = [p for p in self.programs.values() if p.id not in exclude_ids]
        # sort by score, pick a spread: top, middle, bottom slices
        progs.sort(key=self._score, reverse=True)
        picked: List[EvolvedProgram] = []
        n = len(progs)
        if n == 0:
            return picked
        stride = max(1, n // max(1, num))
        i = 0
        while len(picked) < num and i < n:
            cand = progs[i]
            if cand.id not in {p.id for p in picked}:
                picked.append(cand)
            i += stride
        for cand in progs:
            if len(picked) >= num:
                break
            if cand.id not in {p.id for p in picked}:
                picked.append(cand)
        return picked

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = sorted(candidates, key=self._score, reverse=True)
        best = self.get(self.best_id) if self.best_id else None
        if best is None or best.id not in self.programs:
            best = scored[0]

        parent = best
        label = ""
        context: List[EvolvedProgram] = []

        if self.stagnation >= self.STAGNATION_THRESHOLD:
            self.stall_counter += 1
            if self.stall_counter % self.DIVERGE_INTERVAL == 0:
                # Diverge from a mid-tier program (different lineage from last parent)
                mid = scored[len(scored) // 2 : (3 * len(scored)) // 4] or scored
                fresh = [p for p in mid if p.id != self.last_parent_id] or mid
                parent = self.rng.choice(fresh)
                label = self.DIVERGE_LABEL
                context = []  # targeted divergence
            else:
                # Refine the best with top context
                parent = best
                label = self.REFINE_LABEL
                context = self._pick_diverse_context(
                    {parent.id}, max(1, (num_context_programs or 4))
                )
        else:
            # Default: mutate best or a random top-half program, avoid overuse
            top = scored[: max(1, len(scored) // 2)]
            pool = [
                p
                for p in top
                if self.parent_use_count.get(p.id, 0)
                <= min(self.parent_use_count.values()) + 2
            ] or top
            parent = self.rng.choice(pool) if self.rng.random() < 0.7 else self.rng.choice(candidates)
            context = self._pick_diverse_context(
                {parent.id}, max(1, (num_context_programs or 4))
            )

        self.last_parent_id = parent.id
        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1

        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END