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
    """Simple adaptive search database.

    - Parent: mostly top scorers, rotated by usage to avoid overuse;
      occasional mid/low parents for diversity.
    - Context: mix of top performers and diverse others, excluding parent.
    - Stagnation: escalate to REFINE (best) then DIVERGE (least-used lineage).
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score = None
        self.best_id = None
        self.iters_since_improvement = 0
        self.last_label_mode = ""  # "", "refine", "diverge"
        self.parent_usage: Dict[str, int] = {}
        self.recent_scores: List[float] = []

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score") if program.metrics else None
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    def _meaningful(self, new: float, old: float) -> bool:
        return new > old + 0.01 or new > old * 1.01

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
        if s != float("-inf"):
            self.recent_scores.append(s)
            self.recent_scores = self.recent_scores[-20:]

        if self.best_score is None or self._meaningful(s, self.best_score):
            self.best_score = s if self.best_score is None else max(self.best_score, s)
            self.best_id = program.id
            self.iters_since_improvement = 0
        else:
            self.iters_since_improvement += 1

        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = [p for p in self.programs.values() if self._score(p) != float("-inf")]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        candidates.sort(key=self._score, reverse=True)
        n = len(candidates)
        top = candidates[: max(1, n // 3)]
        mid = candidates[max(1, n // 3): max(2, 2 * n // 3)]
        low = candidates[max(2, 2 * n // 3):]

        stuck = self.iters_since_improvement >= 3
        label = ""
        parent = None
        context: List[EvolvedProgram] = []
        num_ctx = num_context_programs or 0

        if stuck:
            if self.last_label_mode != "refine" and self.best_id in self.programs:
                # REFINE the best program, focused context (top peers only)
                parent = self.get(self.best_id)
                label = self.REFINE_LABEL
                self.last_label_mode = "refine"
                for p in top:
                    if p.id != parent.id and len(context) < max(1, num_ctx // 2):
                        context.append(p)
            else:
                # DIVERGE: least-used parent from lower tiers / fresh lineage
                pool = (low + mid) if (low or mid) else candidates
                parent = min(pool, key=lambda p: self.parent_usage.get(p.id, 0))
                label = self.DIVERGE_LABEL
                self.last_label_mode = "diverge"
                context = []
            return {label: parent}, {"": context}

        self.last_label_mode = ""

        # Default: rotate among top scorers weighted by low usage, with
        # occasional mid/low exploration.
        r = self.rng.random()
        if r < 0.65:
            pool = top
        elif r < 0.9 and mid:
            pool = mid
        else:
            pool = low if low else candidates
        parent = min(pool, key=lambda p: (self.parent_usage.get(p.id, 0), -self._score(p)))

        # Context: half top performers (excluding parent), rest diverse others
        seen = {parent.id}
        top_others = [p for p in top if p.id not in seen]
        self.rng.shuffle(top_others)
        for p in top_others[: max(1, num_ctx // 2)]:
            context.append(p)
            seen.add(p.id)
        rest = [p for p in candidates if p.id not in seen]
        self.rng.shuffle(rest)
        for p in rest:
            if len(context) >= num_ctx:
                break
            context.append(p)
            seen.add(p.id)

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END