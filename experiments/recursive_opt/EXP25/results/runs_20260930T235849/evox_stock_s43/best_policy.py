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

    Key principles:
    1) Exploit the newest best: on plateau, REFINE the most recent best
       program with EMPTY context (empty-context refinement on a fresh best
       produced the only breakthroughs in prior runs).
    2) Otherwise, soft-exploit high scorers with varied context, rotating
       parents to avoid overusing any single program.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score: Optional[float] = None
        self.best_id: Optional[str] = None
        self.iters_since_improvement = 0
        self.parent_usage: Dict[str, int] = {}
        self.last_stagnation_action = ""  # "", "refine", "diverge"

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score") if program.metrics else None
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        s = self._score(program)
        if s != float("-inf"):
            # Meaningful improvement: >1% relative or >0.01 absolute
            if self.best_score is None or (
                s > self.best_score + 0.01 or s > self.best_score * 1.01
            ):
                self.best_score = s
                self.best_id = program.id
                self.iters_since_improvement = 0
            else:
                self.iters_since_improvement += 1
        else:
            self.iters_since_improvement += 1

        if program.parent_id:
            self.parent_usage[program.parent_id] = (
                self.parent_usage.get(program.parent_id, 0) + 1
            )

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
        top = candidates[: max(1, n // 4)]

        # Plateau: alternate targeted refinement of the best vs divergence
        if self.iters_since_improvement >= 2:
            if self.last_stagnation_action != "refine" and self.best_id in self.programs:
                # REFINE newest best with empty context (proven breakthrough pattern)
                self.last_stagnation_action = "refine"
                return {self.REFINE_LABEL: self.get(self.best_id)}, {"": []}
            # DIVERGE from a promising but under-explored parent
            pool = top + candidates[n // 2:]
            parent = min(pool, key=lambda p: self.parent_usage.get(p.id, 0))
            self.last_stagnation_action = "diverge"
            return {self.DIVERGE_LABEL: parent}, {"": []}

        self.last_stagnation_action = ""

        # Default: soft exploitation from top tier, penalizing overused parents
        pool = top if (self.rng.random() < 0.7 and top) else candidates
        weights = [1.0 / (1.0 + self.parent_usage.get(p.id, 0)) for p in pool]
        parent = self.rng.choices(pool, weights=weights, k=1)[0]

        # Context: half top performers + half diverse others, excluding parent
        context: List[EvolvedProgram] = []
        seen = {parent.id}
        want = max(0, num_context_programs or 0)
        top_others = [p for p in candidates if p.id not in seen][: want // 2]
        for p in top_others:
            context.append(p)
            seen.add(p.id)
        rest = [p for p in candidates if p.id not in seen]
        self.rng.shuffle(rest)
        for p in rest:
            if len(context) >= want:
                break
            context.append(p)
            seen.add(p.id)

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END