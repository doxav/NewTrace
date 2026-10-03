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


def _score(p: Program) -> float:
    v = p.metrics.get("combined_score", 0.0) if p.metrics else 0.0
    return float(v) if isinstance(v, (int, float)) else 0.0


class EvolvedProgramDatabase(ProgramDatabase):
    """Plateau-tuned strategy.

    Evidence: unlabeled top-tier parents with best-in-context drive gains;
    labels on near-best parents regress. So:
    1. Default: unlabeled parent from top tier (least-recently-used rotation),
       with the best program + diverse context.
    2. Only when deeply stuck: occasional unlabeled DIVERGE-style parent from
       the mid tier (fresh direction), still with best as context.
    """

    STAGNATION_LIMIT = 5  # iterations without meaningful improvement

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score = -1.0
        self.best_id: Optional[str] = None
        self.stagnation = 0
        self.parent_use: Dict[str, int] = {}
        self.diverge_used = 0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        s = _score(program)
        if self.best_score < 0 or s > self.best_score * 1.01 + 0.01:
            self.best_score = s
            self.best_id = program.id
            self.stagnation = 0
        else:
            self.stagnation += 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 0
        ranked = sorted(candidates, key=_score, reverse=True)
        best = self.get(self.best_id) if self.best_id else None
        if best is None or best.id not in self.programs:
            best = ranked[0]

        stuck = self.stagnation >= self.STAGNATION_LIMIT

        # Default: unlabeled parent from top tier, rotating by usage.
        top = ranked[: max(1, len(ranked) // 3)]
        top.sort(key=lambda p: (self.parent_use.get(p.id, 0), -_score(p)))
        parent = top[0]
        label = ""

        # Deeply stuck: occasionally pick a fresh mid-tier parent to spark a
        # new direction (unlabeled — labels regress near the plateau).
        if stuck and self.diverge_used < 2:
            mid = ranked[len(ranked) // 3: 2 * len(ranked) // 3]
            mid = [p for p in mid if p.id != best.id]
            if mid:
                mid.sort(key=lambda p: self.parent_use.get(p.id, 0))
                parent = mid[0]
                self.diverge_used += 1
                self.stagnation = self.STAGNATION_LIMIT // 2  # reset partially

        # Context: best program first, then diverse high scorers not already used.
        ctx_ids: List[str] = [best.id]
        seen = {parent.id, best.id}
        for p in ranked:
            if len(ctx_ids) >= n_ctx:
                break
            if p.id in seen:
                continue
            ctx_ids.append(p.id)
            seen.add(p.id)

        self.parent_use[parent.id] = self.parent_use.get(parent.id, 0) + 1

        examples = [self.get(pid) for pid in ctx_ids]
        examples = [p for p in examples if p is not None and p.id != parent.id][:n_ctx]

        parent_dict = {label: parent}
        context_programs_dict = {"": examples}
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END