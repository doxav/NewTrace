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
    """Search strategy for a plateaued, top-heavy population.

    1. Default: mutate a near-best parent (rotating by usage), with the
       best program plus a diverse mid-tier program as context.
    2. On stagnation: alternate REFINE (deep refinement of the best)
       and DIVERGE from an under-used mid-tier parent with best-in-context.
       Avoid low-tier parents (historically regressive).
    """

    STAGNATION_LIMIT = 3

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score = -1.0
        self.best_id: Optional[str] = None
        self.stagnation = 0
        self.stagnation_round = 0
        self.parent_use: Dict[str, int] = {}

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
        # Meaningful improvement: >1% relative or >0.01 absolute vs best.
        if self.best_score < 0 or s > self.best_score * 1.01 + 0.01:
            self.best_score = s
            self.best_id = program.id
            self.stagnation = 0
            self.stagnation_round = 0
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
        best = self.get(self.best_id) if self.best_id else ranked[0]
        if best is None or best.id not in self.programs:
            best = ranked[0]

        stuck = self.stagnation >= self.STAGNATION_LIMIT
        diverging = stuck and (self.stagnation_round % 2 == 1)

        if diverging:
            # DIVERGE from an under-used MID-tier parent (skip bottom third).
            lo = len(ranked) // 3
            pool = ranked[lo: 2 * lo] if len(ranked) > 3 else ranked[1:]
            pool = [p for p in pool if p.id != best.id] or [p for p in ranked if p.id != best.id] or ranked
            pool.sort(key=lambda p: self.parent_use.get(p.id, 0))
            parent = pool[0]
            label = self.DIVERGE_LABEL
            self.stagnation_round += 1
        else:
            # Exploit: rotate among top-tier parents by least usage.
            top = ranked[: max(2, len(ranked) // 3)]
            top = [p for p in top if p.id != best.id] or top
            top.sort(key=lambda p: (self.parent_use.get(p.id, 0), -_score(p)))
            parent = top[0]
            label = self.REFINE_LABEL if stuck else ""
            if stuck:
                self.stagnation_round += 1

        # Context: best program (anchor) + diverse mid/upper-mid examples.
        ctx_ids: List[str] = [best.id]
        mid_pool = [p for p in ranked[1:] if p.id != parent.id]
        # Prefer upper-mid band for diverse-but-viable ideas.
        band = mid_pool[len(mid_pool) // 3: 2 * len(mid_pool) // 3] or mid_pool
        self.rng.shuffle(band)
        for p in band:
            if len(ctx_ids) >= n_ctx:
                break
            if p.id not in ctx_ids:
                ctx_ids.append(p.id)

        self.parent_use[parent.id] = self.parent_use.get(parent.id, 0) + 1

        examples = [self.get(pid) for pid in ctx_ids]
        examples = [p for p in examples if p is not None and p.id != parent.id][:n_ctx]

        parent_dict = {label: parent}
        context_programs_dict = {"": examples}
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END