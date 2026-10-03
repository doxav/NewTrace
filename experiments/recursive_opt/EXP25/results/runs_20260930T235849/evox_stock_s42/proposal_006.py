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
    """Search strategy for a plateaued, tightly-clustered population.

    Evidence-based principles:
    1. Refining near-best parents regresses; mid-tier parents with top-score
       context produced all recent bests. Default: underused upper-mid parent.
    2. On stagnation, DIVERGE from a fresh mid-tier parent (never low-tier),
       keeping top scorers as contrast context.
    """

    STAGNATION_LIMIT = 3

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score = -1.0
        self.best_id: Optional[str] = None
        self.stagnation = 0
        self.parent_use: Dict[str, int] = {}
        self.recent_parent_ids: List[str] = []

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
        best = self.get(self.best_id) if self.best_id else ranked[0]
        if best is None or best.id not in self.programs:
            best = ranked[0]

        stuck = self.stagnation >= self.STAGNATION_LIMIT

        # Parent pool: mid-tier band (25th-75th percentile scores),
        # excluding the best program (refining it historically regresses).
        n = len(ranked)
        lo, hi = n // 4, max(n // 4 + 1, (3 * n) // 4)
        mid_pool = ranked[lo:hi]
        if not mid_pool:
            mid_pool = ranked

        if stuck:
            # DIVERGE: fresh mid-tier parent, least recently used.
            pool = [p for p in mid_pool if p.id not in self.recent_parent_ids[-3:]] or mid_pool
            pool.sort(key=lambda p: self.parent_use.get(p.id, 0))
            parent = pool[0]
            label = self.DIVERGE_LABEL
        else:
            # Exploit: underused mid-tier parent, slight score preference.
            pool = sorted(mid_pool, key=lambda p: (self.parent_use.get(p.id, 0), -_score(p)))
            parent = pool[0]
            label = ""

        self.parent_use[parent.id] = self.parent_use.get(parent.id, 0) + 1
        self.recent_parent_ids.append(parent.id)
        self.recent_parent_ids = self.recent_parent_ids[-10:]

        # Context: top scorers (incl. best) for contrast with mid-tier parent.
        ctx: List[EvolvedProgram] = []
        seen = {parent.id}
        for p in ranked:
            if p.id not in seen and len(ctx) < n_ctx:
                ctx.append(p)
                seen.add(p.id)

        parent_dict = {label: parent}
        context_programs_dict = {"": ctx}
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END