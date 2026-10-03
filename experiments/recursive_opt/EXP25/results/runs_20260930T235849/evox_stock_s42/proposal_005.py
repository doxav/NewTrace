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

    Evidence from prior runs: ALL breakthroughs came from UNLABELED parents
    (top/mid scorers) with the best program in context; REFINE/DIVERGE labels
    mostly regressed during stalls. So this strategy:
      1. Default: unlabeled parent from the top tier (rotated, slightly
         randomized), with the best program plus diverse context programs.
      2. Labels only when deeply stuck, and used sparingly (DIVERGE on a
         less-explored mid-tier parent with focused context).
    """

    DEEP_STAGNATION = 6

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score = -1.0
        self.best_id: Optional[str] = None
        self.stagnation = 0
        self.parent_use: Dict[str, int] = {}
        self.diverge_since = 0  # iterations since last DIVERGE

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

        deep_stuck = self.stagnation >= self.DEEP_STAGNATION
        label = ""
        parent: Optional[EvolvedProgram] = None

        if deep_stuck and self.diverge_since >= 3:
            # Rare, targeted divergence: less-explored mid-tier parent,
            # focused context (best only) to push a new direction.
            mid = ranked[len(ranked) // 4:] or ranked
            mid = [p for p in mid if p.id != best.id] or ranked
            mid.sort(key=lambda p: (self.parent_use.get(p.id, 0), -_score(p)))
            parent = mid[0]
            label = self.DIVERGE_LABEL
            self.diverge_since = 0
        else:
            # Default exploitation: top-tier parent, rotated with randomness.
            top = ranked[: max(2, len(ranked) // 4)]
            top = sorted(top, key=lambda p: (self.parent_use.get(p.id, 0), -_score(p)))
            # Pick among the two least-used top programs (keeps variety).
            parent = self.rng.choice(top[: min(2, len(top))])
            if deep_stuck:
                label = self.REFINE_LABEL
            self.diverge_since += 1

        self.parent_use[parent.id] = self.parent_use.get(parent.id, 0) + 1

        # Context: best program always included; rest sampled from diverse
        # score tiers for contrast.
        ctx: List[EvolvedProgram] = [best] if best.id != parent.id else []
        pool = [p for p in candidates if p.id not in (parent.id, best.id)]
        if pool and len(ctx) < n_ctx:
            k = min(len(pool), n_ctx - len(ctx))
            ctx.extend(self.rng.sample(pool, k))
        ctx = ctx[:n_ctx]

        parent_dict = {label: parent}
        context_programs_dict = {"": ctx}
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END