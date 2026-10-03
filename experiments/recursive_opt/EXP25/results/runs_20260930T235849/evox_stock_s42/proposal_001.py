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
    """Search strategy tuned for a plateaued, tightly-clustered population.

    Principles:
    1. Exploit the best programs as parents (plateau => refine the leader),
       while rotating context to expose the LLM to diverse alternatives.
    2. When stagnating (>N iterations without meaningful improvement),
       alternate between REFINE (deep refinement of the best) and DIVERGE
       (fundamentally new direction seeded by a mid/low scorer for contrast).
    """

    STAGNATION_LIMIT = 4

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score = -1.0
        self.best_id: Optional[str] = None
        self.stagnation = 0
        self.stagnation_round = 0  # alternates refine/diverge phases
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
        candidates = [p for p in self.programs.values()]
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
            # DIVERGE: pick a fresh parent (less-used, mid/low tier) to
            # explore a fundamentally different approach.
            pool = [p for p in ranked[len(ranked) // 3:] if p.id != best.id] or ranked
            pool.sort(key=lambda p: self.parent_use.get(p.id, 0))
            parent = pool[0] if len(pool) > 1 else self.rng.choice(pool)
            label = self.DIVERGE_LABEL
            # Contrast context: best + a couple of diverse others.
            ctx_ids = [best.id]
            for p in self.rng.sample([p for p in candidates if p.id not in (best.id, parent.id)],
                                     min(2, max(0, len(candidates) - 2))):
                ctx_ids.append(p.id)
            self.stagnation_round += 1
        else:
            # Exploit: mutate the best (or near-best) program.
            top = ranked[: max(1, len(ranked) // 3)]
            # Rotate among top programs weighted by least-used.
            top.sort(key=lambda p: (self.parent_use.get(p.id, 0), -_score(p)))
            parent = top[0]
            if len(top) > 1 and self.parent_use.get(parent.id, 0) > 2:
                parent = self.rng.choice(top)
            label = self.REFINE_LABEL if stuck else ""
            # Context: top scorers + one lower scorer for contrast.
            ctx: List[EvolvedProgram] = []
            for p in ranked:
                if p.id != parent.id and len(ctx) < max(1, n_ctx - 1):
                    ctx.append(p)
            lows = [p for p in reversed(ranked) if p.id not in {parent.id} | {c.id for c in ctx}]
            if lows and n_ctx > 1:
                ctx.append(lows[0])
            ctx_ids = [p.id for p in ctx[:n_ctx]]
            if stuck:
                self.stagnation_round += 1

        self.parent_use[parent.id] = self.parent_use.get(parent.id, 0) + 1

        examples = [self.get(pid) for pid in ctx_ids]
        examples = [p for p in examples if p is not None and p.id != parent.id][:n_ctx]

        parent_dict = {label: parent}
        context_programs_dict = {"": examples}
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END