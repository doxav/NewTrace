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
    """Search strategy informed by history of this population.

    Evidence from prior runs:
    - All recent breakthroughs (0.7051, 0.7052, 0.7131) came from UNLABELED
      diverse parents (mid/high tier) with the best program in context.
    - Late REFINE on the plateaued best always regressed; DIVERGE gave
      mixed results. So labels are used sparingly and never on the best.

    Strategy:
    1. Default: score-weighted stochastic parent choice from the upper half
       of the population (excludes the very best most of the time), with the
       best + diverse others as context. This reproduces the pattern that
       produced every recent best.
    2. On stagnation (>=3 adds without meaningful improvement), occasionally
       DIVERGE from a fresh, under-used mid-tier parent with no context.
    """

    STAGNATION_LIMIT = 3

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score = -1.0
        self.best_id: Optional[str] = None
        self.stagnation = 0
        self.diverge_toggle = False
        self.parent_use: Dict[str, int] = {}
        self.last_parent_id: Optional[str] = None

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
            self.diverge_toggle = False
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
        label = ""
        parent = None

        if stuck and self.diverge_toggle:
            # DIVERGE: fresh under-used mid-tier parent, no context.
            mid = ranked[1: max(2, len(ranked) // 2)]
            if not mid:
                mid = ranked
            mid.sort(key=lambda p: (self.parent_use.get(p.id, 0), -_score(p)))
            parent = mid[0]
            label = self.DIVERGE_LABEL
            self.diverge_toggle = False
            self.parent_use[parent.id] = self.parent_use.get(parent.id, 0) + 1
            return {label: parent}, {}

        # Default: score-weighted stochastic pick from upper half,
        # avoiding repeating the same parent twice in a row.
        upper = ranked[: max(2, len(ranked) // 2)]
        weights = []
        for p in upper:
            w = _score(p) + 0.1
            if p.id == self.last_parent_id:
                w *= 0.25
            # mild penalty for overuse
            w /= 1.0 + 0.3 * self.parent_use.get(p.id, 0)
            weights.append(w)
        total = sum(weights)
        if total <= 0:
            parent = self.rng.choice(upper)
        else:
            r = self.rng.random() * total
            acc = 0.0
            parent = upper[0]
            for p, w in zip(upper, weights):
                acc += w
                if r <= acc:
                    parent = p
                    break
        self.last_parent_id = parent.id

        # Context: best program first (proven driver of breakthroughs),
        # then diverse others spanning the score range, excluding parent.
        ctx: List[EvolvedProgram] = []
        if best.id != parent.id:
            ctx.append(best)
        remaining = [p for p in candidates if p.id not in (parent.id, best.id)]
        # take a spread: a couple of high scorers, one low/mid for contrast
        remaining.sort(key=_score, reverse=True)
        if remaining and n_ctx > len(ctx):
            ctx.append(remaining[0])
        if remaining and n_ctx > len(ctx):
            ctx.append(remaining[len(remaining) // 2])
        for p in self.rng.sample(remaining, min(len(remaining), max(0, n_ctx - len(ctx)))):
            if p.id not in {c.id for c in ctx}:
                ctx.append(p)
            if len(ctx) >= n_ctx:
                break

        if stuck:
            self.diverge_toggle = True

        self.parent_use[parent.id] = self.parent_use.get(parent.id, 0) + 1

        examples = [p for p in ctx if p.id != parent.id][:n_ctx]
        return {"": parent}, {"": examples}


# EVOLVE-BLOCK-END