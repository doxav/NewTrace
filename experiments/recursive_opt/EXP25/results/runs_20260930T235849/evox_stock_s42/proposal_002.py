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
    """Adaptive search: score-weighted stochastic parent selection with
    usage balancing, diverse mixed-score context, and stagnation-triggered
    REFINE/DIVERGE alternation."""

    STAGNATION_LIMIT = 4

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score = -1.0
        self.best_id: Optional[str] = None
        self.stagnation = 0
        self.phase = 0  # alternates refine/diverge during stagnation
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
            self.phase = 0
        else:
            self.stagnation += 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def _weighted_pick(self, pool: List[EvolvedProgram]) -> EvolvedProgram:
        """Score-weighted choice with a least-used bonus to avoid overuse."""
        if len(pool) == 1:
            return pool[0]
        weights = []
        for p in pool:
            w = max(_score(p), 0.05) ** 2.0
            w /= (1.0 + 0.5 * self.parent_use.get(p.id, 0))
            weights.append(w)
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(pool)
        r = self.rng.random() * total
        acc = 0.0
        for p, w in zip(pool, weights):
            acc += w
            if acc >= r:
                return p
        return pool[-1]

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
        diverging = stuck and (self.phase % 2 == 1)

        if diverging:
            # DIVERGE: fresh, under-used parent from mid/low tier.
            pool = [p for p in ranked[len(ranked) // 3:] if p.id != best.id] or ranked
            pool = sorted(pool, key=lambda p: self.parent_use.get(p.id, 0))
            parent = pool[0] if len(pool) > 1 else self.rng.choice(pool)
            label = self.DIVERGE_LABEL
            # Contrast context: best + random diverse others.
            ctx_ids = [best.id]
            others = [p for p in candidates if p.id not in (best.id, parent.id)]
            self.rng.shuffle(others)
            for p in others[: max(0, n_ctx - 1)]:
                ctx_ids.append(p.id)
            self.phase += 1
        else:
            # Exploit: weighted pick across the full population (favors
            # high scorers but keeps mid-tier parents in play).
            parent = self._weighted_pick(candidates)
            if stuck and parent.id == best.id:
                label = self.REFINE_LABEL
            else:
                label = ""
            # Context: top scorers + random lower/mid scorers for diversity.
            ctx: List[EvolvedProgram] = []
            top = [p for p in ranked if p.id != parent.id][: max(1, n_ctx // 2)]
            ctx.extend(top)
            rest = [p for p in ranked if p.id not in {parent.id} | {c.id for c in ctx}]
            self.rng.shuffle(rest)
            for p in rest:
                if len(ctx) >= n_ctx:
                    break
                ctx.append(p)
            ctx_ids = [p.id for p in ctx[:n_ctx]]
            if stuck:
                self.phase += 1

        self.parent_use[parent.id] = self.parent_use.get(parent.id, 0) + 1

        examples = [self.get(pid) for pid in ctx_ids]
        examples = [p for p in examples if p is not None and p.id != parent.id][:n_ctx]

        parent_dict = {label: parent}
        context_programs_dict = {"": examples}
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END