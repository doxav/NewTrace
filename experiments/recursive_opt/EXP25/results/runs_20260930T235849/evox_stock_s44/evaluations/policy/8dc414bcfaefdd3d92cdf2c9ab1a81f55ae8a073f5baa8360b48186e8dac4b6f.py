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


def _score(p: EvolvedProgram) -> float:
    v = p.metrics.get("combined_score", 0.0)
    return float(v) if isinstance(v, (int, float)) else 0.0


class EvolvedProgramDatabase(ProgramDatabase):
    """Simple plateau-aware search strategy.

    Principles:
    1. Refine the best programs (with top-scorer context) most of the time,
       but avoid parents whose recent children regressed.
    2. When progress stalls, alternate DIVERGE (empty context, fresh
       direction) and REFINE (focused polish of the best program).
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score = -1e9
        self.best_id = None
        self.stagnation = 0          # iterations without meaningful improvement
        self.since_diverge = 0
        self.parent_stats: Dict[str, Dict[str, float]] = {}  # id -> {uses, best_child}
        self.last_iteration_seen = 0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
            self.last_iteration_seen = iteration

        if self.config.db_path:
            self._save_program(program)

        # Track parent success to steer future selection
        pid = program.parent_id
        if pid and pid in self.programs:
            st = self.parent_stats.setdefault(pid, {"uses": 0.0, "best_child": -1e9})
            st["uses"] += 1.0
            st["best_child"] = max(st["best_child"], _score(program))

        s = _score(program)
        if s > self.best_score + max(0.01, 0.01 * abs(self.best_score)):
            self.stagnation = 0
            self.best_score = s
            self.best_id = program.id
        else:
            self.stagnation += 1
        self.since_diverge += 1

        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = [p for p in self.programs.values()]
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 0
        ranked = sorted(candidates, key=_score, reverse=True)
        top = ranked[: max(1, min(4, len(ranked)))]

        # --- stagnation handling: alternate DIVERGE / REFINE ---
        if self.stagnation >= 5 and self.best_id in self.programs:
            best = self.programs[self.best_id]
            if self.since_diverge >= 3:
                # Fresh direction: no context, pure divergence
                self.since_diverge = 0
                return {self.DIVERGE_LABEL: best}, {}
            # Focused refinement of the best with top context
            ctx = [p for p in top if p.id != best.id][:n_ctx]
            self.stagnation = max(0, self.stagnation - 2)
            return {self.REFINE_LABEL: best}, {"": ctx}

        # --- default: pick a good parent, avoiding proven-failing ones ---
        def viable(p: EvolvedProgram) -> bool:
            st = self.parent_stats.get(p.id)
            if st is None or st["uses"] < 2:
                return True
            # parent repeatedly used but children never beat ~median -> deprioritize
            return st["best_child"] > 0.5 * self.best_score + 0.5 * _score(ranked[len(ranked) // 2])

        pool = [p for p in top if viable(p)] or top
        # weighted toward the very best, with some randomness
        weights = [1.0 / (i + 1) for i in range(len(pool))]
        parent = self.rng.choices(pool, weights=weights, k=1)[0]

        # context: top scorers (excluding parent) + one diverse mid/low program
        ctx_ids = []
        ctx = []
        for p in top:
            if p.id != parent.id and p.id not in ctx_ids:
                ctx.append(p)
                ctx_ids.append(p.id)
            if len(ctx) >= max(1, n_ctx - 1):
                break
        if n_ctx > len(ctx):
            diverse = [p for p in ranked[len(ranked) // 2:] if p.id not in ctx_ids and p.id != parent.id]
            if diverse:
                pick = self.rng.choice(diverse)
                ctx.append(pick)
                ctx_ids.append(pick.id)

        return {"": parent}, {"": ctx[:n_ctx]}


# EVOLVE-BLOCK-END