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
    """Adaptive search database: rotates parents, escalates on stagnation."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score: float = -1e9
        self.iters_since_improve: int = 0
        self.parent_use: Dict[str, int] = {}      # parent_id -> times used
        self.child_scores: Dict[str, List[float]] = {}  # parent_id -> child combined scores

    def _score(self, p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score", None)
        return float(v) if isinstance(v, (int, float)) else 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        s = self._score(program)
        if s > self.best_score + max(0.01, 0.01 * abs(self.best_score)):
            self.best_score = s
            self.iters_since_improve = 0
        else:
            self.iters_since_improve += 1

        # record child outcome for parent quality tracking
        pid = program.parent_id
        if pid and pid in self.programs:
            self.child_scores.setdefault(pid, []).append(s)
            if len(self.child_scores[pid]) > 6:
                self.child_scores[pid] = self.child_scores[pid][-6:]

        self._update_best_program(program)
        return program.id

    def _parent_quality(self, pid: str) -> float:
        """Average child score of a parent (fallback: parent's own score)."""
        scores = self.child_scores.get(pid, [])
        if scores:
            return sum(scores) / len(scores)
        p = self.get(pid)
        return self._score(p) if p else 0.0

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = [p for p in self.programs.values() if self._score(p) > 0.0]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = sorted(candidates, key=self._score, reverse=True)
        top = scored[: max(1, len(scored) // 4)]
        mid = [p for p in scored if 0.45 <= self._score(p) < self._score(top[-1]) + 1e-9]

        # Overused parents: used >= 3 times with no child improvement
        def overused(p):
            uses = self.parent_use.get(p.id, 0)
            kids = self.child_scores.get(p.id, [])
            return uses >= 3 and all(
                s <= self.best_score - 0.005 for s in kids
            )

        stagnated = self.iters_since_improve >= 5

        label = ""
        parent = None
        if stagnated and self.rng.random() < 0.5:
            # DIVERGE from an under-explored mid-tier parent, clean context
            pool = [p for p in (mid or scored) if not overused(p)]
            if pool:
                pool.sort(key=lambda p: (self.parent_use.get(p.id, 0), -self._score(p)))
                parent = pool[0]
                label = self.DIVERGE_LABEL
        if parent is None:
            # Mix: elite refinement vs mid-tier exploration by parent quality
            elite_pool = [p for p in top if not overused(p)]
            mid_pool = [p for p in (mid or scored) if not overused(p)]
            if self.rng.random() < 0.55 and elite_pool:
                parent = self.rng.choice(elite_pool[:3])
            elif mid_pool:
                parent = self.rng.choice(mid_pool)
            else:
                parent = self.rng.choice(scored[:5])
            if stagnated and self.rng.random() < 0.3:
                label = self.REFINE_LABEL

        self.parent_use[parent.id] = self.parent_use.get(parent.id, 0) + 1

        # Context: top scorers + diverse random others, exclude parent
        rest = [p for p in scored if p.id != parent.id]
        n_ctx = num_context_programs or 0
        ctx: List[EvolvedProgram] = []
        seen = {parent.id}
        for p in rest[:2]:  # two best as reference
            if len(ctx) < n_ctx:
                ctx.append(p)
                seen.add(p.id)
        pool = [p for p in rest if p.id not in seen]
        self.rng.shuffle(pool)
        for p in pool:
            if len(ctx) >= n_ctx:
                break
            ctx.append(p)

        if label == self.DIVERGE_LABEL:
            ctx = []  # clean context for targeted divergence

        return {label: parent}, {"": ctx}


# EVOLVE-BLOCK-END