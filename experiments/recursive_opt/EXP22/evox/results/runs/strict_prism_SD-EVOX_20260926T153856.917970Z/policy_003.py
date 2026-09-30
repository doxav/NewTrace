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
    """Adaptive search database for a converged, plateaued population.

    Strategy:
    - Track best-score improvements in add() to detect stagnation.
    - Parent selection: score-weighted with recency penalty to avoid
      overusing any single program; mostly elite, sometimes second-tier.
    - Context: mix of elite + diverse (mid/low) programs.
    - On deep stagnation, apply DIVERGE to a non-best parent WITH elite
      context (empty-context divergence historically failed).
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score: Optional[float] = None
        self.iters_since_improvement: int = 0
        self.parent_use_count: Dict[str, int] = {}
        self.last_parent_id: Optional[str] = None

    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score") if program.metrics else None
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        # Track meaningful improvements (>1% relative or >0.01 absolute)
        s = self._score(program)
        if s != float("-inf"):
            if self.best_score is None:
                self.best_score = s
                self.iters_since_improvement = 0
            elif s > self.best_score + max(0.01, 0.01 * abs(self.best_score)):
                self.best_score = s
                self.iters_since_improvement = 0
            else:
                self.iters_since_improvement += 1

        # Track parent usage for diversity
        if program.parent_id:
            self.parent_use_count[program.parent_id] = (
                self.parent_use_count.get(program.parent_id, 0) + 1
            )

        self._update_best_program(program)
        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def _weighted_parent(self, candidates: List[EvolvedProgram]) -> EvolvedProgram:
        """Score-weighted choice with recency penalty against overuse."""
        scores = [self._score(p) for p in candidates]
        lo = min(s for s in scores if s != float("-inf")) if candidates else 0.0
        weights = []
        for p, s in zip(candidates, scores):
            if s == float("-inf"):
                weights.append(0.01)
                continue
            base = 1.0 + (s - lo)  # score weight
            if p.id == self.last_parent_id:
                base *= 0.15  # avoid repeating last parent
            base /= 1.0 + 0.35 * self.parent_use_count.get(p.id, 0)  # usage penalty
            weights.append(max(base, 0.01))
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for p, w in zip(candidates, weights):
            acc += w
            if r <= acc:
                return p
        return candidates[-1]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 0
        scored = sorted(candidates, key=self._score, reverse=True)
        best = scored[0]
        stuck = self.iters_since_improvement >= 6

        parent = self._weighted_parent(candidates)
        label = ""

        # Deep stagnation: diverge from a non-best, mid-tier parent, but
        # anchor with elite context (empty-context diverge failed before).
        if stuck and self.rng.random() < 0.4 and len(scored) > 2:
            mid = scored[len(scored) // 3 : 2 * len(scored) // 3]
            if mid:
                parent = self.rng.choice(mid)
                label = self.DIVERGE_LABEL

        # Context: top elites + diverse fill from lower tiers
        context: List[EvolvedProgram] = []
        elite_pool = [p for p in scored[: max(2, len(scored) // 4)] if p.id != parent.id]
        self.rng.shuffle(elite_pool)
        context.extend(elite_pool[: max(1, n_ctx // 2)])
        rest = [p for p in candidates if p.id not in {parent.id} | {c.id for c in context}]
        self.rng.shuffle(rest)
        context.extend(rest[: n_ctx - len(context)])

        self.last_parent_id = parent.id
        return {label: parent}, {"": context[:n_ctx]}


# EVOLVE-BLOCK-END