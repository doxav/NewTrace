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
    """Adaptive search database for a stagnating, converged population.

    Strategy:
    - Default: refine a high-scoring parent (score-weighted among top tier),
      with context mixing top scorers and one diverse mid/low program.
    - When stagnation deepens (no meaningful improvement), escalate:
      alternate REFINE on the best with DIVERGE from strong-but-not-best
      parents to escape the plateau without discarding gains.
    - Usage counts prevent overusing any single program.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score: float = float("-inf")
        self.stagnation: int = 0          # iterations since meaningful improvement
        self.since_diverge: int = 0       # iterations since last DIVERGE
        self.usage: Dict[str, int] = {}   # parent usage counts by id

    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        return float(v) if isinstance(v, (int, float)) else 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        # Track stagnation: meaningful = >1% relative or >0.01 absolute
        s = self._score(program)
        if s > self.best_score:
            meaningful = (s - self.best_score) > 0.01 or (
                self.best_score > 0 and (s - self.best_score) / self.best_score > 0.01
            )
            if meaningful:
                self.stagnation = 0
            else:
                self.stagnation += 1
            self.best_score = max(self.best_score, s)
        else:
            self.stagnation += 1

        self.since_diverge += 1
        logger.debug(f"Added program {program.id} (stagnation={self.stagnation})")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(self._score(p), p) for p in candidates]
        scored.sort(key=lambda t: t[0], reverse=True)
        best_score = scored[0][0]
        n = len(scored)

        # Top tier: within 0.03 of best (the 0.66-0.68 cluster)
        top = [p for s, p in scored if s >= best_score - 0.03]
        mid = [p for s, p in scored if best_score - 0.15 <= s < best_score - 0.03]

        label = ""
        parent = None

        # Escalation ladder as stagnation grows
        if self.stagnation >= 6 and self.since_diverge >= 3 and mid:
            # Diverge from a strong-but-not-best program for a fresh direction
            parent = self.rng.choice(mid[: max(3, len(mid))])
            label = self.DIVERGE_LABEL
            self.since_diverge = 0
        elif self.stagnation >= 3:
            # Refine the best, but vary among top-tier to avoid overuse
            pool = top if top else [p for _, p in scored[:3]]
            parent = min(pool, key=lambda p: self.usage.get(p.id, 0) + self.rng.random())
            label = self.REFINE_LABEL
        else:
            # Score-weighted choice among top tier, penalizing heavy reuse
            pool = top if top else [p for _, p in scored[: max(1, n // 4)]]
            weights = []
            for p in pool:
                s = self._score(p)
                w = max(s, 0.01) / (1.0 + self.usage.get(p.id, 0))
                weights.append(w)
            total = sum(weights)
            r = self.rng.random() * total
            acc = 0.0
            parent = pool[-1]
            for p, w in zip(pool, weights):
                acc += w
                if r <= acc:
                    parent = p
                    break

        self.usage[parent.id] = self.usage.get(parent.id, 0) + 1

        # Context: top scorers for guidance + one diverse mid/low program
        context: List[EvolvedProgram] = []
        seen = {parent.id}
        for s, p in scored:
            if len(context) >= (num_context_programs or 0) - 1:
                break
            if p.id not in seen:
                context.append(p)
                seen.add(p.id)
        # Add one diverse program from below the top tier
        diverse = [p for s, p in scored if p.id not in seen and s < best_score - 0.03]
        if diverse and len(context) < (num_context_programs or 0):
            context.append(self.rng.choice(diverse[-max(1, len(diverse) // 2):]))

        if label == self.DIVERGE_LABEL:
            # Targeted divergence: minimal context
            context = []

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END