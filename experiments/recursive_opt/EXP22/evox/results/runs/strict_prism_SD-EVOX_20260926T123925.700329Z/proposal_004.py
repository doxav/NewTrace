# EVOLVE-BLOCK-START
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


def _score(program: EvolvedProgram) -> float:
    """Safely extract numeric combined_score."""
    value = program.metrics.get("combined_score", None)
    return float(value) if isinstance(value, (int, float)) else float("-inf")


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive search database.

    Strategy (informed by population history):
    - DIVERGE from varied-tier parents with empty context produces
      breakthroughs; repeated REFINE on one best program plateaus.
    - Cycle DIVERGE parents across score tiers; rotate REFINE targets
      among top scorers to avoid overuse.
    - Context: empty on DIVERGE (focused fresh direction); on REFINE
      provide best + one contrasting program.
    """

    STAGNATION_WINDOW = 4

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score: float = float("-inf")
        self.iters_since_improvement: int = 0
        self.stagnation_counter: int = 0
        self.label_flip: bool = False
        self.parent_use_count: Dict[str, int] = {}
        self.context_use_count: Dict[str, int] = {}
        self.diverge_tier_idx: int = 0  # rotate diverge parents across tiers

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add a program and update progress/stagnation state."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        score = _score(program)
        if score > float("-inf"):
            meaningful = self.best_score > float("-inf") and (
                score > self.best_score * 1.01 or score > self.best_score + 0.01
            )
            if meaningful:
                self.iters_since_improvement = 0
                self.stagnation_counter = 0
                self.best_score = max(self.best_score, score)
            elif score > self.best_score:
                self.best_score = score
                self.iters_since_improvement = 0
            else:
                self.iters_since_improvement += 1
                if self.iters_since_improvement >= self.STAGNATION_WINDOW:
                    self.stagnation_counter += 1
                    self.iters_since_improvement = 0
        else:
            self.iters_since_improvement += 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = [p for p in self.programs.values() if _score(p) > float("-inf")]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = sorted(candidates, key=_score, reverse=True)
        n = len(scored)
        third = max(1, n // 3)
        top, mid, bottom = scored[:third], scored[third:2 * third], scored[2 * third:]

        stuck = self.stagnation_counter > 0

        if stuck:
            # Alternate DIVERGE (rotating across tiers) and REFINE
            # (rotating among top scorers) to break the plateau.
            self.label_flip = not self.label_flip
            if self.label_flip:
                parent_label = self.DIVERGE_LABEL
                tiers = [t for t in (top, mid, bottom) if t]
                pool = tiers[self.diverge_tier_idx % len(tiers)]
                self.diverge_tier_idx += 1
                parent = min(pool, key=lambda p: self.parent_use_count.get(p.id, 0))
            else:
                parent_label = self.REFINE_LABEL
                # Rotate REFINE targets among top scorers, least-used first.
                parent = min(top, key=lambda p: (self.parent_use_count.get(p.id, 0),
                                                 -_score(p)))
        else:
            parent_label = ""
            parent = min(top, key=lambda p: (self.parent_use_count.get(p.id, 0),
                                             -_score(p)))

        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1

        context: List[EvolvedProgram] = []
        if parent_label != self.DIVERGE_LABEL:
            want = num_context_programs or 0
            # Best non-parent program + one contrasting low scorer, then fill.
            rest = [p for p in scored if p.id != parent.id]
            rest.sort(key=lambda p: (self.context_use_count.get(p.id, 0), -_score(p)))
            if rest and want > 0:
                context.append(rest[0])
            lows = [p for p in bottom if p.id != parent.id and
                    all(p.id != c.id for c in context)]
            lows.sort(key=lambda p: self.context_use_count.get(p.id, 0))
            if lows and want > 1:
                context.append(lows[0])
            remain = [p for p in scored if p.id != parent.id and
                      all(p.id != c.id for c in context)]
            remain.sort(key=lambda p: (self.context_use_count.get(p.id, 0),
                                       self.rng.random()))
            context.extend(remain[: max(0, want - len(context))])

        for c in context:
            self.context_use_count[c.id] = self.context_use_count.get(c.id, 0) + 1

        return {parent_label: parent}, {"": context}


# EVOLVE-BLOCK-END