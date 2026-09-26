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

    Strategy:
    - Exploit top-tier parents with usage penalties to avoid overuse.
    - Context mixes one elite program, one contrasting scorer, and
      least-used diverse programs.
    - On stagnation, alternate REFINE (on best program) and DIVERGE
      (from upper-mid tier programs, which historically produced
      breakthroughs without the regressions seen from bottom-tier
      divergence).
    """

    STAGNATION_THRESHOLD = 3  # iterations without meaningful improvement

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        # progress tracking (persists via checkpointed attributes)
        self.best_score: float = float("-inf")
        self.iters_since_improvement: int = 0
        self.stagnation_counter: int = 0
        self.label_flip: bool = False
        # usage tracking to avoid overuse
        self.parent_use_count: Dict[str, int] = {}
        self.context_use_count: Dict[str, int] = {}

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

        # --- progress tracking ---
        score = _score(program)
        if score > float("-inf"):
            meaningful = (
                self.best_score > float("-inf")
                and (score > self.best_score * 1.01 or score > self.best_score + 0.01)
            )
            if score > self.best_score:
                self.best_score = score
                self.iters_since_improvement = 0
                if meaningful:
                    self.stagnation_counter = 0
            else:
                self.iters_since_improvement += 1
                if self.iters_since_improvement >= self.STAGNATION_THRESHOLD:
                    self.stagnation_counter += 1
                    self.iters_since_improvement = 0

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
        top = scored[: max(1, n // 3)]
        mid = scored[max(1, n // 3): max(2, 2 * n // 3)] or top
        bottom = scored[-max(1, n // 3):]

        stuck = self.stagnation_counter > 0
        parent_label = ""
        parent = None

        if stuck:
            # Alternate REFINE / DIVERGE; as stagnation deepens, favor REFINE
            # (historically the productive mode) on the best program.
            self.label_flip = not self.label_flip
            if self.label_flip:
                parent_label = self.REFINE_LABEL
                parent = scored[0]
            else:
                parent_label = self.DIVERGE_LABEL
                # Diverge from an under-used upper-mid program, not bottom tier.
                pool = mid if mid else top
                parent = min(pool, key=lambda p: self.parent_use_count.get(p.id, 0))
        else:
            # Exploit top tier with usage penalty for diversity.
            parent = min(
                top,
                key=lambda p: (self.parent_use_count.get(p.id, 0), -_score(p)),
            )

        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1

        # --- context selection ---
        context: List[EvolvedProgram] = []
        k = num_context_programs or 0
        if k > 0 and not (stuck and parent_label == self.DIVERGE_LABEL):
            # one elite (not parent)
            elite_pool = [p for p in scored if p.id != parent.id]
            elite_pool.sort(key=lambda p: (self.context_use_count.get(p.id, 0), -_score(p)))
            if elite_pool:
                context.append(elite_pool[0])
            # one contrasting scorer (mid or bottom, not parent)
            contrast_pool = [p for p in (mid + bottom) if p.id != parent.id]
            contrast_pool.sort(key=lambda p: (self.context_use_count.get(p.id, 0),
                                              self.rng.random()))
            for p in contrast_pool:
                if all(p.id != c.id for c in context):
                    context.append(p)
                    break
            # rest: least-used diverse programs
            rest = [p for p in scored if p.id != parent.id and
                    all(p.id != c.id for c in context)]
            rest.sort(key=lambda p: (self.context_use_count.get(p.id, 0),
                                     self.rng.random()))
            context.extend(rest[: max(0, k - len(context))])
        # when diverging, empty context focuses the LLM on the parent

        for c in context:
            self.context_use_count[c.id] = self.context_use_count.get(c.id, 0) + 1

        return {parent_label: parent}, {"": context}


# EVOLVE-BLOCK-END