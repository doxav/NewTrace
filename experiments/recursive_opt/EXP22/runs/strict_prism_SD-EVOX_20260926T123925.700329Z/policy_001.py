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
    - Track best-score history in add(); detect stagnation (no meaningful
      improvement > 1% / 0.01 for several iterations).
    - Parent selection: exploit best programs normally, but rotate in
      under-explored parents to avoid reuse concentration.
    - Context: mix top scorers with diverse/low scorers for contrast.
    - On stagnation, alternate DIVERGE (from best program, fresh direction)
      and REFINE (on best promising program) labels.
    """

    STAGNATION_WINDOW = 4  # iterations without meaningful improvement -> act

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        # progress tracking
        self.best_score: float = float("-inf")
        self.iters_since_improvement: int = 0
        self.stagnation_counter: int = 0
        self.label_flip: bool = False  # alternate diverge/refine when stuck
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

        # --- progress tracking (persists across resumes) ---
        score = _score(program)
        if score > float("-inf"):
            if self.best_score > float("-inf") and (
                score > self.best_score * 1.01 + 1e-12 or score > self.best_score + 0.01
            ):
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
        candidates = [p for p in self.programs.values() if isinstance(_score(p), float)]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = sorted(candidates, key=_score, reverse=True)
        top = scored[: max(1, len(scored) // 3)]
        bottom = scored[-max(1, len(scored) // 3):]

        stuck = self.stagnation_counter > 0

        # --- parent selection ---
        if stuck:
            # Alternate: diverge from a mid/low-tier program for fresh ideas,
            # or refine the best program to squeeze out improvements.
            self.label_flip = not self.label_flip
            if self.label_flip:
                parent_label = self.DIVERGE_LABEL
                pool = bottom if bottom else scored
                parent = min(pool, key=lambda p: self.parent_use_count.get(p.id, 0))
            else:
                parent_label = self.REFINE_LABEL
                parent = scored[0]
        else:
            parent_label = ""
            # Exploit top tier but prefer least-used parents for diversity.
            parent = min(top, key=lambda p: (self.parent_use_count.get(p.id, 0),
                                             -_score(p)))

        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1

        # --- context selection ---
        context: List[EvolvedProgram] = []
        if not stuck or not self.label_flip:
            n = num_context_programs or 0
            # one best (not parent), one contrasting low scorer, rest diverse
            pool_best = [p for p in scored if p.id != parent.id]
            pool_best.sort(key=lambda p: (self.context_use_count.get(p.id, 0), -_score(p)))
            if pool_best and n > 0:
                context.append(pool_best[0])
            pool_low = [p for p in bottom if p.id not in {parent.id} and
                        all(p.id != c.id for c in context)]
            pool_low.sort(key=lambda p: self.context_use_count.get(p.id, 0))
            if pool_low and n > 1:
                context.append(pool_low[0])
            rest = [p for p in scored if p.id != parent.id and
                    all(p.id != c.id for c in context)]
            rest.sort(key=lambda p: (self.context_use_count.get(p.id, 0),
                                     self.rng.random()))
            context.extend(rest[: max(0, n - len(context))])
        # when diverging, empty context focuses the LLM on the parent

        for c in context:
            self.context_use_count[c.id] = self.context_use_count.get(c.id, 0) + 1

        parent_dict = {parent_label: parent}
        context_programs_dict = {"": context}
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END