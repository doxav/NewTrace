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
    """Adaptive search database: score-weighted parent choice with
    occasional weak-parent + strong-context exploration, and
    stagnation-triggered DIVERGE/REFINE labels."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score = None
        self.stagnation = 0          # iterations since best improved
        self.last_label_iter = -10   # iteration of last label use
        self.recent_parent_ids: List[str] = []

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        # Track meaningful improvement (1% relative or 0.01 absolute)
        score = program.metrics.get("combined_score")
        if isinstance(score, (int, float)):
            score = float(score)
            if self.best_score is None or score > self.best_score:
                if self.best_score is not None:
                    improved = (score - self.best_score) > 0.01 or (
                        self.best_score > 0 and (score - self.best_score) / self.best_score > 0.01
                    )
                    if improved:
                        self.stagnation = 0
                self.best_score = max(self.best_score or score, score)
            else:
                self.stagnation += 1

        self.recent_parent_ids.append(program.parent_id or "")
        if len(self.recent_parent_ids) > 8:
            self.recent_parent_ids.pop(0)

        return program.id

    def _score(self, p: EvolvedProgram) -> float:
        s = p.metrics.get("combined_score")
        return float(s) if isinstance(s, (int, float)) else 0.0

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = sorted(candidates, key=self._score, reverse=True)
        best = scored[0]

        # Stagnation: apply labels sparingly on targeted programs.
        label = ""
        if self.stagnation >= 2 and self.last_iteration - self.last_label_iter >= 2:
            if self.stagnation >= 4:
                # Deeply stuck: diverge from the best program.
                label = self.DIVERGE_LABEL
                parent = best
            else:
                # Mildly stuck: refine a promising but not-best program.
                mid = scored[1] if len(scored) > 1 else best
                label = self.REFINE_LABEL
                parent = mid
            self.last_label_iter = self.last_iteration
        else:
            # Exploit: usually mutate the best; explore: mutate a weak program
            # with strong context (the pattern that produced current best).
            if len(candidates) > 1 and self.rng.random() < 0.3:
                weak_pool = scored[max(1, len(scored) // 2):]
                parent = self.rng.choice(weak_pool)
            else:
                parent = best

        # Context: top scorers excluding parent, plus one diverse/weak program.
        n = num_context_programs or 0
        context: List[EvolvedProgram] = []
        for p in scored:
            if p.id != parent.id and len(context) < max(1, n - 1):
                context.append(p)
        for p in reversed(scored):  # add a bottom program for contrast
            if p.id != parent.id and all(p.id != c.id for c in context):
                context.append(p)
                break
        # Avoid repeating identical context lineages too often
        if len(context) > n:
            context = context[:n]

        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END