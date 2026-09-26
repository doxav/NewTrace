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
    """Simple adaptive search database.

    Strategy:
    - Parent: normally the best program (rotated among near-best to avoid
      overuse); on stagnation alternate DIVERGE (from an under-explored
      distinct program, empty context) and REFINE (on the best, focused).
    - Context: best non-parent + a few programs with distinct scores,
      least-recently-used, to give the LLM contrasting perspectives.
    """

    STAGNATION_WINDOW = 3

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score: float = float("-inf")
        self.iters_since_improvement: int = 0
        self.stagnation_counter: int = 0
        self.label_flip: bool = False
        self.parent_use_count: Dict[str, int] = {}
        self.context_use_count: Dict[str, int] = {}

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
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
            if score > self.best_score:
                meaningful = (
                    self.best_score == float("-inf")
                    or score > self.best_score * 1.01
                    or score > self.best_score + 0.01
                )
                self.best_score = score
                if meaningful:
                    self.iters_since_improvement = 0
                    self.stagnation_counter = 0
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
        stuck = self.stagnation_counter > 0

        # --- parent selection ---
        parent_label = ""
        if stuck:
            self.label_flip = not self.label_flip
            if self.label_flip:
                # Diverge: pick an under-explored program outside the top tier.
                parent_label = self.DIVERGE_LABEL
                pool = scored[max(1, len(scored) // 3):] or scored
                parent = min(pool, key=lambda p: self.parent_use_count.get(p.id, 0))
            else:
                # Refine: focus on the best program.
                parent_label = self.REFINE_LABEL
                parent = scored[0]
        else:
            # Exploit: rotate among the top quartile, preferring least-used.
            top = scored[: max(1, len(scored) // 4)]
            parent = min(top, key=lambda p: (self.parent_use_count.get(p.id, 0), -_score(p)))

        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1

        # --- context selection ---
        context: List[EvolvedProgram] = []
        if parent_label != self.DIVERGE_LABEL:
            n = num_context_programs or 0
            used_ids = {parent.id}
            # 1) the single best non-parent program
            for p in scored:
                if p.id not in used_ids:
                    context.append(p)
                    used_ids.add(p.id)
                    break
            # 2) fill with distinct-score, least-used programs for contrast
            rest = [p for p in scored if p.id not in used_ids]
            rest.sort(key=lambda p: (self.context_use_count.get(p.id, 0), self.rng.random()))
            seen_scores: List[float] = [_score(c) for c in context]
            for p in rest:
                if len(context) >= n:
                    break
                s = _score(p)
                # prefer scores distinct from already-chosen context (contrast)
                if all(abs(s - t) > 0.05 for t in seen_scores) or len(context) < 2:
                    context.append(p)
                    seen_scores.append(s)
                    used_ids.add(p.id)
            # 3) top up with least-used if needed
            for p in rest:
                if len(context) >= n:
                    break
                if p.id not in used_ids:
                    context.append(p)
                    used_ids.add(p.id)

        for c in context:
            self.context_use_count[c.id] = self.context_use_count.get(c.id, 0) + 1

        return {parent_label: parent}, {"": context}


# EVOLVE-BLOCK-END