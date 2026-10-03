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
    """Exploit the top cluster, with stagnation-driven refine/diverge.

    The population shows a sharp gap: a top cluster (~0.67+) far above the
    body (~0.50-0.54). Strategy:
    - Sample parents almost exclusively from the top cluster (soft-randomized
      to avoid always picking the same one).
    - Context = other top performers + one contrasting mid/low program.
    - On stagnation, alternate REFINE (best program, focused context) and
      DIVERGE (fresh lineage / under-explored program) to escape plateaus.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score: Optional[float] = None
        self.best_id: Optional[str] = None
        self.iters_since_improvement = 0
        self.stagnation_counter = 0
        self.last_label_mode = ""  # "", "refine", "diverge"
        self.parent_usage: Dict[str, int] = {}

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score") if program.metrics else None
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    def _is_meaningful_improvement(self, s: float) -> bool:
        if self.best_score is None:
            return True
        return s > self.best_score + 0.01 or s > self.best_score * 1.01

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        s = self._score(program)
        if s != float("-inf") and self._is_meaningful_improvement(s):
            self.best_score = s
            self.best_id = program.id
            self.iters_since_improvement = 0
            self.stagnation_counter = 0
        else:
            self.iters_since_improvement += 1
            if self.iters_since_improvement >= 3:
                self.stagnation_counter += 1
                self.iters_since_improvement = 3  # keep counting stagnation

        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = [p for p in self.programs.values() if self._score(p) != float("-inf")]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        candidates.sort(key=self._score, reverse=True)
        n = len(candidates)
        top = candidates[: max(1, min(5, n // 3))]
        body = candidates[len(top):]

        best = self.get(self.best_id) if self.best_id else None
        if best is None or best.id not in {p.id for p in candidates}:
            best = top[0]

        label = ""
        context: List[EvolvedProgram] = []
        want_ctx = num_context_programs or 0

        # Stagnation handling: alternate REFINE / DIVERGE
        if self.stagnation_counter >= 2:
            if self.last_label_mode != "refine":
                # REFINE the best program, focused context (top peers only)
                parent = best
                label = self.REFINE_LABEL
                self.last_label_mode = "refine"
                seen = {parent.id}
                for p in top:
                    if p.id not in seen and len(context) < max(1, want_ctx // 2):
                        context.append(p)
                        seen.add(p.id)
                return {label: parent}, {"": context}
            else:
                # DIVERGE: pick an under-used program from outside the top cluster
                pool = body if body else candidates
                parent = min(pool, key=lambda p: self.parent_usage.get(p.id, 0))
                label = self.DIVERGE_LABEL
                self.last_label_mode = "diverge"
                return {label: parent}, {"": []}

        # Default: exploit the top cluster with randomization
        if self.rng.random() < 0.3 and best.id in {p.id for p in top}:
            parent = best
        else:
            parent = self.rng.choice(top)

        # Context: other top performers + one contrasting program from the body
        seen = {parent.id}
        top_others = [p for p in top if p.id not in seen]
        self.rng.shuffle(top_others)
        for p in top_others:
            if len(context) >= max(1, want_ctx - 1):
                break
            context.append(p)
            seen.add(p.id)
        if body and want_ctx > len(context):
            contrast = self.rng.choice(body)
            if contrast.id not in seen:
                context.append(contrast)
                seen.add(contrast.id)

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END