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
    """Adaptive search database.

    Strategy:
    - Bias parent selection toward high-scoring programs (soft exploitation),
      with occasional exploration of under-explored / low-scoring parents.
    - Context mixes top programs with diverse mid/low scorers for contrast.
    - On stagnation, alternate between REFINE (best program) and DIVERGE
      (fresh lineage) to escape plateaus.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score = None
        self.best_id = None
        self.iters_since_improvement = 0
        self.stagnation_counter = 0
        self.last_label_mode = ""  # "", "refine", "diverge"
        self.parent_usage: Dict[str, int] = {}
        self.recent_scores: List[float] = []

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
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

        self._update_best_program(program)

        # Track progress state
        s = self._score(program)
        if s != float("-inf"):
            self.recent_scores.append(s)
            self.recent_scores = self.recent_scores[-20:]

        improved = False
        if self.best_score is None or s > self.best_score * 1.01 + 0.01 or (
            s > self.best_score + 0.01
        ):
            if self.best_score is None or s > self.best_score:
                improved = True

        if improved:
            self.best_score = s
            self.best_id = program.id
            self.iters_since_improvement = 0
            self.stagnation_counter = 0
        else:
            self.iters_since_improvement += 1
            if self.iters_since_improvement >= 3:
                self.stagnation_counter += 1

        # Track parent usage
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
        top = candidates[: max(1, n // 3)]
        mid = candidates[max(1, n // 3): max(2, 2 * n // 3)]
        low = candidates[max(2, 2 * n // 3):]

        label = ""
        parent = None

        # Stagnation handling: alternate refine/diverge
        if self.stagnation_counter >= 2:
            if self.last_label_mode == "refine" or self.best_id not in self.programs:
                # Diverge: pick from lower tier or least-used lineage
                pool = low if low else candidates
                parent = min(pool, key=lambda p: self.parent_usage.get(p.id, 0))
                label = self.DIVERGE_LABEL
                self.last_label_mode = "diverge"
            else:
                parent = self.get(self.best_id) or top[0]
                label = self.REFINE_LABEL
                self.last_label_mode = "refine"
            context_programs_dict = {"": []}
            return {label: parent}, context_programs_dict

        # Default: soft exploitation with exploration
        r = self.rng.random()
        if r < 0.55 and top:
            parent = self.rng.choice(top)
        elif r < 0.85 and mid:
            parent = self.rng.choice(mid)
        elif low:
            parent = self.rng.choice(low)
        else:
            parent = self.rng.choice(candidates)

        # Context: mix top performers with diverse others, avoid parent
        context: List[EvolvedProgram] = []
        seen = {parent.id}
        top_others = [p for p in top if p.id not in seen]
        self.rng.shuffle(top_others)
        for p in top_others[: max(1, (num_context_programs or 0) // 2)]:
            context.append(p)
            seen.add(p.id)
        rest = [p for p in candidates if p.id not in seen]
        self.rng.shuffle(rest)
        for p in rest:
            if len(context) >= (num_context_programs or 0):
                break
            context.append(p)
            seen.add(p.id)

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END