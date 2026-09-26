# EVOLVE-BLOCK-START
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


def _score_of(program: EvolvedProgram) -> float:
    """Best available scalar score for a program (defaults to 0.0)."""
    metrics = getattr(program, "metrics", None) or {}
    for key in ("combined_score", "overall_score", "composite_score"):
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return 0.0


class EvolvedProgramDatabase(ProgramDatabase):
    """Rank-based, diversity-aware search strategy database.

    Informed by the measured window (stalled progress, best parent over-reuse):
    - Parents are drawn via rank-based linear weighting. Rank weighting is
      flatter than raw-score softmax, so high scorers are still favoured but
      weaker programs keep real exploration pressure — directly targeting the
      observed stagnation from repeatedly mutating the same best program.
    - A recency guard prevents the exact same program being chosen as parent
      on consecutive samples when alternatives exist.
    - Context programs mix elite anchors with diverse fill drawn from
      under-sampled population members (diversity guidance).
    - Multiobjective mode prefers the global Pareto front with rank weights.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self._last_parent_id: Optional[str] = None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs
    ) -> str:
        """Add a program to the database."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def _rank_weighted_choice(self, candidates: List[EvolvedProgram]) -> EvolvedProgram:
        """Linear rank-based selection (best rank gets weight n, worst 1)."""
        if len(candidates) == 1:
            return candidates[0]
        ordered = sorted(candidates, key=_score_of, reverse=True)
        n = len(ordered)
        weights = [n - i for i in range(n)]  # rank 0 -> n, ..., last -> 1
        total = float(sum(weights))
        r = self.rng.random() * total
        acc = 0.0
        for p, w in zip(ordered, weights):
            acc += w
            if r <= acc:
                return p
        return ordered[-1]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Picks a parent and set of context programs.

        Parent: rank-weighted (Pareto front in multiobjective mode), with a
        recency guard against consecutive reuse of the same parent.
        Context: elite anchors plus diverse fill, excluding the parent.
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 0

        if self.is_multiobjective_enabled():
            front = [p for p in self.get_pareto_front() if p.id in self.programs]
            parent_pool = front if front else candidates
        else:
            parent_pool = candidates

        parent = self._rank_weighted_choice(list(parent_pool))

        # Recency guard: if the same parent was chosen twice in a row and an
        # alternative exists, re-draw once to break exploitation loops.
        if parent.id == self._last_parent_id and len(parent_pool) > 1:
            alt_pool = [p for p in parent_pool if p.id != parent.id]
            if alt_pool:
                alt = self._rank_weighted_choice(alt_pool)
                parent = self.rng.choice([parent, alt])
        self._last_parent_id = parent.id

        # Context: elite anchors first, then diverse random fill.
        others = [p for p in candidates if p.id != parent.id]
        examples: List[EvolvedProgram] = []
        if n_ctx > 0 and others:
            num_elite = max(1, n_ctx // 2)
            elites = sorted(others, key=_score_of, reverse=True)[:num_elite]
            for elite in elites:
                if elite.id not in {e.id for e in examples}:
                    examples.append(elite)
            remaining = [p for p in others if p.id not in {e.id for e in examples}]
            self.rng.shuffle(remaining)
            for p in remaining:
                if len(examples) >= n_ctx:
                    break
                examples.append(p)

        parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END
