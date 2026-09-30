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
    """Search strategy database with tiered, diversity-aware sampling.

    Parent selection is score-weighted over the top half of the population,
    but with a small probability a parent is drawn from the full population
    to escape score plateaus. Context construction always includes the best
    program and then deliberately spans score tiers (upper-mid, mid, and low
    scorers) plus the most recent programs, so the prompt mixes exploitation
    signals with diverse exploration material. Scalar mode uses
    combined_score; multiobjective mode prefers the Pareto front.
    """

    # Probability of sampling the parent from the whole population instead
    # of the score-weighted top half (exploration pressure).
    EXPLORATION_PROB = 0.25

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None

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

    def _score_of(self, program: EvolvedProgram) -> float:
        """Scalar score used for biasing; falls back to 0 when absent."""
        try:
            return float(program.combined_score)
        except (AttributeError, TypeError, ValueError):
            metrics = getattr(program, "metrics", None) or {}
            try:
                return float(metrics.get("combined_score", 0.0))
            except (TypeError, ValueError):
                return 0.0

    def _pareto_front_ids(self) -> List[str]:
        """IDs of the current Pareto front when multiobjective is enabled."""
        if not self.is_multiobjective_enabled():
            return []
        try:
            return [p.id for p in self.get_pareto_front() if p.id in self.programs]
        except Exception:
            return []

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Picks a parent and set of context programs.

        Parent: score-weighted from the top half of the population, or with
        probability ``EXPLORATION_PROB`` drawn from the whole population
        (uniform or Pareto-front-biased in multiobjective mode).
        Context: always includes the best program, then fills with programs
        drawn from distinct score tiers (upper-mid, mid, low) and the most
        recent additions to maximize diversity of the prompt.
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 0

        scored = sorted(candidates, key=self._score_of, reverse=True)
        best = scored[0]

        # ---- Parent selection ----
        front_ids = self._pareto_front_ids()
        if front_ids:
            front_pool = [p for p in scored if p.id in front_ids]
            parent = self.rng.choice(front_pool)
        elif self.rng.random() < self.EXPLORATION_PROB or len(scored) < 4:
            parent = self.rng.choice(candidates)
        else:
            top_k = max(1, len(scored) // 2)
            pool = scored[:top_k]
            weights = [max(self._score_of(p), 1e-6) for p in pool]
            parent = self.rng.choices(pool, weights=weights, k=1)[0]

        # ---- Context selection: best + tiered diversity ----
        chosen: List[EvolvedProgram] = []
        chosen_ids = {parent.id}

        def take(p: Optional[EvolvedProgram]) -> bool:
            if p is not None and p.id not in chosen_ids and len(chosen) < n_ctx:
                chosen.append(p)
                chosen_ids.add(p.id)
                return True
            return False

        if n_ctx > 0:
            take(best)

            # Fill from distinct score tiers below the best.
            if len(scored) > 2:
                upper_mid = scored[1 : max(2, len(scored) // 3)]
                mid = scored[max(2, len(scored) // 3) : max(3, (2 * len(scored)) // 3)]
                low = scored[max(3, (2 * len(scored)) // 3) :]
                for tier in (upper_mid, mid, low):
                    if tier and len(chosen) < n_ctx:
                        take(self.rng.choice(tier))

            # Include recent programs to bias toward current search frontier.
            recent = sorted(
                candidates, key=lambda p: getattr(p, "timestamp", 0.0), reverse=True
            )
            for p in recent:
                if len(chosen) >= n_ctx:
                    break
                take(p)

            # Final fill with random remaining candidates.
            others = [p for p in candidates if p.id not in chosen_ids]
            self.rng.shuffle(others)
            for p in others:
                if len(chosen) >= n_ctx:
                    break
                take(p)

        parent_dict = {"": parent}
        context_programs_dict = {"": chosen}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END
