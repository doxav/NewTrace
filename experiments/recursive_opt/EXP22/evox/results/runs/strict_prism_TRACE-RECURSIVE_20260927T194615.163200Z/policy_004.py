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
    """Search strategy database with stagnation-escalating, diversity-first sampling.

    The latest measured window was a hard plateau (zero improvement over the
    horizon, ~39 iterations without a new best). This policy escalates
    exploration much more aggressively than the previous one: the uniform
    parent probability starts high and saturates quickly as recent programs
    fail to improve on the best score. Context construction strongly avoids
    reusing the single best program (included only ~25% of the time) and
    deliberately mixes recent frontier programs, mid/low scorers, and
    least-reused programs so prompts carry fresh material. Scalar mode uses
    combined_score; multiobjective mode prefers the Pareto front.
    """

    # Exploration pressure (uniform parent over the whole population); grows
    # with stagnation and saturates near full exploration.
    BASE_EXPLORATION_PROB = 0.5
    MAX_EXPLORATION_PROB = 0.9
    # Number of most-recent programs inspected to estimate stagnation.
    STAGNATION_WINDOW = 6

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

    def _exploration_prob(self, candidates: List[EvolvedProgram]) -> float:
        """Exploration pressure that grows quickly when recent offspring stagnate."""
        recent = sorted(
            candidates, key=lambda p: getattr(p, "timestamp", 0.0), reverse=True
        )[: self.STAGNATION_WINDOW]
        if not recent:
            return self.BASE_EXPLORATION_PROB
        best_score = max(self._score_of(p) for p in candidates)
        # A recent program counts as stagnant when it did NOT beat the best
        # score by the improvement threshold (1%).
        stagnant = sum(1 for p in recent if self._score_of(p) <= best_score * 1.01)
        frac = stagnant / len(recent)
        prob = self.BASE_EXPLORATION_PROB + frac * (
            self.MAX_EXPLORATION_PROB - self.BASE_EXPLORATION_PROB
        )
        return min(prob, self.MAX_EXPLORATION_PROB)

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Picks a parent and set of context programs.

        Parent: drawn uniformly (or Pareto-front-biased) with a high,
        stagnation-adaptive probability; otherwise mildly score-weighted over
        the whole population. Context: diversity-first — recent frontier
        programs, mid/low scorers, and least-reused programs — with the best
        program included only ~25% of the time.
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 0

        scored = sorted(candidates, key=self._score_of, reverse=True)

        # ---- Parent selection ----
        front_ids = self._pareto_front_ids()
        if front_ids:
            front_pool = [p for p in scored if p.id in front_ids]
            parent = self.rng.choice(front_pool)
        else:
            explore_prob = self._exploration_prob(candidates)
            if self.rng.random() < explore_prob or len(scored) < 4:
                parent = self.rng.choice(candidates)
            else:
                weights = [max(self._score_of(p), 1e-6) ** 0.5 for p in scored]
                parent = self.rng.choices(scored, weights=weights, k=1)[0]

        # ---- Context selection: diversity-first ----
        chosen: List[EvolvedProgram] = []
        chosen_ids = {parent.id}

        def take(p: Optional[EvolvedProgram]) -> bool:
            if p is not None and p.id not in chosen_ids and len(chosen) < n_ctx:
                chosen.append(p)
                chosen_ids.add(p.id)
                return True
            return False

        if n_ctx > 0:
            # Include the best program only ~25% of the time so the elite
            # template stops dominating every prompt.
            best = scored[0]
            if self.rng.random() < 0.25:
                take(best)

            # Recent frontier programs (current search material).
            recent = sorted(
                candidates, key=lambda p: getattr(p, "timestamp", 0.0), reverse=True
            )
            for p in recent[: max(1, n_ctx // 2)]:
                if len(chosen) >= n_ctx:
                    break
                take(p)

            # Mid and low scorers for diverse exploration material.
            if len(scored) > 2:
                mid = scored[max(1, len(scored) // 3) : max(2, (2 * len(scored)) // 3)]
                low = scored[max(2, (2 * len(scored)) // 3) :]
                for tier in (mid, low):
                    if tier and len(chosen) < n_ctx:
                        take(self.rng.choice(tier))

            # Fill with random remaining candidates.
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
