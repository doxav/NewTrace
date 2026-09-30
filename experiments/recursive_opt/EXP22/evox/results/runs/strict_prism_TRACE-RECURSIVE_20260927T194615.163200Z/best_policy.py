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
    """Search strategy database with elite-exploiting, tiered-diversity sampling.

    The last measured window was again a hard plateau (zero improvement over
    the horizon, ~39 iterations without a new best) under an aggressively
    exploratory policy (uniform parents ~90%, best program in context only
    ~25%). Since broad exploration did not break the plateau, this policy
    rebalances toward the dense elite cluster: parents are drawn from the top
    quartile with score-weighted selection, with a modest exploration
    fallback that only activates under sustained stagnation. Context always
    anchors on the best program (the elite template) and then adds diverse
    material — recent frontier programs, mid/low scorers, and random fills —
    so prompts combine a strong target with fresh variation. Scalar mode uses
    combined_score; multiobjective mode prefers the Pareto front.
    """

    # Parent comes from the top quartile of the population by score.
    TOP_QUARTILE_FRAC = 0.25
    # Exploration fallback: uniform parent probability when stagnating.
    BASE_EXPLORATION_PROB = 0.2
    MAX_EXPLORATION_PROB = 0.6
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
        """Exploration pressure that grows when recent offspring stagnate."""
        recent = sorted(
            candidates, key=lambda p: getattr(p, "timestamp", 0.0), reverse=True
        )[: self.STAGNATION_WINDOW]
        if not recent:
            return self.BASE_EXPLORATION_PROB
        best_score = max(self._score_of(p) for p in candidates)
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

        Parent: score-weighted from the top quartile of the population, or
        drawn uniformly (or Pareto-front-biased) with a stagnation-adaptive
        probability. Context: always includes the best program, then fills
        with recent frontier programs, mid/low scorers, and random candidates.
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
                top_k = max(1, int(len(scored) * self.TOP_QUARTILE_FRAC))
                pool = scored[:top_k]
                weights = [max(self._score_of(p), 1e-6) for p in pool]
                parent = self.rng.choices(pool, weights=weights, k=1)[0]

        # ---- Context selection: best anchor + tiered diversity ----
        chosen: List[EvolvedProgram] = []
        chosen_ids = {parent.id}

        def take(p: Optional[EvolvedProgram]) -> bool:
            if p is not None and p.id not in chosen_ids and len(chosen) < n_ctx:
                chosen.append(p)
                chosen_ids.add(p.id)
                return True
            return False

        if n_ctx > 0:
            # Always anchor the prompt on the best program.
            take(scored[0])

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
