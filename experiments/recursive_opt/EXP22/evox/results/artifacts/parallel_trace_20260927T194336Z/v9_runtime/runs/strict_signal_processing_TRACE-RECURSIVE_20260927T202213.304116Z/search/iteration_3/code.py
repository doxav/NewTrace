# EVOLVE-BLOCK-START
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


def _program_score(p: EvolvedProgram) -> float:
    """Best available scalar score for a program (0.0 if unknown)."""
    metrics = getattr(p, "metrics", None) or {}
    for key in ("combined_score", "overall_score", "composite_score"):
        v = metrics.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


class EvolvedProgramDatabase(ProgramDatabase):
    """Stagnation-aware search database.

    Window evidence: best score plateaued at 0.5387 (0 improvement over the
    last window, 30+ iterations without improvement) while the weakest
    program (initial baseline) remained the most-reused context program and
    parents concentrated on a few top programs whose children rarely beat
    the plateau. Policy changes:
      * Context programs NEVER include bottom-tier (very weak) programs;
        the weakest baseline is no longer re-shown to the model.
      * Parent selection is rank-weighted with a stagnation-adaptive
        epsilon: the longer the search stalls, the more exploratory
        parent choice becomes, escaping stale top lineages.
      * Lineage novelty: parents that have already produced many children
        without improvement are down-weighted; recently added programs get
        a novelty boost so fresh lineages are explored.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        # Base exploration probability for parent choice.
        self._base_epsilon = 0.25
        # Epsilon grows by this amount per stagnant iteration (capped).
        self._epsilon_growth = 0.02
        self._max_epsilon = 0.6
        # Exponent for rank-weighted parent selection (higher = more greedy).
        self._rank_exponent = 1.5
        # Weight of the novelty (child-count) term in parent selection.
        self._novelty_weight = 0.5
        # Fraction of population considered "bottom tier" and excluded
        # from context entirely.
        self._bottom_tier_frac = 0.25
        # Track how often each program is served as context/parent.
        self._usage_counts: Counter = Counter()
        # Track how many children each program has produced.
        self._child_counts: Counter = Counter()
        # Best score seen so far (for stagnation tracking).
        self._best_score: Optional[float] = None
        self._stagnation_iters = 0

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs
    ) -> str:
        """Add a program to the database."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if program.parent_id and program.parent_id in self.programs:
            self._child_counts[program.parent_id] += 1

        # Stagnation tracking: did this program beat the best-so-far?
        score = _program_score(program)
        if self._best_score is None or score > self._best_score + 1e-9:
            self._best_score = score
            self._stagnation_iters = 0
        else:
            self._stagnation_iters += 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def _epsilon(self) -> float:
        """Stagnation-adaptive exploration probability."""
        return min(
            self._max_epsilon,
            self._base_epsilon + self._epsilon_growth * self._stagnation_iters,
        )

    def _rank_weighted_pick(self, pool: List[EvolvedProgram]) -> EvolvedProgram:
        """Rank-weighted choice with lineage-novelty adjustment.

        Weight ~ 1/rank^exponent, further down-weighted for programs that
        already produced many children (exploited lineages), and boosted
        for programs with few children (fresh lineages).
        """
        if len(pool) == 1:
            return pool[0]
        ranked = sorted(pool, key=_program_score, reverse=True)
        max_children = max(1, max(self._child_counts.get(p.id, 0) for p in ranked))
        weights = []
        for i, p in enumerate(ranked):
            base = 1.0 / ((i + 1) ** self._rank_exponent)
            children = self._child_counts.get(p.id, 0)
            novelty = 1.0 + self._novelty_weight * (1.0 - children / max_children)
            weights.append(base * novelty)
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for p, w in zip(ranked, weights):
            acc += w
            if r <= acc:
                return p
        return ranked[-1]

    def _tiered_context(
        self,
        candidates: List[EvolvedProgram],
        parent: EvolvedProgram,
        num_context_programs: Optional[int],
    ) -> List[EvolvedProgram]:
        """Pick context spanning upper score tiers, never bottom-tier programs."""
        k = num_context_programs or 0
        if k <= 0:
            return []
        pool = [p for p in candidates if p.id != parent.id]
        if not pool:
            return []
        pool_sorted = sorted(pool, key=_program_score, reverse=True)
        n = len(pool_sorted)

        # Hard-exclude the bottom tier: weak programs are not useful
        # exemplars and re-showing them caused reuse stagnation.
        cut = max(1, min(n - 1, int(round(n * self._bottom_tier_frac))))
        eligible = pool_sorted[: n - cut] if n > 2 else pool_sorted
        if not eligible:
            eligible = pool_sorted[:1]

        examples: List[EvolvedProgram] = []
        seen = {p.id for p in examples}

        def pick_weighted(sub: List[EvolvedProgram]) -> Optional[EvolvedProgram]:
            sub = [p for p in sub if p.id not in seen]
            if not sub:
                return None
            weights = [1.0 / (1.0 + 0.5 * self._usage_counts.get(p.id, 0)) for p in sub]
            total = sum(weights)
            r = self.rng.random() * total
            acc = 0.0
            for p, w in zip(sub, weights):
                acc += w
                if r <= acc:
                    return p
            return sub[-1]

        # 1) Anchor with the best eligible program.
        best = eligible[0]
        examples.append(best)
        seen.add(best.id)

        # 2) Fill from mid and upper tiers for diversity.
        if len(eligible) > 2 and k > 1:
            mid = eligible[1 : max(1, len(eligible) // 2)]
            upper = eligible[max(1, len(eligible) // 2) :]
            buckets = [mid, upper] if k > 2 else [self.rng.choice([mid, upper])]
            for b in buckets:
                if len(examples) >= k:
                    break
                pick = pick_weighted(b)
                if pick is not None:
                    examples.append(pick)
                    seen.add(pick.id)

        # 3) Fill any remaining slots from the eligible pool.
        rest = eligible
        while len(examples) < k:
            pick = pick_weighted(rest)
            if pick is None:
                break
            examples.append(pick)
            seen.add(pick.id)

        for p in examples:
            self._usage_counts[p.id] += 1
        self._usage_counts[parent.id] += 1

        return examples[:k]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Picks a parent and set of context programs.

        Parent: epsilon-greedy rank-weighted selection with stagnation-
        adaptive exploration and lineage-novelty weighting. Context:
        tier-diverse selection that hard-excludes bottom-tier programs.
        """
        candidates = list(self.programs.values())
        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        parent_pool = candidates
        if self.is_multiobjective_enabled():
            front = [p for p in self.get_pareto_front() if p.id in self.programs]
            if front:
                # Mix front members (weighted 3x) with the general population.
                parent_pool = front + front + front + candidates

        if self.rng.random() < self._epsilon():
            parent = self.rng.choice(candidates)
        else:
            parent = self._rank_weighted_pick(parent_pool)

        examples = self._tiered_context(candidates, parent, num_context_programs)

        parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END
