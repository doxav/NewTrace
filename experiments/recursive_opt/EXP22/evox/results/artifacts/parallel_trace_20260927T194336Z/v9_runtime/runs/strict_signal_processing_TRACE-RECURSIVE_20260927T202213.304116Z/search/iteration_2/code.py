# EVOLVE-BLOCK-START
import logging
import math
from collections import Counter
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
    """Diversity-aware search database.

    Feedback-driven policy (window evidence: best score improved 0.5335 ->
    0.5387, context reuse ratio dropped 0.727 -> 0.524, but 20 iterations
    without improvement and the weakest program still dominates context reuse):
      * Parents are drawn with epsilon-greedy rank-weighted selection, so
        strong programs lead while stale lineages are escaped.
      * Context programs span score tiers AND are penalized by recent reuse
        frequency, so the same small pool (especially the weakest program)
        is not repeatedly shown to the model.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        # Exploration probability for parent choice.
        self._epsilon = 0.3
        # Exponent for rank-weighted parent selection (higher = more greedy).
        self._rank_exponent = 1.5
        # Penalty weight for recently reused context programs.
        self._reuse_penalty = 0.5
        # Track how often each program is served as context/parent.
        self._usage_counts: Counter = Counter()

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

    def _rank_weighted_pick(self, pool: List[EvolvedProgram]) -> EvolvedProgram:
        """Rank-weighted choice: weight ~ 1/rank^exponent over score-sorted pool."""
        if len(pool) == 1:
            return pool[0]
        ranked = sorted(pool, key=_program_score, reverse=True)
        weights = [1.0 / ((i + 1) ** self._rank_exponent) for i in range(len(ranked))]
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
        """Pick context spanning score tiers, avoiding over-reused programs."""
        k = num_context_programs or 0
        if k <= 0:
            return []
        pool = [p for p in candidates if p.id != parent.id]
        if not pool:
            return []
        pool_sorted = sorted(pool, key=_program_score, reverse=True)
        n = len(pool_sorted)

        # Diversity-aware weights: down-weight frequently reused programs.
        def pick_weighted(sub: List[EvolvedProgram]) -> Optional[EvolvedProgram]:
            sub = [p for p in sub if p.id not in seen]
            if not sub:
                return None
            weights = [
                1.0 / (1.0 + self._reuse_penalty * self._usage_counts.get(p.id, 0))
                for p in sub
            ]
            total = sum(weights)
            r = self.rng.random() * total
            acc = 0.0
            for p, w in zip(sub, weights):
                acc += w
                if r <= acc:
                    return p
            return sub[-1]

        examples: List[EvolvedProgram] = []
        seen = set()

        # 1) Anchor with the best program (if not the parent).
        best = pool_sorted[0]
        examples.append(best)
        seen.add(best.id)

        # 2) Fill remaining slots from mid and bottom tiers.
        if n > 2 and k > 1:
            mid = pool_sorted[1 : max(1, n // 2)]
            bottom = pool_sorted[max(1, n // 2) :]
            buckets = [mid, bottom] if k > 2 else [self.rng.choice([mid, bottom])]
            for b in buckets:
                if len(examples) >= k:
                    break
                pick = pick_weighted(b)
                if pick is not None:
                    examples.append(pick)
                    seen.add(pick.id)

        # 3) Fill any remaining slots with weighted picks from the rest.
        rest = pool_sorted
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

        Parent: epsilon-greedy rank-weighted selection (Pareto-front members
        get a boost in multiobjective mode). Context: tier-diverse,
        reuse-penalized selection to combat context reuse stagnation.
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

        if self.rng.random() < self._epsilon:
            parent = self.rng.choice(candidates)
        else:
            parent = self._rank_weighted_pick(parent_pool)

        examples = self._tiered_context(candidates, parent, num_context_programs)

        parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END
