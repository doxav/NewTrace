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
    """Plateau-escaping search database.

    Window evidence (iterations 74-84): the best score improved again
    (0.5392 -> 0.5399) while context reuse of the single most-reused
    program fell to ~0.52 and parent reuse stayed low (~0.11). The recent
    bests all came from freshly sampled, diverse parents, confirming that
    anchor rotation and recency-boosted parent selection are the working
    levers. Policy changes (continuation of the same direction):
      * The rotating anchor tier is widened from 8 to 10 eligible programs
        so anchor rotation spans even more strong programs.
      * The hard context-share cap is tightened from 0.25 to 0.22 so any
        single program is excluded from context sooner.
      * Recency weight raised 0.8 -> 1.0 so recently added fresh lineages
        (which produced the latest bests) are preferred as parents.
      * Base epsilon 0.35 (growth 0.035, cap 0.75) retained: exploration
        probability is not the bottleneck.
      * Lineage saturation exclusion (limit 2), bottom-tier exclusion,
        inverse-usage context weighting and tier-diverse fill retained.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        # Base exploration probability for parent choice.
        self._base_epsilon = 0.35
        # Epsilon grows by this amount per stagnant iteration (capped).
        self._epsilon_growth = 0.035
        self._max_epsilon = 0.75
        # Exponent for rank-weighted parent selection (higher = more greedy).
        self._rank_exponent = 1.5
        # Weight of the novelty (child-count) term in parent selection.
        self._novelty_weight = 1.0
        # A lineage is "saturated" once it has this many children without
        # any child beating the best-so-far score; it is excluded from
        # parent selection.
        self._saturation_limit = 2
        # Fraction of population considered "bottom tier" and excluded
        # from context entirely.
        self._bottom_tier_frac = 0.25
        # Weight of the reuse penalty in context selection.
        self._reuse_penalty = 0.9
        # Hard cap on any single program's share of total context serves.
        self._max_context_share = 0.22
        # Recency boost strength for parent selection tie-breaking.
        self._recency_weight = 1.0
        # Number of top-tier eligible programs among which the context
        # anchor rotates (inverse-usage weighted).
        self._anchor_tier_size = 10
        # Track how often each program is served as context/parent.
        self._usage_counts: Counter = Counter()
        # Track how many children each program has produced.
        self._child_counts: Counter = Counter()
        # Track how many children each program produced that did NOT beat
        # the best-so-far score at their time of addition.
        self._unproductive_children: Counter = Counter()
        # Insertion order for recency tie-breaking.
        self._insertion_order: Dict[str, int] = {}
        self._insert_counter = 0
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
        self._insertion_order[program.id] = self._insert_counter
        self._insert_counter += 1

        if program.parent_id and program.parent_id in self.programs:
            self._child_counts[program.parent_id] += 1
            score = _program_score(program)
            if not (self._best_score is not None and score > self._best_score + 1e-9):
                self._unproductive_children[program.parent_id] += 1

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

    def _is_saturated(self, program_id: str) -> bool:
        """A lineage is saturated if it produced many unproductive children."""
        return self._unproductive_children.get(program_id, 0) >= self._saturation_limit

    def _overused_as_context(self, program_id: str) -> bool:
        """True if this program's context-usage share exceeds the hard cap."""
        total = sum(self._usage_counts.values())
        if total == 0:
            return False
        return self._usage_counts.get(program_id, 0) / total > self._max_context_share

    def _most_overused_id(self, eligible_ids: List[str]) -> Optional[str]:
        """The eligible program with the highest context-usage share (if over cap)."""
        total = sum(self._usage_counts.values())
        if total == 0 or not eligible_ids:
            return None
        best_id = max(eligible_ids, key=lambda i: self._usage_counts.get(i, 0))
        if self._usage_counts.get(best_id, 0) / total > self._max_context_share:
            return best_id
        return None

    def _rank_weighted_pick(self, pool: List[EvolvedProgram]) -> EvolvedProgram:
        """Rank-weighted choice with lineage-novelty and recency adjustment.

        Weight ~ 1/rank^exponent, down-weighted for programs that already
        produced many unproductive children (saturated lineages), boosted
        for programs with few children (fresh lineages), and given a
        recency tie-break so recently added plateau programs are tried.
        """
        if len(pool) == 1:
            return pool[0]
        ranked = sorted(pool, key=_program_score, reverse=True)
        max_children = max(1, max(self._child_counts.get(p.id, 0) for p in ranked))
        max_order = max(self._insertion_order.get(p.id, 0) for p in ranked)
        weights = []
        for i, p in enumerate(ranked):
            base = 1.0 / ((i + 1) ** self._rank_exponent)
            children = self._child_counts.get(p.id, 0)
            novelty = 1.0 + self._novelty_weight * (1.0 - children / max_children)
            saturation = 1.0 / (1.0 + self._unproductive_children.get(p.id, 0))
            # Recency term: newer programs (larger insertion order) get a
            # stronger boost, breaking ties among plateau-scored programs.
            order = self._insertion_order.get(p.id, 0)
            recency = 1.0 + self._recency_weight * (
                order / max_order if max_order > 0 else 0.0
            )
            weights.append(base * novelty * saturation * recency)
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
        """Pick context spanning upper score tiers with a ROTATING anchor.

        The anchor is sampled among the top `_anchor_tier_size` eligible
        programs with inverse-usage weights (instead of always being the
        single best program), so no single program dominates the prompt
        history. Bottom-tier and over-reused programs are hard-excluded
        unless that empties the pool.
        """
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

        # Hard-exclude over-reused programs unless that empties the pool.
        not_overused = [p for p in eligible if not self._overused_as_context(p.id)]
        if not_overused:
            eligible = not_overused

        examples: List[EvolvedProgram] = []
        seen = set()

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

        # 1) ROTATING anchor: among the top tier of eligible programs,
        # pick by inverse usage weight so the anchor rotates across
        # strong programs instead of always being the single best.
        top_tier = eligible[: max(1, min(len(eligible), self._anchor_tier_size))]
        anchor = pick_weighted(top_tier)
        if anchor is None:
            anchor = eligible[0]
        examples.append(anchor)
        seen.add(anchor.id)

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
        adaptive exploration, lineage-novelty weighting, strong recency
        tie-breaks, and hard exclusion of saturated lineages. Context:
        tier-diverse selection with a rotating (usage-weighted) anchor over
        a widened top tier that hard-excludes bottom-tier and over-reused
        programs under a tighter share cap.
        """
        candidates = list(self.programs.values())
        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        # Exclude saturated lineages from the parent pool (keep at least
        # one program so sampling never fails).
        parent_pool = [p for p in candidates if not self._is_saturated(p.id)]
        if not parent_pool:
            parent_pool = candidates

        if self.is_multiobjective_enabled():
            front = [p for p in self.get_pareto_front() if p.id in self.programs]
            front = [p for p in front if not self._is_saturated(p.id)]
            if front:
                # Mix front members (weighted 3x) with the general pool.
                parent_pool = front + front + front + parent_pool

        if self.rng.random() < self._epsilon():
            parent = self.rng.choice(parent_pool)
        else:
            parent = self._rank_weighted_pick(parent_pool)

        examples = self._tiered_context(candidates, parent, num_context_programs)

        parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END
