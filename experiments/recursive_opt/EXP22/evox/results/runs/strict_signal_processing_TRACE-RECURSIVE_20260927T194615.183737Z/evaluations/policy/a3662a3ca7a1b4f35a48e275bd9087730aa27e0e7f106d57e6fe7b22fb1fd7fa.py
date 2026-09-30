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
    """Search strategy database with improvement-aware, novelty-driven sampling.

    Rationale from measured window feedback: the search plateaued (30
    iterations without improvement, window gain 0.0) while the most-reused
    parent (score 0.526, reuse ratio 0.14) and most-reused context (reuse
    ratio 0.29) were mid-tier programs sampled repeatedly. The best new
    programs came from mid-tier parents paired with top-tier, diverse
    context. This policy therefore:
      1. Ranks parents by an improvement-potential score: recent lineage
         gains (child score vs. its parent's score) weighted by recency,
         so saturated lineages are not over-exploited. A small probability
         still exploits the global best, and a small uniform-random share
         injects exploration to escape plateaus.
      2. Penalizes over-reused parents/contexts via tracked sample counts,
         and adds a novelty term that favors solutions textually distinct
         from recently sampled ones.
      3. Builds context from distinct solutions spanning top and mid tiers,
         rotating over-reused context programs out.
    The ProgramDatabase contract (add/sample signatures, initial_program,
    _save_program, _update_best_program) is preserved for both scalar and
    multiobjective modes.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self._sample_counts: Dict[str, int] = {}
        self._recent_solutions: List[str] = []

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
        """Best available scalar score for a program (0.0 if missing)."""
        metrics = getattr(program, "metrics", None) or {}
        for key in ("combined_score", "overall_score", "composite_score"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    def _parent_score_of(self, program: EvolvedProgram) -> float:
        """Score of a program's parent, if resolvable; else its own score."""
        pid = getattr(program, "parent_id", None)
        if pid and pid in self.programs:
            return self._score_of(self.programs[pid])
        return self._score_of(program)

    def _novelty_bonus(self, program: EvolvedProgram) -> float:
        """Bonus if this program's solution differs from recently used ones."""
        sol = getattr(program, "solution", "")
        recent = self._recent_solutions[-10:]
        if not recent:
            return 1.0
        # Cheap textual novelty: fraction of recent solutions it differs from.
        diff = sum(1 for s in recent if s != sol)
        return 0.5 + 0.5 * (diff / len(recent))

    def _potential_weight(self, program: EvolvedProgram) -> float:
        """Improvement potential: recency-weighted lineage gain, reuse-penalized."""
        max_iter = (
            max((getattr(p, "iteration_found", 0) or 0) for p in self.programs.values())
            or 0
        )
        age = max_iter - (getattr(program, "iteration_found", 0) or 0)
        recency = 0.5**age  # exponential decay by iteration age

        # Lineage gain: how much this program improved over its parent.
        gain = self._score_of(program) - self._parent_score_of(program)
        gain_term = 1.0 + max(gain, 0.0) * 10.0  # amplify positive improvements

        # Penalize over-sampled parents to break reuse loops.
        reuse = self._sample_counts.get(program.id, 0)
        reuse_penalty = 1.0 / (1.0 + 0.75 * reuse)

        # Novelty: favor solutions not recently used as parents/context.
        novelty = self._novelty_bonus(program)

        return recency * gain_term * reuse_penalty * novelty

    def _recency_weighted_choice(
        self, candidates: List[EvolvedProgram]
    ) -> EvolvedProgram:
        """Choose a parent by improvement potential; occasionally exploit best."""
        if len(candidates) == 1:
            return candidates[0]

        # Occasionally exploit the global best directly (the best lineage
        # is saturated, so this stays low).
        if self.rng.random() < 0.08:
            return max(candidates, key=self._score_of)

        # Small uniform-random exploration share to escape plateaus.
        if self.rng.random() < 0.07:
            return self.rng.choice(candidates)

        weights = [max(self._potential_weight(p), 1e-6) for p in candidates]
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(candidates)
        r = self.rng.random() * total
        acc = 0.0
        for p, w in zip(candidates, weights):
            acc += w
            if r <= acc:
                return p
        return candidates[-1]

    def _diverse_context(
        self, parent: EvolvedProgram, candidates: List[EvolvedProgram], k: int
    ) -> List[EvolvedProgram]:
        """Pick up to k context programs distinct from parent and each other.

        Strategy: sort by reuse-penalized score, then greedily pick programs
        whose solution text differs from already-picked ones, alternating
        between top and mid-tier scorers; over-reused contexts rotate out.
        """
        if k is None or k <= 0:
            return []

        pool = [p for p in candidates if p.id != parent.id]
        if not pool:
            return []

        # Stronger reuse penalty in the sort key so saturated contexts rotate out.
        pool_sorted = sorted(
            pool,
            key=lambda p: self._score_of(p)
            / (1.0 + 0.5 * self._sample_counts.get(p.id, 0)),
            reverse=True,
        )
        picked: List[EvolvedProgram] = []
        seen_solutions = {parent.solution}

        # Alternate: one from the top of the list, one from the middle band,
        # to mix exploitation exemplars with diverse mid-tier variants.
        top_idx, mid_idx = 0, len(pool_sorted) // 2
        while len(picked) < k and (
            top_idx < len(pool_sorted) or mid_idx < len(pool_sorted)
        ):
            for idx in (top_idx, mid_idx):
                if idx >= len(pool_sorted):
                    continue
                cand = pool_sorted[idx]
                if cand.id in {p.id for p in picked}:
                    continue
                if cand.solution in seen_solutions and len(picked) < k:
                    # Allow duplicates only if we cannot fill otherwise.
                    continue
                picked.append(cand)
                seen_solutions.add(cand.solution)
                if len(picked) >= k:
                    break
            top_idx += 1
            mid_idx += 1

        # Fallback: fill remaining slots with any unused candidates.
        if len(picked) < k:
            for cand in pool_sorted:
                if len(picked) >= k:
                    break
                if cand.id not in {p.id for p in picked}:
                    picked.append(cand)

        return picked[:k]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Picks a parent and set of context programs.

        Parent: improvement-potential-weighted choice (recency x lineage gain,
        reuse-penalized, novelty-boosted, with occasional global-best
        exploitation and uniform exploration). Context: diverse, distinct
        solutions spanning top and mid-tier scorers with reuse-based rotation.
        Sample counts and recent solutions are updated to drive the reuse
        and novelty terms.
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        parent = self._recency_weighted_choice(candidates)
        examples = self._diverse_context(parent, candidates, num_context_programs or 0)

        # Track reuse and recent solutions to feed the penalties/bonuses.
        self._sample_counts[parent.id] = self._sample_counts.get(parent.id, 0) + 1
        for ex in examples:
            self._sample_counts[ex.id] = self._sample_counts.get(ex.id, 0) + 1
        self._recent_solutions.append(getattr(parent, "solution", ""))
        self._recent_solutions = self._recent_solutions[-20:]

        parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END
