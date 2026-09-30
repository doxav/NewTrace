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

    Feedback-driven policy (window evidence: 10 iterations without improvement,
    context reuse ratio 0.727 with the weakest program most reused):
      * Parents are drawn with a softmax over scores plus an exploration
        epsilon, so strong programs lead but stale lineages are escaped.
      * Context programs are chosen across score tiers (best / mid / bottom)
        to maximize strategy diversity in the prompt instead of repeatedly
        reusing the same small pool.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        # Exploration probability for parent choice.
        self._epsilon = 0.25
        # Softmax temperature for score-weighted parent choice.
        self._temperature = 0.01

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

    def _softmax_pick(self, pool: List[EvolvedProgram]) -> EvolvedProgram:
        """Score-weighted (softmax) choice from a pool."""
        if len(pool) == 1:
            return pool[0]
        scores = [_program_score(p) / self._temperature for p in pool]
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        total = sum(exps)
        r = self.rng.random() * total
        acc = 0.0
        for p, e in zip(pool, exps):
            acc += e
            if r <= acc:
                return p
        return pool[-1]

    def _tiered_context(
        self,
        candidates: List[EvolvedProgram],
        parent: EvolvedProgram,
        num_context_programs: Optional[int],
    ) -> List[EvolvedProgram]:
        """Pick context spanning score tiers for maximal diversity."""
        k = num_context_programs or 0
        if k <= 0:
            return []
        pool = [p for p in candidates if p.id != parent.id]
        if not pool:
            return []
        pool_sorted = sorted(pool, key=_program_score, reverse=True)
        n = len(pool_sorted)

        examples: List[EvolvedProgram] = []
        seen = set()

        # 1) Always anchor with the best program (if not the parent).
        best = pool_sorted[0]
        examples.append(best)
        seen.add(best.id)

        # 2) Fill remaining slots by sampling from mid and bottom tiers.
        if n > 2 and k > 1:
            mid = pool_sorted[1 : max(1, n // 2)]
            bottom = pool_sorted[max(1, n // 2) :]
            buckets = [mid, bottom] if k > 2 else [self.rng.choice([mid, bottom])]
            for b in buckets:
                if len(examples) >= k:
                    break
                b = [p for p in b if p.id not in seen]
                if b:
                    pick = self.rng.choice(b)
                    examples.append(pick)
                    seen.add(pick.id)

        # 3) Fill any remaining slots with random unseen programs.
        rest = [p for p in pool_sorted if p.id not in seen]
        self.rng.shuffle(rest)
        for p in rest:
            if len(examples) >= k:
                break
            examples.append(p)
            seen.add(p.id)

        return examples[:k]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Picks a parent and set of context programs.

        Parent: softmax over scores with epsilon exploration (Pareto-front
        members get a boost in multiobjective mode). Context: tier-diverse
        selection to combat context reuse stagnation.
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
            parent = self._softmax_pick(parent_pool)

        examples = self._tiered_context(candidates, parent, num_context_programs)

        parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END
