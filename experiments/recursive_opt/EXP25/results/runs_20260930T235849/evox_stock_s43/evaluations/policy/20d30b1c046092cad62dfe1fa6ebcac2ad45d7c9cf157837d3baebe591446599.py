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
    - Parent selection: score-weighted with an explicit anti-reuse penalty,
      so the monoculture around a single parent is broken and under-explored
      high/mid scorers get mutation attempts.
    - Context: always mixes top performers with diverse/low scorers for contrast,
      avoiding the parent itself.
    - Stagnation ladder (tracked in add()): escalate from normal sampling to
      REFINE(best) to DIVERGE(under-used lineage) to DIVERGE(low-scoring
      contrast parent), resetting when a meaningful improvement appears.
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
        self.recent_parent_ids: List[str] = []

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score") if program.metrics else None
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    def _meaningful(self, new: float, old: float) -> bool:
        return new > old + 0.01 or (old > 0 and new > old * 1.01)

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        s = self._score(program)
        if s != float("-inf"):
            self.recent_scores.append(s)
            self.recent_scores = self.recent_scores[-20:]

        improved = self.best_score is not None and self._meaningful(s, self.best_score)
        if self.best_score is None or s > self.best_score:
            if self.best_score is None or improved or s > self.best_score:
                if improved or self.best_score is None:
                    self.best_score = s
                    self.best_id = program.id
                    self.iters_since_improvement = 0
                    self.stagnation_counter = 0
        if not improved:
            self.iters_since_improvement += 1
            if self.iters_since_improvement >= 3:
                self.stagnation_counter += 1

        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1
            self.recent_parent_ids.append(program.parent_id)
            self.recent_parent_ids = self.recent_parent_ids[-10:]

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def _usage_penalty(self, pid: str) -> float:
        # Strong penalty for recently used parents, mild for historical reuse.
        recent = 2.0 if pid in self.recent_parent_ids[-5:] else 0.0
        return 1.0 / (1.0 + self.parent_usage.get(pid, 0) + recent)

    def _weighted_parent(self, candidates: List[EvolvedProgram]) -> EvolvedProgram:
        weights = []
        for p in candidates:
            s = self._score(p)
            base = max(s, 0.0) + 0.05  # floor so zero-score programs remain possible
            weights.append(base * self._usage_penalty(p.id))
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
        context: List[EvolvedProgram] = []

        stag = self.stagnation_counter
        if stag >= 4 and self.last_label_mode != "diverge2":
            # Deep stagnation: diverge from an under-used lineage entirely
            pool = sorted(candidates, key=lambda p: self.parent_usage.get(p.id, 0))
            parent = pool[0]
            label = self.DIVERGE_LABEL
            self.last_label_mode = "diverge2"
            return {label: parent}, {"": []}
        elif stag >= 2:
            if self.last_label_mode != "refine" and self.best_id in self.programs:
                # First escalate: refine the best program without distraction
                parent = self.get(self.best_id)
                label = self.REFINE_LABEL
                self.last_label_mode = "refine"
                return {label: parent}, {"": []}
            else:
                # Diverge using a contrasting low/mid scorer as parent
                pool = low if low else mid
                parent = self._weighted_parent(pool if pool else candidates)
                label = self.DIVERGE_LABEL
                self.last_label_mode = "diverge"
        else:
            self.last_label_mode = ""
            # Normal: anti-reuse weighted sampling across tiers
            r = self.rng.random()
            if r < 0.6 and top:
                parent = self._weighted_parent(top)
            elif r < 0.9 and mid:
                parent = self._weighted_parent(mid)
            else:
                parent = self._weighted_parent(candidates)

        # Context: top performers + diverse others, excluding parent
        k = num_context_programs if num_context_programs else 4
        seen = {parent.id}
        top_others = [p for p in top if p.id not in seen]
        self.rng.shuffle(top_others)
        for p in top_others[: max(1, k // 2)]:
            context.append(p)
            seen.add(p.id)
        rest = [p for p in candidates if p.id not in seen]
        self.rng.shuffle(rest)
        for p in rest:
            if len(context) >= k:
                break
            context.append(p)
            seen.add(p.id)

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END