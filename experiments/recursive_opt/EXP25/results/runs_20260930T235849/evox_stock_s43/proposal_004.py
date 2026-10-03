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
    """Elite-rotation search database.

    Strategy:
    - With a dense elite band, rotate parents among top scorers so each
      refinement sees a different parent + context combination.
    - Context = top peers (excluding parent) plus one contrasting low/mid
      scorer, chosen to avoid recent repeats.
    - Stagnation: REFINE best (top-peer context), then DIVERGE from an
      under-explored mid-tier lineage.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score: Optional[float] = None
        self.best_id: Optional[str] = None
        self.iters_since_improvement = 0
        self.parent_usage: Dict[str, int] = {}
        self.context_usage: Dict[str, int] = {}
        self.recent_scores: List[float] = []
        self.last_mode = ""  # "", "refine", "diverge"
        self.last_parent_id: Optional[str] = None

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score") if program.metrics else None
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    def _meaningful(self, s: float) -> bool:
        if self.best_score is None:
            return True
        return s > self.best_score + 0.01 or s > self.best_score * 1.01

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        s = self._score(program)
        if s != float("-inf"):
            self.recent_scores.append(s)
            self.recent_scores = self.recent_scores[-30:]

        if self._meaningful(s):
            self.best_score = max(s, self.best_score if self.best_score is not None else s)
            self.best_id = program.id
            self.iters_since_improvement = 0
        else:
            self.iters_since_improvement += 1

        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1
        for cid in program.other_context_ids or []:
            self.context_usage[cid] = self.context_usage.get(cid, 0) + 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

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
        k_elite = max(3, min(6, n // 4))
        elite = candidates[:k_elite]
        mid = candidates[k_elite: max(k_elite + 1, 2 * n // 3)]
        low = candidates[max(k_elite + 1, 2 * n // 3):]
        n_ctx = num_context_programs or 4

        stuck = self.iters_since_improvement >= 3
        label = ""
        parent = None

        if stuck and self.last_mode != "diverge":
            # DIVERGE: under-explored mid-tier lineage (mid-tier diverges
            # previously unlocked new score tiers).
            pool = mid if mid else candidates
            parent = min(pool, key=lambda p: self.parent_usage.get(p.id, 0))
            label = self.DIVERGE_LABEL
            self.last_mode = "diverge"
            return {label: parent}, {"": []}

        if stuck:
            # REFINE best, but with fresh top-peer context.
            parent = self.get(self.best_id) if self.best_id else None
            if parent is None:
                parent = elite[0]
            label = self.REFINE_LABEL
            self.last_mode = "refine"
        else:
            # Rotate among elite, weighted against recently-used parents.
            pool = elite if self.rng.random() < 0.8 else (mid or elite)
            parent = min(pool, key=lambda p: self.parent_usage.get(p.id, 0)
                         + self.rng.random() * 0.5)

        self.last_parent_id = parent.id

        # Context: top peers (excluding parent), preferring least-used,
        # plus one contrasting low/mid scorer.
        context: List[EvolvedProgram] = []
        seen = {parent.id}
        peers = [p for p in elite if p.id not in seen]
        peers.sort(key=lambda p: self.context_usage.get(p.id, 0) + self.rng.random())
        for p in peers[: max(1, n_ctx - 1)]:
            context.append(p)
            seen.add(p.id)
        contrast = (low or mid) or []
        contrast = [p for p in contrast if p.id not in seen]
        if contrast and len(context) < n_ctx:
            context.append(min(contrast, key=lambda p: self.context_usage.get(p.id, 0)
                               + self.rng.random()))
            seen.add(context[-1].id)

        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END