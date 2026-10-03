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
    """Search strategy: exploit best programs, diversify context,
    and adaptively use REFINE/DIVERGE labels on stagnation."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score: float = -1e18
        self.best_id: Optional[str] = None
        self.stagnation: int = 0          # iterations since meaningful improvement
        self.since_diverge: int = 999     # iterations since last DIVERGE label
        self.parent_use: Dict[str, int] = {}  # parent_id -> times used
        self.improved_since_diverge: bool = False

    # ---------- helpers ----------

    @staticmethod
    def _score(program: Optional[EvolvedProgram]) -> float:
        if program is None:
            return -1e18
        val = program.metrics.get("combined_score") if program.metrics else None
        if isinstance(val, (int, float)):
            return float(val)
        return -1e18

    def _meaningful(self, new_score: float, ref_score: float) -> bool:
        return (new_score - ref_score) > 0.01 or (
            ref_score > 0 and (new_score - ref_score) / ref_score > 0.01
        )

    # ---------- add ----------

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        score = self._score(program)
        if score > self.best_score:
            if self.best_id is not None and self._meaningful(score, self.best_score):
                self.stagnation = 0
                self.improved_since_diverge = True
            else:
                self.stagnation += 1
            self.best_score = score
            self.best_id = program.id
        else:
            self.stagnation += 1

        self.since_diverge += 1

        # Track parent usage for diversity
        pid = program.parent_id
        if pid:
            self.parent_use[pid] = self.parent_use.get(pid, 0) + 1

        self._update_best_program(program)
        return program.id

    # ---------- sample ----------

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 0
        ranked = sorted(candidates, key=self._score, reverse=True)
        best = self.get(self.best_id) if self.best_id else ranked[0]
        if best is None:
            best = ranked[0]

        label = ""
        context: List[EvolvedProgram] = []

        # --- Parent selection ---
        if self.stagnation >= 4 and self.since_diverge >= 3:
            # Deeply stuck: diverge from the best with a fresh direction,
            # no context to avoid anchoring.
            label = self.DIVERGE_LABEL
            parent = best
            self.since_diverge = 0
            self.stagnation = max(0, self.stagnation - 2)
            return {label: parent}, {}
        elif self.stagnation >= 2:
            # Mildly stuck: refine the best program with top context.
            label = self.REFINE_LABEL
            parent = best
        else:
            # Normal: mostly best, occasionally a diverse underused parent
            # from the top half to keep exploring variants.
            if len(ranked) > 2 and self.rng.random() < 0.3:
                top_half = ranked[: max(2, len(ranked) // 2)]
                # prefer less-used parents
                top_half.sort(key=lambda p: self.parent_use.get(p.id, 0))
                parent = top_half[0]
                if parent.id == best.id and len(top_half) > 1:
                    parent = top_half[1]
            else:
                parent = best

        # --- Context selection ---
        pool = [p for p in ranked if p.id != parent.id]
        # Top scorers as primary context
        top = pool[: max(1, n_ctx - 1)]
        context.extend(top)
        # One low-scorer for contrasting insight (occasionally)
        if len(pool) > n_ctx and self.rng.random() < 0.4:
            low = pool[-min(3, len(pool)):]
            pick = self.rng.choice(low)
            if pick.id not in {c.id for c in context}:
                context.append(pick)
        # Fill if short
        for p in pool:
            if len(context) >= n_ctx:
                break
            if p.id not in {c.id for c in context}:
                context.append(p)

        self.parent_use[parent.id] = self.parent_use.get(parent.id, 0) + 1

        return {label: parent}, {"": context[:n_ctx]}


# EVOLVE-BLOCK-END