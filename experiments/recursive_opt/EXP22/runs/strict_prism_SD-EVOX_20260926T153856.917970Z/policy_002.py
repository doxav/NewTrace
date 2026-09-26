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

    Strategy: exploit the elite tier with rotating parents and tiered context;
    when progress stalls, alternate DIVERGE (fresh direction) and REFINE
    (polish the best) instead of repeating the same parent/context pair.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score: float = float("-inf")
        self.stagnation: int = 0          # iterations since meaningful improvement
        self.parent_usage: Dict[str, int] = {}   # program_id -> times used as parent
        self.ctx_usage: Dict[str, int] = {}      # program_id -> times used as context
        self.diverge_count: int = 0
        self.refine_count: int = 0

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score") if program.metrics else None
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        # Track meaningful progress (>1% relative or >0.01 absolute).
        s = self._score(program)
        if s > float("-inf"):
            if s > self.best_score * 1.01 + 0.01 or s > self.best_score + 0.01:
                self.stagnation = 0
                self.best_score = max(self.best_score, s)
            elif s > self.best_score:
                self.best_score = s
            else:
                self.stagnation += 1

        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        ranked = sorted(candidates, key=self._score, reverse=True)
        n = len(ranked)
        elite = ranked[: max(1, n // 4)]          # top quartile
        mid = ranked[max(1, n // 4): max(2, n // 2)]
        weak = ranked[max(2, n // 2):]

        parent_dict: Dict[str, EvolvedProgram]
        context_dict: Dict[str, List[EvolvedProgram]] = {}

        # --- Stalled: alternate DIVERGE / REFINE on the best program ---
        if self.stagnation >= 5:
            best_prog = ranked[0]
            if self.stagnation % 2 == 1:
                # DIVERGE: fresh direction from a strong base, no context bias.
                parent_dict = {self.DIVERGE_LABEL: best_prog}
                self.diverge_count += 1
                return parent_dict, {}
            parent_dict = {self.REFINE_LABEL: best_prog}
            self.refine_count += 1
            # Give the best a couple of elite companions for refinement.
            ctx = [p for p in elite if p.id != best_prog.id][:2]
            context_dict = {"": ctx}
            return parent_dict, context_dict

        # --- Normal phase: rotate elite parents, penalize overuse ---
        def pick(pool: List[EvolvedProgram]) -> EvolvedProgram:
            if not pool:
                return self.rng.choice(ranked)
            weights = [1.0 / (1.0 + self.parent_usage.get(p.id, 0)) for p in pool]
            total = sum(weights)
            r = self.rng.random() * total
            acc = 0.0
            for p, w in zip(pool, weights):
                acc += w
                if r <= acc:
                    return p
            return pool[-1]

        # Mostly exploit elites, occasionally explore mid/weak stock for diversity.
        roll = self.rng.random()
        if roll < 0.70 or not mid:
            parent = pick(elite)
        elif roll < 0.90:
            parent = pick(mid)
        else:
            parent = pick(weak)
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Context: top performers + one diverse (mid/weak) perspective.
        k = num_context_programs or 0
        ctx_pool = [p for p in ranked if p.id != parent.id]
        ctx_pool.sort(key=lambda p: self.ctx_usage.get(p.id, 0))
        chosen: List[EvolvedProgram] = []
        seen = {parent.id}
        # one diverse pick first (from mid+weak) if available
        diverse = [p for p in (mid + weak) if p.id not in seen]
        if diverse and k > 1:
            dp = self.rng.choice(diverse)
            chosen.append(dp)
            seen.add(dp.id)
        for p in ctx_pool:
            if len(chosen) >= k:
                break
            if p.id not in seen:
                chosen.append(p)
                seen.add(p.id)
        for p in chosen:
            self.ctx_usage[p.id] = self.ctx_usage.get(p.id, 0) + 1

        parent_dict = {"": parent}
        context_dict = {"": chosen}
        return parent_dict, context_dict


# EVOLVE-BLOCK-END