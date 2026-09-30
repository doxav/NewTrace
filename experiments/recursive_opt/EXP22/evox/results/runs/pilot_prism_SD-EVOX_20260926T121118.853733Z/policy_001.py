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
    """Simple adaptive search database.

    Strategy: with a tiny population and short window, exploit the best
    program as the parent (refinement is the highest-value move), while
    giving the LLM the best and a diverse low-scoring program as context
    for contrast. Track stagnation in add(); if no meaningful improvement
    for a few iterations, apply REFINE_LABEL to the best program to focus
    the LLM on polishing the leader.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score = None
        self.best_id = None
        self.stagnation = 0

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

        # Track meaningful improvement (>1% relative or >0.01 absolute).
        s = self._score(program)
        if self.best_score is not None and s > self.best_score:
            meaningful = (s - self.best_score) > 0.01 or (
                self.best_score > 0 and (s - self.best_score) / self.best_score > 0.01
            )
            if meaningful:
                self.stagnation = 0
            else:
                self.stagnation += 1
        if self.best_score is None or s > self.best_score:
            self.best_score = s
            self.best_id = program.id

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = [p for p in self.programs.values() if isinstance(self._score(p), float)]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        # Parent: the best program (exploit leader in tiny population).
        parent = max(candidates, key=self._score)

        # Context: best + a couple of diverse/other programs for contrast.
        others = sorted(
            [p for p in candidates if p.id != parent.id], key=self._score, reverse=True
        )
        examples = []
        if others:
            examples.append(others[0])  # second-best
            if len(others) > 1:
                examples.append(others[-1])  # worst, for contrast
            # fill remaining slots randomly
            rest = [p for p in others if p not in examples]
            if rest:
                self.rng.shuffle(rest)
                examples.extend(rest)
        examples = examples[: num_context_programs or 0]

        label = ""
        if self.stagnation >= 2:
            label = self.REFINE_LABEL
            self.stagnation = 0

        return {label: parent}, {"": examples}


# EVOLVE-BLOCK-END