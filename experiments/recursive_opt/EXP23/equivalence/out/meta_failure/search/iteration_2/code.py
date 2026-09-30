# EVOLVE-BLOCK-START
from dataclasses import dataclass
from typing import Optional

from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.base_database import Program, ProgramDatabase


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(self, num_context_programs: Optional[int] = 4, **kwargs):
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")
        k = num_context_programs or 0
        rng = self.rng

        def score(p):
            value = p.metrics.get("combined_score") if p.metrics else None
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else float("-inf")

        parent = max(candidates, key=score)
        examples = [p for p in candidates if p.id != parent.id][:k]
        label = self.REFINE_LABEL
        return {label: parent}, {"": examples}
# EVOLVE-BLOCK-END