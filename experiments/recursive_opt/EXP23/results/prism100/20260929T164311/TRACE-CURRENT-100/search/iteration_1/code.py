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
    """Initial search strategy database.

    When ``config.pareto_objectives`` is set, sampling prefers members of the
    global Pareto front (issue #42) while still mixing in non-front programs
    for exploration. Scalar mode keeps the original uniform random sample.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
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

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Picks a parent and set of context programs.

        Multiobjective: parent is drawn from the Pareto front when available;
        context mixes front members with the broader population.
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        front: List[EvolvedProgram] = []
        if self.is_multiobjective_enabled():
            front = [p for p in self.get_pareto_front() if p.id in self.programs]

        if front:
            parent = self.rng.choice(front)
            # Prefer other front members for context, then fill from population.
            pool = [p for p in front if p.id != parent.id]
            if len(pool) < (num_context_programs or 0):
                extras = [p for p in candidates if p.id != parent.id and p not in pool]
                self.rng.shuffle(extras)
                pool.extend(extras)
            examples = pool[:num_context_programs]
            if len(examples) < (num_context_programs or 0):
                # Extremely small front — allow duplicates avoidance only.
                examples = [p for p in candidates if p.id != parent.id][:num_context_programs]
        else:
            parent = candidates[_select_index(self, candidates)]
            sample_size = min((num_context_programs or 0) + 1, len(candidates))
            examples = self.rng.sample(candidates, sample_size)
            examples = [p for p in examples if p.id != parent.id][:num_context_programs]

        parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


def select_parent(members, rng):
    """Return the index of the parent to mutate.

    members: list of dicts with keys score, rank (0 = best), rank_pct (1 = best),
    uses (times already used as parent), age (iterations since creation).
    rng: random.Random; use it for every random choice.
    """
    n = len(members)
    if n == 1:
        return 0
    # Exploit the current best sometimes; otherwise rank-weighted sampling.
    if rng.random() < 0.35:
        best = min(range(n), key=lambda i: members[i]["rank"])
        return best
    # Weight proportional to 1/(rank+1), reduced for heavily used members.
    weights = []
    for m in members:
        w = 1.0 / (m["rank"] + 1.0)
        w /= 1.0 + 0.5 * m["uses"]
        weights.append(w)
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return n - 1


def _select_index(db, candidates):
    """Build the members view and delegate to select_parent; fall back to uniform on failure."""
    uses = db.__dict__.setdefault('_parent_uses', {})
    scores = [(p.metrics or {}).get('combined_score') for p in candidates]
    scores = [s if isinstance(s, (int, float)) else float('-inf') for s in scores]
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    rank = {i: r for r, i in enumerate(order)}
    size = max(1, len(candidates) - 1)
    last = max((p.iteration_found or 0) for p in candidates)
    members = [{'score': scores[i], 'rank': rank[i], 'rank_pct': 1 - rank[i] / size, 'uses': uses.get(p.id, 0), 'age': last - (p.iteration_found or 0)} for i, p in enumerate(candidates)]
    try:
        index = select_parent(members, db.rng)
        if not isinstance(index, int) or not 0 <= index < len(candidates):
            raise ValueError('index out of range')
    except Exception:
        index = db.rng.randrange(len(candidates))
    uses[candidates[index].id] = uses.get(candidates[index].id, 0) + 1
    return index

# EVOLVE-BLOCK-END
