"""Selection policies as validated code, and hot swapping with rollback.

Policy contract (the O1 code surface). The source defines::

    class Policy:
        def __init__(self, labels): ...          # labels: {name: instruction text}
        def observe(self, candidate): ...        # called once per candidate, in population order
        def sample(self, population, rng, num_context):
            return parent, contexts, label_name  # label_name in labels or ''

The population is shared state; a policy may keep its own state (archives, islands,
counters) but must not mutate candidates. ``rng`` is owned by the deployed policy and
seeded at deployment, like SkyDiscover's per-database ``random.Random(seed)``.
"""

from __future__ import annotations

import builtins
import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .state import Candidate, Population


class PolicyContractError(ValueError):
    """A policy failed validation or could not be instantiated/migrated."""


class PolicyRuntimeError(RuntimeError):
    """A deployed policy raised or returned an invalid selection while running."""


@dataclass
class Selection:
    parent: Candidate
    contexts: List[Candidate]
    label: str = ''


# Port of SkyDiscover's initial EvolvedProgramDatabase.sample (scalar branch).
UNIFORM_POLICY_SOURCE = '''class Policy:
    """Uniform random parent; context = a random sample of the population (SkyDiscover/EvoX stock)."""

    def __init__(self, labels):
        self.labels = labels

    def observe(self, candidate):
        pass

    def sample(self, population, rng, num_context):
        candidates = population.members
        if not candidates:
            raise ValueError("No candidates available for sampling")
        parent = rng.choice(candidates)
        sample_size = min((num_context or 0) + 1, len(candidates))
        examples = rng.sample(candidates, sample_size)
        examples = [c for c in examples if c.id != parent.id][:num_context]
        return parent, examples, ""
'''


def _instantiate(source: str, labels: Mapping[str, str], namespace: Optional[Mapping[str, Any]] = None) -> Any:
    scope: Dict[str, Any] = {'__builtins__': builtins, '__name__': 'coevolution_policy', **dict(namespace or {})}
    try:
        exec(compile(source, '<policy>', 'exec'), scope)  # noqa: S102 - policies are code by design (as in EvoX)
    except Exception as error:  # noqa: BLE001
        raise PolicyContractError(f'{type(error).__name__}: {error}') from None
    policy_class = scope.get('Policy')
    if not isinstance(policy_class, type):
        raise PolicyContractError('policy source must define class Policy')
    for method in ('observe', 'sample'):
        if not callable(getattr(policy_class, method, None)):
            raise PolicyContractError(f'Policy must define {method}()')
    try:
        return policy_class(dict(labels))
    except Exception as error:  # noqa: BLE001
        raise PolicyContractError(f'Policy(labels) raised {type(error).__name__}: {error}') from None


def check_selection(result: Any, population: Population, labels: Mapping[str, str]) -> Selection:
    """Validate one sample() return value against the contract."""
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise ValueError('sample() must return (parent, contexts, label)')
    parent, contexts, label = result
    if getattr(parent, 'id', None) not in population or population.get(parent.id) is not parent:
        raise ValueError(f'sampled parent {getattr(parent, "id", parent)!r} is not in the population')
    if not isinstance(contexts, (list, tuple)):
        raise ValueError('contexts must be a list of candidates')
    for context in contexts:
        if getattr(context, 'id', None) not in population:
            raise ValueError(f'context {getattr(context, "id", context)!r} is not in the population')
    if not isinstance(label, str) or (label and label not in labels):
        raise ValueError(f'label {label!r} is not one of {sorted(labels)} or ""')
    return Selection(parent, list(contexts), label)


def _synthetic(size: int, score_key: str, rng: random.Random) -> Population:
    population = Population(score_key)
    for i in range(size):
        metrics = {score_key: rng.random(), 'other': float(i)} if i % 5 != 3 else {score_key: 0.0, 'error': 'boom'}
        population.add(Candidate(f's{size}_{i}', f'def f(): return {i}', metrics, i, parent_id=f's{size}_{i-1}' if i else None))
    return population


def validate_policy_source(source: str, labels: Mapping[str, str], score_key: str = 'combined_score', namespace: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """Return None if the source satisfies the contract, else a readable error (EvoX validator analogue)."""
    try:
        for size in (1, 2, 7):
            policy = _instantiate(source, labels, namespace)
            probe = random.Random(size)
            population = _synthetic(size, score_key, probe)
            frozen = {c.id: copy.deepcopy(c.metrics) for c in population.members}
            for candidate in population.members:
                policy.observe(candidate)
            for _ in range(5):
                check_selection(policy.sample(population, probe, 4), population, labels)
            for candidate in population.members:
                if candidate.metrics != frozen[candidate.id]:
                    return f'policy modified the metrics of candidate {candidate.id}'
    except PolicyContractError as error:
        return str(error)
    except Exception as error:  # noqa: BLE001
        return f'{type(error).__name__}: {error}'
    return None


@dataclass
class _Deployed:
    entry_id: str
    source: str
    policy: Any
    rng: random.Random
    seen: set = field(default_factory=set)


class PolicySlot:
    """Active policy + fallback (+ optional challenger) over one shared population."""

    def __init__(self, labels: Mapping[str, str], seed: Optional[int] = 42, namespace: Optional[Mapping[str, Any]] = None, score_key: str = 'combined_score') -> None:
        self.labels, self.seed, self.namespace, self.score_key = dict(labels), seed, namespace, score_key
        self.active: Optional[_Deployed] = None
        self.fallback: Optional[_Deployed] = None
        self.challenger: Optional[_Deployed] = None
        self.population: Optional[Population] = None
        self.log: List[Dict[str, Any]] = []

    def _make(self, source: str, population: Population, entry_id: str, validate: bool) -> _Deployed:
        if validate:
            error = validate_policy_source(source, self.labels, self.score_key, self.namespace)
            if error:
                raise PolicyContractError(error)
        deployed = _Deployed(entry_id, source, _instantiate(source, self.labels, self.namespace), random.Random(self.seed))
        try:
            for candidate in population.members:  # migration: replay the shared population in order
                deployed.policy.observe(candidate)
                deployed.seen.add(candidate.id)
        except Exception as error:  # noqa: BLE001
            raise PolicyContractError(f'migration failed: {type(error).__name__}: {error}') from None
        self.population = population
        return deployed

    def deploy(self, source: str, population: Population, entry_id: str, validate: bool = True) -> None:
        """Hot-swap: validate, instantiate and migrate; the previous policy becomes the fallback."""
        deployed = self._make(source, population, entry_id, validate)
        self.fallback, self.active = self.active, deployed
        self.log.append({'event': 'deploy', 'entry_id': entry_id})

    def deploy_challenger(self, source: str, population: Population, entry_id: str, validate: bool = True) -> None:
        """Paired deployment: the challenger runs alongside the incumbent until resolve()."""
        self.challenger = self._make(source, population, entry_id, validate)
        self.log.append({'event': 'challenger', 'entry_id': entry_id})

    def resolve(self, promote: bool) -> Optional[str]:
        """End a paired block: promote the challenger or keep the incumbent."""
        challenger, self.challenger = self.challenger, None
        if challenger is not None and promote:
            self.fallback, self.active = self.active, challenger
        self.log.append({'event': 'resolve', 'promoted': bool(challenger and promote)})
        return challenger.entry_id if challenger else None

    def _deployed(self, role: str) -> _Deployed:
        deployed = self.challenger if role == 'challenger' else self.active
        if deployed is None:
            raise PolicyRuntimeError(f'no {role} policy deployed')
        return deployed

    def sample(self, population: Population, num_context: int, role: str = 'active') -> Selection:
        deployed = self._deployed(role)
        try:
            return check_selection(deployed.policy.sample(population, deployed.rng, num_context), population, self.labels)
        except Exception as error:  # noqa: BLE001
            raise PolicyRuntimeError(f'{role} policy {deployed.entry_id} failed in sample(): {type(error).__name__}: {error}') from error

    def observe(self, candidate: Candidate) -> None:
        for role, deployed in (('active', self.active), ('challenger', self.challenger)):
            if deployed is None or candidate.id in deployed.seen:
                continue
            try:
                deployed.policy.observe(candidate)
                deployed.seen.add(candidate.id)
            except Exception as error:  # noqa: BLE001
                raise PolicyRuntimeError(f'{role} policy {deployed.entry_id} failed in observe(): {type(error).__name__}: {error}') from error

    def rollback(self) -> Optional[str]:
        """Restore the fallback policy after a runtime failure; it catches up on new candidates."""
        if self.fallback is None:
            return None
        restored, self.fallback = self.fallback, None
        self.active = restored
        if self.population is not None:
            for candidate in self.population.members:
                if candidate.id not in restored.seen:
                    try:
                        restored.policy.observe(candidate)
                        restored.seen.add(candidate.id)
                    except Exception:  # noqa: BLE001 - mirror SkyDiscover: failed migrations are skipped
                        pass
        self.log.append({'event': 'rollback', 'entry_id': restored.entry_id})
        return restored.entry_id

    @property
    def active_source(self) -> Optional[str]:
        return self.active.source if self.active else None
