"""Online co-evolution engine: an O0 solution loop and an O1 policy loop sharing one population.

Per iteration: the deployed policy selects (parent, contexts, label); the O0 operator
generates, evaluates and retries; the child joins the shared population and every
deployed policy observes it. When the trigger fires, the pending policy's window is
closed and scored (deferred evaluation), the archive selects a parent policy, the meta
proposer writes a new one, and it is hot-swapped in (``replace``) or run alongside the
incumbent (``paired``). A policy that fails at runtime is rolled back.

``evox_preset()`` reproduces SkyDiscover EvoX (see README "Co-evolution"). One deliberate
deviation: an LLM transport failure is not treated as a policy failure (EvoX restores the
previous policy when *any* pre-generation error occurs after a switch).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from .feedback import META_SYSTEM, FeedbackComposer
from .operator import DEFAULT_LABELS, PopulationOperator, generate_labels
from .policy import UNIFORM_POLICY_SOURCE, PolicyContractError, PolicyRuntimeError, PolicySlot, validate_policy_source
from .proposers import LLMRewriteProposer, TraceProposer
from .scheduling import DeferredEvaluation, GainScorer, LogWindowScorer, PairedScorer, StrategyArchive, interleave, make_trigger, resolve_patience
from .state import Candidate, Population

LLMText = Callable[[str, str], str]
Evaluate = Callable[[str], Tuple[Dict[str, Any], Dict[str, Any]]]

_CHOICES = {
    'trigger': ('stagnation', 'periodic', 'never'),
    'operator_mode': ('diff', 'rewrite'),
    'proposer': ('llm_rewrite', 'trace'),
    'meta_parent': ('best', 'current'),
    'window_scorer': ('log_window', 'gain', 'paired'),
    'deployment': ('replace', 'paired'),
}


@dataclass
class CoevolutionConfig:
    horizon: int = 100
    trigger: str = 'stagnation'
    patience: Union[int, str] = 'auto'
    patience_ratio: float = 0.10
    improvement_threshold: float = 0.01
    operator_mode: str = 'diff'
    retries: int = 3
    num_context: int = 4
    num_previous_attempts: int = 3
    max_solution_chars: int = 60000
    language: str = 'python'
    score_key: str = 'combined_score'
    seed: Optional[int] = 42
    labels: Optional[Dict[str, str]] = None
    generate_labels: bool = True
    label_packages: Optional[Tuple[str, ...]] = None  # None: native label prompt; tuple: stock EvoX package-aware prompt
    initial_policy: str = UNIFORM_POLICY_SOURCE
    proposer: str = 'llm_rewrite'
    meta_retries: int = 3
    meta_feed_errors: bool = False
    meta_parent: str = 'best'
    meta_num_context: int = 2
    archive_seed: Optional[int] = None
    proposer_memory: int = 5
    window_scorer: str = 'log_window'
    deployment: str = 'replace'
    rollback: bool = True
    summaries: bool = True
    strict_budget: bool = False
    meta_system_prompt: Optional[str] = None
    evaluator_timeout_s: Optional[float] = None

    def __post_init__(self) -> None:
        for name, allowed in _CHOICES.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f'{name} must be one of {allowed}, got {getattr(self, name)!r}')
        if not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError('horizon must be a positive integer')
        for name in ('retries', 'meta_retries', 'num_context', 'meta_num_context'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < (1 if 'retries' in name else 0):
                raise ValueError(f'{name} is out of range')
        if (self.deployment == 'paired') != (self.window_scorer == 'paired'):
            raise ValueError("deployment='paired' requires window_scorer='paired' and vice versa")
        resolve_patience(self.patience, self.horizon, self.patience_ratio)


# Fields added after plans were fingerprinted. Omitted from presets while unset, so existing specs keep their
# plan fingerprint (the control plane hashes the full engine config). EXP25's fingerprint pins this.
_UNSET_OMITTED = {'label_packages': None}


def evox_preset(horizon: int = 100, summaries: bool = True, generate_labels: bool = True, retries: int = 3, operator_mode: str = 'diff') -> Dict[str, Any]:
    """Configuration reproducing SkyDiscover EvoX's co-evolution controller."""
    return _omit_unset(CoevolutionConfig(horizon=horizon, trigger='stagnation', patience='auto', patience_ratio=0.10, improvement_threshold=0.01,
                                    operator_mode=operator_mode, retries=retries, num_context=4, num_previous_attempts=3, max_solution_chars=60000,
                                    seed=42, generate_labels=generate_labels, proposer='llm_rewrite', meta_retries=3, meta_feed_errors=False,
                                    meta_parent='best', meta_num_context=2, window_scorer='log_window', deployment='replace', rollback=True,
                                    summaries=summaries))


def _omit_unset(config: CoevolutionConfig) -> Dict[str, Any]:
    return {k: v for k, v in asdict(config).items() if not (k in _UNSET_OMITTED and v == _UNSET_OMITTED[k])}


class _Counted:
    def __init__(self, llm: Optional[LLMText], counter: Dict[str, int], role: str) -> None:
        self.llm, self.counter, self.role = llm, counter, role

    def __call__(self, system: str, user: str) -> str:
        self.counter[self.role] += 1
        return self.llm(system, user)


class CoevolutionEngine:
    def __init__(self, config: CoevolutionConfig, solution_llm: LLMText, meta_llm: LLMText, evaluate: Evaluate, initial_source: str,
                 system_message: str, feedback_llm: Optional[LLMText] = None, problem_description: Optional[str] = None,
                 evaluator_context: str = '', population: Optional[Population] = None, policy_namespace: Optional[Mapping[str, Any]] = None,
                 on_event: Optional[Callable[[Dict[str, Any]], None]] = None, projections: Optional[List[Callable[[str], Tuple[str, str]]]] = None) -> None:
        self.config = config
        self.llm_calls = {'solution': 0, 'meta': 0, 'feedback': 0}
        self.solution_llm = _Counted(solution_llm, self.llm_calls, 'solution')
        self.meta_llm = _Counted(meta_llm, self.llm_calls, 'meta')
        self.feedback_llm = _Counted(feedback_llm, self.llm_calls, 'feedback') if feedback_llm else None
        self.evaluate, self.initial_source, self.system_message = evaluate, initial_source, system_message
        self.problem_description = problem_description if problem_description is not None else system_message
        self.evaluator_context, self.namespace, self.on_event = evaluator_context, policy_namespace, on_event
        self.population = population if population is not None else Population(config.score_key)
        self.events: List[Dict[str, Any]] = []
        self.proposer = None
        self.projections = list(projections or [])

    # ---- helpers
    def _event(self, **event: Any) -> None:
        self.events.append(event)
        if self.on_event:
            self.on_event(event)

    @property
    def meta_prompts(self) -> List[str]:
        return [user for _, user in getattr(self.proposer, 'prompts', [])]

    def _best(self) -> float:
        return self.population.best_score()

    # ---- run
    def run(self) -> Dict[str, Any]:
        cfg, population = self.config, self.population
        patience = resolve_patience(cfg.patience, cfg.horizon, cfg.patience_ratio)
        operator_projections = PopulationOperator(None, None, '', projections=self.projections)
        if not len(population):
            deployable, notes = operator_projections.project(self.initial_source)
            metrics, artifacts = self.evaluate(deployable)
            artifacts = dict(artifacts or {})
            if notes:
                artifacts['projection'] = '; '.join(notes)
            population.add(Candidate('initial', self.initial_source, dict(metrics), 0, artifacts=artifacts,
                                     metadata={'deployable': deployable} if deployable != self.initial_source else {}))
        if cfg.labels is not None:
            labels = dict(cfg.labels)
        elif cfg.generate_labels and self.feedback_llm is not None:
            # Stock EvoX's controller passes no initial program to its label generator; neither do we.
            labels = generate_labels(self.feedback_llm, self.system_message, self.evaluator_context, packages=cfg.label_packages)
        else:
            labels = dict(DEFAULT_LABELS)
        self.labels = labels
        slot = PolicySlot(labels, cfg.seed, self.namespace, cfg.score_key)
        slot.deploy(cfg.initial_policy, population, entry_id='initial')
        archive = StrategyArchive(cfg.archive_seed)
        scorer = GainScorer(patience) if cfg.window_scorer == 'gain' else LogWindowScorer(patience)
        deferred = DeferredEvaluation(scorer)
        trigger = make_trigger(cfg.trigger, patience, cfg.improvement_threshold)
        operator = PopulationOperator(self.solution_llm, self.evaluate, self.system_message, cfg.operator_mode, cfg.retries, labels,
                                      cfg.num_previous_attempts, cfg.max_solution_chars, cfg.language, cfg.score_key, cfg.evaluator_timeout_s,
                                      self.projections)
        composer = FeedbackComposer(self.feedback_llm if cfg.summaries else None, cfg.meta_system_prompt or META_SYSTEM, self.problem_description, self.evaluator_context)
        self.composer = composer
        if cfg.proposer == 'trace':
            self.proposer = TraceProposer(self.meta_llm, cfg.proposer_memory, cfg.meta_retries, initial_source=cfg.initial_policy)
        else:
            self.proposer = LLMRewriteProposer(self.meta_llm, cfg.meta_retries, cfg.meta_feed_errors, cfg.language)
        validate = lambda source: validate_policy_source(source, labels, cfg.score_key, self.namespace)  # noqa: E731
        paired, paired_scorer = cfg.deployment == 'paired', PairedScorer()
        state: Dict[str, Any] = {'active': None, 'challenger': None, 'tags': [], 'records': [], 'meta_counter': 0, 'meta_failures': 0}
        start_stats = population.statistics(improvement_threshold=cfg.improvement_threshold)
        deferred.reset(self._best(), None)
        curve: List[float] = []
        attempts_total = 0

        def resolve_paired(completed: int, stats_now: Dict[str, Any]) -> None:
            entry = state['challenger']
            if entry is None:
                return
            score = paired_scorer.score_records(state['records'])
            promoted = score > 0
            slot.resolve(promoted)
            entry.metrics.update({'combined_score': score, 'window_start_iteration': entry.iteration, 'paired_decisions': len(state['records'])})
            entry.status = 'scored'
            entry.metadata['end_stats'] = stats_now
            archive.add(entry)
            if promoted:
                state['active'] = entry
            self._event(type='paired_resolution', iteration=completed, entry=entry.id, score=score, promoted=promoted)
            state['challenger'], state['records'], state['tags'] = None, [], []

        def generate(completed: int, stats_now: Dict[str, Any]) -> None:
            window = {'window_start_iteration': completed, 'total_iterations': cfg.horizon, 'horizon': patience, 'improvement_threshold': cfg.improvement_threshold}
            parent, context = archive.select(cfg.meta_num_context, cfg.meta_parent, state['active'])
            previous = max(archive.entries.values(), key=lambda e: e.iteration)
            if cfg.proposer == 'trace':
                feedback = composer.compose(parent, context, previous, window, stats_now)[1]
                result = self.proposer.propose(parent_source=parent.source, validate=validate, feedback=feedback, context=[c.source for c in context])
            else:
                result = self.proposer.propose(build_prompt=lambda errors: composer.compose(parent, context, previous, window, stats_now, errors), validate=validate)
            self._event(type='proposal', iteration=completed, parent=parent.id, context=[c.id for c in context], attempts=result.attempts, ok=result.ok, errors=result.errors)
            if not result.ok:
                state['meta_failures'] += 1
                return
            state['meta_counter'] += 1
            entry = archive.new_entry(result.source, state['meta_counter'], parent_id=parent.id)
            entry.metadata['start_stats'] = stats_now
            try:
                if paired:
                    slot.deploy_challenger(result.source, population, entry.id, validate=False)
                else:
                    slot.deploy(result.source, population, entry.id, validate=False)
            except PolicyContractError as error:
                state['meta_failures'] += 1
                self._event(type='deploy', iteration=completed, entry=entry.id, ok=False, error=str(error))
                return
            self._event(type='deploy', iteration=completed, entry=entry.id, ok=True)
            if paired:
                state['challenger'], state['records'] = entry, []
                state['tags'] = interleave(cfg.horizon + cfg.retries, (cfg.seed, completed))
            else:
                deferred.attach(entry)
                state['active'] = entry
                deferred.reset(self._best(), start_iteration=completed)

        def evolve(completed: int) -> None:
            best, stats_now = self._best(), population.statistics(improvement_threshold=cfg.improvement_threshold)
            self._event(type='trigger', iteration=completed, best=best)
            if not len(archive):
                metrics = deferred.window_metrics(best, start_iteration=0)
                if paired:
                    metrics['combined_score'] = 0.0
                entry = archive.new_entry(cfg.initial_policy, 0, source_kind='initial')
                entry.metrics, entry.status = metrics, 'scored'
                entry.metadata.update(start_stats=start_stats, end_stats=stats_now)
                archive.add(entry)
                state['active'] = entry
                self._event(type='scored', iteration=completed, entry=entry.id, metrics=dict(metrics))
                deferred.reset(best, None)
                generate(completed, stats_now)
                return
            if paired:
                resolve_paired(completed, stats_now)
            elif deferred.pending is not None:
                entry = deferred.close(best)
                entry.metadata['end_stats'] = stats_now
                archive.add(entry)
                self._event(type='scored', iteration=completed, entry=entry.id, metrics=dict(entry.metrics))
            deferred.reset(best, None)
            generate(completed, stats_now)

        iteration, total = 1, 1 + cfg.horizon
        while iteration < total:
            role = 'active'
            if paired and state['challenger'] is not None:
                position = len(state['records'])
                role = 'challenger' if state['tags'][position % len(state['tags'])] == 'challenger' else 'active'
            try:
                selection = slot.sample(population, cfg.num_context, role)
            except PolicyRuntimeError as error:
                if cfg.rollback and role == 'challenger':
                    entry = state['challenger']
                    slot.resolve(False)
                    state['challenger'], state['records'] = None, []
                    self._event(type='rollback', iteration=iteration, entry=entry.id if entry else None, error=str(error), restored=slot.active.entry_id)
                    continue
                if cfg.rollback and slot.fallback is not None:
                    restored = slot.rollback()
                    deferred.pending = None
                    state['active'] = archive.entries.get(restored, state['active'])
                    self._event(type='rollback', iteration=iteration, error=str(error), restored=restored)
                    continue
                self._event(type='iteration', iteration=iteration, attempts=1, error=str(error), policy=slot.active.entry_id)
                deferred.record(self._best())
                curve.append(self._best())
                attempts_total += 1
                completed, iteration = iteration, iteration + 1
                if iteration < total and trigger.update(self._best()):
                    evolve(completed)
                continue
            best_before = self._best()
            retries = min(cfg.retries, total - iteration) if cfg.strict_budget else cfg.retries
            result = operator.run(selection, population, iteration, retries)
            child = result.candidate
            if child is not None:
                population.add(child)
                try:
                    slot.observe(child)
                except PolicyRuntimeError as error:
                    population.remove(child.id)
                    if cfg.rollback and slot.fallback is not None:
                        restored = slot.rollback()
                        deferred.pending = None
                        state['active'] = archive.entries.get(restored, state['active'])
                        self._event(type='rollback', iteration=iteration, error=str(error), restored=restored)
                        continue
                    self._event(type='iteration', iteration=iteration, attempts=result.attempts_used, error=str(error), policy=slot.active.entry_id)
                    iteration += 1  # SkyDiscover: database.add failure advances without recording the window
                    continue
            if paired and state['challenger'] is not None:
                score = population.score(child) if child is not None else None
                state['records'].append({'tag': 'challenger' if role == 'challenger' else 'incumbent', 'new_best': score is not None and score > best_before})
            self._event(type='iteration', iteration=iteration, attempts=result.attempts_used, error=result.error, role=role,
                        policy=(slot.challenger if role == 'challenger' else slot.active).entry_id,
                        parent_iteration=selection.parent.iteration, label=selection.label, context_iterations=[c.iteration for c in selection.contexts],
                        child_score=population.score(child) if child is not None else None)
            for _ in range(result.attempts_used):
                deferred.record(self._best())
                curve.append(self._best())
            attempts_total += result.attempts_used
            completed, iteration = iteration, iteration + result.attempts_used
            if iteration < total and trigger.update(self._best()):
                evolve(completed)
        final_stats = population.statistics(improvement_threshold=cfg.improvement_threshold)
        if paired:
            resolve_paired(iteration - 1, final_stats)
        elif deferred.pending is not None:
            entry = deferred.close(self._best())
            entry.metadata['end_stats'] = final_stats
            archive.add(entry)
            self._event(type='scored', iteration=iteration - 1, entry=entry.id, metrics=dict(entry.metrics))
        best = population.best()
        self.archive, self.slot = archive, slot
        return {'best_score': self._best(), 'best_source': best.metadata.get('deployable', best.content) if best else None,
                'best_editable_source': best.content if best else None, 'best_metrics': dict(best.metrics) if best else {},
                'solution_attempts': attempts_total, 'curve': curve, 'events': self.events, 'archive': [e.to_dict() for e in archive.entries.values()],
                'active_policy': slot.active_source, 'meta_failures': state['meta_failures'], 'llm_calls': dict(self.llm_calls),
                'feedback_calls': dict(composer.calls), 'labels': labels, 'population_size': len(population)}
