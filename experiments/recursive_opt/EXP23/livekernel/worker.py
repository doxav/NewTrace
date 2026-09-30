"""Live SkyDiscover workers for A4 (paired co-evolution), A5 (amortized) and A6 (task level).

Run as a script: EXP22's modules (stock-EvoX audit controller, process evaluator, Novita-pinned
transport) are imported from the EXP22 root, so this file must not import EXP23's ``src``.
Schedule: (1 + 2) x 4 = 12 solution iterations, policy switches forced after iterations 4 and 8.

Arms
  A4/SD-FIXED        stock EvoX kernel, stock policy for all 12 iterations
  A4/SD-EVOX         stock EvoX meta proposer (full EvolvedProgramDatabase source), LogWindowScorer
  A4/SD-EVOX-PAIRED  stock EvoX meta proposer; each block interleaves challenger and incumbent
                     (2 + 2 iterations); archive score = paired new-best rate; promote if > 0
  A4/TRACE-PAIRED    persistent OptoPrimeV2 on a compact select_parent(members, rng) surface injected
                     into the stock strategy; evidence = the decisions recorded from the executed block
  A5/AMORTIZED       PrioritySearch (3 steps = 2 proposals) on select_parent; each evaluation is a
                     live 4-iteration EvoX episode executed inside a Trace bundle
  A6/TRACE-TASK      PrioritySearch + OptoPrimeV2 directly on the solution program (12 proposals), no meta
"""

import argparse
import asyncio
import copy
import importlib.util
import json
import math
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

EXP23 = Path(__file__).resolve().parents[1]
EXP22 = EXP23.parent / 'EXP22'
sys.path.insert(0, str(EXP22))

from opto import trace
from opto.optimizers.optoprime_v2 import OptoPrimeV2
from opto.utils.llm import DummyLLM
from skydiscover.optimize.config import LLMModelConfig
from skydiscover.optimize.search.registry import (
    create_database,
    get_program,
)
from skydiscover.optimize.search.utils.discovery_utils import (
    SerializableResult,
)
from skydiscover.optimize.utils.metrics import get_score
from src.evaluation import SKY, TASKS, TraceEvaluatorAdapter

from src import kernel as K
from src import transport as T

_spec = importlib.util.spec_from_file_location('exp23_live', EXP23 / 'src' / 'live.py')
LIVE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LIVE)

CALLS: list[dict[str, Any]] = []
SWITCH_AT = (4, 8)
BLOCK = 4


def model_config() -> LLMModelConfig:
    """EXP22 GLM/Novita identity with bounded SDK retries (EXP22 stopped on single 429s)."""
    return LLMModelConfig(name=T.MODEL, api_base=T.BASE_URL, temperature=0.7, max_tokens=32000, timeout=600, retries=3, retry_delay=10, init_client=T.SkyOpenRouter)


K.model_config = model_config
# EXP22's usage summary needs HTTP_RECORDS from EXP22's own runner; calls are recorded in CALLS instead.
K.usage_summary = lambda records, solution_limit=None: {'total_calls': len(records)}
_original_call = T.SkyOpenRouter._call_api


async def _recorded_call(self, params: dict[str, Any]) -> str:
    """Record every SkyDiscover request (role, latency, outcome) without content."""
    started, record = time.time(), {'role': T.HTTP_ROLES.get(id(self.client._client), 'unknown')}
    try:
        text = await _original_call(self, params)
        record.update(ok=True, text_chars=len(text or ''))
        return text
    except Exception as error:
        record.update(ok=False, error_type=type(error).__name__)
        raise
    finally:
        record['latency_s'] = round(time.time() - started, 2)
        CALLS.append(record)


T.SkyOpenRouter._call_api = _recorded_call

# ------------------------------------------------------------------ compact select_parent surface

SELECT_STOCK = '''def select_parent(members, rng):
    """Return the index of the parent to mutate.

    members: list of dicts with keys score, rank (0 = best), rank_pct (1 = best),
    uses (times already used as parent), age (iterations since creation).
    rng: random.Random; use it for every random choice.
    """
    return rng.randrange(len(members))
'''
HOOK_OLD = '''            parent = self.rng.choice(candidates)
            sample_size'''
HOOK_NEW = '''            parent = candidates[_select_index(self, candidates)]
            sample_size'''
HELPER = '''

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
'''


def strategy_source(select_code: str) -> str:
    """Stock EvolvedProgramDatabase with its scalar parent choice delegated to select_parent."""
    stock = K.POLICY.read_text()
    if HOOK_OLD not in stock:
        raise RuntimeError('stock strategy changed; select_parent hook not found')
    body = stock.replace(HOOK_OLD, HOOK_NEW)
    return body.replace('# EVOLVE-BLOCK-END', select_code.rstrip() + '\n' + HELPER + '\n# EVOLVE-BLOCK-END')


def extract_select(text: str) -> str:
    """Recover the select_parent function from an optimizer answer."""
    text = LIVE.extract_policy(text) if '```' in text else text
    return text if 'def select_parent' in text else SELECT_STOCK


# ------------------------------------------------------------------ paired block database

class PairedDB:
    """Challenger and incumbent strategies share every new program; parent choice alternates by block tag."""

    OWN = frozenset({'incumbent', 'challenger', 'tags', 'k', 'records', 'pending'})

    def __init__(self, incumbent: Any, challenger: Any, tags: list[str]) -> None:
        self.incumbent, self.challenger, self.tags, self.k = incumbent, challenger, tags, 0
        self.records: list[dict[str, Any]] = []
        self.pending: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__['challenger'], name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.OWN:
            object.__setattr__(self, name, value)
            return
        for db in {id(self.incumbent): self.incumbent, id(self.challenger): self.challenger}.values():  # e.g. EvoX labels
            setattr(db, name, value)

    def _best(self) -> float:
        best = self.challenger.get_best_program()
        return get_score(best.metrics) if best else float('-inf')

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        if self.pending is not None:  # previous child never added: invalid generation
            self.records.append({**self.pending, 'outcome': 'invalid'})
        tag = self.tags[min(self.k, len(self.tags) - 1)]
        self.k += 1
        db = self.challenger if tag == 'challenger' else self.incumbent
        parent_dict, context = db.sample(*args, **kwargs)
        parent = next(iter(parent_dict.values())) if isinstance(parent_dict, dict) else parent_dict
        scores = sorted((get_score(p.metrics) for p in db.programs.values()), reverse=True)
        self.pending = {'tag': tag, 'parent_rank': scores.index(get_score(parent.metrics)), 'parent_score': get_score(parent.metrics), 'best_before': self._best()}
        return parent_dict, context

    def add(self, program: Any, iteration: int | None = None, **kwargs: Any) -> str:
        pending, self.pending = self.pending, None
        self.incumbent.add(copy.deepcopy(program), iteration=iteration, **kwargs)
        result = self.challenger.add(program, iteration=iteration, **kwargs)
        if pending is not None:
            score = get_score(program.metrics)
            outcome = 'new_best' if score > pending['best_before'] else 'beat_parent' if score > pending['parent_score'] else 'worse'
            self.records.append({**pending, 'child_score': score, 'outcome': outcome})
        return result

    def paired_score(self) -> float:
        if self.pending is not None:
            self.records.append({**self.pending, 'outcome': 'invalid'})
            self.pending = None
        rate = {t: [r['outcome'] == 'new_best' for r in self.records if r['tag'] == t] for t in ('challenger', 'incumbent')}
        if not rate['challenger'] or not rate['incumbent']:
            return 0.0
        return sum(rate['challenger']) / len(rate['challenger']) - sum(rate['incumbent']) / len(rate['incumbent'])


class RecordingDB(PairedDB):
    """Single active strategy; records each parent decision and outcome as traced evidence."""

    def __init__(self, db: Any) -> None:
        super().__init__(db, db, ['active'])

    def add(self, program: Any, iteration: int | None = None, **kwargs: Any) -> str:
        pending, self.pending = self.pending, None
        result = self.challenger.add(program, iteration=iteration, **kwargs)
        if pending is not None:
            score = get_score(program.metrics)
            outcome = 'new_best' if score > pending['best_before'] else 'beat_parent' if score > pending['parent_score'] else 'worse'
            self.records.append({**pending, 'child_score': score, 'outcome': outcome})
        return result


class Scheduled:
    """Force policy updates after iterations 4 and 8 (FORCED), else the stock EvoX stagnation trigger."""

    FORCED = True

    def _should_evolve_search(self) -> bool:
        if not self.FORCED:
            return super()._should_evolve_search()
        fired = self.__dict__.setdefault('_fired', set())
        n = len(self.curve)
        if n in SWITCH_AT and n not in fired:
            fired.add(n)
            return True
        return False


class Paired(Scheduled):
    """Deploy each proposal as a paired challenger; score the archive with the paired rate."""

    def _init_paired(self) -> None:
        self.__dict__.setdefault('paired_log', [])
        self.__dict__.setdefault('incumbent_code', self._active_search_algorithm_code)

    def _compute_search_metrics(self, start_score=None, best_scores=None, horizon=None, start_iteration=None):
        metrics = super()._compute_search_metrics(start_score, best_scores, horizon, start_iteration)
        if start_iteration == 0:  # stock strategy on the paired scale: itself vs itself
            metrics['combined_score'] = 0.0
        return metrics

    def _resolve_block(self) -> float | None:
        self._init_paired()
        if not isinstance(self.database, PairedDB):
            return None
        paired, score = self.database, self.database.paired_score()
        promoted = score > 0
        self.database = paired.challenger if promoted else paired.incumbent
        if promoted:
            self.incumbent_code = self._active_search_algorithm_code
        else:
            self._active_search_algorithm_code = self.incumbent_code
        self.paired_log.append({'iteration': len(self.curve), 'paired_score': score, 'promoted': promoted, 'records': paired.records})
        return score

    def _assign_search_score(self) -> bool:
        score = self._resolve_block()
        if not self._pending_search_result or score is None:
            return super()._assign_search_score()
        child = self._pending_search_result.child_program_dict or {}
        child.setdefault('metrics', {})['combined_score'] = score
        self._pending_search_result.child_program_dict = child
        new_best = self._best_search_score is not None and score > self._best_search_score
        if new_best or self._best_search_score is None:
            self._best_search_score = score
        return new_best

    def _switch_to_new_search_algorithm(self, result: SerializableResult) -> bool:
        self._init_paired()
        incumbent = self.database
        activated = super()._switch_to_new_search_algorithm(result)
        if activated:
            import random
            length = BLOCK if self.FORCED else max(BLOCK, self.total_solution_iterations)  # stagnation blocks can span the run
            tags = ['challenger', 'incumbent'] * (length // 2)
            random.Random(f'{len(self.curve)}').shuffle(tags)
            self.database = PairedDB(incumbent, self.database, tags)
        return activated


class EvoxScheduled(Scheduled, K.AuditController):
    """Stock EvoX meta proposer on the forced schedule."""


class EvoxPaired(Paired, K.AuditController):
    """Stock EvoX meta proposer with paired deployment and archive score."""


@trace.bundle()
def executed_block(selection_policy, decisions):
    """Decisions made by selection_policy (as challenger) and by the incumbent during the last live block.

    One line per solution iteration: 'role parent_rank=<0 is best> outcome'. Outcomes: new_best,
    beat_parent, worse, invalid (generation or evaluation failed).
    """
    return decisions


class TracePaired(Paired, K.AuditController):
    """Persistent OptoPrimeV2 proposes select_parent code; evidence is the executed paired block."""

    def __init__(self, controller_input: Any, llm: Any) -> None:
        super().__init__(controller_input)
        self.node = trace.node(SELECT_STOCK, trainable=True, name='select_parent_code', description=LIVE.SURFACE_HELP['code'])
        self.optimizer = OptoPrimeV2([self.node], llm=DummyLLM(llm), memory_size=5, log=False, max_tokens=8000)
        self.incumbent_select = SELECT_STOCK
        self._policy_proposer = self._trace_proposal
        self.trace_log: list[dict[str, Any]] = []

    async def _trace_proposal(self, start_iteration: int, max_iterations: int, post_process_result: bool = True) -> SerializableResult:
        last = self.paired_log[-1] if getattr(self, 'paired_log', None) else None
        if last is not None and last['promoted']:
            self.incumbent_select = self.trace_log[-1]['proposed']
        if last is None:
            decisions = [f"stock parent_rank=? {'new_best' if c['best_score'] > p['best_score'] + 1e-12 else 'no_new_best'}" for p, c in zip(self.curve, self.curve[1:])]
            feedback = f'Stock policy alone for {len(self.curve)} iterations; best score {self._get_best_score():.4f}. Propose a select_parent that finds new global bests faster than uniform random parent choice.'
        else:
            decisions = [f"{r['tag']} parent_rank={r['parent_rank']} {r['outcome']}" for r in last['records']]
            feedback = (f"Paired block: challenger minus incumbent new-best rate = {last['paired_score']:+.3f} "
                        f"({'challenger promoted' if last['promoted'] else 'incumbent kept'}). Best score {self._get_best_score():.4f}.\n"
                        f"Incumbent select_parent:\n{self.incumbent_select}")
        self.node._data = self.trace_log[-1]['proposed'] if self.trace_log else SELECT_STOCK
        output = executed_block(self.node, decisions)
        self.node._data = self.incumbent_select
        self.optimizer.zero_feedback()
        self.optimizer.backward(output, feedback)
        self.optimizer.step()
        proposed = extract_select(str(self.node.data))
        source = strategy_source(proposed)
        self.trace_log.append({'iteration': len(self.curve), 'proposed': proposed, 'feedback': feedback, 'decisions': decisions})
        validation = await self.search_controller.evaluator.evaluate_program(source)
        candidate = get_program(self.search_controller.config, source, str(uuid.uuid4()), validation.metrics, start_iteration)
        if validation.metrics.get('validity') != 1:
            return SerializableResult(iteration=start_iteration, child_program_dict=candidate.to_dict(), error='Stock policy validator rejected Trace proposal')
        return SerializableResult(iteration=start_iteration, child_program_dict=candidate.to_dict())


class TracePaired100(TracePaired):
    """TRACE-PAIRED on the stock stagnation trigger (100-iteration runs)."""

    FORCED = False


class TraceCurrent(K.AuditController):
    """Persistent OptoPrimeV2 on select_parent; continue from the active policy, deploy every valid proposal.

    Stock stagnation trigger. Evidence = the decisions the active policy actually made in its window.
    """

    def __init__(self, controller_input: Any, llm: Any) -> None:
        super().__init__(controller_input)
        self.database = RecordingDB(self.database)
        self.node = trace.node(SELECT_STOCK, trainable=True, name='select_parent_code', description=LIVE.SURFACE_HELP['code'])
        self.optimizer = OptoPrimeV2([self.node], llm=DummyLLM(llm), memory_size=5, log=False, max_tokens=8000)
        self.active_select = SELECT_STOCK
        self._policy_proposer = self._trace_proposal
        self.trace_log: list[dict[str, Any]] = []

    def _switch_to_new_search_algorithm(self, result: SerializableResult) -> bool:
        activated = super()._switch_to_new_search_algorithm(result)
        if activated:
            self.active_select = self.trace_log[-1]['proposed']
            self.database = RecordingDB(self.database)
        return activated

    async def _trace_proposal(self, start_iteration: int, max_iterations: int, post_process_result: bool = True) -> SerializableResult:
        db = self.database if isinstance(self.database, RecordingDB) else None
        records = list(db.records) if db else []
        if db:
            db.records = []
        window = self.window_observation.get('window_metrics', {})
        counts = {o: sum(r['outcome'] == o for r in records) for o in ('new_best', 'beat_parent', 'worse')}
        decisions = [f"active parent_rank={r['parent_rank']} {r['outcome']}" for r in records][-40:]
        feedback = (f"Window of the active select_parent: {len(records)} decisions, outcomes {counts}. Best score "
                    f"{window.get('search_window_start_score', float('nan')):.4f} -> {window.get('search_window_end_score', float('nan')):.4f} "
                    f"(EvoX window score {window.get('combined_score', 0.0):.4f}); this update was triggered by 10 iterations without a gain > 0.01. "
                    f"Overall best {self._get_best_score():.4f} after {len(self.curve)} iterations. Improve select_parent to find new global bests.")
        self.node._data = self.active_select
        output = executed_block(self.node, decisions)
        self.optimizer.zero_feedback()
        self.optimizer.backward(output, feedback)
        self.optimizer.step()
        proposed = extract_select(str(self.node.data))
        source = strategy_source(proposed)
        self.trace_log.append({'iteration': len(self.curve), 'proposed': proposed, 'feedback': feedback, 'decisions': decisions})
        validation = await self.search_controller.evaluator.evaluate_program(source)
        candidate = get_program(self.search_controller.config, source, str(uuid.uuid4()), validation.metrics, start_iteration)
        if validation.metrics.get('validity') != 1:
            return SerializableResult(iteration=start_iteration, child_program_dict=candidate.to_dict(), error='Stock policy validator rejected Trace proposal')
        return SerializableResult(iteration=start_iteration, child_program_dict=candidate.to_dict())


# ------------------------------------------------------------------ runs

async def run_coevolution(task: str, arm: str, output: Path, horizon: int = 12, llm: Any = None, initial_select: str | None = None) -> dict[str, Any]:
    """One live run of the stock kernel with the requested arm (mirrors EXP22 run_kernel)."""
    config = K.configuration(task, output, fixed=arm == 'SD-FIXED')
    database = create_database('evox', config.search.database)
    benchmark = SKY / TASKS[task]
    inputs = K.DiscoveryControllerInput(config, str(benchmark / 'evaluator/evaluator.py'), database, output_dir=str(output))
    traced = {'TRACE-PAIRED': TracePaired, 'TRACE-PAIRED-100': TracePaired100, 'TRACE-CURRENT-100': TraceCurrent}
    controller = traced[arm](inputs, llm) if arm in traced else {'SD-FIXED': K.AuditController, 'SD-EVOX': EvoxScheduled, 'SD-EVOX-PAIRED': EvoxPaired}[arm](inputs)
    source = (benchmark / 'initial_program.py').read_text()
    initial_metrics = (await controller.evaluator.evaluate_program(source)).metrics
    initial = get_program(config, source, str(uuid.uuid4()), initial_metrics, 0)
    database.add(initial, iteration=0)
    database.initial_program_id, database.initial_program_score = initial.id, get_score(initial_metrics)
    if initial_select is not None:  # A5 episodes: deploy a fixed select_parent policy from the start
        policy = get_program(controller.search_controller.config, strategy_source(initial_select), str(uuid.uuid4()), {'validity': 1}, 0)
        if not controller._switch_to_new_search_algorithm(SerializableResult(child_program_dict=policy.to_dict())):
            raise RuntimeError('select_parent strategy failed to load')
    started = time.monotonic()
    best = await controller.run_discovery(1, horizon)
    result = {'task': task, 'arm': arm, 'initial_score': get_score(initial_metrics), 'final_best_score': get_score(best.metrics), 'iterations': len(controller.curve),
              'curve': [row['best_score'] for row in controller.curve], 'valid_candidates': sum(controller.attempt_validity), 'wall_s': round(time.monotonic() - started, 1),
              'policy_events': [{k: e.get(k) for k in ('iteration', 'activated', 'validation')} for e in controller.policy_events], 'meta_failures': controller._meta_evolution_failures,
              'paired_log': [{k: v for k, v in e.items()} for e in getattr(controller, 'paired_log', [])], 'trace_log': getattr(controller, 'trace_log', []),
              'gate_failure': controller.gate_failure, 'best_solution': best.solution}
    controller.close()
    controller.search_controller.close()
    return result


class SolutionProgram(trace.Module):
    """A6 surface: the complete benchmark program."""

    def __init__(self, source: str) -> None:
        super().__init__()
        self.program = trace.node(source, trainable=True, name='program', description='Complete Python program; keep GPU_MEM_SIZE/imports, the EVOLVE-BLOCK markers and every function signature.')

    def forward(self, evaluator: Any):
        return evaluate_program(self.program, evaluator)


@trace.bundle()
def evaluate_program(program, evaluator):
    """Run the stock benchmark evaluator (cascade, timeouts) on the program; returns its metrics."""
    return asyncio.run(evaluator.evaluate(program))


class MetricGuide:
    """Score = combined_score (invalid -> task initial score minus a margin); feedback = metrics."""

    def __init__(self, floor: float) -> None:
        self.floor = floor

    def __call__(self, task: Any, response: Any, info: Any, **kwargs: Any) -> tuple[float, str]:
        metrics = response.data if hasattr(response, 'data') else response
        score = metrics.get('combined_score')
        valid = isinstance(score, (int, float)) and math.isfinite(score) and metrics.get('validity', 1) != 0
        text = json.dumps({k: v for k, v in metrics.items() if isinstance(v, (int, float, str))})[:3000]
        return (float(score) if valid else self.floor), f'Evaluator metrics (higher combined_score is better): {text}'

    def get_feedback(self, query: Any, response: Any, reference: Any = None, **kwargs: Any) -> tuple[float, str]:
        return self(query, response, reference)


def run_trace_task(task: str, output: Path, llm: Any, proposals: int = 12) -> dict[str, Any]:
    """A6: PrioritySearch + OptoPrimeV2 on the solution program; num_steps = proposals + 1."""
    from opto.trainer.algorithms.priority_search import PrioritySearch
    from opto.trainer.guide import Guide

    class _Guide(Guide, MetricGuide):
        def __init__(self, floor: float) -> None:
            Guide.__init__(self)
            MetricGuide.__init__(self, floor)

        def get_feedback(self, query: Any, response: Any, reference: Any = None, **kwargs: Any) -> tuple[float, str]:
            return MetricGuide.__call__(self, query, response, reference)

    adapter = TraceEvaluatorAdapter(task)
    source = (SKY / TASKS[task] / 'initial_program.py').read_text()
    initial = asyncio.run(adapter.evaluate(source))
    module = SolutionProgram(source)
    optimizer = OptoPrimeV2(module.parameters(), llm=DummyLLM(llm), memory_size=5, log=False, max_tokens=32000, initial_var_char_limit=100000)
    trainer = PrioritySearch(module, optimizer, num_threads=1)
    scores: list[float] = []
    guide = _Guide(floor=float(initial.get('combined_score', 0.0)) - 1.0)
    original = guide.get_feedback

    def recording(*args: Any, **kwargs: Any) -> tuple[float, str]:
        score, text = original(*args, **kwargs)
        scores.append(score)
        return score, text
    guide.get_feedback = recording
    started = time.monotonic()
    trainer.train(guide=guide, train_dataset={'inputs': [adapter], 'infos': [None]}, num_steps=proposals + 1, num_epochs=0, num_candidates=1, num_proposals=1,
                  batch_size=1, num_batches=1, num_threads=1, test_frequency=None, log_frequency=1, save_frequency=None, validate_exploration_candidates=True,
                  use_best_candidate_to_explore=True, decouple_optimizers=False)
    final = asyncio.run(adapter.evaluate(str(module.program.data)))
    adapter.close()
    return {'task': task, 'arm': 'TRACE-TASK', 'initial_score': initial.get('combined_score'), 'evaluated_scores': scores, 'best_evaluated': max(scores) if scores else None,
            'final_module_score': final.get('combined_score'), 'optimizer_calls': llm.calls, 'wall_s': round(time.monotonic() - started, 1)}


def run_amortized(task: str, output: Path, llm: Any, steps: int = 3) -> dict[str, Any]:
    """A5: PrioritySearch over select_parent; every evaluation is a live 4-iteration EvoX episode."""
    from opto.trainer.algorithms.priority_search import PrioritySearch
    from opto.trainer.guide import Guide
    episodes: list[dict[str, Any]] = []

    @trace.bundle()
    def live_episode(select_parent_code, episode):
        """Run a live 4-iteration stock-EvoX episode from the initial program with select_parent deployed; return its outcome."""
        code = extract_select(str(select_parent_code))
        result = asyncio.run(run_coevolution(task, 'SD-FIXED', output / f'episode_{len(episodes):02d}', horizon=BLOCK, initial_select=code))
        summary = {'initial': result['initial_score'], 'final_best': result['final_best_score'], 'curve': [round(x, 6) for x in result['curve']], 'valid': result['valid_candidates']}
        episodes.append({**summary, 'policy': code})
        return summary

    class Policy(trace.Module):
        def __init__(self) -> None:
            super().__init__()
            self.select_parent_code = trace.node(SELECT_STOCK, trainable=True, name='select_parent_code', description=LIVE.SURFACE_HELP['code'])

        def forward(self, episode: Any):
            return live_episode(self.select_parent_code, episode)

    class EpisodeGuide(Guide):
        def get_feedback(self, query: Any, response: Any, reference: Any = None, **kwargs: Any) -> tuple[float, str]:
            data = response.data if hasattr(response, 'data') else response
            gain = data['final_best'] - data['initial']
            return gain, f"Episode best-score gain {gain:+.4f} over the initial program in {BLOCK} iterations (curve {data['curve']}, valid children {data['valid']})."

    module = Policy()
    optimizer = OptoPrimeV2(module.parameters(), llm=DummyLLM(llm), memory_size=5, log=False, max_tokens=8000)
    trainer = PrioritySearch(module, optimizer, num_threads=1)
    started = time.monotonic()
    trainer.train(guide=EpisodeGuide(), train_dataset={'inputs': ['episode'], 'infos': [None]}, num_steps=steps, num_epochs=0, num_candidates=1, num_proposals=1,
                  batch_size=1, num_batches=1, num_threads=1, test_frequency=None, log_frequency=1, save_frequency=None, validate_exploration_candidates=True,
                  use_best_candidate_to_explore=True, decouple_optimizers=False)
    return {'task': task, 'arm': 'AMORTIZED', 'episodes': episodes, 'solution_iterations': sum(len(e['curve']) for e in episodes),
            'final_policy': extract_select(str(module.select_parent_code.data)), 'optimizer_calls': llm.calls, 'wall_s': round(time.monotonic() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=tuple(TASKS), required=True)
    parser.add_argument('--arm', choices=('SD-FIXED', 'SD-EVOX', 'SD-EVOX-PAIRED', 'TRACE-PAIRED', 'TRACE-PAIRED-100', 'TRACE-CURRENT-100', 'AMORTIZED', 'TRACE-TASK'), required=True)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    llm = LIVE.LiveOptimizerLLM('novita', output / 'optimizer_calls.jsonl', max_tokens=32000 if args.arm == 'TRACE-TASK' else 8000)
    started = time.time()
    try:
        if args.arm == 'TRACE-TASK':
            result = run_trace_task(args.task, output, llm)
        elif args.arm == 'AMORTIZED':
            result = run_amortized(args.task, output, llm)
        else:
            result = asyncio.run(run_coevolution(args.task, args.arm, output, horizon=args.horizon, llm=llm))
        result['error'] = None
    except Exception as error:  # noqa: BLE001 - reported in the result file
        result = {'task': args.task, 'arm': args.arm, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()[-4000:]}
    result.update(wall_total_s=round(time.time() - started, 1), skydiscover_calls=CALLS, optimizer_calls_n=llm.calls)
    (output / 'result.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps({k: result.get(k) for k in ('task', 'arm', 'error', 'initial_score', 'final_best_score', 'best_evaluated', 'wall_total_s')}))


if __name__ == '__main__':
    main()
