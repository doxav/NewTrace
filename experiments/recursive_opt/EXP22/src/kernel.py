"""Stock EvoX execution with measured Trace proposals at the meta boundary."""

import copy
import hashlib
import json
import math
import time
import uuid
from importlib import import_module
from pathlib import Path
from typing import Any

from skydiscover.optimize.config import Config, EvoxDatabaseConfig, load_config
from skydiscover.optimize.search.default_discovery_controller import (
    DiscoveryControllerInput,
)
from skydiscover.optimize.search.evox.controller import CoEvolutionController
from skydiscover.optimize.search.registry import create_database, get_program
from skydiscover.optimize.search.utils.discovery_utils import SerializableResult
from skydiscover.optimize.utils.metrics import get_score
from src.accounting import usage_summary
from src.evaluation import SKY, TASKS, ProcessEvaluator
from src.transport import HTTP_RECORDS, HTTP_ROLES, model_config

from opto import trace
from opto.optimizers.optoprime_v2 import OptoPrimeV2

ROOT = Path(__file__).resolve().parents[1]
POLICY = SKY / 'skydiscover/optimize/search/evox/database/initial_search_strategy.py'


def digest(source: str) -> str:
    """Identify exactly the source that was proposed, evaluated or deployed."""
    return hashlib.sha256(source.encode()).hexdigest()


def append_json(path: Path, record: dict[str, Any]) -> None:
    """Append one finite machine-readable event to an immutable run directory."""
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False, default=str) + '\n')


def configuration(task: str, output: Path, fixed: bool = False) -> Config:
    """Apply only required transport, EvoX and sequential execution settings."""
    import_module('skydiscover.optimize.search.route')
    if task not in TASKS:
        raise ValueError('Unknown EXP22 task')
    config = load_config(SKY / TASKS[task] / 'config.yaml')
    config.search.type = 'evox'
    config.search.database = EvoxDatabaseConfig(random_seed=42)
    config.search.share_llm = True
    config.search.output_dir = str(output / 'search')
    config.search.switch_interval = 101 if fixed else None
    config.max_parallel_iterations = 1
    config.llm.models = [model_config()]
    config.llm.evaluator_models = [model_config()]
    config.llm.guide_models = [model_config()]
    config.llm.api_key = None
    config.llm.api_base = config.llm.models[0].api_base
    return config


@trace.bundle()
def measured_window(policy_source: str, observation: dict[str, Any]) -> dict[str, Any]:
    """Associate measured downstream feedback with its actual deployed policy."""
    if digest(policy_source) != observation['active_policy_hash']:
        raise ValueError('Feedback does not belong to the trainable active policy')
    return observation


@trace.model
class PolicyModule(trace.Module):
    """Expose the complete EvolvedProgramDatabase source as one trainable value."""

    def __init__(self, source: str) -> None:
        """Seed the parameter from the exact stock strategy source."""
        super().__init__()
        self.policy_source = trace.node(source, trainable=True, name='policy_source', description='Complete executable Python EvolvedProgramDatabase source; preserve the ProgramDatabase contract.')

    def forward(self, observation: dict[str, Any]) -> Any:
        """Connect observed policy-window measurements to optimizer feedback."""
        return measured_window(self.policy_source, observation)


class AuditController(CoEvolutionController):
    """Record stock behavior without replacing candidate or policy selection."""

    def __init__(self, controller_input: DiscoveryControllerInput) -> None:
        """Initialize stock pools and attach per-run audit state."""
        super().__init__(controller_input)
        self.directory = Path(controller_input.output_dir)
        for controller, role in ((self, 'solution'), (self.search_controller, 'policy')):
            controller.evaluator = ProcessEvaluator.replace(controller.evaluator, self.directory / 'evaluations' / role)
        self.curve: list[dict[str, Any]] = []
        self.policy_events: list[dict[str, Any]] = []
        self.window_observation: dict[str, Any] = {}
        self.gate_failure: str | None = None
        self.pending_attempt_scores: list[float | None] = []
        self.attempt_validity: list[bool] = []
        self.pending_attempt_validity: list[bool] = []
        self.current_candidate_score: float | None = None
        self.stop_on_trace_failure = False
        self._policy_proposer = self.search_controller.run_discovery
        self.search_controller.run_discovery = self._record_policy_proposal
        for controller in (self, self.search_controller):
            for role, pool in (('solution' if controller is self else 'meta', controller.llms), ('evaluator', controller.evaluator_llms), ('guide', controller.guide_llms)):
                if any(type(client).__name__ != 'SkyOpenRouter' for client in pool.models):
                    raise ValueError('A nested EvoX model escaped the required custom client')
                pool.random_state.seed(42)
                for client in pool.models:
                    HTTP_ROLES[id(client.client._client)] = role

    async def _evolve_search(self, solution_iter: int) -> None:
        """Capture feedback before the stock controller resets its window."""
        if self.gate_failure:
            return
        self.window_observation = {
            'active_policy_hash': digest(self._active_search_algorithm_code),
            'window_metrics': self._compute_search_metrics(horizon=self._switch_interval),
            'search_stats': self._build_search_stats(solution_iter),
            'policy_history': [program.to_dict() for program in self.search_controller.database.programs.values()],
        }
        append_json(self.directory / 'window_history.jsonl', {'iteration': len(self.curve), 'active_policy_hash': self.window_observation['active_policy_hash'], 'phase': 'before_meta', **self.window_observation['window_metrics']})
        await super()._evolve_search(solution_iter)
        if any(not row['passed'] for row in HTTP_RECORDS):
            self.gate_failure = 'Meta HTTP transport or serving-identity validation failed'
            self.shutdown_event.set()

    async def _record_policy_proposal(self, start_iteration: int, max_iterations: int, post_process_result: bool = True) -> SerializableResult | None:
        """Persist the unchanged stock/Trace proposal and validation before deployment."""
        try:
            result = await self._policy_proposer(start_iteration=start_iteration, max_iterations=max_iterations, post_process_result=post_process_result)
        except Exception as error:
            append_json(self.directory / 'policy_proposals.jsonl', {'iteration': len(self.curve), 'proposal': start_iteration, 'error_type': type(error).__name__, 'valid': False})
            raise
        candidate = result.child_program_dict if result else None
        source = (candidate or {}).get('solution')
        source_path = f'proposal_{start_iteration:03d}.py' if source else None
        if source_path:
            (self.directory / source_path).write_text(source)
        append_json(self.directory / 'policy_proposals.jsonl', {'iteration': len(self.curve), 'proposal': start_iteration, 'source_path': source_path, 'source_hash': digest(source) if source else None, 'validation': (candidate or {}).get('metrics'), 'valid': bool(result and not result.error and candidate and candidate['metrics'].get('validity') == 1), 'error': result.error if result else 'No proposal returned'})
        return result

    def _record_search_window_step(self) -> None:
        """Record the stock best-so-far curve at each consumed solution attempt."""
        super()._record_search_window_step()
        prior = self.pending_attempt_scores.pop(0) if self.pending_attempt_scores else None
        valid = self.pending_attempt_validity.pop(0) if self.pending_attempt_validity else False
        self.attempt_validity.append(valid)
        record = {'iteration': len(self.curve) + 1, 'best_score': self._get_best_score() if prior is None else prior, 'population_size': len(self.database.programs), 'active_policy_hash': digest(self._active_search_algorithm_code), 'policy_switches': sum(event['activated'] for event in self.policy_events)}
        if not math.isfinite(record['best_score']):
            raise ValueError('Non-finite solution fitness')
        self.curve.append(record)
        record.update({'current_score': self.current_candidate_score if valid else None, 'valid_candidates': sum(self.attempt_validity), 'invalid_candidates': len(self.attempt_validity) - sum(self.attempt_validity), 'policy_validation_failures': self._meta_evolution_failures, 'llm_usage': usage_summary(HTTP_RECORDS, solution_limit=len(self.curve))})
        append_json(self.directory / 'solution_curve.jsonl', record)
        if any(not row['passed'] for row in HTTP_RECORDS):
            self.gate_failure = 'HTTP transport or serving-identity validation failed'
        elif self.stop_on_trace_failure and len(self.attempt_validity) >= 10 and not any(self.attempt_validity[-10:]):
            self.gate_failure = 'No valid solution over ten consecutive generation attempts'
        elif self.stop_on_trace_failure and self._meta_evolution_failures and not any(event['activated'] for event in self.policy_events) and len(self.curve) >= 30:
            self.gate_failure = 'No policy accepted after failed meta proposals by checkpoint 30'
        if self.gate_failure:
            self.shutdown_event.set()
        if len(self.curve) in (10, 30, 60, 100):
            (self.directory / f'checkpoint_{len(self.curve):03d}.json').write_text(json.dumps({**record, 'gate_failure': self.gate_failure}, indent=2, allow_nan=False) + '\n')

    async def _run_iteration(self, iteration: int, retry_times: int = 3) -> SerializableResult:
        """Reuse the entire stock generation/evaluation path and record outcomes."""
        before = self._get_best_score()
        used = sum(row['role'] == 'solution' for row in HTTP_RECORDS)
        if HTTP_RECORDS and used != len(self.curve):
            self.gate_failure = 'HTTP solution calls and recorded attempt budget diverged'
            self.shutdown_event.set()
            raise ValueError(self.gate_failure)
        remaining = self.total_solution_iterations - iteration
        result = await super()._run_iteration(iteration, retry_times=min(retry_times, remaining))
        self.current_candidate_score = get_score(result.child_program_dict['metrics']) if result.child_program_dict else None
        if self.current_candidate_score is not None and not math.isfinite(self.current_candidate_score):
            self.gate_failure = 'Candidate evaluator returned non-finite fitness'
            self.shutdown_event.set()
            result.error = self.gate_failure
            self.current_candidate_score = None
        self.pending_attempt_scores = [before] * (getattr(result, 'attempts_used', 1) - 1) + [None]
        self.pending_attempt_validity = [False] * (getattr(result, 'attempts_used', 1) - 1) + [not result.error and bool(result.child_program_dict)]
        append_json(self.directory / 'candidate_history.jsonl', {'iteration': iteration, 'error': result.error, 'attempts_used': getattr(result, 'attempts_used', 1), 'candidate': result.child_program_dict, 'parent_id': result.parent_id, 'context_ids': result.other_context_ids})
        return result

    def _switch_to_new_search_algorithm(self, result: SerializableResult) -> bool:
        """Audit stock validation and migration without resetting the population."""
        before = {key: copy.deepcopy(program.to_dict()) for key, program in self.database.programs.items()}
        parent_hash = digest(self._active_search_algorithm_code)
        activated = super()._switch_to_new_search_algorithm(result)
        if activated:
            for key, previous in before.items():
                current = self.database.get(key)
                if current is None or current.solution != previous['solution'] or current.metrics != previous['metrics']:
                    self.gate_failure = 'Policy switch changed or lost the existing population'
                    self.shutdown_event.set()
                    raise ValueError('Policy switch changed or lost the existing population')
        source = (result.child_program_dict or {}).get('solution', '')
        event = {'iteration': len(self.curve), 'parent_policy_hash': parent_hash, 'candidate_policy_hash': digest(source), 'activated': activated, 'validation': (result.child_program_dict or {}).get('metrics'), 'feedback': self.window_observation, 'population_before': len(before), 'population_after': len(self.database.programs)}
        self.policy_events.append(event)
        append_json(self.directory / 'policy_history.jsonl', event)
        if source:
            (self.directory / f'policy_{len(self.policy_events):03d}.py').write_text(source)
        return activated


class TraceMetaCoEvolutionController(AuditController):
    """Replace only stock meta candidate production with a real OptoPrimeV2 step."""

    def __init__(self, controller_input: DiscoveryControllerInput, module: PolicyModule, optimizer_llm: Any) -> None:
        """Keep inherited validation, scoring, variation and migration unchanged."""
        super().__init__(controller_input)
        self.stop_on_trace_failure = controller_input.config.search.switch_interval != 101
        self.policy_module = module
        self.optimizer_llm = optimizer_llm
        self._policy_proposer = self._trace_proposal

    async def _trace_proposal(self, start_iteration: int, max_iterations: int, post_process_result: bool = True) -> SerializableResult:
        """Produce one policy from measured feedback, then use the stock evaluator."""
        if max_iterations != 1:
            raise ValueError('Trace must produce exactly one policy per evolution event')
        self.policy_module.policy_source._data = self._active_search_algorithm_code
        if self.window_observation.get('active_policy_hash') != digest(self._active_search_algorithm_code):
            self.gate_failure = 'Trace policy update is disconnected from measured active-policy feedback'
            self.shutdown_event.set()
            raise ValueError(self.gate_failure)
        optimizer = OptoPrimeV2(self.policy_module.parameters(), llm=self.optimizer_llm, max_tokens=32000, log=False, initial_var_char_limit=100000, objective='Improve downstream solution search by rewriting the complete EvolvedProgramDatabase Python source. Preserve all interfaces and solution metrics. Use the measured window feedback; return exactly one policy proposal.')
        output = self.policy_module(self.window_observation)
        optimizer.zero_feedback()
        optimizer.backward(output, json.dumps(self.window_observation, default=str))
        optimizer.step()
        source = str(self.policy_module.policy_source.data)
        if digest(source) == digest(self._active_search_algorithm_code):
            self.gate_failure = 'Trace optimizer produced no policy-source change'
            self.shutdown_event.set()
            raise ValueError('Trace optimizer produced no policy-source change')
        validation = await self.search_controller.evaluator.evaluate_program(source)
        candidate = get_program(self.search_controller.config, source, str(uuid.uuid4()), validation.metrics, start_iteration)
        if validation.metrics.get('validity') != 1:
            (self.directory / f'rejected_policy_{start_iteration:03d}.py').write_text(source)
            append_json(self.directory / 'policy_history.jsonl', {'iteration': len(self.curve), 'activated': False, 'candidate_policy_hash': digest(source), 'validation': validation.metrics})
            return SerializableResult(iteration=start_iteration, child_program_dict=candidate.to_dict(), error='Stock policy validator rejected Trace proposal')
        return SerializableResult(iteration=start_iteration, child_program_dict=candidate.to_dict())


async def run_kernel(task: str, arm: str, horizon: int, output: Path, module: PolicyModule | None = None, optimizer_llm: Any = None) -> dict[str, Any]:
    """Run a persistent stock EvoX kernel with optional Trace meta proposals."""
    if arm not in {'SD-FIXED', 'SD-EVOX', 'TRACE-FIXED', 'TRACE-RECURSIVE'} or horizon < 1 or horizon > 100:
        raise ValueError('Invalid arm or solution horizon')
    config = configuration(task, output, fixed=arm.endswith('FIXED'))
    database = create_database('evox', config.search.database)
    benchmark = SKY / TASKS[task]
    inputs = DiscoveryControllerInput(config, str(benchmark / 'evaluator/evaluator.py'), database, output_dir=str(output))
    controller = TraceMetaCoEvolutionController(inputs, module, optimizer_llm) if arm.startswith('TRACE') else AuditController(inputs)
    source = (benchmark / 'initial_program.py').read_text()
    initial_metrics = (await controller.evaluator.evaluate_program(source)).metrics
    initial = get_program(config, source, str(uuid.uuid4()), initial_metrics, 0)
    database.add(initial, iteration=0)
    database.initial_program_id = initial.id
    database.initial_program_score = get_score(initial_metrics)
    started = time.monotonic()
    best = await controller.run_discovery(1, horizon)
    if module is not None:
        module.policy_source._data = controller._active_search_algorithm_code
    result = {'task': task, 'arm': arm, 'horizon': horizon, 'initial_score': get_score(initial_metrics), 'final_best_score': get_score(best.metrics), 'final_metrics': best.metrics, 'iterations_observed': len(controller.curve), 'wall_time': time.monotonic() - started, 'policy_switches': sum(e['activated'] for e in controller.policy_events), 'gate_failure': controller.gate_failure}
    result['meta_proposal_failures'] = controller._meta_evolution_failures
    result['final_window_metrics'] = controller._compute_search_metrics(horizon=controller._switch_interval)
    append_json(output / 'window_history.jsonl', {'iteration': len(controller.curve), 'active_policy_hash': digest(controller._active_search_algorithm_code), 'phase': 'final', **result['final_window_metrics']})
    (output / 'best_solution.py').write_text(best.solution)
    (output / 'best_policy.py').write_text(controller._active_search_algorithm_code)
    (output / 'kernel_result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    controller.close()
    controller.search_controller.close()
    return result
