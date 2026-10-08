"""Control-plane specs for EXP29 numeric runs: the O0 child (Trace learns an optimizer program) and episodes."""
from __future__ import annotations

import copy

from opto.features.recursive_opt import spec as S

from . import task as T

MODEL = 'z-ai/glm-5.3-flash'
PROFILE = {'provider': 'openrouter', 'model': MODEL, 'max_tokens': 16000, 'temperature': 0.7, 'request_timeout_s': 300,
           'request_params': {'extra_body': {'reasoning': {'effort': 'low'}, 'provider': {'sort': 'price'}}}}
OBJECTIVE_TEXT = ('The variable is a Python black-box minimization program `propose(history, bounds, seed) -> list[float]`. '
                  'Each call must return the next point to evaluate inside `bounds`, using only the standard library, '
                  'deterministically from (history, seed). It is scored on several shifted, scaled test functions (smooth and '
                  'multimodal, 2-5 dimensions, 64 evaluations each) by the mean log10 normalized best-so-far regret: find good '
                  'points early and refine them precisely. Keep the exact signature.')

# Episodes: (train, validation, holdout) strata; each stratum = (family, dimension). Indexes differ by split.
EPISODES = {
    'mixA': {'train': [('sphere', 5), ('rastrigin', 2), ('rosenbrock', 5), ('ackley', 2)], 'validation': [('sphere', 2), ('rastrigin', 5)]},
    'mixB': {'train': [('ackley', 5), ('sphere', 2), ('rastrigin', 5), ('rosenbrock', 2)], 'validation': [('ackley', 2), ('sphere', 5)]},
    'mixC': {'train': [('rastrigin', 2), ('ackley', 2), ('sphere', 5), ('rosenbrock', 5)], 'validation': [('rosenbrock', 2), ('ackley', 5)]},
}


def episode(name: str, seed: int = 0) -> dict:
    """Datasets for one episode; holdout = new instances of the train and validation strata."""
    plan = EPISODES[name]
    def items(strata, split, count):
        return [item for f, d in strata for item in T.items((f,), (d,), f'{name}-{split}-s{seed}', count, seed)]
    return {'train': items(plan['train'], 'train', 1), 'validation': items(plan['validation'], 'validation', 1),
            'holdout': items(plan['train'] + plan['validation'], 'holdout', 1)}


def o0_spec(datasets: dict, *, seed: int, iterations: int = 8, trainer: str = 'PrioritySearch', trainer_kwargs: dict | None = None,
            patches: list | None = None, objective_text: str = OBJECTIVE_TEXT, offline: bool = False) -> dict:
    return {'schema_version': S.SCHEMA_VERSION, 'kind': S.SPEC_KIND,
            'runtime': {'offline': offline, 'seed': seed, 'test_mode': offline},
            'llm_profiles': {'glm': copy.deepcopy(PROFILE)},
            'levels': [{'id': 'O0', 'surface': {'kind': 'module', 'targets': ['optimizer']},
                        'module': {'ref': 'recursive_opt.module.reasoning_workflow@1', 'config': {'components': {'optimizer': T.SEED_SOURCE}}, 'inputs': {}},
                        'engine': {'name': 'trace', 'config': {'trainer': trainer, 'optimizer': 'OptoPrimeV2', 'iterations': iterations, 'num_candidates': 1,
                                                              'trainer_kwargs': {'batch_size': len(datasets['train']), **(trainer_kwargs or {})},
                                                              'optimizer_kwargs': {'objective': objective_text, 'memory_size': 0},
                                                              'patches': copy.deepcopy(patches or [])}},
                        'objective': {'evaluator_ref': 'exp29.evaluator.bbo@1', 'intent': 'Maximize the score (minus mean log10 regret).',
                                      'metrics': {'score': {'direction': 'maximize', 'source': 'evaluation.metrics.score', 'aggregate_examples': 'mean'}},
                                      'selection': {'mode': 'scalar', 'score_key': 'score'}},
                        'llm_roles': {'optimizer': 'glm'},
                        'datasets': copy.deepcopy(datasets)}]}


# ---- O1: discover the trainer's search policy (T2) and the optimizer instruction (T1) through child_spec@1
MODE_TARGET = 'opto.trainer.algorithms.variation_search:VariationSearch._next_mode'
TEXTS_TARGET = 'opto.trainer.algorithms.variation_search:VARIATION_INSTRUCTIONS'
O1_OBJECTIVE = ('You are optimizing HOW a lower-level optimizer learns, not a solution. The variables are (a) `next_mode`, the '
                'VariationSearch method that picks the mutation intent of each step (it must return one of "free", "refine", '
                '"diverge", "combine" and may read self._variation_stall, self._variation_step, self._variation_best, '
                'self.patience, self.period, self.variation_schedule and self.variation_log), (b) `diverge_text`, the text appended '
                'to the lower optimizer prompt on "diverge" steps, and (c) `objective`, the '
                'lower optimizer\'s task instruction. Each evaluation runs the whole lower-level learning of a numeric black-box '
                'optimizer program on new test functions and returns its held-out score (higher is better). Keep the code valid '
                'Python with the same signature; prefer general changes that help any task.')


def o1_spec(train_episodes: list, validation_episodes: list, holdout_episodes: list, *, seed: int, child_iterations: int,
            o1_iterations: int, slots: tuple = ('next_mode', 'diverge_text', 'objective'), timeout_s: float = 3600, offline: bool = False) -> dict:
    from opto.features.recursive_opt import patches as P
    from opto.trainer.algorithms import variation_search as V
    child = o0_spec(episode('mixA', 0), seed=seed, iterations=child_iterations, trainer='VariationSearch',
                    trainer_kwargs={'num_threads': 8, 'variation_seed': seed},
                    patches=[{'target': MODE_TARGET, 'source': P.default_source(MODE_TARGET)},
                             {'target': TEXTS_TARGET, 'value': dict(V.VARIATION_INSTRUCTIONS)}], offline=offline)
    paths = {'next_mode': 'levels.O0.engine.config.patches.0.source', 'diverge_text': 'levels.O0.engine.config.patches.1.value.diverge',
             'combine_text': 'levels.O0.engine.config.patches.1.value.combine',
             'objective': 'levels.O0.engine.config.optimizer_kwargs.objective'}
    wrap = lambda names: [{'episodes': [{'levels.O0.datasets': episode(n, s), 'runtime.seed': s} for n, s in names]}]  # noqa: E731  (parallel)
    return {'schema_version': S.SCHEMA_VERSION, 'kind': S.SPEC_KIND,
            'runtime': {'offline': offline, 'seed': seed, 'test_mode': offline},
            'llm_profiles': {'glm': copy.deepcopy(PROFILE)},
            'levels': [{'id': 'O1', 'surface': {'kind': 'module', 'targets': list(slots)},
                        'module': {'ref': 'recursive_opt.module.child_spec@1', 'inputs': {},
                                   'config': {'child': child, 'slots': {k: paths[k] for k in slots}, 'example_paths': ['levels.O0.datasets', 'runtime.seed'],
                                              'timeout_s': timeout_s}},
                        'engine': {'name': 'trace', 'config': {'trainer': 'PrioritySearch', 'optimizer': 'OptoPrimeV2', 'iterations': o1_iterations,
                                                              'num_candidates': 1, 'trainer_kwargs': {'batch_size': 1, 'num_threads': 1},
                                                              'optimizer_kwargs': {'objective': O1_OBJECTIVE, 'memory_size': 0}}},
                        'objective': {'evaluator_ref': 'recursive_opt.evaluator.child_spec@1', 'intent': 'Maximize the child held-out score.',
                                      'metrics': {'score': {'direction': 'maximize', 'source': 'evaluation.metrics.score', 'aggregate_examples': 'mean'}},
                                      'selection': {'mode': 'scalar', 'score_key': 'score'}},
                        'llm_roles': {'optimizer': 'glm'},
                        'datasets': {'train': wrap(train_episodes), 'validation': wrap(validation_episodes), 'holdout': wrap(holdout_episodes)}}]}
