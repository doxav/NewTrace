"""Offline tests for recursive_opt.coevolution: online co-evolution of a selection policy
with a population-conditioned O0 operator (EvoX-equivalent semantics, generic components)."""

import math
import random

import pytest

from opto.features.recursive_opt.coevolution import (
    Candidate,
    CoevolutionConfig,
    CoevolutionEngine,
    DeferredEvaluation,
    LogWindowScorer,
    PairedScorer,
    PeriodicTrigger,
    PolicyContractError,
    PolicyRuntimeError,
    PolicySlot,
    Population,
    PopulationOperator,
    StagnationTrigger,
    StrategyArchive,
    UNIFORM_POLICY_SOURCE,
    apply_search_replace,
    evox_preset,
    parse_full_rewrite,
    resolve_patience,
    validate_policy_source,
)

GREEDY = '''
class Policy:
    def __init__(self, labels):
        self.labels = labels
    def observe(self, candidate):
        pass
    def sample(self, population, rng, num_context):
        members = population.members
        parent = max(members, key=lambda c: population.score(c))
        others = [c for c in members if c.id != parent.id][:num_context]
        return parent, others, "refine" if "refine" in self.labels else ""
'''


def pop_with(scores, score_key='combined_score'):
    population = Population(score_key)
    for i, s in enumerate(scores):
        population.add(Candidate(id=f'c{i}', content=f'v{i}', metrics={score_key: s}, iteration=i, parent_id=f'c{i-1}' if i else None))
    return population


# ------------------------------------------------------------------ shared state

def test_population_statistics_execution_trace_and_stagnation():
    population = Population()
    population.add(Candidate('a', 'x', {'combined_score': 1.0}, 0))
    population.add(Candidate('b', 'y', {'combined_score': 2.0}, 1, parent_id='a', context_ids=('a',), label='diverge', context_labels=('',)))
    population.add(Candidate('c', 'z', {'combined_score': 2.005}, 2, parent_id='b', label='refine'))
    stats = population.statistics(improvement_threshold=0.01)
    trace = stats['recent_solution_stats']['execution_trace']
    assert [t['iteration'] for t in trace] == [0, 1, 2]
    assert trace[1]['parent'] == ('diverge', 'a', 1.0)
    assert trace[1]['context'] == [('', 'a', 1.0)]
    assert trace[2]['parent'] == ('refine', 'b', 2.0)
    # EvoX: iterations since the first program within threshold of the best (b at iteration 1)
    assert stats['recent_solution_stats']['iterations_without_improvement'] == 1
    assert stats['solution_score_summary']['best'] == 2.005
    assert stats['population_size'] == 3


def test_population_snapshot_restore():
    population = pop_with([1.0, 2.0])
    snap = population.snapshot()
    population.add(Candidate('x', 'x', {'combined_score': 9.0}, 5))
    assert population.best_score() == 9.0
    population.restore(snap)
    assert population.best_score() == 2.0 and len(population.members) == 2


# ------------------------------------------------------------------ trigger / patience

def test_stagnation_trigger_matches_evox_counting():
    trigger = StagnationTrigger(patience=3, threshold=0.01)
    bests = [1.0, 1.0, 1.005, 1.005, 2.0, 2.0, 2.0, 2.0]
    fired = [trigger.update(b) for b in bests]
    # first call initializes; +0.005 is not an improvement; 3 stagnant calls fire and reset
    assert fired == [False, False, False, True, False, False, False, True]


def test_periodic_trigger_and_patience_resolution():
    trigger = PeriodicTrigger(every=4)
    assert [trigger.update(0.0) for _ in range(8)] == [False, False, False, True, False, False, False, True]
    assert resolve_patience('auto', 100, ratio=0.1) == 10
    assert resolve_patience('auto', 5, ratio=0.1) == 1
    assert resolve_patience(7, 100) == 7


# ------------------------------------------------------------------ deferred evaluation

def test_log_window_scorer_formula_uses_constant_horizon():
    scorer = LogWindowScorer(horizon=10)
    metrics = scorer.score(start=21.0, steps=[21.0, 22.0, 21.5], start_iteration=3)
    expected = (22.0 - 21.0) * (1 + math.log(1 + 21.0)) / math.sqrt(10)
    assert metrics['combined_score'] == pytest.approx(expected)
    assert metrics['search_window_end_score'] == 22.0 and metrics['search_horizon'] == 10 and metrics['window_start_iteration'] == 3


def test_deferred_evaluation_scores_only_on_close():
    archive = StrategyArchive()
    deferred = DeferredEvaluation(LogWindowScorer(horizon=4))
    deferred.reset(start=1.0, start_iteration=None)
    entry = archive.new_entry(source='p1', iteration=1)
    deferred.attach(entry)
    deferred.reset(start=1.0, start_iteration=7)
    for best in (1.0, 1.5, 1.5):
        deferred.record(best)
    assert entry.status == 'pending' and entry.score is None
    closed = deferred.close(current_best=1.5)
    assert closed is entry and entry.status == 'scored'
    assert entry.metrics['search_window_start_score'] == 1.0 and entry.metrics['search_window_end_score'] == 1.5
    # empty window: EvoX falls back to [current best]
    other = archive.new_entry(source='p2', iteration=9)
    deferred.attach(other)
    deferred.reset(start=1.5, start_iteration=9)
    deferred.close(current_best=2.0)
    assert other.metrics['search_window_end_score'] == 2.0


def test_paired_scorer_prefers_challenger_new_bests():
    records = [{'tag': 'challenger', 'new_best': True}, {'tag': 'incumbent', 'new_best': False},
               {'tag': 'challenger', 'new_best': False}, {'tag': 'incumbent', 'new_best': False}]
    assert PairedScorer().score_records(records) == pytest.approx(0.5)


# ------------------------------------------------------------------ archive

def test_archive_best_parent_and_random_context_excluding_parent():
    archive = StrategyArchive(seed=0)
    entries = []
    for score in (0.5, None, 3.0, 1.0):
        entry = archive.new_entry(source=f's{score}', iteration=0)
        entry.metrics = {'combined_score': score} if score is not None else {'combined_score': 'bad'}
        entry.status = 'scored'
        archive.add(entry)
        entries.append(entry)
    parent, context = archive.select(num_context=2)
    assert parent is entries[2]
    assert parent not in context and 1 <= len(context) <= 2


# ------------------------------------------------------------------ policy contract / hot swap

def test_policy_contract_validation():
    assert validate_policy_source(UNIFORM_POLICY_SOURCE, labels={'diverge': 'D', 'refine': 'R'}) is None
    assert 'SyntaxError' in validate_policy_source('class Policy(:\n  pass', labels={})
    foreign = GREEDY.replace('return parent, others', 'from types import SimpleNamespace\n        return SimpleNamespace(id="ghost"), others')
    assert 'not in the population' in validate_policy_source(foreign, labels={})
    mutating = GREEDY.replace('pass', 'candidate.metrics["combined_score"] = 0')
    assert 'metrics' in validate_policy_source(mutating, labels={})
    bad_label = GREEDY.replace('"refine" if "refine" in self.labels else ""', '"nonexistent"')
    assert 'label' in validate_policy_source(bad_label, labels={'refine': 'R'})


def test_uniform_policy_reproduces_stock_evox_rng_sequence():
    population = pop_with([1.0, 3.0, 2.0, 5.0, 4.0, 0.5])
    slot = PolicySlot(labels={}, seed=42)
    slot.deploy(UNIFORM_POLICY_SOURCE, population, entry_id='stock')
    ours = [slot.sample(population, num_context=4) for _ in range(5)]
    # literal port of skydiscover EvolvedProgramDatabase.sample (scalar branch) with random.Random(42)
    rng = random.Random(42)
    candidates = population.members
    for selection in ours:
        parent = rng.choice(candidates)
        examples = rng.sample(candidates, min(5, len(candidates)))
        examples = [p for p in examples if p.id != parent.id][:4]
        assert selection.parent.id == parent.id
        assert [c.id for c in selection.contexts] == [c.id for c in examples]
        assert selection.label == ''


def test_hot_swap_migrates_then_rolls_back_on_runtime_error():
    population = pop_with([1.0, 2.0])
    seen = []
    recorder = GREEDY.replace('pass', 'SEEN.append(candidate.id)')
    slot = PolicySlot(labels={}, seed=0, namespace={'SEEN': seen})
    slot.deploy(recorder, population, entry_id='p0', validate=False)  # validation is a dry run that would also append to SEEN
    assert seen == ['c0', 'c1']  # migration replays the population in order
    raising = GREEDY.replace('members = population.members', 'raise RuntimeError("boom")')
    slot.deploy(raising, population, entry_id='p1', validate=False)
    population.add(Candidate('c2', 'v', {'combined_score': 3.0}, 2))
    slot.observe(population.members[-1])
    with pytest.raises(PolicyRuntimeError):
        slot.sample(population, num_context=2)
    assert slot.rollback() == 'p0'
    assert seen[-1] == 'c2'  # restored policy catches up on candidates added meanwhile
    assert slot.sample(population, num_context=2).parent.id == 'c2'
    assert slot.fallback is None


def test_deploy_rejects_invalid_policy():
    slot = PolicySlot(labels={}, seed=0)
    with pytest.raises(PolicyContractError):
        slot.deploy('def nope(:', pop_with([1.0]), entry_id='bad')


# ------------------------------------------------------------------ O0 operator

class ScriptedLLM:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, system, user):
        self.prompts.append((system, user))
        return self.replies.pop(0)


def toy_evaluate(source):
    namespace = {}
    try:
        exec(source, namespace)
        score = float(namespace['SCORE'])
    except Exception as error:  # noqa: BLE001
        return {'validity': 0, 'combined_score': 0, 'error': str(error)}, {}
    return {'combined_score': score}, {'feedback': f'score {score}'}


def test_search_replace_and_full_rewrite_parsing_match_skydiscover():
    original = 'a = 1\nb = 2\n# end'
    diff = '<<<<<<< SEARCH\n# end\n=======\nc = 3\n# end\n>>>>>>> REPLACE'
    assert apply_search_replace(original, diff)[0] == 'a = 1\nb = 2\nc = 3\n# end'
    assert apply_search_replace(original, 'no blocks')[1] == 'No valid diffs found in response'
    miss = '<<<<<<< SEARCH\nzzz\n=======\nq\n>>>>>>> REPLACE'
    assert 'did not match' in apply_search_replace(original, miss)[1]
    assert parse_full_rewrite('text\n```python\nx = 1\n```') == 'x = 1'
    assert parse_full_rewrite('plain') == 'plain'


def test_operator_retries_with_errors_labels_and_context_in_prompt():
    population = Population()
    population.add(Candidate('p', 'SCORE = 1.0\n# end', {'combined_score': 1.0}, 0, artifacts={'feedback': 'ok'}))
    population.add(Candidate('q', 'SCORE = 0.5\n# end', {'combined_score': 0.5}, 1))
    llm = ScriptedLLM(['no diff here', '<<<<<<< SEARCH\n# end\n=======\nSCORE = SCORE + "x"\n# end\n>>>>>>> REPLACE',
                       '<<<<<<< SEARCH\n# end\n=======\nSCORE = SCORE + 0.5\n# end\n>>>>>>> REPLACE'])
    operator = PopulationOperator(llm, toy_evaluate, system_message='TASK', mode='diff', retries=3, labels={'refine': 'REFINE TEXT'})
    from opto.features.recursive_opt.coevolution import Selection
    result = operator.run(Selection(population.get('p'), [population.get('q')], 'refine'), population, iteration=2)
    assert result.attempts_used == 3 and result.error is None
    assert result.candidate.metrics['combined_score'] == 1.5
    assert result.candidate.parent_id == 'p' and result.candidate.context_ids == ('q',) and result.candidate.label == 'refine'
    system, user = llm.prompts[0]
    assert system == 'TASK' and 'REFINE TEXT' in user and 'SCORE = 0.5' in user and 'Evaluator Feedback' in user
    assert 'No valid diffs found' in llm.prompts[1][1]           # parse error fed back
    assert 'could only concatenate' in llm.prompts[2][1] or 'unsupported operand' in llm.prompts[2][1]  # evaluation error fed back


def test_operator_exhausted_retries_report_error():
    population = Population()
    population.add(Candidate('p', 'SCORE = 1.0\n# end', {'combined_score': 1.0}, 0))
    from opto.features.recursive_opt.coevolution import Selection
    operator = PopulationOperator(ScriptedLLM(['x', 'y']), toy_evaluate, system_message='T', mode='diff', retries=2)
    result = operator.run(Selection(population.get('p'), [], ''), population, iteration=1)
    assert result.candidate is None and result.attempts_used == 2 and 'after 2 attempts' in result.error


# ------------------------------------------------------------------ engine

def diff_add(delta):
    return f'<<<<<<< SEARCH\n# end\n=======\nSCORE = SCORE + {delta}\n# end\n>>>>>>> REPLACE'


class CountingLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def __call__(self, system, user):
        self.calls += 1
        return self.reply(self.calls, system, user)


def test_engine_evox_semantics_trigger_propose_deploy_and_score():
    solution = CountingLLM(lambda n, s, u: diff_add(1.0 if n <= 3 else 0.0))
    meta = CountingLLM(lambda n, s, u: '```python\n' + GREEDY + '\n```')
    config = CoevolutionConfig(**{**evox_preset(horizon=12, summaries=False, generate_labels=False, retries=1), 'patience': 3, 'initial_policy': GREEDY})
    engine = CoevolutionEngine(config, solution_llm=solution, meta_llm=meta, evaluate=toy_evaluate, initial_source='SCORE = 1.0\n# end', system_message='T')
    report = engine.run()
    # bests: 2,3,4,4,4,4 -> first trigger after 3 stagnant checks
    triggers = [e for e in report['events'] if e['type'] == 'trigger']
    assert triggers[0]['iteration'] == 6
    deploys = [e for e in report['events'] if e['type'] == 'deploy']
    assert deploys and deploys[0]['ok']
    scored = report['archive']
    assert scored[0]['source_kind'] == 'initial' and scored[0]['metrics']['combined_score'] > 0
    assert report['solution_attempts'] == 12 and report['best_score'] == 4.0


def test_engine_rolls_back_policy_that_fails_at_runtime():
    raising = GREEDY.replace('members = population.members', 'if len(population.members) > 8:\n            raise RuntimeError("late failure")\n        members = population.members')
    solution = CountingLLM(lambda n, s, u: diff_add(0.0))
    meta = CountingLLM(lambda n, s, u: '```python\n' + raising + '\n```')
    config = CoevolutionConfig(**{**evox_preset(horizon=10, summaries=False, generate_labels=False, retries=1), 'patience': 2})
    engine = CoevolutionEngine(config, solution_llm=solution, meta_llm=meta, evaluate=toy_evaluate, initial_source='SCORE = 1.0\n# end', system_message='T')
    report = engine.run()
    assert any(e['type'] == 'rollback' for e in report['events'])
    assert report['solution_attempts'] == 10  # a rollback retries the same iteration without consuming budget


def test_engine_meta_retries_invalid_proposals_blind_in_evox_mode():
    solution = CountingLLM(lambda n, s, u: diff_add(0.0))
    meta = CountingLLM(lambda n, s, u: '```python\nclass Policy(:\n```' if n < 3 else '```python\n' + GREEDY + '\n```')
    config = CoevolutionConfig(**{**evox_preset(horizon=6, summaries=False, generate_labels=False), 'patience': 2})
    engine = CoevolutionEngine(config, solution_llm=solution, meta_llm=meta, evaluate=toy_evaluate, initial_source='SCORE = 1.0\n# end', system_message='T')
    report = engine.run()
    first = next(e for e in report['events'] if e['type'] == 'proposal')
    assert first['attempts'] == 3 and first['ok']
    assert all('SyntaxError' not in p for p in engine.meta_prompts[1:3])  # EvoX does not feed validator errors back


def test_engine_paired_deployment_promotes_or_reverts():
    solution = CountingLLM(lambda n, s, u: diff_add(0.0))
    meta = CountingLLM(lambda n, s, u: '```python\n' + GREEDY + '\n```')
    config = CoevolutionConfig(**{**evox_preset(horizon=12, summaries=False, generate_labels=False, retries=1), 'patience': 2, 'deployment': 'paired', 'window_scorer': 'paired'})
    engine = CoevolutionEngine(config, solution_llm=solution, meta_llm=meta, evaluate=toy_evaluate, initial_source='SCORE = 1.0\n# end', system_message='T')
    report = engine.run()
    resolved = [e for e in report['events'] if e['type'] == 'paired_resolution']
    assert resolved and all(e['promoted'] is False for e in resolved)  # no new bests: incumbent kept


def _optoprime_constructible():
    try:
        import litellm.types.llms.openai  # noqa: F401  (OptoPrime builds a default LLM through litellm)
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.mark.skipif(not _optoprime_constructible(), reason='litellm is incompatible with the installed openai package (pre-existing; also breaks test_22j)')
def test_trace_meta_proposer_keeps_optimizer_memory():
    from opto.features.recursive_opt.coevolution import TraceProposer
    replies = []

    def llm(system, user):
        import re
        replies.append(user)
        name = re.search(r'<variable name="(\w+)"', user).group(1)
        body = GREEDY + f'\n# revision {len(replies)}\n'
        return f'<reasoning>r</reasoning>\n<variable>\n<name>{name}</name>\n<value>\n{body}\n</value>\n</variable>'
    proposer = TraceProposer(llm, memory_size=3)
    first = proposer.propose(parent_source=UNIFORM_POLICY_SOURCE, context=[], feedback='window score 0.0', validate=lambda s: None)
    second = proposer.propose(parent_source=first.source, context=[], feedback='window score 1.0', validate=lambda s: None)
    assert first.ok and second.ok and 'class Policy' in second.source
    assert proposer.memory_size == 3 and len(proposer.optimizer.memory) >= 1 and len(replies) == 2


# ------------------------------------------------------------------ control plane

def test_control_plane_engine_compiles_and_runs_offline():
    from opto.features.recursive_opt import spec as S
    from opto.features.recursive_opt.coevolution import control_plane as CP
    CP.register()
    S.register_evaluator('test.coevolution.toy_evaluator@1', CP.program_evaluator(toy_evaluate))
    raw = CP.coevolution_spec(level_id='evox', program='SCORE = 1.0\n# end', evaluator_ref='test.coevolution.toy_evaluator@1',
                              system_message='T', engine_config={**evox_preset(horizon=6, summaries=False, generate_labels=False), 'patience': 2})
    plan = S.compile_plan(raw)
    replies = {'forward': lambda n, s, u: diff_add(0.5), 'optimizer': lambda n, s, u: '```python\n' + GREEDY + '\n```', 'feedback': lambda n, s, u: 'summary'}

    def factory(profile, role):
        counter = {'n': 0}

        def client(messages=None, **kwargs):
            counter['n'] += 1
            return replies[role](counter['n'], messages[0]['content'], messages[-1]['content'])
        return client
    (result,) = S.execute_plan(plan, {'llm_factory': factory})
    assert result.status == 'success', result.error
    assert result.evaluation.metrics['combined_score'] > 1.0
    assert result.metadata['report']['solution_attempts'] == 6
    assert 'SCORE = SCORE + 0.5' in result.artifact['program']


# ------------------------------------------------------------------ guides, diagnostics, projections

from opto.features.recursive_opt.coevolution import (  # noqa: E402
    CompileCheck,
    FallbackWrapper,
    GuidedEvaluator,
    ProjectionError,
    format_case_diagnostics,
    make_projection,
    register_projection,
)


def test_guided_evaluator_separates_guide_score_and_enforces_constraints():
    def evaluate(source):
        return {'combined_score': 30.0, 'success_rate': 0.06}, {}
    penalize = GuidedEvaluator(evaluate, hard_constraints={'success_rate': ('>=', 1.0)}, violation='penalize', penalty_score=0.0)
    metrics, artifacts = penalize('x')
    assert metrics['guided_score'] == 0.0 and metrics['combined_score'] == 30.0
    assert 'success_rate' in artifacts['constraints']
    reject = GuidedEvaluator(evaluate, hard_constraints={'success_rate': ('>=', 1.0)}, violation='reject')
    metrics, _ = reject('x')
    assert metrics['validity'] == 0 and 'success_rate' in metrics['error']
    custom = GuidedEvaluator(lambda s: ({'a': 2.0}, {'cases': [1]}), score_fn=lambda m, a: m['a'] * 10)
    assert custom('x')[0]['guided_score'] == 20.0


def test_case_diagnostics_highlights_worst_gaps_and_failures():
    cases = [{'case': i, 'value': 1.0 + i / 10, 'bound': 1.0, 'ok': True} for i in range(10)]
    cases.append({'case': 10, 'ok': False, 'error': 'ZeroDivisionError: float division by zero'})
    cases.append({'case': 11, 'ok': False, 'error': 'ZeroDivisionError: float division by zero'})
    text = format_case_diagnostics(cases, worst=3, lower_is_better=True)
    assert 'case 9' in text and 'case 0' not in text.split('Worst')[1]
    assert '2 case(s) failed' in text and 'ZeroDivisionError' in text


def test_fallback_wrapper_projects_failures_onto_baseline():
    source = 'def solve(x):\n    return 1 / x\n'
    wrapper = FallbackWrapper(entry='solve', fallback_source='def solve(x):\n    return -1\n', check_source='def check(result, x):\n    return result < 100\n')
    projected, notes = wrapper(source)
    namespace = {}
    exec(projected, namespace)
    assert namespace['solve'](2) == 0.5          # candidate kept when valid
    assert namespace['solve'](0) == -1           # exception -> fallback
    assert namespace['solve'](0.001) == -1       # invalid result -> fallback
    assert namespace['_PROJECTION_EVENTS'] == ['ZeroDivisionError', 'invalid']
    assert 'fallback' in notes


def test_compile_check_rejects_before_evaluation():
    check = CompileCheck(required=('solve',))
    assert check('def solve():\n    return 1\n')[0].startswith('def solve')
    with pytest.raises(ProjectionError, match='SyntaxError'):
        check('def solve(:\n')
    with pytest.raises(ProjectionError, match='solve'):
        check('def other():\n    return 1\n')


def test_projection_registry_builds_from_config():
    register_projection('test.double_newline@1', lambda config: (lambda source: (source + '\n' * config['n'], 'padded')))
    projection = make_projection({'ref': 'test.double_newline@1', 'config': {'n': 2}})
    assert projection('x')[0] == 'x\n\n'
    assert make_projection({'ref': 'recursive_opt.projection.compile_check@1', 'config': {'required': ['f']}})('def f(): pass')[0]


def test_operator_projects_before_evaluation_and_keeps_editable_content():
    population = Population()
    population.add(Candidate('p', 'SCORE = 1.0\n# end', {'combined_score': 1.0}, 0))
    seen = []

    def evaluate(source):
        seen.append(source)
        return toy_evaluate(source)
    projection = lambda source: (source + '\nSCORE = SCORE + 10', 'bonus')  # noqa: E731
    llm = ScriptedLLM(['<<<<<<< SEARCH\n# end\n=======\nSCORE = SCORE + 0.5\n# end\n>>>>>>> REPLACE'])
    operator = PopulationOperator(llm, evaluate, system_message='T', mode='diff', retries=1, projections=[projection])
    from opto.features.recursive_opt.coevolution import Selection
    result = operator.run(Selection(population.get('p'), [], ''), population, iteration=1)
    child = result.candidate
    assert seen[0].endswith('SCORE = SCORE + 10')
    assert 'SCORE = SCORE + 10' not in child.content and child.metadata['deployable'].endswith('SCORE = SCORE + 10')
    assert child.metrics['combined_score'] == 11.5 and child.artifacts['projection'] == 'bonus'
    rejecting = PopulationOperator(ScriptedLLM(['<<<<<<< SEARCH\n# end\n=======\nSCORE = (\n# end\n>>>>>>> REPLACE']), evaluate, system_message='T', retries=1, projections=[CompileCheck()])
    failed = rejecting.run(Selection(population.get('p'), [], ''), population, iteration=2)
    assert failed.candidate is None and 'SyntaxError' in failed.error and len(seen) == 1  # rejected before evaluation


def test_engine_reports_deployable_best_program():
    solution = CountingLLM(lambda n, s, u: diff_add(1.0))
    config = CoevolutionConfig(**{**evox_preset(horizon=3, summaries=False, generate_labels=False, retries=1), 'trigger': 'never'})
    engine = CoevolutionEngine(config, solution_llm=solution, meta_llm=lambda s, u: '', evaluate=toy_evaluate, initial_source='SCORE = 1.0\n# end',
                               system_message='T', projections=[lambda source: (source + '\n# deployed', 'tag')])
    report = engine.run()
    assert report['best_source'].endswith('# deployed') and not report['best_editable_source'].endswith('# deployed')


def _capturing_llm(reply="=== DIVERGE ===\nd\n=== REFINE ===\nr"):
    seen = []
    def llm(system, user):
        seen.append((system, user))
        return reply
    return llm, seen


def test_generate_labels_default_prompt_is_unchanged():
    """packages=None must send exactly the pre-EXP26 native prompt, so EXP23 equivalence and past runs stay valid."""
    from opto.features.recursive_opt.coevolution.operator import LABEL_GENERATION_SYSTEM, generate_labels
    llm, seen = _capturing_llm()
    assert generate_labels(llm, 'TASK', 'EVAL') == {'diverge': 'd', 'refine': 'r'}
    assert seen == [(LABEL_GENERATION_SYSTEM, '## Problem\nTASK\n\n## Evaluator\nEVAL')]


def test_generate_labels_package_aware_mode_matches_stock_contract():
    from opto.features.recursive_opt.coevolution.operator import LABEL_GENERATION_SYSTEM, LIBRARY_RULES, generate_labels
    llm, seen = _capturing_llm()
    generate_labels(llm, 'TASK', 'EVAL', packages=('numpy', 'scipy'), initial_source='def f(): pass')
    system, user = seen[0]
    assert system == LABEL_GENERATION_SYSTEM + LIBRARY_RULES
    assert '## Available Packages in Environment\nnumpy\nscipy' in user
    assert '## Initial Program (reference implementation)\n```python\ndef f(): pass\n```' in user

    llm, seen = _capturing_llm()
    generate_labels(llm, 'TASK', packages=())
    assert seen[0][1].endswith('## Available Packages in Environment\nNo packages found')


def test_engine_forwards_label_packages_only_when_configured():
    import opto.features.recursive_opt.coevolution.engine as E
    calls = []
    real = E.generate_labels
    E.generate_labels = lambda *a, **k: calls.append(k) or {'diverge': 'd', 'refine': 'r'}
    try:
        for packages in (None, ('scipy',)):
            config = E.CoevolutionConfig(**{**E.evox_preset(horizon=1, summaries=False, generate_labels=True, retries=1),
                                            'label_packages': packages, 'trigger': 'never'})
            engine = E.CoevolutionEngine(config, lambda s, u: '', lambda s, u: '', lambda src: ({'combined_score': 0.0}, {}),
                                         'def f(): pass', 'T', feedback_llm=lambda s, u: '')
            try:
                engine.run()
            except Exception:
                pass  # only the label call matters here
    finally:
        E.generate_labels = real
    assert [c['packages'] for c in calls] == [None, ('scipy',)]
    assert all('initial_source' not in c for c in calls)  # stock parity: no initial program in label generation


def test_evox_preset_key_set_is_frozen_for_plan_fingerprints():
    """The control plane hashes the full engine config, so a new CoevolutionConfig field silently changes every
    preset-based plan fingerprint (it changed EXP25's). New fields must be added to _UNSET_OMITTED, or this set
    updated deliberately, knowing that existing fingerprints change."""
    from opto.features.recursive_opt.coevolution.engine import CoevolutionConfig, evox_preset
    assert set(evox_preset()) == {  # keys of the preset at fb9ad806, the code EXP25 ran
        'archive_seed', 'deployment', 'evaluator_timeout_s', 'generate_labels', 'horizon', 'improvement_threshold',
        'initial_policy', 'labels', 'language', 'max_solution_chars', 'meta_feed_errors', 'meta_num_context',
        'meta_parent', 'meta_retries', 'meta_system_prompt', 'num_context', 'num_previous_attempts', 'operator_mode',
        'patience', 'patience_ratio', 'proposer', 'proposer_memory', 'retries', 'rollback', 'score_key', 'seed',
        'strict_budget', 'summaries', 'trigger', 'window_scorer'}
    assert CoevolutionConfig(**evox_preset()).label_packages is None


def test_injected_labels_reach_the_solution_prompt():
    """EXP26's native_stocklabels arm injects stock-generated labels via config.labels. The uniform initial policy
    never selects a label, so prove the path with a policy that always picks 'diverge'."""
    from opto.features.recursive_opt.coevolution.engine import CoevolutionConfig, CoevolutionEngine, evox_preset
    always_diverge = UNIFORM_POLICY_SOURCE.replace('return parent, examples, ""', 'return parent, examples, "diverge"')
    prompts = []

    def solution_llm(system, user):
        prompts.append(f'{system}\n{user}')
        return ''
    config = CoevolutionConfig(**{**evox_preset(horizon=2, summaries=False, generate_labels=True, retries=1), 'trigger': 'never',
                                  'initial_policy': always_diverge, 'labels': {'diverge': 'INJECTED-DIVERGE', 'refine': 'INJECTED-REFINE'}})
    generated = []
    engine = CoevolutionEngine(config, solution_llm, lambda s, u: '', lambda src: ({'combined_score': 0.0}, {}), 'x = 1', 'T',
                               feedback_llm=lambda s, u: generated.append(1) or '')
    engine.run()
    assert not generated, 'labels were injected, so no label-generation call may happen'
    assert prompts and all('INJECTED-DIVERGE' in p and 'INJECTED-REFINE' not in p for p in prompts)
