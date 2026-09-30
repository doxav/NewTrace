"""Run the recursive_opt ``coevolution`` engine (control plane v2, ``evox_preset``) on the toy task.

Runs against ~/code/Trace (put it first on PYTHONPATH). Same scripted LLMs as run_stock.py.
"""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import script  # noqa: E402
from opto.features.recursive_opt import spec as S  # noqa: E402
from opto.features.recursive_opt.coevolution import control_plane as CP  # noqa: E402
from opto.features.recursive_opt.coevolution import evox_preset  # noqa: E402
from opto.features.recursive_opt.coevolution.feedback import BATCH_SYSTEM, PROBLEM_SYSTEM, STATS_SYSTEM  # noqa: E402

_spec = importlib.util.spec_from_file_location('toy_evaluator', HERE / 'toy' / 'evaluator.py')
TOY = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TOY)
CALLS: Counter = Counter()


def toy_evaluate(source):
    metrics = TOY.score_source(source)
    return metrics, {}


def factory(profile, role):
    def client(messages=None, **kwargs):
        system = messages[0]['content'] if messages and messages[0]['role'] == 'system' else ''
        if role == 'forward':
            CALLS['solution'] += 1
            return script.solution_reply(CALLS['solution'])
        if role == 'optimizer':
            CALLS['meta'] += 1
            return '```python\n' + script.v2_policy_source(script.meta_policy(CALLS['meta'])) + '\n```'
        kind = {STATS_SYSTEM: 'stats_insight', PROBLEM_SYSTEM: 'problem_context', BATCH_SYSTEM: 'batch_summary'}.get(system, 'feedback_other')
        CALLS[kind] += 1
        return 'summary'
    return client


def normalize(events):
    out = []
    for e in events:
        if e['type'] == 'iteration':
            event = {'type': 'iteration', 'iteration': e['iteration'], 'attempts': e['attempts'], 'ok': e['error'] is None}
            if e['error'] is None:
                event.update(parent_iteration=e['parent_iteration'], label=e['label'], context_iterations=e['context_iterations'], child_score=round(e['child_score'], 9))
            out.append(event)
        elif e['type'] == 'trigger':
            out.append({'type': 'trigger', 'iteration': e['iteration'], 'best': round(e['best'], 9)})
        elif e['type'] == 'proposal':
            out.append({'type': 'proposal', 'attempts': e['attempts'], 'ok': e['ok']})
        elif e['type'] == 'deploy':
            out.append({'type': 'deploy', 'ok': e['ok']})
        elif e['type'] == 'rollback':
            out.append({'type': 'rollback'})
        elif e['type'] == 'scored':
            out.append({'type': 'scored', 'score': round(float(e['metrics']['combined_score']), 9)})
    return out


def main(horizon: int, out: Path) -> None:
    CP.register()
    S.register_evaluator('exp23.equivalence.toy_evaluator@1', CP.program_evaluator(toy_evaluate))
    config_text = (HERE / 'toy' / 'config.yaml').read_text()
    system_message = config_text.split('system_message: "', 1)[1].split('"', 1)[0]
    raw = CP.coevolution_spec(level_id='evox_toy', program=(HERE / 'toy' / 'initial_program.py').read_text(), evaluator_ref='exp23.equivalence.toy_evaluator@1',
                              system_message=system_message, engine_config=evox_preset(horizon=horizon, summaries=True, generate_labels=False),
                              problem_description=system_message, evaluator_context=(HERE / 'toy' / 'evaluator.py').read_text())
    plan = S.compile_plan(raw)
    (out / 'v2_raw_spec.json').write_text(json.dumps(raw, indent=1, default=str) + '\n')
    (result,) = S.execute_plan(plan, {'llm_factory': factory})
    if result.status != 'success':
        raise SystemExit(f'control plane run failed: {result.error}')
    report = result.metadata['report']
    trace = {'framework': 'trace-recursive_opt-coevolution', 'events': normalize(report['events']), 'curve': [round(x, 9) for x in report['curve']],
             'best_score': round(report['best_score'], 9), 'calls': dict(CALLS), 'feedback_calls': report['feedback_calls'], 'meta_failures': report['meta_failures'],
             'plan_fingerprint': plan.explain()['fingerprint']}
    (out / 'v2_trace.json').write_text(json.dumps(trace, indent=1) + '\n')
    print(json.dumps({k: trace[k] for k in ('best_score', 'calls', 'feedback_calls', 'meta_failures')}))


if __name__ == '__main__':
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'out'
    out.mkdir(parents=True, exist_ok=True)
    main(horizon, out)
