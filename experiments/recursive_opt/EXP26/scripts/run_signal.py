"""EXP26 native arms: EXP25's ``trace_exp24`` configuration with only the label source varied.

Arms (100 solution calls, strict budget, seeds, transport retries up to one hour):
  native              native label generator (EXP25 baseline, re-run in the same window)
  native_pkg          + label_packages from stock SkyDiscover's own get_available_packages()
  native_stocklabels  labels drawn by stock's own generator (gen_stock_labels.py), injected via `labels`
The configuration is built by EXP25's runner itself (imported, not copied), so nothing else can drift.
Every LLM call is logged with full content in transcripts.jsonl; EXP25 logged metadata only.

Usage (EXP22 venv; OPENROUTER_API_KEY unless --mock):
  python run_signal.py --arm native_pkg --seed 42 --out DIR [--labels FILE] [--horizon 100] [--mock]
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP26 = HERE.parent
_spec = importlib.util.spec_from_file_location('exp25_run_signal', EXP26.parent / 'EXP25' / 'scripts' / 'run_signal.py')
E25 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E25)
W, S, CP = E25.W, E25.S, E25.CP

ARMS = ('native', 'native_pkg', 'native_stocklabels')
BASE_ARM = 'trace_exp24'


def stock_packages() -> list:
    """Exactly what stock EvoX puts in its label prompt: same function, same problem directory."""
    from skydiscover.optimize.search.evox.utils.variation_operator_generator import get_available_packages
    return get_available_packages(problem_dir=str(W.SKY / 'evaluator'))


class Transcribed:
    """Wrap any client and append role, messages and response to ``path``."""

    def __init__(self, client, role: str, path: Path) -> None:
        self.client, self.role, self.path = client, role, path

    def __call__(self, messages=None, **kwargs):
        response = self.client(messages=messages, **kwargs)
        with self.path.open('a') as stream:
            stream.write(json.dumps({'role': self.role, 'messages': messages, 'response': response}) + '\n')
        return response


def build_spec(arm: str, seed: int, horizon: int, offline: bool, labels: dict = None) -> dict:
    raw = E25.build_spec(BASE_ARM, seed, horizon, offline)
    level = raw['levels'][0]
    level['id'] = f'exp26_signal_{arm}'
    level['objective']['evaluator_ref'] = f'exp26.signal.{arm}@1'
    config = level['engine']['config']
    if arm == 'native_pkg':
        config['label_packages'] = stock_packages()
    elif arm == 'native_stocklabels':
        if not labels or not labels.get('diverge') or not labels.get('refine'):
            raise ValueError('native_stocklabels needs --labels with non-empty diverge and refine')
        config['labels'] = {'diverge': labels['diverge'], 'refine': labels['refine']}
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--provider', default='novita')
    parser.add_argument('--labels', help='stock label draw (JSON) for native_stocklabels')
    parser.add_argument('--out', required=True)
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    CP.register()
    evaluator = E25.GuidedEvaluator(E25.audited(W.evaluate, out, keep_artifacts=True), score_key='guided_score', source_key='valid_score')
    S.register_evaluator(f'exp26.signal.{args.arm}@1', CP.program_evaluator(evaluator))
    labels = json.loads(Path(args.labels).read_text()) if args.labels else None
    raw = build_spec(args.arm, args.seed, args.horizon, offline=args.mock, labels=labels)
    plan = S.compile_plan(raw)
    config = raw['levels'][0]['engine']['config']
    manifest = {'experiment': 'EXP26', 'arm': args.arm, 'seed': args.seed, 'horizon': args.horizon, 'model': E25.MODEL, 'provider': args.provider,
                'mock': args.mock, 'base_configuration': f'EXP25 {BASE_ARM}', 'label_packages': config.get('label_packages'),
                'labels_file': args.labels, 'plan_fingerprint': plan.explain()['fingerprint'], 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'coevolution_sha256': E25.sha256_tree(E25.TRACE_ROOT / 'opto' / 'features' / 'recursive_opt' / 'coevolution'),
                'signal_adapter_sha256': E25.sha256_tree(E25.EXP25 / 'signal')}
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    (out / 'raw_spec.json').write_text(json.dumps(raw, indent=1, default=str) + '\n')
    events_path, transcripts = out / 'events.jsonl', out / 'transcripts.jsonl'

    def on_event(event):
        with events_path.open('a') as stream:
            stream.write(json.dumps(event, default=str) + '\n')
    inner = E25.mock_factory if args.mock else (lambda profile, role: E25.RoleClient(role, args.provider, out / 'calls.jsonl'))

    def factory(profile, role):
        return Transcribed(inner(profile, role), role, transcripts)
    started = time.time()
    (result,) = S.execute_plan(plan, {'llm_factory': factory, 'capture': {'coevolution_on_event': on_event}})
    report = dict(result.metadata.get('report') or {}) if result.metadata else {}
    summary = {'status': result.status, 'error': result.error, 'arm': args.arm, 'seed': args.seed, 'wall_s': round(time.time() - started, 1),
               'best_metrics': report.get('best_metrics'), 'solution_attempts': report.get('solution_attempts'), 'llm_calls': report.get('llm_calls'),
               'feedback_calls': report.get('feedback_calls'), 'meta_failures': report.get('meta_failures')}
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    (out / 'labels.json').write_text(json.dumps(report.get('labels') or {}, indent=1) + '\n')
    (out / 'best_program.py').write_text(report.get('best_source') or '')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
