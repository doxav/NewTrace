"""EXP28 recursive_opt arms: EXP27 Part D's cue-hidden Trace configuration (EXP26 `native`) plus one exploration change.

  brief   CoevolutionConfig.meta_brief = EVOX_POLICY_BRIEF: Trace's policy proposer (OptoPrimeV2, O1) gets EvoX's
          label / context / diversity rules as its instruction (policy-level fix)
  guard   CoevolutionConfig.diverge_guard = {'patience': 5, 'num_context': 0}: after 5 iterations without improvement the
          next selection is relabelled DIVERGE with no context, whatever the policy chose (engine-level fix)

Usage: python run_coevo_variation.py --arm brief --seed 42 --out DIR [--horizon 100] [--mock]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp27_cue', HERE.parents[1] / 'EXP27' / 'scripts' / 'run_trace_cue.py')
CUE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CUE)
E26, E25 = CUE.E26, CUE.E25
from opto.features.recursive_opt.coevolution.feedback import EVOX_POLICY_BRIEF  # noqa: E402

ARMS = {'brief': {'meta_brief': EVOX_POLICY_BRIEF}, 'guard': {'diverge_guard': {'patience': 5, 'num_context': 0}}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=sorted(ARMS), required=True)
    args, rest = parser.parse_known_args()
    build = E26.build_spec

    def build_spec(*a, **k):
        raw = build(*a, **k)
        raw['levels'][0]['engine']['config'].update(ARMS[args.arm])
        raw['levels'][0]['id'] = f'exp28_signal_{args.arm}'
        return raw
    E26.build_spec = build_spec
    audited = E25.audited  # cue hidden, exactly as EXP27 Part D
    E25.audited = lambda evaluate, out, keep_artifacts: CUE.hide(audited(evaluate, out, keep_artifacts))
    E25.RoleClient = CUE.DeadlineClient
    sys.argv = [sys.argv[0], '--arm', 'native', *rest]
    try:
        E26.main()
    finally:
        out = Path(rest[rest.index('--out') + 1])
        path = out / 'run_manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            manifest.update(experiment='EXP28', arm=f'coevo_{args.arm}', hidden_metrics=list(CUE.HIDDEN), engine_overrides=ARMS[args.arm],
                            runner='EXP26 run_signal.py --arm native + EXP27 cue hiding + EXP28 engine override')
            path.write_text(json.dumps(manifest, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
