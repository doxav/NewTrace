"""EXP27 Part D: full Trace runs with the causal cue hidden or shown, everything else = EXP26 `native` (EXP25 trace_exp24).

  --cue on   identical to EXP26 native (control)
  --cue off  `causal_fraction` is removed from the metrics the engine receives (so from every LLM-visible prompt);
             evaluations.jsonl still records it, so causality is audited exactly as before.
The plan is built by EXP26's runner unchanged, so the plan fingerprint equals EXP26 native's. Each LLM call also gets
a wall-clock deadline (hung OpenRouter requests otherwise stall a run for hours; see Part C).

Usage: python run_trace_cue.py --cue off --seed 42 --out DIR [--horizon 100] [--mock]
"""
import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('exp26_run_signal', HERE.parents[1] / 'EXP26' / 'scripts' / 'run_signal.py')
E26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E26)
E25 = E26.E25
HIDDEN = ('causal_fraction',)


def hide(evaluate):
    def wrapped(source):
        metrics, artifacts = evaluate(source)
        return {k: v for k, v in metrics.items() if k not in HIDDEN}, artifacts
    return wrapped


class DeadlineClient(E25.RoleClient):
    """RoleClient with a wall-clock deadline per request; a hung request is abandoned and retried."""
    deadline_s, attempts = 600.0, 4
    _pool = ThreadPoolExecutor(max_workers=16)

    def __call__(self, messages=None, **kwargs):
        for attempt in range(self.attempts):
            try:
                return self._pool.submit(super().__call__, messages, **kwargs).result(timeout=self.deadline_s)
            except FutureTimeout:
                with self.log.open('a') as stream:
                    stream.write(json.dumps({'role': self.role, 'deadline_exceeded': attempt}) + '\n')
        raise TimeoutError(f'LLM call exceeded {self.deadline_s}s on {self.attempts} attempts')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cue', choices=('on', 'off'), required=True)
    args, rest = parser.parse_known_args()
    if args.cue == 'off':
        audited = E25.audited
        E25.audited = lambda evaluate, out, keep_artifacts: hide(audited(evaluate, out, keep_artifacts))
    E25.RoleClient = DeadlineClient
    sys.argv = [sys.argv[0], '--arm', 'native', *rest]
    try:
        E26.main()
    finally:
        out = Path(rest[rest.index('--out') + 1])
        path = out / 'run_manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            manifest.update(experiment='EXP27', arm=f'trace_cue_{args.cue}', hidden_metrics=list(HIDDEN) if args.cue == 'off' else [],
                            runner='EXP26 run_signal.py --arm native, unchanged; evaluator-side metric hiding only')
            path.write_text(json.dumps(manifest, indent=1) + '\n')


if __name__ == '__main__':
    main()
