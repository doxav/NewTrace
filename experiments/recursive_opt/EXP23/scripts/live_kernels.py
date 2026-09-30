"""Launch every live SkyDiscover job (A4 arms, A5, A6) as its own worker process, in parallel.

Needs OPENROUTER_API_KEY in the environment; workers use z-ai/glm-5.3-flash pinned to Novita.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT.parent / 'EXP22' / '.venv' / 'bin' / 'python')
ARMS = ('SD-FIXED', 'SD-EVOX', 'SD-EVOX-PAIRED', 'TRACE-PAIRED', 'AMORTIZED', 'TRACE-TASK')


def main() -> None:
    stamp = time.strftime('%Y%m%dT%H%M%S')
    base = ROOT / 'results' / 'live_kernels' / stamp
    jobs = []
    for task in ('prism', 'signal_processing'):
        for arm in ARMS:
            out = base / f'{task}__{arm}'
            out.mkdir(parents=True, exist_ok=True)
            log = (out / 'worker.log').open('w')
            jobs.append((task, arm, out, subprocess.Popen([PY, '-I', str(ROOT / 'livekernel' / 'worker.py'), '--task', task, '--arm', arm, '--out', str(out)], stdout=log, stderr=log)))
    for task, arm, out, process in jobs:
        process.wait()
        print(task, arm, 'exit', process.returncode, flush=True)
    summary = {}
    for task, arm, out, _ in jobs:
        path = out / 'result.json'
        r = json.loads(path.read_text()) if path.exists() else {'error': 'no result.json'}
        calls = r.get('skydiscover_calls', [])
        summary[f'{task}/{arm}'] = {k: r.get(k) for k in ('error', 'initial_score', 'final_best_score', 'best_evaluated', 'final_module_score', 'iterations', 'valid_candidates', 'wall_total_s', 'optimizer_calls_n', 'meta_failures')}
        summary[f'{task}/{arm}'].update(sky_calls=len(calls), sky_errors=sum(not c['ok'] for c in calls), roles=sorted({c['role'] for c in calls}),
                                         paired=[(e['paired_score'], e['promoted']) for e in r.get('paired_log', [])],
                                         policy_events=[(e['iteration'], e['activated']) for e in r.get('policy_events', [])])
    (base / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    sys.exit(main())
