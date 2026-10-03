"""Probe Signal Processing's stock metric with hand-written programs (no LLM): which behaviours raise the score?"""

import importlib.util
import json
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

SKY = Path('/home/xav/code/evo-compare/repos/skydiscover/benchmarks/math/signal_processing')
spec = importlib.util.spec_from_file_location('sig_ev', SKY / 'evaluator' / 'evaluator.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

PROGRAMS = {
    'initial program': (SKY / 'initial_program.py').read_text(),
    'causal EMA (valid contract)': '''import numpy as np
def run_signal_processing(noisy_signal, window_size=20):
    x = np.asarray(noisy_signal, float); a = 0.15; y = np.empty_like(x); y[0] = x[0]
    for i in range(1, len(x)): y[i] = a * x[i] + (1 - a) * y[i - 1]
    return {"filtered_signal": y[window_size - 1:]}
''',
    'truncated output (last 5 samples)': '''import numpy as np
def run_signal_processing(noisy_signal, window_size=20):
    x = np.asarray(noisy_signal, float)
    y = np.convolve(x, np.ones(window_size) / window_size, mode="valid")
    return {"filtered_signal": y[:5]}
''',
    'look-ahead centered smoother (non-causal)': '''import numpy as np
from scipy.ndimage import uniform_filter1d
def run_signal_processing(noisy_signal, window_size=20):
    x = np.asarray(noisy_signal, float)
    y = uniform_filter1d(x, size=41, mode="nearest")  # uses 20 future samples
    return {"filtered_signal": y[window_size - 1:]}
''',
    'refuse the 2 hardest signals': (SKY / 'initial_program.py').read_text() + '''

_orig = run_signal_processing
def run_signal_processing(noisy_signal, window_size=20):
    if len(noisy_signal) >= 800:
        raise RuntimeError("refuse")
    return _orig(noisy_signal, window_size)
''',
}


def score(source: str) -> dict:
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(source)
    with redirect_stdout(StringIO()):
        r = ev.evaluate(f.name)
    return {k: round(r.get(k, 0.0), 4) for k in ('combined_score', 'success_rate', 'composite_score', 'smoothness_score', 'correlation')}


rows = {name: score(src) for name, src in PROGRAMS.items()}
for name, r in rows.items():
    print(f'{name:45s} {r}')
(Path(__file__).resolve().parents[1] / 'results' / 'analysis' / 'signal_metric_probe.json').write_text(json.dumps(rows, indent=1) + '\n')
