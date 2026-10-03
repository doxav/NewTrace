"""Independent check of prism_exact.py: exhaustive enumeration of all partitions (restricted growth strings)
of the models into at most gpu_num GPUs, for the 5-GPU cases (10 models)."""

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ev', '/home/xav/code/evo-compare/repos/skydiscover/benchmarks/ADRS/prism/evaluator/evaluator.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
saved = json.loads((HERE.parent / 'results' / 'analysis' / 'prism_exact_optimum.json').read_text())['per_case']


def brute(gpu_num, models):
    items = [(m.req_rate / m.slo, m.model_size) for m in models]
    best = [float('inf')]
    load, size = [0.0] * gpu_num, [0] * gpu_num

    def rec(k, used):
        if k == len(items):
            value = max((load[g] / (80 - size[g]) for g in range(used)), default=0.0)
            best[0] = min(best[0], value)
            return
        w, s = items[k]
        for g in range(min(used + 1, gpu_num)):
            if size[g] + s < 80:
                load[g] += w
                size[g] += s
                if max(load[h] / (80 - size[h]) for h in range(max(used, g + 1))) < best[0]:
                    rec(k + 1, max(used, g + 1))
                load[g] -= w
                size[g] -= s
    rec(0, 0)
    return best[0]


checked = []
for index, (gpu_num, models) in enumerate(ev.generate_test_gpu_models()):
    if gpu_num == 5:
        value = brute(gpu_num, models)
        checked.append((index, value, saved[index]))
        print(f'case {index}: brute force {value:.10f}  bisection solver {saved[index]:.10f}  diff {abs(value - saved[index]):.2e}', flush=True)
print(json.dumps({'cases_checked': len(checked), 'max_abs_diff': max(abs(a - b) for _, a, b in checked)}))
