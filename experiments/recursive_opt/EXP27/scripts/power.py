"""Power of the EXP27 Part C design by simulation (prompt-clustered binary outcome).

Each prompt has its own base logit (normal, sd = HET); each factor multiplies the odds by OR. A main effect is
declared when the 95% prompt-bootstrap CI of the pooled difference excludes 0, and the point estimate is >= 0.05
(the protocol rule). Prints power for each (prompts, samples per cell) design.
"""
import numpy as np

rng = np.random.default_rng(0)


def simulate(prompts, samples, base, odds_ratio, het, reps=400, boot=400):
    hits = 0
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    for _ in range(reps):
        b = logit(base) + rng.normal(0, het, prompts)
        # factor A (the tested one) on/off, factor B (other) on/off; B has no effect here
        y = {(a, c): rng.binomial(samples, sig(b + a * np.log(odds_ratio))) for a in (0, 1) for c in (0, 1)}
        diff_p = (y[1, 0] + y[1, 1] - y[0, 0] - y[0, 1]) / (2 * samples)  # per-prompt difference
        est = diff_p.mean()
        bs = np.array([diff_p[rng.integers(0, prompts, prompts)].mean() for _ in range(boot)])
        lo = np.quantile(bs, 0.025)
        hits += est >= 0.05 and lo > 0
    return hits / reps


if __name__ == '__main__':
    for base, oratio in ((0.03, 3.6), (0.05, 2.5), (0.10, 2.0)):
        for het in (0.7, 1.2):
            row = []
            for prompts, samples in ((12, 6), (12, 10), (16, 10), (20, 10), (24, 10)):
                row.append(f'{prompts}x{samples}:{simulate(prompts, samples, base, oratio, het):.2f}')
            print(f'base={base} OR={oratio} het={het}  ' + '  '.join(row), flush=True)
