# Optimizer-program discovery — EXP-15

**Completed preregistered result: positive signal versus the seed; added value over
independent generation remains inconclusive.**

Can recursive_opt discover a deployable black-box policy that improves on its seed
and equal-budget independent LLM code search? FunSearch/OpenEvolve are not compared yet.

The artifact is one portable Python file:

```python
def propose(history, bounds, seed):
    # history contains only prior {"x": [...], "value": number} observations
    return next_point
```

The program receives no objective code, function name, hidden parameters, known
optimum or split identity. It cannot call an LLM. Each proposal runs in a fresh
subprocess, with deterministic replay and a two-second timeout per execution.

| Benchmark | Frozen design |
|---|---|
| Families | shifted Sphere, anisotropic Quadratic, Rosenbrock |
| Dimensions / bounds | 2 and 4; [-5,5] in each coordinate |
| Training | 6 instances: one per family/dimension combination |
| Validation | 6 different instances, same balance |
| ID holdout | 12 new instances: two per combination |
| Inner budget | 32 objective evaluations per trajectory |
| Primary metric | normalized best-so-far regret averaged over all 32 evaluations; lower is better |

Normalization uses an independent fixed uniform reference design. Family/dimension
strata have equal weight. The known optimum and normalization remain host-side.
No OOD result is planned.

A0 is the unchanged handwritten policy: uniform exploration mixed with decreasing
Gaussian perturbations around the incumbent. A1 generates eight independent
DeepSeek programs per outer seed, without earlier code, scores or feedback.
A2 generates eight updates through the production recursive_opt/Trace search path,
using previous code and training observations. Both final pools include the seed.

All generation uses OpenRouter `deepseek/deepseek-v4-flash-0731`, temperature .6,
top_p 1, 8,000 completion tokens, native low reasoning, concurrency 1 and no response
cache or automatic empty-response replacement. Outer seeds are 11, 23, 37, 41 and 53:
**80 completed response slots**, including invalid or empty responses. Each generative
arm has 17,280 search objective allocations and 1,920 holdout allocations. Shared
normalization preparation allocates 3,072 reference evaluations.
Equalized proposals do not imply equal realized token expenditure or cost.

Selection uses validation only. All five seeds' selections and the representative
are frozen before any holdout evaluation. If deployment fails, the trajectory
permanently switches to the seed with its actual history and remaining budget.

| Held-out primary result | A0 seed | A1 independent | A2 recursive |
|---|---:|---:|---:|
| Mean normalized regret AUC | 0.139579 | 0.121748 | 0.116680 |
| Median | 0.133280 | 0.126687 | 0.120232 |

The paired mean A2−A0 delta is **−0.022899**, with a 95% paired bootstrap interval
**[−0.042456, −0.003341]**: a positive signal under the registered rule.
The central A2−A1 delta is **−0.005068**, interval **[−0.037728, +0.024551]**:
**inconclusive**. A2 beats A1 in three pairs and loses in two. The five outer seeds,
not individual tasks or trajectory points, are the replication units; uncertainty
is fragile. A1 has better mean final regret (0.010110 versus A2's 0.012166) and
target attainment (81.7% versus 76.7%). No novelty, recursion-depth or amortization
claim follows.

All 80 responses remain. A1 has 9/40 invalid candidates; A2 has 12/40: seven missing
sources, four syntax errors and one program using unseeded randomness. A2 selects
the unchanged seed in two of five replications. All 180 selected-policy holdout
trajectories complete; none needs deployment fallback. There are 81 transport
attempts, 573,191 reported tokens and USD 0.086824 reported cost, excluding unknown
billing for one timed-out attempt. Actual shared execution uses 28,800 objective
calls and 57,624 subprocesses; cached and unused allocations remain explicit.

The representative, selected on validation before holdout, is seed41/slot5:
**Halton exploration plus incumbent perturbations and a regularized quadratic fit**.
Its lineage is seed → slot4 → slot5. It clips proposals to bounds but does not check
that the fitted stationary point is a minimum. Its exact evaluated source is
[A2_seed_41.py.gz](../../EXP15/results/selected/A2_seed_41.py.gz), SHA-256
`1684f91acdc36c0ca6aac70afeb9cc2c4eed7ab847926d5880590e059266abb7`.
Decompress it to optimizer.py without editing. All selected sources and attempted
diffs are in [the export index](../../EXP15/results/selected/index.json).

The separate four-response engineering pilot is excluded from confirmation.
Full per-seed results, invalid records, accounting and limitations are in
[EXP15_REPORT.md](EXP15_REPORT.md); raw data recompute exactly and 772 offline tests pass.

The sanitized subprocess boundary is **not an operating-system security sandbox**
and does not establish adversarial filesystem confinement.

Proposed next question:

> Can we plug your FunSearch/OpenEvolve search into this exact optimizer.py contract and evaluator, keeping the artifact, tasks and budgets fixed?

Patrick is **not being asked to adopt recursive_opt infrastructure**. The evaluator
is engine-independent; a thin source-producing adapter can use the same contract
and selection protocol in a new preregistered equal-budget experiment. This brief
has not been sent.
