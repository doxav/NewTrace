# Optimizer-program discovery — EXP-15

**Draft: confirmation is running. Results and the representative program are pending.**

Can iterative code generation through recursive_opt discover a deployable black-box
optimization policy that improves on its starting policy and on independent LLM
code search with the same proposal budget? This first experiment compares those
procedures. It does not compare FunSearch/OpenEvolve yet.

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
**80 completed response slots**, including invalid or empty responses. Transport
retries are counted separately. Each generative arm has 17,280 search objective
allocations and 1,920 holdout allocations; actual calls and cache hits are reported
separately. Shared normalization preparation allocates 3,072 reference evaluations.
Equalized proposals do not imply equal realized token expenditure or cost.

Selection uses validation only. All five seeds' selections and the representative
are frozen before any holdout evaluation. An invalid search may retain the seed.
If a selected program fails during deployment, the trajectory permanently switches
to the seed with its actual history and remaining budget. Candidate invalidity and
fallback frequency are reported alongside the deployment metric.

**Confirmatory results and uncertainty: pending.** The registered comparisons are
A2−A0 and A2−A1; negative regret deltas favor A2. A paired bootstrap uses the five
outer seeds as its replication units. Its uncertainty is fragile at this sample
size. A null, negative or fallback-dominated result is a valid scientific outcome.
No conclusion about novelty, additional recursion depth or amortization follows.

**Validation-selected optimizer example: pending.** The exact evaluated source,
its hash and lineage will accompany the final result. No representative will be
chosen by holdout performance. The separate engineering pilot retained the seed
in both search arms; two of four pilot responses yielded eligible programs. Pilot
results are excluded from confirmation.

The execution boundary sanitizes the credential environment and separates the
candidate API from objective data. It is **not an operating-system security sandbox**
and does not establish adversarial filesystem confinement.

Proposed next question:

> Can we plug your FunSearch/OpenEvolve search into this exact optimizer.py contract and evaluator, keeping the artifact, tasks and budgets fixed?

Patrick is **not being asked to adopt recursive_opt infrastructure**. The evaluator
is engine-independent; a thin source-producing adapter can use the same contract
and selection protocol in a new preregistered equal-budget experiment. This brief
has not been sent.
