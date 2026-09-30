# S2 — observed proposal budgets, availability and late improvements

Read-only retrospective analysis of the ten preserved EXP-15 train/validation
pools. No holdout file, model call or new objective evaluation is used. All eight
completed response slots and all five outer seeds remain represented per arm.
The [protocol](S2_PROTOCOL.md), [script](audit_budget.py) and [complete output](results.json)
include input-file hashes, every prefix value and every retained seed.

| Observed proposal prefix N | A1 best validation AUC, mean | A2 best validation AUC, mean | A1 eligible generated, mean | A2 eligible generated, mean | A1 seed selections/5 | A2 seed selections/5 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | .122891 | .136940 | .8 | .8 | 2 | 4 |
| 2 | .113841 | .136940 | 1.6 | 1.6 | 0 | 4 |
| 3 | .090474 | .136940 | 2.6 | 2.0 | 0 | 4 |
| 4 | .090474 | .118997 | 3.4 | 2.8 | 0 | 2 |
| 5 | .090474 | .118997 | 4.0 | 3.6 | 0 | 2 |
| 6 | .090474 | .103114 | 4.6 | 4.2 | 0 | 2 |
| 7 | .089273 | .099353 | 5.4 | 4.8 | 0 | 2 |
| 8 | .088602 | .099353 | 6.2 | 5.6 | 0 | 2 |

These validation curves are monotone **by construction** because the candidate
pool only expands. They are not new search experiments or evidence that more
proposals improve independent performance. A1 still has the smaller final mean
validation AUC. From N4 to N8, the mean decreases .001872 for A1 and .019645 for
A2; this describes those particular retained candidates, not an expected return
from doubling the budget in a future run.

First prefix attaining the final validation minimum by outer seed:

| Outer seed | A1 | A2 |
|---:|---:|---:|
| 11 | 2 | 1, unchanged seed |
| 23 | 3 | 1 |
| 37 | 2 | 1, unchanged seed |
| 41 | 1 | 6 |
| 53 | 8 | 7 |

The two later selected A2 artifacts appear on responses6 and7. Thus a pilot with
only two or four proposals does not exercise the refinement depth that produced
those observed artifacts. This supports retaining enough iterations in an actual
production-path pilot. It does **not** project an advantage at N16, and does not
show that increasing N alone remedies the mechanism.

## Availability is different from quality and diversity

At N1,1/5 pools have no eligible generated candidate in each arm. From N2 onward,
all ten pools contain at least one eligible generated program. At N8 the A2 seed
is selected in2/5 runs despite six eligible alternatives in outer11 and four in
outer37. The claim “A2 fell back because no program was available” is false for
these completed searches. It selected the seed by the registered validation rule.

Across N8 runs, mean eligible counts are6.2 A1 and5.6 A2. After excluding exact
seed-source copies, distinct source counts average5.8 A1 and5.6 A2. After additionally
grouping candidates whose **complete observed training trajectories** are identical
and excluding the seed's observed behavior, both arms average **5.4 distinct
observed behaviors**. A1 has exact and cosmetic seed copies; A2 has two distinct
sources implementing the same observed midpoint behavior in outer23. This is a
finite-fixture comparison, not a proof of global algorithmic equivalence. It
nevertheless shows why source validity alone is an incomplete account of effective
search diversity.

The empirical empty-pool rate at N8 is0/5 per arm. That does not demonstrate a zero
population failure probability: under an independent outer-run binomial model, the
one-sided exact95% upper bound is45.1%. If one further, unjustifiably assumed IID
responses at the observed invalid fractions, the formula `p_invalid**8` would give
6.57e-6 for A1 and6.56e-5 for A2. Those are **hypothetical arithmetic illustrations,
not forecasts**, because responses are adaptive, routes drift, invalidity clusters,
and all rates were estimated from small samples. Use actual availability evidence
to interpret EXP-15; do not use the IID illustration to promise future reliability.

Verification: two tests pass, covering invalid response-slot preservation,
eligibility versus seed selection, and deterministic ties. Black and Ruff pass.
The broader S0/S1 plus reused benchmark/contract regression reports49 passed in8.38s.
