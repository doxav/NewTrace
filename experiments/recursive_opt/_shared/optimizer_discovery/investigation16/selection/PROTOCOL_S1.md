# EXP-16 / S1 — fresh fixed-bank selection and local-randomness diagnostic

Registered before any S1 trajectory. This stage tests whether more task instances
or local optimizer seeds improve reliable policy selection. It is a fixed-policy
diagnostic, not live A1/A2 generation, and cannot establish recursive feedback gain.
EXP-15 selections supply policies chosen previously without using any S1 result.

## Fixed artifact bank and evaluator

Use the exact unique sources of all EXP-15 validation-selected A1/A2 policies,
including the unchanged seed, plus two prespecified diagnostics: always midpoint
and independent uniform sampling with `random.Random(seed+len(history))`.
This yields eleven source hashes: five A1 selections, three distinct generated
A2 selections, the seed, midpoint and uniform. Preserve exact source bytes in
`sources/<hash>.py.gz`; record provenance and deterministic order in `freeze.json`.
Order: seed first, then first-seen unique selections in outer-seed order
11,23,37,41,53 and arm order A1,A2, then midpoint and uniform. Lowest bank position
breaks exact score ties. No candidate editing, repair, extra LLM call or tuning.

Reuse EXP-15's `benchmark.evaluate`, source screening, deterministic subprocess
replay, timeout2s and budget32. Candidate API remains history/bounds/local seed.
No extra task identity or normalization is exposed. Environment/source hashes and
the exact G.fresh_tasks/G.local_seed function text are frozen before evaluation.
This source bank is not asserted novel, optimal, representative of the literature,
or an unbiased sample of all possible optimizers.

## New tasks and randomness

Reconstruct tasks using `generation.fresh_tasks("S1","train",4)` (24 tasks) and
`fresh_tasks("S1","audit",2)` (12 tasks), six balanced family/dimension strata.
Check semantic disjointness from EXP-15 pilot/confirmation and between both S1 splits.
Use four separate local-seed blocks [16101,16102,16103,16104], with the stable
`generation.local_seed("S1", block, task)` derivation shared across policies.
Thus 11×36×4=1,584 allocated trajectories, 50,688 objective evaluations and up to
101,376 subprocess executions for valid deterministic trajectories. Reference
normalization: 36×128=4,608 shared benchmark-preparation objective calls, separately
counted. Four offline worker threads maximum; live generation is in another process.

Train every bank policy on all24 training tasks and all4 local seeds. Each exact
result gets an immutable file with full raw observations/errors/captured output,
source/task/local identity and timestamps. Completed trajectories are never replaced.
Resume checks freeze/config/source integrity and skips completed files. Interrupted,
unfinished local trajectories may replay deterministically under their same IDs;
they are not model calls and cannot change a completed outcome. Retain failure
records; do not recycle unused allocations. An evaluator or trusted-seed defect
blocks the stage rather than becoming a policy score.

## Selection schedule, frozen before independent audit evaluation

Compare every Cartesian configuration:

- instances per stratum m∈{1,2,4};
- local seeds per task k∈{1,2,4}.

Generate200 paired subsample permutations using `random.Random(161516)`. For each
draw independently shuffle the four training-instance indices within each of six
strata, then shuffle the four local-seed-block indices once. For configuration
(m,k), take the first m instance indices per stratum and first k local indices.
All policies receive identical selected rows. Nesting within a draw makes size
comparisons paired. Preserve these permutations in the pre-evaluation freeze.

Select the eligible policy with the smallest equal-stratum mean normalized anytime
regret-AUC, averaging local seeds within instance, then instances within stratum,
then strata. Eligibility requires every selected training trajectory to be valid;
invalid rows have no numerical regret. The trusted seed is always in the bank and
must be valid. Persist all1,800 selections and their source hashes after training,
before **any** independent audit trajectory is opened. Do not use audit to alter
the bank, subset schedule, selection rule, sample size or outcome interpretation.

Evaluate every bank policy on12 audit tasks×4 local seeds only after the global
selection freeze. A selected policy's audit failure uses EXP-15's permanent seed
fallback on the accumulated history and remaining budget. Apply that same deployment
rule to all bank audit evaluations. Report candidate-only validity/fallback separately.
The fixed bank is fully audited to define an empirical reference; no audit-selected
policy becomes a new confirmatory artifact or a retrospectively changed selection.

## Frozen descriptive analysis

For each of nine panel configurations report selected-policy frequencies, average
and median independent-audit AUC of the200 selections, and excess above the lowest
audit AUC among the fixed eleven policies. The latter is a finite-bank, finite-audit
reference, not the unknown true optimum or a novel winner. Report paired changes
as m/k increase; no rule assumes they must improve performance. Compare training
ranking against independent-audit ranking with tie-aware Spearman correlation.

Describe task and local variance separately within each policy/stratum using the
4×4 training grid: mean variance across local seeds within each task; variance of
four task means; and raw method-of-moments task component
`var(task_means) − mean(within_task_variance)/4`. Do not silently clip negative
variance-component estimates; these indicate imprecision. Report the unchanged
seed separately. These small-grid estimates are diagnostics, not population CIs.

The200 subsets reuse one finite grid and are **not200 independent replications**.
Report their dispersion as selection sensitivity, not confidence intervals on a
population treatment effect. All analyses are exploratory evidence for choosing
measurement design, followed by fresh live intervention and later confirmation.

Include raw call/unused-allocation counts, invalid and fallback trajectories,
source-hash verification, timestamps proving the audit barrier, runtime, commands
and tests. No EXP-15 artifacts, sources or metrics may change.
