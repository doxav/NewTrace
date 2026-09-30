# EXP-16 / S0 — retrospective statistical and evidence audit

All findings here are **exploratory descriptions of already observed EXP-15**.
They do not replace its registered results, select a new winner, or establish that
any proposed change will make feedback beat independent generation. No new model
or objective calls were made. `audit.py` reproduces `diagnostics.json` from 256
preserved raw files; their exact hashes are included in that output.

## What the inconclusive result does and does not say

A2−A1 mean regret-AUC is −0.005068 with sample paired SD 0.038660 and SE 0.017289.
The registered paired-bootstrap interval is [−0.037728,+0.024551]. Removing outer
seed53 changes the mean to **+0.010065**, favoring A1. Other leave-one-out means
remain negative. This is not evidence of a reliably zero effect, or of reliable
feedback superiority. The exploratory exact two-sided sign-flip p-value is .8125.
This procedure was not the registered decision rule; it does not retract the
registered interpretation. Even five differences all sharing one sign have a
minimum two-sided sign-flip p-value of 2/32=.0625. The A2−A0 exploratory p-value is
.25 because two pairs are exactly zero; EXP-15's registered bootstrap rule still
reports a positive signal versus A0. These distinctions illustrate fragile n=5
inference and should prevent “statistically significant” wording.

For planning only, plugging this very uncertain SD into the normal approximation
`n=(z_.975+z_.8)^2 SD^2/effect^2` yields about 470,118,30,14 pairs for absolute AUC
effects .005,.01,.02,.03 respectively. These are **not measured future gains or
reliable sample-size guarantees**. Simply changing n from5 to7 is unlikely to
clarify an effect as small and variable as the observed central contrast. More
accurate within-search evaluation may reduce variance before buying many outer
replications; S1 tests that separately.

## Generation is constrained, but the cap is not a sufficient relative explanation

Sixteen of80 responses finish `length` with no usable code: A1 9/40, A2 7/40.
Reasoning consumes 87.66% of A1 completion tokens and 87.34% of A2 completion
tokens. Increasing the cap is a credible throughput intervention, requiring fresh
paired requests. A2 actually loses fewer slots to length than A1, so these counts
alone cannot explain why A2 failed to establish added value.

The other four A2 source failures are actual mismatched parentheses/brackets,
with `stop` finishes at 4,418,5,531,5,593,7,071 completion tokens. They are not
cap-truncated code. One further A2 program fails deterministic replay due to
unseeded global randomness. A larger token allowance does not automatically fix
these faults. Eligible fractions remain A1 31/40, A2 28/40.

Routing is another competing explanation, not a demonstrated cause. DigitalOcean
returns four length failures in four calls (three A1, one A2); Baidu accounts for
39 calls (17 A1,22 A2) and28 eligible candidates. Fifteen upstream identities and
small, nonrandom provider subgroups prevent attributing differences to provider
quality. Test routing only prospectively and symmetrically.

Two exact seed-source copies occur in **A1**, at23/slot01 and37/slot00. A2 has zero
exact seed copies. A2/23/slots03 and04 have different sources but identical
midpoint behavior. Source diversity therefore overstates behavioral diversity.

## There is feedback, but it omits useful information and the optimized criterion

All40 A2 feedback payloads are complete JSON: 2,894–6,001 characters, below12,000.
There was no character-truncation failure. Every current-parent panel contains
six tasks. The “at least6/7 examples” hypothesis must distinguish number of tasks
from useful observations within each task: **only four of32 observations per task
are shown**, the first two and last two. Across240 current-parent task records,
167 (69.6%) omit the actual incumbent's point from those observations. A scalar
best value alone remains present. Across the35 previous-attempt panels,76 of144
valid task records likewise omit the incumbent point.

The median ratio between largest and smallest task `best_observed_value` within
a current-parent panel is9,570; the maximum is approximately5.09e17 (near-zero
values create very large ratios). These are raw, incomparable objective scales.
The model sees neither normalized AUC nor a description of the anytime criterion.
The common prompt says to minimize the function in32 evaluations; feedback gives
final best values and sparse endpoint observations. The production engine ranks
by an anytime metric that the model is not told to optimize. This is an explicit
information-design defect to ablate, not proof of a consequent performance gain.

Seven of35 previous-attempt feedback panels are exactly identical to the current
parent's panel (the preceding improvement was accepted). The remaining feedback
does not include the rejected attempt's source, nor detailed syntax locations or
exception diagnostics. Eleven previous attempts appear only as repeated per-task
failure statuses: six missing-source, four syntax-error, one nondeterministic.
Thus rejection feedback communicates failure type but often lacks the failed code
needed to associate that failure with a particular change. A fixed bounded schema
with explicit candidate identity, source change and actionable error could test
this mechanism without extra model calls or access to validation.

The metric really does emphasize early behavior: first8 evaluations contribute
75.6%/80.1%/80.2% of total A0/A1/A2 holdout AUC; first4 contribute56.6%/58.7%/59.2%.
This is a valid registered anytime metric, but it can reward initialization much
more than late history adaptation. First8 contribute −.003950 of the total
A2−A1 difference −.005068. Test criterion explanation and chronological improvement
checkpoints while preserving all generic information equally across arms.

## The task is deterministic; policy evaluation is still stochastic

The exact same A0 source on the exact same six train tasks obtains AUCs
[.207912,.070397,.171432,.126563,.248884] across the five local-seed panels.
SD=.06955; range=.17849. Its six-task validation panel has SD=.06291 and range=.13913.
These differences require no stochastic objectives, model calls or code changes.
One local seed per instance is enough for deterministic replay, but not for a
precise estimate of a stochastic policy's expected performance.

Train and validation winners differ in7/10 pools (A1 3/5, A2 4/5). The mean
within-pool Spearman correlations are .511 A1 and .605 A2. Thus noisy/rank-changing
evaluation is real, but these retrospective numbers do not establish a uniquely
A2 failure. For11/A2, the unchanged seed is last of7 on train and first on validation.
For41/A2 and53/A2, the eventual selected child is worse than its parent on train
but better on validation. A greedy training incumbent can therefore discard
promising lineages even while final pool selection properly retains all candidates.

Omitting one of six validation strata changes A1 selections in6/30 comparisons,
A2 in2/30. This **does not demonstrate that A2 validation selection is more unstable**.
Omission changes stratum weights and is only sensitivity analysis, not a confidence
interval or a recommended selection rule. Fresh balanced tasks and independent
local replication are needed to isolate evaluation noise from distribution shift.

Twenty of40 A2 requests still use the original seed as parent. Seed37 never accepts
any generated train replacement. The selected artifacts' source depths are
[0,1,0,2,3], and only7/28 eligible generated candidates improve their parent on
training AUC. Eight proposal slots thus do not imply eight successful refinement
steps or evidence concerning greater recursion depth.

## Heterogeneity is a hypothesis for fresh validation, not a license to prune tasks

| Stratum | A0 AUC | A1 AUC | A2 AUC | A2−A1 |
|---|---:|---:|---:|---:|
| Quadratic2 | .089094 | .054204 | .095405 | +.041200 |
| Quadratic4 | .139184 | .139842 | .098538 | −.041304 |
| Rosenbrock2 | .064630 | .045277 | .078691 | +.033414 |
| Rosenbrock4 | .163935 | .139250 | .096638 | −.042612 |
| Sphere2 | .115767 | .109750 | .097801 | −.011949 |
| Sphere4 | .264863 | .242165 | .233009 | −.009156 |

A2 beats A1 in4/5 outer pairs on dimension4 and only1/5 on dimension2. This is
exploratory and could reflect selected algorithm behavior, local randomness,
landscape differences or initial-point luck. Do not replace the balanced benchmark
with dimension4 based on these outcomes. Register dimensional interaction tests on
fresh instances if investigating this lead. The final-regret and target-attainment
means favor A1, even though primary anytime AUC slightly favors A2; retain both.

## Fresh-data tests justified by this audit

1. Cap8k versus32k at fixed prompts, counterbalanced routing/time, same model; test
   validity and actual cost first. Distinguish stop syntax errors from length loss.
2. Fixed policy bank on fresh tasks, factorial instance counts1/2/4 per stratum and
   local seed counts1/2/4; freeze selections before independent audit evaluation.
3. Sparse original feedback versus explicit anytime criterion plus bounded,
   scale-aware training summaries and incumbent/improvement points; equal information
   available to A1 outside evaluated feedback, no validation leakage or extra calls.
4. Only after that, equal-budget greedy-parent versus archive/population selection
   through the existing production path. This isolates search strategy from trace
   quality; arbitrary “more recursion” is not itself a remedy.

Validation: `python -m pytest -q artifacts/optimizer_discovery/investigation16/statistics/test_audit.py`
reports4 passed. Black (target py313) and Ruff pass. A null-content diagnostic bug
was reproduced and fixed before producing final diagnostics; it affected no EXP-15
data or scientific value. All raw response slots remain represented, including
missing source, syntax errors and the execution-invalid candidate.

Independent rerunning of `audit.main()` reproduces `diagnostics.json` byte for byte:
SHA256 `24721eed96ec7063147df4bc91ed115762232eb861ead60c21df091e62f0f0e9`.
