# P1 supplementary fixed-initialization control

Prospective supplementary reference, declared before any main P1 generation.
It does not change P1's A0/I/C/R/W arms, seed, prompts, feedback, selection pools,
representative, budgets, primary comparisons or analysis. B2 was developed and
checked on separate public diagnostic fixtures; it is development evidence, not
an independently discovered P1 winner. This control tests its transfer to the
already frozen P1 audit distribution and qualifies the strength of the starting
baseline. It is not a new generative search and is not conditional on R−I's sign.

## Registration and barriers

Fix the exact `benchmark/b2/optimizer.py` bytes, SHA256
`958fbb12279a15966bf1ffa45ddffc7cd5f946ca800a3fe2b8e132a150c5c190`.
It changes the unchanged seed's initial uniform point to the legal box midpoint;
subsequent exploration and incumbent perturbation behavior remains that source's
fixed implementation. Do not modify this artifact after registration.

After the main P1 freeze exists, but before `generation_started.json`, any request
start or any response, run the separate helper's `prepare`. It freezes its source,
tests, this protocol, evaluator dependencies, B2 source, unchanged fallback seed,
exact P1 freeze digest and all144 deterministic audit task/local-seed jobs. No
objective or audit-result read occurs in preparation. Refuse late registration.

Do not read primary audit results or evaluate B2 until all primary generation,
selection and audit completion barriers exist. Then run the frozen primary
`analysis.read_bundle` and `analysis.summarize` checks, including S's manifest,
chronology, slot, selection and source-integrity checks. If they fail, the control
cannot proceed. Verify that control registration preceded primary generation.
A completed negative primary outcome passes this gate when scientifically valid.

## Fixed evaluation

Use P1 outer seeds [16411,16423,16437,16441,16453,16467], the same12 audit instances,
and the same two local optimizer seeds per instance/outer seed. There are exactly
6×12×2=144 B2 trajectories, B32, eight offline workers, hard proposal timeout2s,
standard deterministic replay, and no LLM calls. Reuse the unchanged evaluator.
The exact seed derivation is P1's `G.local_seed('P1', int(B.digest([outer,replicate])
[:15],16), task)` with replicate0/1.

Apply the same deployment fallback: on a B2 proposal failure, permanently switch
to the unchanged P1 seed with actual accumulated history and remaining budget.
No reset, extra objective evaluation, code repair or dropped trajectory. Candidate
failure remains visible even when fallback yields a valid deployment score.
Failure of the trusted evaluator/seed/fallback is an infrastructure defect and
blocks completion; preserve its sanitized record and unfinished state.

Allocated candidate objective calls are144×32=4,608. Successful deterministic replay
uses9,216 subprocesses; one failed B2 proposal may add at most two executions per
trajectory, giving a maximum9,504 with successful fallback infrastructure. The
12 audit tasks use a1,536-evaluation reference design already shared scientifically
with P1. A new Python process may reconstruct these constants; record observed
reference-cache misses per completed control invocation separately. Unrecorded
work interrupted before persistence remains unknown, not zero. No A0/R optimizer
is rerun, and no primary cache/raw/result file is written by this helper.

## Storage, resume and analysis

Store freeze, per-job raw rows, attempt starts, sanitized infrastructure failures,
invocation clocks, primary verification hashes and aggregate results under
`investigation16/production_baseline_control/`, outside the P1 run tree. Completed
rows are immutable and hash checked. Resume only unfinished jobs under identical
frozen inputs. Retain started-but-uncheckpointed attempts and possible wasted work;
never replace a completed unfavorable or fallback trajectory. A process lock
prevents overlapping control executions. Refuse missing or extra final jobs.

Recompute every outer-seed B2 normalized anytime AUC and final regret using the
unchanged equal-stratum aggregation. Reuse preserved primary A0/R audit outcomes
as comparators, without candidate or objective reevaluation. Report B2−A0 and
R−B2 with the same `E.paired` 10,000 outer-seed bootstrap, seed1515, interpolated
2.5/97.5 percentiles and descriptive interval labels. Negative deltas favor the
first policy. Retain every seed, both contrasts, validity/fallback, actual objective
calls, subprocesses and execution time. Report source hashes and primary evidence
hashes. All results are supplementary exploratory comparisons, with fragile n=6
uncertainty conditional on the common audit panel. No multiplicity-adjusted or
novelty claim is made.

B2 outperforming R would qualify the practical strength of P1's unchanged baseline;
R outperforming B2 would not replace the central R−I comparison or isolate feedback's
causal mechanism. Failure of B2's large public-fixture effect to transfer is valid
evidence. No outcome triggers new generative search or retrospective modification
of the primary experiment. This supplementary control cannot turn the original
primary comparison into a confirmatory comparison against a stronger seed.

Commands (preparation only after main freeze; execution only after primary audit):

```bash
python -m artifacts.optimizer_discovery.investigation16.production.baseline_control prepare
python -m artifacts.optimizer_discovery.investigation16.production.baseline_control run
python -m artifacts.optimizer_discovery.investigation16.production.baseline_control analyze
```
