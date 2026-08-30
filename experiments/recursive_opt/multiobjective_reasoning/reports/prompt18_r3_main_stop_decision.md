# Prompt 18-R3 main stop decision

## Cost authorization and frozen main lock

The user explicitly waived a numeric main monetary ceiling on 2026-08-24. This
removed the only pre-main operational blocker without changing the task, P0,
dataset pools, objective, hard constraint, arms, models, seeds, budgets,
holdout policy, or success criteria.

The main design was frozen before the first provider call as 5 paired seeds ×
2 candidate budgets (6 and 12) × A/B/C/D, using the complete frozen GSM8K
16/12/24 pools. The implementation state was commit
`a3267c3cfbc028e96809c20a329c59d6b458fc46`; required GitHub Actions run
`32735198613`, job `97456226110`, completed successfully. The authoritative
digests were:

- runtime tree: `ba4836d9f43cffcd0271086932745b270d75478b5287a7d8100be4928b623cbc`
- control-plane registry: `f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`
- Experiment-0 source: `434bc450e09c97c0b7158043b12b616b6f5044bd1b4a2469eaba87c2bed093a7`
- Experiment-0 registry: `18a7efd58f3d265a723fa62efb89d2dc07082afbcdf16c9470e6d4eb93a77121`
- frozen preregistration: `e6954f457132518e9d62f9d0ee2dd0f7d73b49669f3deb905ae481a609c8b8ee`
- main lock: `f9863918d561b4b612f5332b8f7bd7dbf9aff7e18328104bd34323a9375cb40c`

## Required stop after the first main unit

The first frozen unit was seed 0, proposal budget 6, arm A (fixed P0). It
evaluated the complete 24-example holdout and stopped with
`hard constraints not satisfied`:

- accuracy: `0.9166666666666666` (22/24 exact answers)
- invalid rate: `0.041666666666666664` (1/24)
- mean forward-token ratio: `1.072793249592918`
- mean latency: `8.302135325041794` seconds/example
- forward calls: `48`
- evaluator runs: `24`
- prompt/completion/total tokens: `8,343 / 4,642 / 12,985`
- provider-reported cost: unavailable
- token-priced cost proxy: `$0.00150300`
- optimizer calls, proposals, evaluations, and semantic retries: all `0`
- wall time: `199.278981` seconds

The invalid result was holdout sample `gsm8k:test:915` (content SHA-256
`926bb31a00a4f5f0eb2862e68b07028e65963f7f932537fd24179a66e6c059f0`).
The expected answer was `23`; deterministic extraction persisted an empty
answer. The raw provider response text is not persisted, so the exact upstream
formatting cause is unknown and is not inferred. A separate valid-but-wrong
answer occurred on `gsm8k:test:35` (`0` instead of `9`).

All non-safety infrastructure observations passed: exact provider/model and
request parameters, one workflow forward per evaluator example, exact-output
identity, forward call/token reconciliation, source locks, no cache sharing,
no hidden environment override, and no holdout access before final evaluation.
Persistence/resume was not attempted because the result failed the mandatory
validity prerequisite; this is not classified as an independent resume defect.

## Scientific classification

This is not a monetary blocker and not a demonstrated control-plane defect.
It is a preregistered scientific stop: fixed P0 violates the hard
`invalid_rate <= 0` requirement on the first complete holdout evaluation.
Rerunning the same arm to seek a stochastic pass, changing P0, weakening the
constraint, changing the evaluator, or removing the revealed holdout example
would move the goalposts after observing holdout performance and is prohibited.

The run stopped after 1/40 units. No Trace, GEPA, or no-validation main arm was
started. Therefore no paired main statistics, engine-efficacy comparison,
validation-gate ablation, seed-expansion decision, second-task run, candidate
trajectory audit, or episode export was performed. The completed pilot remains
valid pilot evidence but is not combined with this incomplete main run.

## Decision

`FIX_EXPERIMENT_ONLY`

Experiment 0 v2 cannot resume under its frozen hard constraint. Any follow-up
must be a prospective, versioned experiment amendment with a new independent
holdout; it must not redesign P0 or the task using this revealed holdout result.
No recursive-opt core change is justified by this evidence.
