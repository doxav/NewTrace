# Experiment 0 — stopped main run

## Experiment purpose and frozen design

Experiment 0 tests one fixed compound-reasoning module under A fixed, B Trace,
C GEPA, and D Trace without validation gating. The main matrix was frozen as
GSM8K 16/12/24, P0 unchanged, five paired seeds, candidate budgets 6/12, and
the preregistered weighted accuracy/token objective with `invalid_rate <= 0`.
The user's explicit monetary-gate waiver changed no scientific field.

## Control-plane readiness

- implementation HEAD: `a3267c3cfbc028e96809c20a329c59d6b458fc46`
- runtime tree: `ba4836d9f43cffcd0271086932745b270d75478b5287a7d8100be4928b623cbc`
- control-plane registry: `f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`
- Experiment-0 source: `434bc450e09c97c0b7158043b12b616b6f5044bd1b4a2469eaba87c2bed093a7`
- required CI: run `32735198613`, job `97456226110`, completed/success

Local readiness passed 497 unit tests (3 accepted optional skips), 212 focused
tests, 29 Experiment-0 tests, all 20 offline assertions, Ruff, and diff checks.

## Main result and stop gate

| arm | seed | budget | accuracy | invalid rate | token ratio | forward calls/tokens | optimizer calls/tokens | token-priced USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A fixed | 0 | 6 | 0.916667 | 0.041667 | 1.072793 | 48 / 12,985 | 0 / 0 | 0.00150300 |

Sample `gsm8k:test:915` expected `23` but produced no extractable numeric answer.
The raw provider text was not persisted, so no upstream formatting cause is
claimed. The complete holdout was used only for this permitted final
evaluation; no holdout leakage occurred.

The mandatory hard constraint failed, so execution stopped after 1/40 units.
No B/C/D main arm, paired statistics, validation-gate ablation, second task,
seed expansion, or episode export was run. Rerunning or changing P0/evaluator/
constraint after this holdout observation would be post-hoc goal shifting.

## Cost and infrastructure

Provider monetary cost was unavailable. The token-price proxy was `$0.00150300`.
There were no optimizer calls, proposals, semantic retries, or budget overruns.
All other completed-unit infrastructure checks passed.

## Decision

`FIX_EXPERIMENT_ONLY`

The control plane is not implicated by this result. Experiment 0 v2 cannot
resume under its frozen hard constraint. A prospective follow-up would require
a separately versioned protocol and a new independent holdout, without using
this revealed holdout to redesign P0 or task selection.
