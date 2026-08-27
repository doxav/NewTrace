# Experiment 0 — main report

## 1. Experiment purpose

Cross-engine portability of one frozen compound-reasoning module under fixed, Trace, GEPA, and Trace-without-validation arms.

## 2. Control-plane lock/provenance

- runtime tree: `420b5351063a56b0ad274a6c39b6aaa4dc95b9094434600e89ea79f3eccc8872`
- registry: `c7cd082c9e177180c448e11285bae9d23d53a4f48cfdeba886d7d6121d2d6d8b` (arm-specific plan registry is retained per run)
- Experiment-0 source: `851c47563a1365733988babf49cdbf5be98d9a7be534b8aaf82cf7f70550c6b0`

## 3. Readiness and skips

The required CI, pilot, local readiness, optional-skip classification, and every main infrastructure gate passed before analysis.

## 4–13. Frozen design and execution

GSM8K used the frozen 16/12/24 train/validation/holdout pools, P0, exact-output evaluator, OpenRouter DeepSeek v4 Flash profiles, weighted accuracy/token objective, five paired seeds, and candidate budgets 6/12. Holdout was unavailable during optimization.

## 14. Main results

| arm | runs | accuracy mean | token ratio mean | invalid mean | unsafe runs | forward calls/tokens | optimizer calls/tokens | selected changed | token-priced USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 10 | 0.991667 | 1.012482 | 0.004167 | 1 | 480/125054 | 0/0 | 0 | 0.01403122 |
| B | 10 | 0.979167 | 0.879019 | 0.004167 | 1 | 4612/1059267 | 98/597892 | 8 | 0.18181192 |
| C | 10 | 0.950000 | 0.755077 | 0.004167 | 1 | 2232/462510 | 90/170530 | 9 | 0.07338530 |
| D | 10 | 0.887500 | 0.629026 | 0.004167 | 1 | 3944/710588 | 93/589132 | 10 | 0.14508390 |

## 15. Trace vs GEPA and fixed

| comparison | accuracy delta [95% CI] | token-ratio delta [95% CI] | quality success | efficiency success |
|---|---:|---:|---|---|
| B-A | -0.012500 [-0.041667, 0.016667] | -0.133463 [-0.248967, -0.022951] | False | False |
| C-A | -0.041667 [-0.095833, 0.004167] | -0.257405 [-0.344160, -0.172819] | False | False |
| B-C | 0.029167 [-0.012500, 0.079167] | 0.123942 [-0.000031, 0.233117] | False | False |
| D-B | -0.091667 [-0.166667, -0.025000] | -0.249994 [-0.399754, -0.102257] | False | False |

## 16. Validation-gate ablation

Arm D vs B is reported above. Proposal-level accepted/rejected/harmful/rollback counts are not inferable from current persisted artifacts and were not fabricated.

## 17–18. Optional ablations

Pareto and heterogeneous-artifact ablations were not part of the frozen primary run and were not executed.

## 19. Cost accounting

Provider-reported monetary cost was unavailable. The frozen token-price proxy totals `$0.41431234` and includes semantic retries.

## 20. Failure analysis

All main infrastructure gates passed. Scientific safety failures remain in every paired statistic and do not erase quality or efficiency evidence.

SAFETY: **FAILED**; runs with any invalid output by arm: `{'A': 1, 'B': 1, 'C': 1, 'D': 1}`.

## 21. Episode dataset quality

Candidate-trajectory provenance ready: **False**. No episodes were exported because missing candidate artifact/parent/evaluation/selection records cannot be reconstructed without inference.

## 22. Limitations

One task was eligible; no second task was introduced post hoc. Statistical uncertainty uses paired seed-budget blocks. Provider billing fields were unavailable.

## 23. Decision for next experiment

`RETURN_TO_CONTROL_PLANE_FOR_TRAJECTORY_PROVENANCE`

Main execution and statistics completed, but Prompt-19 episode export is blocked by missing proposal-level candidate trajectory provenance.
