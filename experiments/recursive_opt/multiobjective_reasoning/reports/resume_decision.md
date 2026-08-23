# Experiment 0 v2 resume decision

## A. Why v1 blocked

Experiment 0 v1 correctly stopped at task eligibility because none of its
three candidates satisfied the frozen v1 rule. GSM8K had no pooled accuracy
spread despite a large token spread, BBEH boolean expressions was excluded by
the maximum invalid rate across all probes, and BBEH object counting did not
show an adequate signal. The stop occurred before micro-smoke, optimizer
comparison, pilot, or holdout-informed selection. The original evidence is
unchanged and copied byte-for-byte under `manifests/v1/` and `reports/v1/`,
with hashes in `manifests/v1/evidence_hashes.json`.

## B. Why the eligibility repair is not post-hoc goal shifting

V1 exposed a measurement-design false negative before any optimizer result was
available. Its experiment objective rewards either quality improvement or
accuracy-preserving token efficiency, but its task gate could recognize only
accuracy variation. V2 makes the task gate measure the already-preregistered
scientific question: a feasible validation quality signal OR a feasible probe
within 0.02 accuracy of P0 using at most 80% of P0's forward tokens. It does not
change P0, P1-P3, the objective, weights, invalidity constraint, arms, paired
seeds, validation-gate ablation, or eventual success criteria.

## C. Exact v1 to v2 differences

- V1 files remain authoritative immutable evidence; v2 uses new manifest and
  report names rather than overwriting them.
- Eligibility uses validation metrics rather than pooled train and validation
  metrics, and requires a P0 validation accuracy in `[0.20, 0.95]`, P0 invalid
  rate zero, and nonempty mutually disjoint pools.
- Informativeness is validation quality change OR accuracy-preserving token
  efficiency. Invalid non-baseline probes retain diagnostics but do not exclude
  a task by task-level maximum invalidity.
- Provider cost unavailable for nonzero-token runs is `null`, and task ranking
  falls back transparently to mean forward tokens per evaluated example.
- Dataset pools are deterministic seeded selections (`seed=1803`) from the
  pinned sources, with 16 train, 12 validation, and 24 holdout rows per task.
  Eligibility uses the first 8 train and 8 validation rows and zero holdout
  evaluations.
- Experiment-facing BigBenchExtraHard names are
  `bbeh_object_counting` and `bbeh_boolean_expressions`; v1 names are untouched.
- Stage limits now select explicit subsets from the frozen v2 pools instead of
  retaining the v1 4/2/2 assumption.
- After task selection, failed live micro attempts revealed that the original
  768-token optimizer cap produced reasoning with no final content. Before any
  optimizer proposal or engine comparison, the experiment-only profile was
  transparently amended to the same model with `reasoning.effort=low` and an
  8,192-token per-call cap. Aggregate stage budgets and every scientific gate
  remain unchanged. All failed attempts and the content-free response audit are
  retained under `reports/v2_attempts/`.

## D. Evidence that holdout did not affect eligibility

`reports/task_eligibility_v2.json` records
`holdout_evaluator_invocations: 0` and `holdout_used_for_eligibility: false`.
The v2 eligibility runner receives only the frozen train and validation subset
IDs. The 24 holdout rows were selected and hashed without evaluating answers.
The later P0 baseline-token manifest measured only per-example token
denominators and records no holdout accuracy. Holdout evaluation began only in
the subsequently authorized one-unit live micro-smoke and did not redesign the
task, probes, thresholds, P0, or pools.

## E. Selected task and remaining blocker

GSM8K is selected. It was the only task mechanically classified near-eligible
from preserved v1 evidence and was therefore the only task recalibrated. On the
expanded validation subset P0 accuracy is `0.875` with invalid rate `0.0`;
P1 and P3 produce a quality signal, validation accuracy spread is `0.25`, and
all split-integrity checks pass. Monetary cost was unavailable, so selection
used the declared forward-token proxy. The baseline manifest freezes all 52
per-example denominators (16/12/24) with content digest
`6fbdc131c24fefc84e62c10a8698e65a83946110e17d7febf2c870a1260a16fe`.

The offline A/B/C/D contract passes all 20 assertions. In the live micro-smoke,
arms A and B pass every check. Arm C reaches GEPA reflection, where GEPA 0.1.4
calls its reflection function with a plain string. The locked recursive-opt
client forwards that positional string to LiteLLM, whose messages validator
requires mappings and raises `AttributeError: 'str' object has no attribute
'get'`. GEPA handles the reflection failure and retains P0, so C correctly
fails the nontrivial-proposal and optimizer-usage gates. No cost forecast or
pilot was started after this micro gate failed. Repairing that seam would
require a recursive-opt core change, which this task explicitly forbids.

## F. Resume status

Corrected v2 task eligibility passes, the baseline token manifest is frozen,
and Experiment 0 resumed at the first selected-task-dependent phase. It reached
and stopped at the live C-arm micro-smoke gate; the primary task does not need
replacement, but the locked GEPA reflection-client seam must be repaired before
the pilot can run.

RESUME_PROMPT_18_R3
