# recursive_opt use cases: current limits and root-cause analysis

This report analyzes the latest live Stage-2 run:

`examples/notebook_outputs/recursive_opt_use_cases/three_way_stage2_20260624_150102/`

Scope: use cases where recursive optimization had negative, tied, low, or non-promotable progress versus the standard Trace optimizer. The analysis only uses measurable evidence from the saved `three_way_report.json`, `promotion_decision.json`, learning curves, budget snapshots, and artifact diffs.

## Method

Tests applied to each use case:

1. Reliability gate: `n >= 3`; if `abs(mean recursive-standard delta) <= std(seed deltas)`, treat as a tie, not a real win/loss.
2. Promotion gate: inspect `promotion_decision.json`; promotion requires recursive lower-confidence bound to beat standard by the configured margin.
3. Headroom test: check whether both arms already reached the maximum score or a near-flat surface.
4. Curve/cost test: compare best candidate units, optimizer LLM calls, elapsed time, and whether recursive reaches standard's best earlier.
5. Structural test: identify whether the recursive arm is genuinely different: same-level warm code restart, O2->O3 family prior, numeric route, or held-out transfer.
6. Artifact test: inspect saved `diffs/seed0_standard_to_recursive.diff` for concrete behavioral differences.

Verification run after the notebook fixes:

```bash
PYTHONPATH=. python -m pytest tests/unit_tests/test_recursive_opt.py tests/unit_tests/test_recursive_spec.py -q
# 110 passed
```

## Summary table

| UC | Standard | Recursive | Delta | Seed-delta std | Reliability | Promotion | Root status |
|---|---:|---:|---:|---:|---|---|---|
| UC1 code BBEH solver | 1.000 | 1.000 | 0.000 | 0.000 | tie | hold | saturated score; cost-only recursive win |
| UC2 QASPER config | 0.292 | 0.265 | -0.027 | 0.050 | tie | reject | noisy/low-signal; no reliable recursive gain |
| UC5 tool policy code | 0.750 | 0.792 | +0.042 | 0.629 | tie | hold | high seed variance; no promotable gain |
| UC8 campaign policy code | 0.517 | 0.445 | -0.072 | 0.063 | standard | reject | real loss for same-level warm code restart |
| UC9 agentic policy code | 0.760 | 0.843 | +0.083 | 0.112 | tie | promote | score promising, but paired reliability gate says tie |
| UC10 promotion policy code | 0.342 | 0.398 | +0.057 | 0.049 | recursive | hold | positive but not LCB-safe |
| UC11 prompt emitter code | 0.499 | 0.381 | -0.118 | 0.085 | standard | reject | real loss for same-level warm code restart |
| UC13 numeric head-to-head | -0.158 | -0.160 | -0.002 | 0.004 | tie | hold | flat numeric objective; speed-only win |
| UC14 code transfer | 0.643 | 0.573 | -0.070 | 0.000 | standard | reject | held-out code transfer fails |

Positive control:

| UC | Standard | Recursive | Delta | Promotion | Budget stability |
|---|---:|---:|---:|---|---|
| UC4 family policy prior | 0.013 | 0.176 | +0.163 | promote | stable |
| UC4 sweep N=6 | 0.019 | 0.180 | +0.161 | promote | agrees |
| UC4 sweep N=12 | 0.029 | 0.190 | +0.161 | promote | agrees |

UC4 is the current robust pattern: structurally different O2->O3 family-policy prior, validated by a budget sweep.

## UC1: code BBEH solver

Observed:

- Initial: `0.625`
- Standard: `1.000`
- Recursive: `1.000`
- Delta: `0.000`
- Both arms have `std_final = 0.0`
- Recursive uses fewer optimizer LLM calls: `2` per seed versus standard `4` per seed.

Hypotheses:

1. The task is saturated, leaving no score headroom.
   - Validated. Both standard and recursive reach `1.000` for all three seeds. More score improvement is mathematically impossible under this evaluator.
2. Recursive is not doing anything different.
   - Rejected. Artifact diffs are non-empty and recursive uses fewer optimizer LLM calls. It is doing work, but the score ceiling hides any score difference.
3. More budget would show a recursive score win.
   - Rejected for score. Both arms already hit the maximum score.

Root cause:

UC1 is useful as a code-rewrite and cost demonstration, but not as a recursive-performance benchmark. It is saturated.

Solution:

Keep UC1 as a "recursive can match standard with fewer optimizer calls" example. Do not use it as evidence of higher recursive final score. For performance claims, replace or augment it with a harder BBEH/code task where standard does not reach `1.000` at `N=6`.

## UC2: QASPER config/prompt

Observed:

- Standard: `0.292`
- Recursive: `0.265`
- Delta: `-0.027`
- Seed-delta std: `0.050`
- Per-seed deltas: approximately `+0.002`, `-0.085`, `+0.002`
- Recursive artifact adds `batch_design: random` and `batch_size: 4` in seed 0.

Hypotheses:

1. Recursive is genuinely worse.
   - Not validated. The negative mean is smaller than the paired seed-delta std, so the reliability gate classifies this as a tie.
2. QASPER is noisy/slow enough that `n=3, N=6` is insufficient.
   - Validated. Standard std is `0.040`, recursive std is `0.016`, and per-arm wall times are around 155-191 seconds. One seed dominates the mean.
3. The recursive arm is over-configured or adding weak active fields.
   - Partly validated. Recursive adds numeric/config fields, but the observed selected values do not improve score. The data does not prove the fields are inactive, only that this configuration did not help.
4. The budget is too small.
   - Not resolved. Both standard and recursive best units occur at candidate `6`, so a larger `N` could change the result, but no UC2 budget sweep was run.

Root cause:

The current UC2 result is a noisy tie with a weak recursive configuration. It is not a robust recursive loss, but it also gives no evidence of recursive advantage.

Solution or next experiment:

Run a UC2-specific budget sweep at `N=6` and `N=12`, or split the experiment into isolated factors:

- starting artifact only
- numeric fields only
- warm prior only
- combined recursive arm

Do not promote UC2 until it passes the reliability gate and LCB promotion gate.

## UC5: optimizer tool policy code

Observed:

- Standard: `0.750`
- Recursive: `0.792`
- Delta: `+0.042`
- Seed-delta std: `0.629`
- Per-seed deltas: `+0.625`, `+0.125`, `-0.625`
- Promotion: `hold`
- Recursive uses fewer optimizer LLM calls: `2` versus standard `4`.

Hypotheses:

1. Recursive is better.
   - Not validated. The mean is positive, but the seed variance is much larger than the effect.
2. The evaluator is too discrete/high-variance.
   - Validated. Seeds jump between baseline-level and perfect/near-perfect policies. A single seed reversal changes the verdict.
3. The recursive two-phase split is under-budgeted.
   - Partly validated. Recursive uses fewer optimizer LLM calls and one seed stays at the baseline score `0.375` while standard reaches `1.000`. The current data supports "budget split can fail", but does not prove which split would fix it.
4. The use case is badly designed.
   - Not fully validated. The surface has signal: both arms improve substantially over initial `0.375`. The problem is reliability, not total absence of signal.

Root cause:

UC5 is a high-variance discrete policy task. Recursive can win cheaply, but the current three-seed result is not statistically stable.

Solution or next experiment:

Increase seeds or make the evaluator less binary by adding more policy cases. If the goal is recursive advantage, compare:

- cold standard at `N=6`
- recursive warm at `N=6`
- recursive warm with larger phase-1 or phase-2 split

Do not claim a recursive score win from the current run. A cost-efficiency claim is plausible but should be separated from the score promotion gate.

## UC8: campaign policy code

Observed:

- Standard: `0.517`
- Recursive: `0.445`
- Delta: `-0.072`
- Seed-delta std: `0.063`
- Per-seed deltas: `0.000`, `-0.095`, `-0.120`
- Promotion: `reject`
- Both arms improve over initial `0.360`, but standard improves more.

Hypotheses:

1. Recursive is genuinely worse at this setting.
   - Validated. All seed deltas are non-positive, and the mean loss is larger than the paired seed-delta std threshold used for this analysis.
2. Same-level warm code restart is the wrong recursive structure for this use case.
   - Supported. UC8 uses the same two-phase warm code restart that also loses in UC11 and UC14. In contrast, UC4's structurally different O2->O3 prior wins robustly.
3. The evaluator has no signal.
   - Rejected. Standard improves by `+0.157` over initial; there is enough signal for optimization.
4. Budget starvation explains the loss.
   - Not resolved. Both arms reach best at candidate `6`; a budget sweep was not run for UC8.
5. The recursive artifact is worse because it over-simplifies the decision policy.
   - Supported by artifact diff. The recursive seed0 policy removes several more detailed campaign checks from the standard artifact and replaces them with a simpler policy. This is consistent with lower score.

Root cause:

The current recursive structure is not appropriate for UC8. It spends part of the budget discovering a warm code prior, but the cold standard optimizer uses the full budget directly on the target policy and wins.

Solution or next experiment:

Do not promote UC8. To continue investigating, run a budget sweep and/or redesign it as a family-policy/prior experiment rather than a same-level code warm restart. The most scientifically useful next comparison is UC8 cold standard versus a structurally different prior level, not another same-level warm restart.

## UC9: agentic policy code

Observed:

- Standard: `0.760`
- Recursive: `0.843`
- Delta: `+0.083`
- Seed-delta std: `0.112`
- Per-seed deltas: `+0.040`, `0.000`, `+0.210`
- Promotion: `promote`

Hypotheses:

1. Recursive is a robust score winner.
   - Not validated by the stricter reliability gate. The promotion gate promotes because recursive LCB beats standard using recursive variance only, but the paired delta variance is larger than the mean gain.
2. The promotion gate can be too optimistic when standard variance is high.
   - Supported. Standard std is `0.079`; recursive std is `0.031`; paired delta std is `0.112`. The current promotion decision does not fully account for standard-arm or paired-delta uncertainty.
3. The use case has real optimization signal.
   - Validated. Initial is `0.210`, standard is `0.760`, recursive is `0.843`.
4. More seeds could turn this into a robust win.
   - Plausible but unvalidated. The current data suggests promise, not convergence.

Root cause:

UC9 is promising but not settled. The current promotion gate says promote, while the stricter paired reliability gate says tie. This is a methodological limit, not a code failure.

Solution:

Before using UC9 as a headline result, rerun with at least five seeds or update the promotion gate/reporting to include paired-delta uncertainty. Until then, report UC9 as "promising, not reliability-stable."

## UC10: promotion policy code

Observed:

- Standard: `0.342`
- Recursive: `0.398`
- Delta: `+0.057`
- Seed-delta std: `0.049`
- Promotion: `hold`
- Per-seed deltas: `+0.080`, `0.000`, `+0.090`

Hypotheses:

1. The guarded decision policy improved recursive behavior.
   - Partly validated. Recursive is positive on two seeds and tied on one seed.
2. The result is strong enough to promote.
   - Rejected. Recursive std is `0.149`, and the LCB is below standard. Promotion correctly holds.
3. Remaining failures are due to incomplete learned guard order.
   - Supported by artifact diff, not fully proven. The recursive seed0 artifact learns a simple `mean_score > 0.05` promote threshold. That is better than unconditional promote, but it is not the full guard order: invalid syntax -> reject, saturated control -> control/archive, low support -> retest, regression -> rollback, then promote.
4. More budget could help the policy learn the full guard order.
   - Not resolved. No UC10 budget sweep was run in this stage.

Root cause:

UC10 shows the guarded schema helps, but the learned artifacts are still partial guard implementations. The result is positive but not confidence-safe.

Solution or next experiment:

Keep UC10 as a guarded-policy progress example, not a promoted result. Run a budget sweep or increase guard-case diversity if the goal is a promotable artifact. The current data does not justify changing core recursive_opt; it justifies a richer UC10 evaluator or more budget/seeds.

## UC11: prompt emitter code

Observed:

- Standard: `0.499`
- Recursive: `0.381`
- Delta: `-0.118`
- Seed-delta std: `0.085`
- Per-seed deltas: `-0.210`, `-0.101`, `-0.043`
- Promotion: `reject`
- This is slow: per-arm seed runs are around 149-220 seconds.

Hypotheses:

1. Recursive is genuinely worse.
   - Validated. All seed deltas are negative, and the mean loss is larger than the reliability threshold.
2. Same-level warm code restart is the wrong structure.
   - Supported. UC11 uses the same recursive code-arm structure that loses in UC8 and UC14.
3. The standard artifact is more target-specific than recursive.
   - Supported by artifact diff. The standard seed0 prompt explicitly preserves QASPER format/label hints, while the recursive artifact becomes more generic. The generic prompt scores worse on the QASPER artifact-injection evaluator.
4. The evaluator is too noisy to trust.
   - Not enough to explain the loss. The evaluator is noisy, but all three seeds are negative.
5. More budget would fix it.
   - Not validated. No UC11 budget sweep was run. Standard already wins at the same candidate budget.

Root cause:

UC11 is a real failure of the same-level warm code restart on a target-specific prompt-emitter task. The warm prior appears to generalize away details that the target evaluator rewards.

Solution:

Do not promote UC11. If kept, use it as a negative control showing that recursive warm-starts can hurt target-specific artifacts. A better recursive design would need a held-out validation structure or family-policy prior, not just a two-phase restart.

## UC13: numeric optimizer head-to-head

Observed:

- Standard: `-0.158`
- Recursive: `-0.160`
- Delta: `-0.002`
- Seed-delta std: `0.004`
- Promotion: `hold`
- Recursive reaches standard's best by candidate unit `1`, with `0` optimizer LLM calls.
- Standard uses `8` optimizer LLM calls and about 148-167 seconds per seed; recursive takes about 38-44 seconds per seed.

Hypotheses:

1. The numeric optimizer is failing.
   - Not validated. It is faster and cheaper, and scores are within noise.
2. The numeric objective surface is flat.
   - Validated. Initial, standard, and recursive all sit within roughly `0.005` score. No method has meaningful score headroom here.
3. This is a good score-performance benchmark.
   - Rejected. It is a cost/speed demonstration, not a score-gain demonstration.

Root cause:

UC13's current task does not expose a useful numeric landscape. It proves routing/cost behavior, not recursive quality.

Solution:

Keep UC13 for "zero optimizer-LLM numeric routing" demonstration. For performance, replace the task with one where `batch_design` or `batch_size` causally changes score by more than run noise.

## UC14: code transfer

Observed after fixing notebook wiring:

- Initial: `0.210`
- Standard held-out score: `0.643`
- Recursive transfer held-out score: `0.573`
- Delta: `-0.070`
- Seed-delta std: `0.000`
- Promotion: `reject`
- Every seed has the same held-out gap: standard beats recursive by `0.070`.

Bug found and fixed:

- The original UC14 cell used `make_code_evaluator(...)`, but that helper only supports `batch_design` and `trace_summarizer`. It did not invoke the candidate callable for policy components, so every arm failed with "Component evaluator did not invoke the candidate callable".
- The notebook now uses the real executable policy evaluators: `evaluate_optimizer_tool_policy` for train and `evaluate_agentic_trace_policy` for held-out scoring.

Hypotheses:

1. UC14 transfer arm beats standard on held-out and should generalize to other code UCs.
   - Rejected. The held-out transfer arm loses on all three seeds and promotion rejects it.
2. The train and held-out policy families are too far apart.
   - Supported. Recursive trains on optimizer-tool policy and scores on agentic-trace policy. Cold standard trains directly on the held-out target and consistently wins.
3. The failure is caused by evaluator wiring.
   - Rejected after fix. The fixed run has no errors and still loses.
4. The recursive arm is cheaper but lower quality.
   - Validated. Recursive uses fewer optimizer LLM calls, but the held-out score is lower.
5. More budget would reverse the result.
   - Not validated. The current deterministic seed gap argues against assuming this, but a UC14 budget sweep would be required to prove budget sensitivity.

Root cause:

UC14 shows that code-surface transfer is not currently robust. The O2->O3 family-policy prior from UC4 transfers; a two-phase code artifact trained on a different policy family does not.

Solution:

Do not generalize the UC14 code-transfer arm to UC5/UC8/UC9 yet. If testing code transfer further, use closer train/held-out policy families or add a held-out validation objective during training, then rerun a budget sweep.

## Cross-UC conclusions

1. Same-level warm code restart is not a robust general recursive pattern.
   - Evidence: UC8, UC11, and UC14 lose; UC5 and UC9 are noisy/tied by the paired reliability gate; UC10 is positive but not promotable.
   - Interpretation: spending part of the budget on a prior and then warm-starting the same code surface often gives fewer target-specific updates than cold standard optimization.

2. O2->O3 family-policy prior is the current robust recursive pattern.
   - Evidence: UC4 wins at `N=6` and `N=12`, with promotion at both points and consistent delta around `+0.161`.
   - Interpretation: recursion is valuable when the recursive level changes the structure of the problem: family policy followed by prior induction, not just a same-level restart.

3. Promotion must not be read before reliability.
   - Evidence: UC9 promotes by recursive LCB, but paired seed-delta std is larger than the mean gain. UC5 has a positive mean but very high seed variance and correctly holds.
   - Solution: final analysis should apply reliability first, then budget stability, then promotion.

4. Some UCs are demonstrations, not benchmarks.
   - UC1: saturated score; useful cost demo.
   - UC13: flat numeric score; useful speed/cost demo.
   - UC10: useful guarded-decision demo; not yet promotable.

5. The evaluation harness should keep failed arms visible.
   - UC14 initially exposed a real notebook wiring bug. Keeping failed rows and exact errors prevented a false performance interpretation.

## Recommended next actions

1. Keep UC4 as the flagship recursive result and continue using budget sweeps for claims.
2. Do not promote UC8, UC11, or UC14; they are current negative controls for same-level code warm restart and code transfer.
3. Treat UC9 as promising but unresolved; rerun with more seeds or paired-delta-aware promotion before using it as a headline.
4. Treat UC5 as high variance; add more evaluator cases or more seeds before interpreting it.
5. Keep UC1 and UC13 as cost/speed demonstrations, not score superiority claims.
6. If adding core changes, prioritize reporting/gating improvements over new optimization machinery:
   - report paired delta mean/std and reliability classification directly in `three_way_report.json`;
   - make promotion optionally use paired-delta or two-arm uncertainty;
   - keep Trace validation as the source of re-evaluation, rather than creating a separate recursive_opt re-test engine.

