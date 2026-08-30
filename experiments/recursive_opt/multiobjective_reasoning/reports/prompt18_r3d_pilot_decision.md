# Prompt 18-R3D pilot decision

## Infrastructure hotfix

The failure was confirmed in the pre-hotfix `OptoPrimeV2` path. The optimizer
read `response.choices[0].message.content` and then evaluated
`"TERMINATE" in response`; a successful provider response with `content=None`
therefore raised `TypeError: argument of type 'NoneType' is not iterable`.
The preserved failed seed-0/budget-6/B run and the earlier preserved micro
attempt contain the same exception. They do not contain sufficient
finish-reason or reasoning-token metadata to establish the upstream reason,
so reasoning-token exhaustion remains unproven.

The recursive-opt optimizer-role boundary now requires non-empty final text
and retries the identical request at most once. Both attempts pass through the
same guarded role client, so calls, tokens, cost, and wall time remain metered.
The same private boundary supplies Trace/OptoPrimeV2 and GEPA reflection.
Reasoning text is never substituted for final content. Direct OptoPrimeV2 use
now raises an explicit missing-text error instead of reaching membership on
`None`.

The scientific protocol was unchanged: GSM8K, frozen 16/12/24 pools, pilot
subset, P0, model and request parameters, reasoning settings, objective and
weights, invalidity constraint, seeds, budgets, A/B/C/D definitions, and
holdout policy are identical to the preserved preregistration.

## Control-plane identity and validation

- implementation commit: `d63746afbb88d6193cbfedf2932b256d9f33b6e4`
- relock/evidence commit: `1eef831b0bbaf34e6fa33145f4600896936e2f80`
- runtime tree: `ba4836d9f43cffcd0271086932745b270d75478b5287a7d8100be4928b623cbc`
- control-plane registry: `f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`
- Experiment-0 source: `7ce17813cd86526009d21a97da5b6eb6c5f931854bcf4fc8410b3a14da7d41d6`
- Experiment-0 registry: `18a7efd58f3d265a723fa62efb89d2dc07082afbcdf16c9470e6d4eb93a77121`
- implementation CI: run `32669603929`, job `97268256178`, completed/success
- relock-head CI: run `32669791486`, job `97268727291`, completed/success

Provider-free validation passed: 6 causal semantic-response tests; 212 focused
control-plane/hardening/spec/objective/vector/multi-objective tests; 317
mandated recursive tests with 2 optional graph/telemetry skips; 497 complete
unit tests with those 2 skips plus the accepted missing-Graphviz-executable
skip; 23 Experiment-0 tests including all 20 offline assertions; and the
clean-kernel notebook. Ruff, `git diff --check`, and the credential scan passed.
No Trace, GEPA, or Experiment-0 test skipped.

## Fresh A/B/C micro-smoke

The post-hotfix 1/1/1 micro-smoke passed under the new lock.

| arm | accuracy | forward-token ratio | forward calls/tokens | optimizer calls/tokens | proposed/evaluated | selected artifact changed | empty/retries |
|---|---:|---:|---:|---:|---:|---|---:|
| A fixed | 1.0 | 1.088384 | 2 / 862 | 0 / 0 | 0 / 0 | false | 0 / 0 |
| B Trace | 1.0 | 1.008838 | 18 / 4,845 | 1 / 11,497 | 3 / 6 | true | 0 / 0 |
| C GEPA | 1.0 | 1.166667 | 8 / 2,516 | 1 / 1,688 | 3 / 3 | false | 0 / 0 |

Both optimized arms have `proposal_path_exercised=true`. Selection change was
diagnostic. Usage reconciliation, exact provider/model/request parameters,
source locks, holdout isolation, and persistence/resume passed.

## Preserved partial pilot

The pre-hotfix 6/24 pilot is preserved under
`reports/pre_empty_text_retry_runtime/`. Its efficacy classification remains
incomplete/provisional and it was not combined statistically with this pilot.
Its infrastructure-failure evidence remains valid.

## Restarted 24-run pilot

The frozen pilot restarted from run 1 and completed all 24 runs. Every pilot
gate passed, including real Trace and GEPA proposals, source provenance,
paired evaluator/forward budgets, token reconciliation, environment/cache
isolation, holdout isolation, valid selected artifacts, and output resume.

| arm | runs | mean accuracy | mean forward-token ratio | forward calls/tokens | optimizer calls/tokens | proposed/evaluated | selection changed runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| A fixed | 6 | 1.000000 | 1.042186 | 96 / 26,866 | 0 / 0 | 0 / 0 | 0 |
| B Trace | 6 | 0.958333 | 0.999712 | 640 / 144,661 | 33 / 243,519 | 50 / 224 | 1 |
| C GEPA | 6 | 0.979167 | 0.833305 | 368 / 86,062 | 31 / 57,947 | 136 / 136 | 5 |
| D Trace, validation-gate ablation | 6 | 0.916667 | 0.815156 | 490 / 101,082 | 32 / 226,082 | 53 / 197 | 6 |

These pilot summaries are descriptive stop-gate results, not authorization to
reinterpret preregistered success criteria or begin the main experiment.

## Empty-text and retry accounting

Six empty final-text responses occurred. Each was followed by exactly one
successful, identically parameterized metered retry. No optimizer request had
two consecutive empty responses, and no candidate was fabricated by a retry.

| arm | empty responses | semantic retries | retry prompt tokens | retry completion tokens | retry total tokens | retry cost proxy USD |
|---|---:|---:|---:|---:|---:|---:|
| A | 0 | 0 | 0 | 0 | 0 | 0.00000000 |
| B | 3 | 3 | 12,767 | 13,180 | 25,947 | 0.00339376 |
| C | 1 | 1 | 644 | 1,164 | 1,808 | 0.00026104 |
| D | 2 | 2 | 8,792 | 4,089 | 12,881 | 0.00143938 |
| total | 6 | 6 | 22,203 | 18,433 | 40,636 | 0.00509418 |

The proxy uses the frozen OpenRouter metadata rates of $0.08/M prompt tokens
and $0.18/M completion tokens. Provider monetary cost fields were unavailable
and remain null/zero-unavailable in canonical usage rather than being treated
as authoritative billed cost.

## Cost gate and decision

Actual complete-pilot usage, including retries, was 541,783 prompt tokens,
344,436 completion tokens, 886,219 total tokens, and a token-priced cost proxy
of `$0.10534112`. Scaling actual pilot usage by the frozen sample and proposal
units gives a full-v2-pool main forecast of 8,524,215.92 tokens and
`$1.01395901`.

No user-acknowledged `MAIN_COST_CEILING_USD` exists. With the transparent 20%
safety margin, the recommended minimum ceiling is `$1.21675081`. The main run
was not started.

`PILOT_COMPLETE_MAIN_AUTH_REQUIRED`
