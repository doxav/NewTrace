# Prompt 18-R3F main completion and trajectory stop

## Execution outcome

The frozen main matrix completed 40/40 canonical units. Every infrastructure
gate passed. The model/provider recorded by the canonical runs is
`openrouter` / `deepseek/deepseek-v4-flash-0731`.

The interrupted seed-1 / budget-12 / Trace-B attempt caused by the temporary
local Internet outage is preserved under
`main_after_transport_resilience_fix/pre_temporary_internet_outage/`. It is
infrastructure-failure evidence only. The replacement unit restarted from zero;
no partial optimizer state, candidates, evaluations, usage, or scientific
metrics from the interrupted attempt entered the 40-run matrix.

Control-plane runtime tree:
`420b5351063a56b0ad274a6c39b6aaa4dc95b9094434600e89ea79f3eccc8872`.

Control-plane registry:
`f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`.

Experiment source:
`851c47563a1365733988babf49cdbf5be98d9a7be534b8aaf82cf7f70550c6b0`.

Experiment registry:
`18a7efd58f3d265a723fa62efb89d2dc07082afbcdf16c9470e6d4eb93a77121`.

## Canonical 40-unit matrix

`invalid` is a completed scientific constraint failure, not an infrastructure
failure. `selected` reports whether the final artifact differs from P0.

| seed | budget | arm | status | accuracy | invalid rate | token ratio | forward calls/tokens | optimizer calls/tokens | proposed/evaluated | selected |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 6 | A | success | 1.000000 | 0.000000 | 0.975747 | 48/12092 | 0/0 | 0/0 | no |
| 0 | 6 | B | success | 1.000000 | 0.000000 | 1.044149 | 346/89371 | 6/31892 | 7/125 | no |
| 0 | 6 | D | invalid | 0.625000 | 0.041667 | 0.394195 | 292/37990 | 6/37326 | 10/122 | yes |
| 0 | 6 | C | invalid | 0.958333 | 0.041667 | 0.989086 | 168/39146 | 6/8969 | 60/60 | no |
| 0 | 12 | A | invalid | 0.916667 | 0.041667 | 1.011203 | 48/12390 | 0/0 | 0/0 | no |
| 0 | 12 | B | success | 1.000000 | 0.000000 | 1.009624 | 570/103448 | 15/77585 | 17/237 | no |
| 0 | 12 | D | success | 0.791667 | 0.000000 | 0.471726 | 502/81417 | 13/93580 | 19/227 | yes |
| 0 | 12 | C | success | 1.000000 | 0.000000 | 0.741390 | 240/51631 | 12/25179 | 96/96 | yes |
| 1 | 6 | B | success | 1.000000 | 0.000000 | 0.822279 | 352/73960 | 7/45649 | 10/128 | yes |
| 1 | 6 | C | success | 1.000000 | 0.000000 | 0.704830 | 168/31730 | 6/11721 | 60/60 | yes |
| 1 | 6 | A | success | 1.000000 | 0.000000 | 1.002973 | 48/12399 | 0/0 | 0/0 | no |
| 1 | 6 | D | success | 0.958333 | 0.000000 | 0.823086 | 288/62798 | 6/36149 | 8/120 | yes |
| 1 | 12 | B | invalid | 0.916667 | 0.041667 | 0.662384 | 568/116507 | 13/80613 | 16/236 | yes |
| 1 | 12 | C | success | 0.958333 | 0.000000 | 0.811089 | 288/72963 | 12/21292 | 120/120 | yes |
| 1 | 12 | A | success | 1.000000 | 0.000000 | 1.021526 | 48/12697 | 0/0 | 0/0 | no |
| 1 | 12 | D | success | 0.708333 | 0.000000 | 0.383590 | 498/63416 | 13/76572 | 17/225 | yes |
| 2 | 6 | C | success | 0.958333 | 0.000000 | 0.812444 | 192/43348 | 6/11678 | 72/72 | yes |
| 2 | 6 | D | success | 0.916667 | 0.000000 | 0.650961 | 290/62056 | 7/54450 | 9/121 | yes |
| 2 | 6 | B | success | 0.916667 | 0.000000 | 0.556539 | 350/78627 | 6/35162 | 9/127 | yes |
| 2 | 6 | A | success | 1.000000 | 0.000000 | 0.991922 | 48/12273 | 0/0 | 0/0 | no |
| 2 | 12 | C | success | 0.791667 | 0.000000 | 0.512476 | 264/43212 | 12/25820 | 108/108 | yes |
| 2 | 12 | D | success | 1.000000 | 0.000000 | 0.800348 | 496/97265 | 12/70120 | 16/224 | yes |
| 2 | 12 | B | success | 1.000000 | 0.000000 | 0.815833 | 580/129798 | 12/80044 | 22/242 | yes |
| 2 | 12 | A | success | 1.000000 | 0.000000 | 1.037135 | 48/12788 | 0/0 | 0/0 | no |
| 3 | 6 | D | success | 0.916667 | 0.000000 | 0.658735 | 292/54597 | 6/35792 | 10/122 | yes |
| 3 | 6 | A | success | 1.000000 | 0.000000 | 1.049708 | 48/13009 | 0/0 | 0/0 | no |
| 3 | 6 | C | success | 0.833333 | 0.000000 | 0.633620 | 192/35936 | 6/11904 | 72/72 | yes |
| 3 | 6 | B | success | 0.958333 | 0.000000 | 0.733058 | 352/73041 | 6/36260 | 10/128 | yes |
| 3 | 12 | D | success | 0.958333 | 0.000000 | 0.572974 | 498/82601 | 12/76778 | 17/225 | yes |
| 3 | 12 | A | success | 1.000000 | 0.000000 | 0.999382 | 48/12336 | 0/0 | 0/0 | no |
| 3 | 12 | C | success | 1.000000 | 0.000000 | 0.718367 | 264/49066 | 12/20588 | 108/108 | yes |
| 3 | 12 | B | success | 1.000000 | 0.000000 | 1.075062 | 570/148538 | 12/71629 | 17/237 | yes |
| 4 | 6 | D | success | 1.000000 | 0.000000 | 0.671200 | 292/59361 | 6/36724 | 10/122 | yes |
| 4 | 6 | A | success | 1.000000 | 0.000000 | 1.043455 | 48/12821 | 0/0 | 0/0 | no |
| 4 | 6 | C | success | 1.000000 | 0.000000 | 0.802478 | 168/35960 | 6/10417 | 60/60 | yes |
| 4 | 6 | B | success | 1.000000 | 0.000000 | 1.030169 | 350/91925 | 8/50532 | 9/127 | yes |
| 4 | 12 | D | success | 1.000000 | 0.000000 | 0.863442 | 496/109087 | 12/71641 | 16/224 | yes |
| 4 | 12 | A | success | 1.000000 | 0.000000 | 0.991768 | 48/12249 | 0/0 | 0/0 | no |
| 4 | 12 | C | success | 1.000000 | 0.000000 | 0.824988 | 288/59518 | 12/22962 | 120/120 | yes |
| 4 | 12 | B | success | 1.000000 | 0.000000 | 1.041095 | 574/154052 | 13/88526 | 19/239 | yes |

## Aggregate scientific results

| arm | runs | accuracy mean | token-ratio mean | invalid-rate mean | unsafe runs | selected changed |
|---|---:|---:|---:|---:|---:|---:|
| A | 10 | 0.991667 | 1.012482 | 0.004167 | 1 | 0 |
| B | 10 | 0.979167 | 0.879019 | 0.004167 | 1 | 8 |
| C | 10 | 0.950000 | 0.755077 | 0.004167 | 1 | 9 |
| D | 10 | 0.887500 | 0.629026 | 0.004167 | 1 | 10 |

For B versus A, the paired accuracy delta is -0.012500 with 95% CI
[-0.041667, 0.016667], and the token-ratio delta is -0.133463 with 95% CI
[-0.248967, -0.022951]. For C versus A, the corresponding deltas are
-0.041667 [-0.095833, 0.004167] and -0.257405
[-0.344160, -0.172819]. Neither optimized engine meets the frozen quality or
efficiency success criterion. Both engine-efficacy conclusions are uncertain.

Safety failed: four of 40 runs had one invalid output (invalid rate 1/24), one
run in each arm. These completed scientific failures remain in every paired
statistic.

## Transport and semantic-response diagnostics

| arm | transient failures | retry attempts | recovered | exhausted | empty text | semantic retries | semantic-retry tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B | 7 | 7 | 7 | 0 | 8 | 8 | 47727 |
| C | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| D | 3 | 3 | 3 | 0 | 3 | 3 | 16625 |

The canonical matrix recorded zero classified connection resets and zero
classified server disconnects. All 11 transient transport failures recovered;
none exhausted the frozen three-attempt policy. The earlier local DNS-outage
attempt exhausted independently and is excluded from the matrix.

Provider-reported monetary cost was unavailable. The frozen token-price proxy,
which includes semantic-retry usage, totals `$0.41431234`.

## Candidate trajectory audit and stop decision

The audit covered all 10 Trace and all 10 GEPA optimized runs. Both engines
reported zero persisted `candidate_trajectory` records. Candidate artifact or
hash, parent/seed relation, canonical per-candidate evaluation, and
selected/rejected status are therefore not recoverable without inference.
No Prompt-19 episodes were exported and no trajectory data were fabricated.

The smallest required control-plane extension is to persist one
`candidate_trajectory` record per Trace and GEPA proposal with those four
fields, without changing proposal, evaluation, or selection behavior.

Final status:

`RETURN_TO_CONTROL_PLANE_FOR_TRAJECTORY_PROVENANCE`
