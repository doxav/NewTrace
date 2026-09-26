STATUS: PARTIAL

Low reasoning effort resolved the two observed PRISM empty-output failures. Two independent SkyDiscover PRISM one-attempt checks and one Trace check produced valid candidates; completion lengths were 886–996 tokens, including 51–74 reasoning tokens. The earlier default-reasoning failures consumed 32,000 completion tokens each and produced no code. This establishes usable generation, not a guarantee against later failures.

All S0–S5 gates passed. All eight five-attempt pilots completed. The first strict run, PRISM SD-EVOX, is in progress; the validated runtime is frozen in `artifacts/strict_source_hashes.json`.

| Task | Arm | Best score | Valid / 5 | Policy switches |
|---|---|---:|---:|---:|
| prism | SD-EVOX | 24.46738724 | 2 | 1 |
| prism | SD-FIXED | 25.80596345 | 3 | 0 |
| prism | TRACE-FIXED | 24.04223387 | 4 | 0 |
| prism | TRACE-RECURSIVE | 23.31466986 | 5 | 2 |
| signal_processing | SD-EVOX | 0.50011137 | 5 | 3 |
| signal_processing | SD-FIXED | 0.50493377 | 5 | 0 |
| signal_processing | TRACE-FIXED | 0.50325736 | 4 | 0 |
| signal_processing | TRACE-RECURSIVE | 0.59696759 | 5 | 0 |

These pilots validate execution only. They do not establish relative optimizer quality. One Signal Trace S4 attempt failed because generated SEARCH blocks did not match the parent; the bounded retry passed, and both attempts remain recorded. Trace PRISM deployed two real optimizer proposals with the solution population preserved. Signal Trace made no switch because stock stagnation conditions did not fire.

PRISM combined score must be read alongside success rate: pilot final success rates range from 0.56 to 1.00. A higher combined score can reflect successful placements on fewer cases. Full comparisons will retain native component metrics.

Current Novita/low series: 105 HTTP attempts, 462716 reported tokens, $0.06353948 reported cost; 0 calls lack cost. Earlier provider/default-reasoning series are archived separately.

Trace uses CP-A with explicit nonportable/nonpromotable control-plane override. CP-B is excluded from live execution because its empty-response fallback changes the frozen token limit. Model, provider, session, temperature, token ceiling and timeout otherwise remain fixed.

Strict and advanced outcomes: not yet measured. No conclusion about Trace versus fixed policy, EvoX versus fixed policy, or additional freedom is warranted.

Validation: 30 EXP22 unit tests and 162 targeted Trace tests passed (six integration cases deselected). All eight pilot gates passed. See `artifacts/exp22_unit_tests.txt`, `artifacts/s4_validation.json`, and `artifacts/s5_validation.json`.

Reporting audit: stock EvoX also creates a guide pool inside its context builder. Raw HTTP records retain the fallback role label for that pool. Derived analysis resolves those records only by exact matches against the source-hashed stock guide prompts, and stores request-index evidence. Total calls and cost already include these requests. No runtime code changed.
