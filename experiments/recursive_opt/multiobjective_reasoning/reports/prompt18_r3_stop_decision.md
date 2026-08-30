# Prompt 18-R3 stop decision

## A. Exact control-plane identity

- git HEAD: `8077703218d087c8b458d029ca581ce7dcd745b6`
- runtime tree: `5b460d771ca0b0f9bd914b2c8330860e6f5771a8447d40e50db0d554986e0642`
- registry: `f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`
- required workflow run: `32583578664`
- required job: `97056444766`, completed / success

## B. Readiness

- Exact required focused matrix: `202 passed`.
- Experiment eligibility tests: `13 passed`.
- Selected-task offline A/B/C/D contract: `20/20` assertions passed.
- Strict evaluator registry mode: `output`.
- Accepted optional skips: two optional OTEL/system-monitoring telemetry tests
  listed with exact node IDs and reasons in `preflight_skips.json`.
- Critical or unexplained skips affecting Experiment 0: zero.
- Ruff and `git diff --check`: pass.

## C. Experiment configuration reached

- Selected task: GSM8K.
- Frozen pools: 16 train, 12 validation, 24 holdout; eligibility used 8/8/0.
- Micro subset: 1 train, 1 validation, 1 holdout.
- Model for forward and optimizer roles:
  `openrouter/deepseek/deepseek-v4-flash-0731`, temperature zero.
- Forward: 384 maximum tokens, reasoning disabled.
- Optimizer: 8,192 maximum tokens, reasoning effort low (enabled by the
  documented effort control); the amendments and failed attempts predate any
  successful cross-engine comparison and are preserved.
- Objective: weighted accuracy `1.0` and minimized per-example forward-token
  ratio `0.10`, with invalid rate `<= 0` as a hard constraint.
- Arms: A fixed, B Trace + OptoPrimeV2, C GEPA OptimizeAnything, D Trace
  without validation gating.
- Micro seed count: one. Pilot/main seed count executed: zero.
- Monetary cost: unavailable (`null`); no forecast was run after the failed
  micro gate.

## D. Results at the stop point

| arm | gate | accuracy | token ratio | forward calls/tokens | optimizer calls/tokens | status |
|---|---|---:|---:|---:|---:|---|
| A fixed | pass | 1.0 | 0.9116 | 2 / 722 | 0 / 0 | valid baseline |
| B Trace | pass | 1.0 | 1.0114 | 18 / 5,185 | 1 / 11,200 | nontrivial proposal |
| C GEPA | fail | 1.0 | 0.9116 | 6 / 1,570 | 2 attempted / 0 reported | retained P0 after reflection failure |
| D no gate | not run | unavailable | unavailable | unavailable | unavailable | pilot not authorized |

These are one-unit plumbing measurements, not scientific engine-effect
estimates. No paired delta, uncertainty interval, validation-gate conclusion,
or optimizer winner is estimable.

## E. Mechanistic conclusions

- The fixed and Trace execution/evaluation, output anchoring, accounting,
  artifact persistence, and resume paths work in the live micro-smoke.
- Trace produced a nontrivial candidate under the real production path.
- GEPA did not produce a candidate because its string reflection-call contract
  is not adapted to the messages/`ModelResponse` contract of the guarded role
  client. The provider-free reproducer is in
  `gepa_reflection_client_blocker.md`.
- Validation-gate effect, Trace-versus-GEPA effect, and cost/quality tradeoff
  remain unknown.

## F. Data quality

- Canonical Prompt-19 episodes exported: zero.
- Valid cross-engine trajectories: zero.
- Control-plane and experiment provenance at the stop point: complete.
- Holdout leakage: none observed.
- Dataset ready for Prompt 19: no; GEPA and D trajectories and paired pilot
  evidence do not exist.
- A/B micro outputs are retained as provisional infrastructure evidence. C is
  invalid as optimizer-efficacy evidence.

## G. Decision

The Prompt 18-R3 stop condition "GEPA cannot produce a real proposal" is met.
The defect is in the frozen control-plane reflection boundary, so Experiment 0
cannot fix it. Cost forecast, pilot, main experiment, ablations, statistics,
and episode export remain prohibited until a separate minimal hotfix is proven
and required CI/readiness are green again.

RETURN_TO_CONTROL_PLANE
