# Prompt 18-R3F main stop decision

## Locked implementation and external gate

The transport/runtime fix, Experiment-0 execution amendment, watchdog child
initialization fix, and versioned lock are preserved in branch HEAD
`e0faed98da54c19f1a6da3f1fb8bbf786a5fe04b`. Required workflow run
`32859625717`, job `97840091924` (`recursive-opt v2 offline (required)`),
completed successfully at that exact HEAD.

The locked runtime digest is
`420b5351063a56b0ad274a6c39b6aaa4dc95b9094434600e89ea79f3eccc8872`.
The Experiment-0 source digest is
`851c47563a1365733988babf49cdbf5be98d9a7be534b8aaf82cf7f70550c6b0`.
The control-plane registry digest remains
`f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`.
The Experiment-0 registry digest remains
`18a7efd58f3d265a723fa62efb89d2dc07082afbcdf16c9470e6d4eb93a77121`.

Every persisted live provider response identifies provider `openrouter` and
model `deepseek/deepseek-v4-flash-0731`. The API credential was loaded from the
local ignored environment reference and is not persisted in this evidence.

## Revalidation gates

The fresh A/B/C micro-smoke passed under the new lock:

| arm | accuracy | invalid rate | token ratio | forward calls/tokens | optimizer calls/tokens | proposed/evaluated | artifact changed |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 1.0 | 0.0 | 1.090909 | 2 / 864 | 0 / 0 | 0 / 0 | false |
| B | 1.0 | 0.0 | 0.570707 | 18 / 4,754 | 1 / 7,821 | 3 / 6 | false |
| C | 1.0 | 0.0 | 1.095960 | 8 / 2,360 | 1 / 2,595 | 3 / 3 | false |

Both optimized arms exercised a real proposal path. No transport or semantic
retry occurred in the micro-smoke.

The main-size Trace transport stress gate also passed with full frozen train
and validation pools, one diagnostic holdout example, `num_threads=4`, and the
frozen transport policy. It produced accuracy `1.0`, invalid rate `0.0`, token
ratio `1.13636`, 120 forward calls / 32,014 forward tokens, one optimizer call /
4,871 optimizer tokens, two proposed candidates, 35 evaluated candidates, and
no transport retry or hang. It is infrastructure-only evidence.

## Preserved partial main matrix

The new main namespace contains 12 complete canonical units. They are
incomplete/provisional scientific evidence and must not be combined with a
future restarted matrix.

| seed | budget | arm | accuracy | invalid rate | token ratio | status | artifact changed |
|---:|---:|---|---:|---:|---:|---|---|
| 0 | 6 | A | 1.000000 | 0.000000 | 0.975747 | success | false |
| 0 | 6 | B | 1.000000 | 0.000000 | 1.044149 | success | false |
| 0 | 6 | D | 0.625000 | 0.041667 | 0.394195 | invalid | true |
| 0 | 6 | C | 0.958333 | 0.041667 | 0.989086 | invalid | false |
| 0 | 12 | A | 0.916667 | 0.041667 | 1.011203 | invalid | false |
| 0 | 12 | B | 1.000000 | 0.000000 | 1.009624 | success | false |
| 0 | 12 | D | 0.791667 | 0.000000 | 0.471726 | success | true |
| 0 | 12 | C | 1.000000 | 0.000000 | 0.741390 | success | true |
| 1 | 6 | B | 1.000000 | 0.000000 | 0.822279 | success | true |
| 1 | 6 | C | 1.000000 | 0.000000 | 0.704830 | success | true |
| 1 | 6 | A | 1.000000 | 0.000000 | 1.002973 | success | false |
| 1 | 6 | D | 0.958333 | 0.000000 | 0.823086 | success | true |

Scientific invalidity in three units was recorded without stopping the matrix,
which causally revalidates the earlier scientific-vs-infrastructure correction.
No quality, efficiency, engine, or global safety conclusion is valid from the
incomplete matrix.

Across the 12 completed units, Trace arm B recorded four recovered empty-text
responses (23,029 retry tokens), and Trace arm D recorded one recovered
empty-text response (4,243 retry tokens). D also recorded three recovered
transport requests. No completed unit exhausted its transport policy.

## Blocking transport failure

The thirteenth unit, seed `1`, proposal budget `12`, arm `B` / Trace, stopped
during optimizer iteration 5. The canonical recursive-opt result is complete
and records:

```text
TransportRetryError: LiteLLM_completion: transient transport failure after 3 attempts:
litellm.APIError: APIError: OpenrouterException - [Errno -3] Temporary failure in name resolution
```

The optimizer usage records exactly three transient failures, two retry
attempts, one exhausted request, and zero recovered optimizer requests for the
failed logical call. Four earlier forward transport failures in the same unit
were each recovered. At failure, the unit had 206 forward calls, five metered
optimizer logical calls, 91 candidate evaluations, 141,009 accounted tokens,
and 1,440.42 seconds of active wall time. No resource budget was exceeded.

This is the frozen policy's intended bounded failure: the identical provider
request was attempted at most three times, did not recover, and stopped the
matrix. Retry attempts were not increased and the request timeout was not
changed.

## Secondary stop-reporting defect

After receiving the canonical infrastructure-error result, the Experiment-0
parent correctly identified that infrastructure checks failed. While building
the progress document, however, `_progress_document()` treated every
`safety_passed=false` run as if `metrics.invalid_rate` existed. Infrastructure
error results have an empty metrics mapping, so the parent raised:

```text
KeyError: 'invalid_rate'
```

The last atomic main checkpoint therefore remains at 12/40 with
`stopped_after=null`; the failed unit's canonical run result, usage, budget,
and error artifacts remain intact in its output tree. This is a separate
Experiment-0 stop-report persistence defect and must be fixed before a future
restart. The failed unit must not be resumed or interpreted as efficacy
evidence.

## Consequences

- Main completion: 12/40 canonical scientific units; the thirteenth unit is
  infrastructure failure evidence only.
- The complete paired analysis and retry aggregation were not run.
- The candidate-trajectory audit was not run because it follows 40/40
  infrastructure-complete units.
- The matrix must not resume from unit 13 under this source lock.
- A future repair must first make infrastructure-error stop persistence robust,
  then revalidate and restart the full frozen matrix without changing the
  scientific protocol or transport attempt ceiling.

`BLOCKED_TRANSPORT_INFRASTRUCTURE`
