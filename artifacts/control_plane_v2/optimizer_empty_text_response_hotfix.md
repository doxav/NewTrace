# Optimizer empty-text response hotfix

## Pre-patch causal finding

The failure is confirmed at `opto/optimizers/optoprime_v2.py` in the
`OptoPrimeV2.call_llm()` / `_step()` boundary. The failed Experiment-0 arm is
configured with `engine.config.optimizer = "OptoPrimeV2"`. Before this hotfix,
`call_llm()` assigned `response.choices[0].message.content` directly to
`response` (line 676 at the failed runtime) and returned it without validating
its type or content. `_step()` then evaluated `"TERMINATE" in response` (line
623). A provider-style response with `content = None` therefore returned
`None` from `call_llm()` and raised exactly:

```text
TypeError: argument of type 'NoneType' is not iterable
```

The provider-free reproducer used a fake response with
`choices[0].message.content = None`. It observed `OptoPrimeV2.call_llm()`
returning `None`; applying the next production membership test produced the
exact exception above.

Trace's LiteLLM backend calls `retry_with_exponential_backoff()`. That helper
returns immediately whenever the provider callable returns normally. A fake
successful response object without final text was returned after one call even
when `max_retries=3`; only raised provider exceptions enter its retry branch.
The existing provider retry therefore cannot repair this semantic response
failure.

Repository search found one analogous `"TERMINATE" in response` site in the
older `OptoPrime` implementation. It is not on this arm's execution path,
because the preserved raw pilot spec selects `OptoPrimeV2`. No other source
location on the configured execution path was found that both consumes this
provider response and can produce the exact observed membership error.

## Evidence classification

The same exception is preserved in the earlier micro attempt and in the
seed-0, proposal-budget-6 Trace pilot arm. An identical later micro execution
succeeded. The defect is intermittent because final text is absent only in
some otherwise successful provider responses; normal responses continue down
the unchanged parsing path.

Proven:

- the failed arm selected `OptoPrimeV2`;
- `content = None` reproduces the exact exception at its next statement;
- the provider-exception retry layer does not retry successful response
  objects with missing text;
- repeated live runs have observed both this failure and normal textual
  responses under the same frozen model/profile.

Not proven for the stopped pilot response:

- its provider `finish_reason`;
- whether its reasoning-token allowance was exhausted;
- the upstream reason the provider omitted final content.

One older diagnostic artifact records `finish_reason = "length"`, reasoning
presence, and missing content for a separate occurrence. That artifact supports
the general failure shape but does not establish the cause of the later pilot
response. The preserved earlier classification naming reasoning exhaustion is
therefore historical evidence, not a causal claim for this hotfix.

## Repair boundary

This is an infrastructure reliability repair. The recursive-opt optimizer-role
boundary will require non-empty final text and retry the identical guarded
request once. Both attempts remain metered by the existing guarded client, and
the same boundary serves Trace and GEPA. `OptoPrimeV2.call_llm()` will also
reject missing text explicitly when invoked outside recursive-opt, without an
independent retry loop.

## Implemented response contract

The private recursive-opt optimizer-role adapter accepts either a non-empty
direct string or non-empty `choices[0].message.content`. It never substitutes
`reasoning`, `reasoning_content`, or an object representation. When final text
is absent, it sends the same arguments through the same `_GuardedRoleClient`
exactly once more. The guarded layer therefore charges both returned attempts
to optimizer calls, tokens, provider-reported cost, and wall time. A second
empty response raises:

```text
RuntimeError: optimizer LLM returned no final textual content after 2 metered attempts
```

The wrapper is installed only for the optimizer role. Trace receives it as the
`OptoPrimeV2` LLM; GEPA receives the same wrapper beneath its text/chat
reflection adapter. Forward, judge, and feedback clients are unchanged.

Canonical optimizer usage now includes `empty_text_responses`,
`semantic_retries`, and retry-only prompt/completion/total token and cost
counters. Safe attempt metadata records model, finish reason, content presence,
reasoning-presence boolean, token counts, and a reasoning-token count when the
provider exposes one. It does not retain prompts or reasoning text. Direct
`OptoPrimeV2.call_llm()` validates final text and raises explicitly but does not
add a second retry loop.

The task, GSM8K pools, P0 artifact, models, request parameters, reasoning
settings, objective and weights, invalidity constraint, arms, seeds, proposal
budgets, holdout policy, and success criteria are unchanged.

## Provider-free causal verification

- empty then text: both identical guarded requests were made and all 26 fake
  tokens were charged; one empty response and one semantic retry were recorded;
- empty then empty: exactly two guarded calls and 14 tokens were recorded,
  followed by the explicit error with no candidate fabricated;
- normal text: one call, no retry, unchanged response;
- real Trace path: the first empty response retried, a valid second response
  produced/evaluated a changed candidate, and both calls/tokens reconciled;
- installed `gepa==0.1.4`: actual `optimize_anything()` reflection retried the
  empty response, proposed/evaluated a better candidate, and kept holdout out of
  optimization with sockets blocked;
- direct `OptoPrimeV2`: `content=None` raises a missing-text `RuntimeError`, not
  a `NoneType` membership error.

Pre-CI results with provider keys removed and external sockets blocked:

- six causal seams: **6 passed**;
- focused control-plane/hardening/spec/objective matrix: **212 passed**;
- mandated recursive regression: **317 passed, 2 skipped, 1 warning**;
- complete unit suite: **497 passed, 3 skipped, 1 warning**;
- Experiment-0 tests and full offline contract: **23 passed**, with all 20
  offline assertions green;
- clean-kernel notebook: **1 passed**;
- changed-file Ruff, `git diff --check`, and credential scan: **passed**.

The two common skips require unavailable optional graph/telemetry backends. The
complete-suite-only third skip requires the Graphviz `dot` executable. No
Trace, GEPA, or Experiment-0 test skipped.

The authoritative runtime digest is
`ba4836d9f43cffcd0271086932745b270d75478b5287a7d8100be4928b623cbc`;
the registry digest remains
`f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`.
Required Actions run `32669603929`, job `97268256178`, completed successfully
for implementation commit `d63746afbb88d6193cbfedf2932b256d9f33b6e4`.
Digest-based readiness is promoted to true for that CI-verified implementation.

## Preserved pilot evidence

The six-run stopped pilot is versioned under
`reports/pre_empty_text_retry_runtime/`. Its pilot report SHA-256 remains
`bb24afbae79e945de0eeab97a971f50a5fee63fb619fc37dde9df0d4123a3d04`.
Its efficacy evidence is incomplete/provisional and will not be combined with
the replacement pilot; its infrastructure-failure evidence remains valid.
