# Unactivated proposal: bounded recovery from Novita solution rate limits

Status: prepared for review only. No live/root/v9 runtime files were modified; no API requests were made. This candidate deliberately has no execution gates or source lock and cannot be launched through run_stage.py as supplied.

The current v9 runs stopped correctly when Novita refused solution requests with HTTP 429 and error metadata limit_source=upstream_provider_shared_pool. This proposed amendment permits only that specifically identified refusal to consume a failed solution attempt and continue.

## Exact proposed behavior

- The request must have role solution, HTTP status 429, error code 429, error metadata provider_name=Novita and limit_source=upstream_provider_shared_pool.
- Its outbound model, provider routing, session, reasoning effort, temperature 0.7, and max_tokens 32000 must match the frozen configuration exactly.
- The response must have empty content_lengths and finish_reasons, no returned model/provider/generation ID, and passed=False. The original HTTP record and error remain unchanged.
- At most two qualifying refusals are recoverable across the entire run, including nonconsecutive failures. Pause 30 seconds after each qualifying failed generation batch. This is an ordinary next budgeted iteration, not a free request retry or an increased SDK retry count.
- Every consumed HTTP attempt remains in candidate history and the solution curve. The 100 actual solution-call budget stays unchanged. Existing accounting prevents a provider refusal from restoring the fallback database or retrying the same iteration for free.
- A third qualifying refusal is recorded and stops the run without another pause or request. Every other HTTP failure, ambiguous response, missing status, serving-identity violation, meta-call refusal, or altered request retains the existing stop behavior.
- One shared predicate controls both kernel HTTP guards and run_stage final run acceptance. A completed run may therefore pass with up to two clearly recorded failed transport attempts; those HTTP records never become successful records. Successful-response identity checks in the observer are unchanged.

## Review artifacts and verification

review.patch contains the three proposed runtime changes and the new offline regression test. Existing runtime dependencies are copied only to make the candidate independently testable; worktrees is a read-only-use link to the existing frozen framework checkouts.

The candidate test suite passed all 11 tests (see testlog.txt), including the unchanged kernel regressions. New tests exercise the actual stock outer loop: refusal then valid candidate gives two calls/two curve points and preserves fallback; three refusals yield three accounted attempts and stop; another provider error stops immediately. Sleep and generation are mocked, so no API requests or real 30-second waits occur. Predicate tests reject wrong providers/models, HTTP 400, unknown status, changed routing/parameters, unexpected output, and meta-call refusals. Ruff passed across all candidate source, script, and test files.

Activation would require explicit approval of this changed stop condition, a new documented protocol/source lock, and fresh recursive runs. Existing partial results remain separate; no stopped run is relabeled as completed. This proposal does not alter the model, provider, low reasoning effort, evaluator, lossless feedback encoding, policy triggers, or solution budget.
