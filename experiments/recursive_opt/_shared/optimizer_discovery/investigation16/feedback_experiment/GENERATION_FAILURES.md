# F1 — three designated generation failures, format audit only

This note examines only the three completed responses designated during the run.
It opens no validation results, ranks no condition, and does not repair, execute,
or select any code fragment. All three responses remain completed, consumed,
invalid proposal slots under the unchanged F1 protocol. Machine-readable
diagnostics and every fragment's hash are in `generation_failure_audit.json`.

| Slot | Recorded finish / completion tokens | Final content | Independent AST inspection | Recorded provider |
|---|---|---|---|---|
| 16301 / anytime_code | stop / 26,594 | 29,373 characters, 17 recognized fenced spans | 14 spans produce an AST; none exports the required `propose` API | DigitalOcean |
| 16302 / anytime_code | length / 32,000 | 39,763 characters, 10 recognized fenced spans; terminal code is cut off | 5 spans produce an AST; none exports the required `propose` API | DigitalOcean |
| 16301 / anytime_rich | length / 32,000 | `content=null`; no final text exposed | No final code span to inspect | DeepInfra |

**The first two outputs contain code; they do not contain a usable contracted
optimizer.** The first repeatedly develops `choose(...)` with different arguments,
`ss` history and `loss` fields. Its last span still exports `choose`, not
`propose(history, bounds, seed)`. Several drafts use 200/300-step schedules despite
the 32-evaluation request. The second drifts toward a complete
`solve(x0, bounds, budget, objective)` routine that calls an objective, interleaving
helper functions, prose and incomplete drafts. This is a different interface from
the required single black-box proposal function. AST parsability alone does not
establish successful compilation, execution, deterministic behavior or validity.

The frozen extractor deliberately accepts plain source or **exactly one Python
code fence**. It rejects these multi-span responses before any source is evaluated.
This strictness is a proximal reason for `parse_status=unparsable`, but relaxing
it to “take the last block” would not fix either response's missing entry point.
Combining fragments, renaming functions, rewriting history fields or choosing a
preferred draft would be manual program construction, not recovery of the
registered proposal. No such change was made.

**The prompt contract is explicit.** The first message names the exact signature
and three positional parameters, the `{x,value}` history, the 32-evaluation budget,
and exactly one complete Python code block without explanations. The final user
message supplies a parent which itself exports the required `propose` function.
Total prompt text is 3,195 characters for 16301/code and 8,128 for 16302/code; these
failures are not evidence that a huge training trace hid the API. Both have no
evaluated feedback in their request. Placing a concise signature/output reminder
after the parent could be tested prospectively, symmetrically across arms; its
benefit is a hypothesis, not an established correction.

**These fragments are final-channel text as exposed by the client.**
`opto/features/recursive_opt/spec.py::_optimizer_response_text` reads only
`choices[0].message.content`; it does not substitute `reasoning` or
`reasoning_content`. F1's recorder uses that function. The visible repeated
planning prose therefore cannot be blamed on the research code intentionally
promoting a reasoning channel into source. The record does not preserve the full
provider reasoning payload or wire response, so it cannot distinguish model
instruction-following failure from upstream channel/response handling with
certainty. The null rich response is accurately described as **no final text
exposed**, not “the model generated no internal code.”

The cap is binding for the two `length` outcomes. It is not binding for the first
`stop` outcome, which ends below 32,000 while still using the wrong API. More tokens
alone cannot be projected to solve format and contract drift. For the rich
response, native completion is reported as 32,000 while native reasoning is
33,818; the receipt also gives 33,818 in its other completion-token field. Preserve
these different counters; do not add reasoning to completion, assume a comparable
subset, or infer a max-token violation from the larger count.

Provider names above are descriptive associations in three deliberately selected
failures, not a provider comparison. Request IDs, actual model identifiers, usage,
receipts and timing remain preserved. The run includes independently documented
host suspension, including an additional 5,577.196 seconds between clock
observations. Calendar elapsed time is not endpoint latency. No provider pinning,
model change, prompt edit or extra generation was introduced by this audit.

Content SHA256 values (the extracted scientific source is empty in all three):

- 16301/code: `5af3725deca52b7e3baa53bfdfe5a59bd531ccc4529ebeaa9884a120e6063a41`
- 16302/code: `7fe20101e4f46fc89081fb1b5c1977a3be2a6db4b8bb18f3a01db7a06ccf1524`
- 16301/rich: content is JSON null; no text hash is invented.

The common extracted-source SHA256 is
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
Canonical response/request JSON hashes and all block hashes are in the audit JSON.

Verification: 17 format/final-audit unit tests pass, including real null final
content, multiple independently parsable blocks, wrong APIs, and an infinite
top-level statement inspected only as AST. The null-content fixture first exposed
and then verified a reporting-only `len(None)` correction in the **new, unfrozen
analysis helper**; no execution code, raw response, or scientific value changed.
Black/Ruff checks pass. Final efficacy analysis still waits for all 24 responses
and complete validation.

## Additional designated case: completed provider-error response, 16304/rich

The later `16304/Aanytime_rich/slot_00` response is a different failure mechanism:
**`finish_reason="error"`, final content JSON null**, 18,186 reported completion
tokens and 18,186 reasoning tokens, against a configured 32,000-token limit.
Its matching provider receipt identifies **Morph**, repeats the error finish and
the counters, and reports `native_finish_reason=null`. Response cost,
receipt `total_cost` and `upstream_inference_cost` all report zero; those values
are preserved without interpreting them as a general billing guarantee.

The SDK's recorded monotonic duration is 4,229.333818 seconds (about 70.49 minutes).
The receipt records `generation_time=4228514` and `latency=3860` verbatim. This is
an observed provider-error termination, not a reported token-limit termination
or a demonstrated syntax error in generated code. No final source is exposed,
so the exact frozen parser reports `unparsable` and the source validator reports
`missing_source`. These are downstream usability states; they do not identify
the cause of the upstream error. Detailed upstream failure information and
internal reasoning text are not preserved in this receipt.

`G.complete_slot`, reused by the frozen exact recorder, records a returned
response as completed when the client did not raise an exception. Only raised
client exceptions enter its transport-retry branch. Here the attempt record is
`completed`, has the same response ID, and is attempt 1. F1's registered budget
is 24 responses, with no manual repair or replacement. Its frozen `_response`
integrity check verifies identity and source extraction without requiring a
normal finish. The observed handling therefore follows the frozen procedure;
**no slot is replaced or retrospectively converted into an unfinished request**.

This is nevertheless material to scientific attribution. It is a failure of
the provider/generation procedure to deliver a usable proposal, not evidence
that trajectory feedback produced a syntactically bad optimizer. The declared
deployment comparison retains the common fallback and this end-to-end reliability
variation. A final result cannot cleanly attribute that variation to feedback
content or to Morph from this one routed response. No local execution/metric
defect is demonstrated by this record. All 24 responses and validation must still
complete before efficacy comparisons.

The unfrozen final-analysis helper now reports raw finish-reason counts and
orthogonal termination categories (`provider_error`, `token_limit`, `normal_stop`,
unreported/other), alongside the original parsing and execution validity. Six
new tests first failed and then passed. A complete synthetic 24-slot fixture
checks that the annotation retains the response and leaves all paired metrics
unchanged. Combined format/final-analysis verification: **23 tests passed in
9.71 seconds**. The three earlier designated-failure JSON records remain unchanged;
the new raw-only diagnostic is [provider_error_16304_audit.json](provider_error_16304_audit.json).

Response ID: `gen-1788959588-rNlO6GxQ5YdHztz7POE1`. No candidate execution,
fitness/validation access, generation replacement or frozen-file modification
was performed by this additional audit.
