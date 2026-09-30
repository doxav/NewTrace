# EXP-16 / P1-E1 — production long-prompt engineering check

Prospective engineering check, separate from every efficacy comparison. Freeze
this document, the driver, owner, analysis code, evaluator, prompts and environment
before generation. This is not a confirmatory result or a candidate-selection pilot.

Use the actual production R path for exactly two completed responses, outer16601,
namespace `P1-E1`. The configuration records the common four-arm machinery, but
this check executes **R only**. Do not execute independent, code-only or width-two
generation here. Do not evaluate validation or audit tasks. No generated program
from this check enters P1.

Use 24 fresh balanced training instances, two independently derived local seeds
each, B32. Use the unchanged seed optimizer and the common Phase-0 subprocess
contract. Evaluation workers8; live generation concurrency1. The R prompt consumes
actual propagated current-parent Trace feedback, with indexed observations,
incumbent/improvement history, the full best-so-far curve and one aggregate training
normalized-regret-AUC scalar. No per-task normalization constants or hidden data.
Maximum total request text524288 characters; reject a larger request before sending.

Model `deepseek/deepseek-v4-flash-0731` through OpenRouter, temperature0.6,
top_p1, max_tokens32000, native `reasoning.effort=low`, timeout300s, cachefalse,
empty_response_retries0, wrapper max_retries1, client num_retries0. Preserve the
existing bounded transport retry policy, every completed response and safe receipts.
Do not replace empty, invalid, truncated or poor completed responses.

Check exact request/source hashes, actual callback and completed-response counts,
prompt sizes, source/execution validity, typed failure propagation, allocated and
actual objective calls, cache reuse, output limits, tokens and elapsed time. Resume
the completed run without a callable client and verify no response changes or new
generation occurs. This is a real production check in addition to mocked unit tests.

Feasibility passes if both allocated callbacks/responses complete, the full feedback
fits, all evidence and replay checks pass, and at least one generated program is
eligible on all training trajectories. Do not rank generated policies to choose
P1 settings. If either response is ordinarily invalid, preserve it. An engineering
defect requires a tested correction and a new documented engineering version.
If both programs are invalid, inspect generation feasibility before authorizing the
larger diagnostic; do not relabel invalidity as an implementation bug.

The driver records wall, monotonic and boot clocks so host suspension is visible.
The maximum logical training allocation is3 pool entries ×48 trajectories ×32
=4608 objective calls, or9216 deterministic-replay subprocesses, before cache
savings/invalid early termination. Repeated internal Trace lookups use the same
complete evaluation cache. Normalization reference calls are separate.

## Resource-accounting clarifications before freeze

The installed client forwards timeout300 to HTTPX connect/read/write/pool
operations. A loopback test through the actual project/OpenRouter adapter confirms
that continuing response bytes can keep a request alive longer than300 seconds.
This is not an absolute generation deadline, and the parameter is not silently
dropped. Preserve active/elapsed durations and any suspension separately.

The48 unique P1 tasks imply6144 reference objective calls for one complete unique
normalization design. Physical reconstructions across separate processes/resumes
are not fully instrumented and must not be mislabeled as exactly6144 physical
calls. Candidate objective allocations/cached outcomes are counted separately.
For P1-E1 only24 training instances are evaluated; its unique preparation design
is3072 reference calls, with the same reconstruction caveat.
