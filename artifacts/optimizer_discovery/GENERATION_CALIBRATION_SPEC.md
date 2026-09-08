# Phase 0 generation-readiness calibration — preregistration

Authorized on 2026-09-08 after the original 0/3 generation failures. This is a
new engineering calibration; it does not replace or reinterpret those failures.
The original Phase-0 specification, requests, fixture and midpoint smoke remain
immutable. No model replacement, holdout access or performance claim is authorized
by this protocol. The matching JSON fixes request labels, settings and thresholds.

## Question and controlled changes

Can the original open-ended prompt produce executable, history-responsive optimizer
programs with a larger completion allowance, retaining the exact DeepSeek model?
OpenRouter's dated metadata snapshot advertises reasoning enabled by default at
high effort, with low effort supported. That is metadata, not proof of the historical
requests' reasoning allocation. Preserve numeric reasoning usage in new responses.

All requests retain the original prompt (SHA-256 in JSON), temperature 0.6, top_p
1.0, concurrency 1, no cache, no empty-output retries, and the same provider/model.
Use a common 300-second request timeout in this separate calibration to avoid
confounding a larger token allowance with the old 120-second timeout. First run
one 3,000-token control with LLM seed 17. It is diagnostic, not a selection gate.

Then run all three pilot seeds 17/18/19 at 8,000 tokens with default reasoning.
If its pilot gate fails, run 8,000 tokens with explicit low reasoning. If that
fails, run 16,000 tokens with low reasoning. Select the first qualifying setting;
do not choose by objective score. If none qualify, report NO-GO. This changes one
generation setting between successive pilot configurations. No prompt tuning.

Before confirmation, persist selection and its pilot evidence. Run exactly ten
fresh LLM seeds 101–110 at that one setting, including failures in the denominator.
Do not try another configuration after seeing a failed confirmation batch.
Maximum 20 logical requests (one control, up to nine pilots, ten confirmations).
Only transient transport failures receive the existing three retries (2/4/8s);
all attempts remain. Worst-case allowance: 80 attempts, 1,280,000 completion tokens.
Report measured latency, token usage and provider-reported cost where available;
do not substitute estimated cost for missing billed cost.

## Prospective readiness gates

Each generation must finish normally (`stop`), pass the unchanged portable program
validator, and complete exactly eight objective calls at each fixture seed 0/1/2.
The same 2-second proposal timeout, deterministic replay, bounds, finite outputs,
signature and sanitized subprocess environment apply. No score threshold.

Additionally probe two synthetic histories of equal length at each length 2 and 8,
using the same coordinates and reversed value rankings. Coordinates for entry i
are [-4 + 8*i/(n-1), 3 - 6*i/(n-1)]; values are i in one history and n-1-i in the
other. Use local seeds 0/1/2, unchanged bounds [-5,5]^2, and existing deterministic
proposal validation. All twelve probe outputs must be valid. A program is
history-responsive if at least one matched pair changes its proposed point.
These are public engineering inputs, consume no objective calls, and are never
shown to the generating model. This finite probe is not a universal semantic test.

Pilot qualification: 3/3 fully valid generations, at least 2/3 history-responsive,
and effective menu size at least 2 on the actual common fixture trajectories.
Confirmation green: at least 9/10 fully valid, at least 8/10 fully valid and
history-responsive, and effective menu size at least 2. Use the existing menu
measurement; code differences alone never establish behavioral diversity.
Unknown behavior equivalence cannot pass. Successful attempts must take at most
300 seconds and report completion usage within the configured token allowance.
Missing required token telemetry fails readiness, rather than silently passing.

These thresholds screen out a pipeline dominated by unusable or trivial outputs
while allowing one observed generation failure. Ten trials do not establish a
population reliability of 90%; publish that statistical limitation explicitly.
No early stopping of a pilot or confirmation batch based on poor outputs.

## What green permits

Phase 0 green requires this confirmation gate plus passing affected offline tests,
lint/format checks, preserved historical evidence and unchanged execution validity.
It permits freezing generation settings and preparing a preregistered Phase-1
experiment. Phase-1 benchmark execution separately requires task-family/split,
baselines, compute accounting, replication and success criteria to be registered.
It does not establish performance superiority, recursion benefit or transfer.

Tests must cover registered-setting dispatch, telemetry preservation, history
response versus constant/length-only output, invalid probes, gate boundary cases,
missing/duplicate batch members and behavior collapse. Commit this specification
before live requests, then commit the tested implementation before live requests.
