# Separate interface-only engineering smoke, preregistered before its request

The frozen Phase-0 generation_1, generation_2 and clean_smoke requests each
returned finish_reason=length, 3000 completion tokens and no parseable source.
All three remain failed calibration observations; they are not rerun, replaced,
excluded, or presented as successful discovery. No provider outage is claimed.

This separate pilot addresses only whether the exact model/request configuration
can deliver a trivial code artifact through the contract. It is not a search run,
quality improvement, or continuation of the failed generation task. The goal
objective explicitly permits separate pilot/calibration surfaces for engineering
work and requires generating or improving a trivial implementation.

One request, label interface_smoke. Prompt frozen in engineering_smoke_spec.json:
ask for a complete propose function returning the midpoint of each supplied bound.
No objective source, shift, data labels or holdout is sent. OpenRouter model,
temperature=.6, top_p=1, max_tokens=3000, seed=17, timeout=120s, concurrency=1
and retry policy remain exactly those in phase0_spec.json. Same parser, validator,
canonical evaluator, seeds [0,1,2] and eight-evaluation fixture budget.

Acceptance is parseable valid source that executes the contract; there is no
performance threshold. No valid poor or invalid program may be resampled.
Preserve the complete outcome even if it fails. Successful execution establishes
only the tiny interface smoke; open-ended optimizer discovery under the frozen
prompt/config remains unready on the three failed observations. Phase 1 must
preregister a separate generation calibration before launching a search benchmark.
