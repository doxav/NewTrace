# Phase 0 preregistration

Purpose: finish the measurement instrument and a portable optimizer-program interface,
then establish readiness with a tiny calibration. No performance or recursion claim.
Non-goals: Phase-1 benchmark, FunSearch/OpenEvolve, sandbox infrastructure, new memory.

## Contract and validity
One UTF-8 optimizer.py exports exactly `propose(history, bounds, seed)` (three positional
parameters). Minimization; history is JSON records `{x: [float, ...], value: float}` of
prior evaluations only; bounds is a nonempty list of finite [low, high] pairs; seed is
an integer. Return one finite, correctly dimensioned, in-bounds numeric list/tuple.
Only standard-library imports; no objective source, split labels, LLM or credentials
in the API. Fresh isolated Python process, clean temporary directory, allowlisted
environment and hard 2-second timeout per invocation. This is NOT a security sandbox.
Syntax/import/signature errors, exceptions, timeouts, invalid outputs and observed
seed nondeterminism are typed invalid; objective values are absent, never penalties.
Two fresh executions per proposal test exact local reproducibility; this detects
observed nondeterminism, not a proof of universal determinism. Logs are bounded.

## Deterministic calibration
Public engineering fixture only: 2-D shifted sphere, bounds [-5,5]^2, shift
[1.25,-0.75], optimum 0, eight objective evaluations, seeds [0,1,2]. No holdout.
Each successful proposal consumes one objective evaluation; invalid proposal stops
that trajectory, retaining prior evaluations and consumed budget. Reproducibility
checks do not call the objective. No quality threshold; a poor valid program stays.

## Menu evidence
Persist actual evaluation observations automatically in canonical Trace/GEPA results
and legacy runs. Record declared_menu_size (null for adaptive/unknown menus),
evaluated_candidate_count, valid_candidate_count, effective_menu_size,
menu_collapsed, basis_of_equivalence. Duplicates and invalid observations remain.
Compare candidates only on shared evaluated inputs/phase; never mix holdout into
search evidence. Prefer evaluator-declared behavior signatures (e.g. actual proposal
trajectories or choices/rankings); otherwise report observed metric-vector equivalence
explicitly, without asserting behavioral equivalence. Missing comparable evidence
is unknown (null), never a pass. Source bytes identify artifacts, not behavior.
No generic inference that distinct code, scores on different tasks, or stochastic
prose outcomes establish search headroom. Existing result files remain readable.

## Frozen live configuration and run set
OpenRouter, deepseek/deepseek-v4-flash-0731, OPENROUTER_API_KEY loaded privately
from .env or its OPENROUTER_API_KEY_SOURCE. Existing project LLM abstraction.
Temperature 0.6, top_p 1.0, max_tokens 3000, concurrency 1, request timeout 120s,
seed 17. Two identical generation requests test observed provider reproducibility;
a third clean smoke request uses the same settings. Every response is parsed by the
same parser and evaluated at seeds [0,1,2], budget 8. All attempts retained.
Up to three engineering retries after an initial transient failure, backoff 2/4/8s;
never retry valid poor or invalid generated programs. No model substitution or
parameter escalation. Preserve safe request metadata, usage, response, code,
validity and evaluations. If seed is explicitly unsupported, retain the rejection
and report external stochasticity; no silent settings change. An unavailable exact
model or persistent external outage is a recorded blocker after offline completion.

## Tests and acceptance
Tests first: syntax/import/missing function/signature/exception/timeout, shape,
NaN/inf/bounds, nondeterminism, valid program, sanitization, input validation,
exact accounting, parser, and trainable artifact/control-plane integration.
Menu tests: byte-different equivalent code/prose, invalids, duplicates, score ties,
true distinct behavior, incomparable panels and persistence. Run nine requested
baseline suites, all new and affected tests, broad offline recursive tests; lint
changed code and git diff --check. Refresh source readiness with honest pre-CI state.
A–L acceptance is exactly the objective file: offline green, trustworthy menu evidence,
documented executable contract, typed validity, timeout/environment, exact fixture
budget, canonical integration, real model validation or genuine provider blocker,
uncontaminated history, final green and explicit Phase-1 readiness. Phase 1 may begin
protocol/benchmark design after these gates; no full scientific benchmark is supplied.

## Historical constraints
H1 ceiling refuted; H2 conditional amortisation supported; H3 signature-bound artifact
transfer failed; H5 shared routing optimum supported. No changes to those claims.
Prose noise, single-example/inert knobs and EXP-12/13 remain historical limitations.
Historical probes document this same model, but complete frozen request metadata and
EXP-13's ephemeral found-prompt input are not established reproducible. Fresh calibration
is independent and will not be appended to EXP-12/13. MC-b/d fixes concern future runs.
