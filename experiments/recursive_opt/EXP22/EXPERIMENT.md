# EXP22 protocol v1 — 2026-09-26

Current status: Novita continuation under amendment v5; see RESULTS.md for observed gates.
The initial S0 source stop is historical; no full benchmark has begun. This document does not certify an implemented benchmark.
All experiment files belong under this directory. Source repositories remain unchanged.

## Objective and hypotheses

Compare stock SkyDiscover EvoX and Trace recursive_opt meta-policy proposal on
stock PRISM and Signal Processing, holding the solution-evolution kernel fixed.
Test the gain of meta-evolution over a fixed policy within each framework, then
compare frameworks at 100 solution generations. No directional advantage is assumed.

## Source identity gate

Required Trace checkout: `/home/xav/code/Trace`, branch `recursive_opt`.
The reference currently resolves to `846580defe935195c1f5f39d6336079ef6ff1e10`.
Observed checkout: `codex/exp17-parent-selection-exp18-memory-pareto`,
HEAD `7e701b40485b9880ccfbb64c1faaabb22401c294`, with relevant dirty files.
These are distinct versions, separated by 40 commits. A detached worktree of
current HEAD would isolate dirty changes but would not satisfy the required
branch identity. Do not silently substitute it or rewind the user's checkout.

SkyDiscover: `/home/xav/code/evo-compare/repos/skydiscover`, detached HEAD
`3f7a611fe83980970dd14f1d65c49e40de61b7df`. Use only stock files at this coherent
revision. Ignore the untracked benchmark ZIP archives. `manifest.json` records
all six benchmark hashes and comparison against the supplied locks;
`artifacts/source_hashes.json` records relevant implementation/config hashes
at HEAD and on disk. Timestamped evidence includes sanitized relevant diffs.

The current branch mismatch fails S0. Implementation and paid execution stop
until this non-negotiable input is resolved. No branch switching, source patches,
or dependency changes are performed by the audit.

The latest local research/status material was inspected, including the
2026-09-24 recursive_opt study and subsequent RESEARCH_LOG and o1_learning/EXP22
status. That historical EXP22 is a different HotpotQA experiment, not this study.
Its results are not imported as evidence here.

## Intended transport and control plane

Every role must request OpenRouter `z-ai/glm-5.3-flash`, with
`provider={"only":["DeepInfra"]}` and exactly
`session_id="benchmark-PRIMS-SIGNAL-run-001"`. Read the credential from
`OPENROUTER_API_KEY`; never serialize it. This explicit protocol supersedes the
older Experiment-0 model instruction for this experiment only.
Planned generation settings: temperature 0.7 subject to stock-default inspection,
max_tokens 32000, timeout 600 seconds, no request seed.

Before optimizer calls, intercept final SDK requests for every framework and
verify serialized payloads. Three tiny live requests (direct, SkyDiscover,
Trace) must establish the actual model and DeepInfra serving provider using
response or generation metadata. A routing request alone is insufficient.
No transport claim is made until this evidence exists.

SkyDiscover must use its `init_client` extension with `search.type=evox`,
`share_llm=true`; test inheritance into meta/evaluator/guide pools and reject
other model instantiation. CP-A would use a resource-injected Trace llm_factory,
explicitly marking test mode/non-portable/non-promotable when required.
CP-B would use an isolated worktree with a narrowly normalized upstream-routing
field, keeping provider identity validation intact. Compare both using identical
mocked policy proposals, feedback, validation and persistent DB migration.
Prefer CP-B only after targeted tests pass. Neither variant has been built or
selected because S0 failed. No trace_cp_b.patch is claimed.

Trace must register versioned module/engine/evaluator/dataset refs and execute
through normalize_spec → compile_plan → run_spec/execute_plan. Persist raw and
normalized specs and resolved plans before optimizer calls. Use real code-artifact
optimizer machinery; surface.kind metadata alone cannot establish dispatch.

## Execution and evaluator controls

Reuse stock CoEvolutionController, solution generator, benchmark prompt,
initial EvolvedProgramDatabase source, policy validator, LogWindowScorer,
variation operators, fallback behavior and population migration. Replace only
the meta-policy proposer. The solution population must survive switches.
Give Trace only observations available to EvoX and one proposal per switch.

Use exact stock PRISM evaluation and Signal cascade (EVAL-A) on both arms.
If impossible, preregister forced full evaluation on both as separate EVAL-B.
Before optimization, evaluate each stock initial program ten times sequentially
through direct stock, Trace adapter and any execution boundary. Require scalar
non-timing agreement within 1e-12 and equivalent broken-candidate ranking.
Expected combined scores: PRISM 21.891622105209393; Signal full
0.49904861783269006. These are supplied expectations, not measured results here.

Create an owned environment and resolve both editable projects plus required
benchmark dependencies. Document conflicts instead of silently changing package
constraints. Record exact package/Python versions. The current audit uses only
the standard library and does not establish benchmark dependency compatibility.

## Arms, budget and order

Four arms per task: SD-FIXED, SD-EVOX, TRACE-FIXED, TRACE-RECURSIVE.
Fixed arms retain all machinery but disable meta switches for the horizon.
Each full arm has 100 solution generations; this does not equalize total compute.
Search and evaluation remain sequential, concurrency 1; use local RNG seed 42
where supported, without adding an LLM request seed.

PRISM order: SD-EVOX, TRACE-RECURSIVE, TRACE-FIXED, SD-FIXED.
Signal order: TRACE-RECURSIVE, SD-EVOX, SD-FIXED, TRACE-FIXED.
Shared provider/session caching confounds cost and latency even with balanced order.

Mandatory gates: S0 provenance/dependencies; S1 evaluator parity; S2 transport;
S3 every intended Trace plan compiles; S4 one changed, evaluable finite-scoring
candidate per task/framework; S5 five-generation pilots per active arm, with
connected policy optimization and persistent population. No full runs before all pass.

## Stopping, metrics and analysis

Check structural integrity at generations 10, 30, 60 and 100. Stop for evaluator
or routing divergence, broken accounting, population reset, disconnected/no-op
policy updates, sustained ten-attempt invalidity, or universally rejected policies.
Flat quality alone is not a stop condition. Record reasons and evidence in STOP.json.

Record initial/final/gain/relative gain, best-score AUC, first improvement and best
iteration; policy switches, validation failures, window gains; solution/meta/guide
calls, transport/semantic retries, tokens including cache, cost and wall time.
Compare SD-EVOX−SD-FIXED, TRACE-RECURSIVE−TRACE-FIXED,
TRACE-RECURSIVE−SD-EVOX and TRACE-FIXED−SD-FIXED. Report PRISM native pressure
metrics and Signal composite, correlation, noise reduction, slope changes,
lag error and success rate. Timing is not quality. Plot trajectories with policy
switches and cumulative calls/tokens/cost only when measured data exist.

One run per arm is a controlled single-run benchmark, not statistical replication.
Evaluator cases are not independent LLM optimization replicates; no pseudo p-values.

The exploratory phase is conditional on valid strict comparison, deployed Trace
policies and a positive within-Trace effect or credible adaptation signal. It is
not warranted by instrumentation alone. If eligible, preregister incremental
switch-rule, scorer, then variation-operator freedoms, prove causal sensitivity,
and label all findings exploratory. No advanced phase is authorized by this audit.

## Layout and rerun commands

`scripts/preflight.py` and `tests/test_preflight.py` implement only the source
audit, branch gate and secret-safe evidence checks. `src/`, `configs/` and `runs/`
are empty because the experiment stopped before implementation. `artifacts/`
contains immutable timestamped provenance, current pointers and STOP.json.
Future actual runs require config, source/environment/transport manifests,
solution/candidate/policy histories, LLM usage, final result, best sources and logs.

From `/home/xav/code/Trace-experiment0`:

```bash
python3 experiments/recursive_opt/EXP22/scripts/preflight.py
python3 -m unittest discover -s experiments/recursive_opt/EXP22/tests -v
ruff check experiments/recursive_opt/EXP22/scripts/preflight.py experiments/recursive_opt/EXP22/tests/test_preflight.py
```

Preflight exits 2 for the observed branch failure; exit 0 would mean only this
source gate passed, never permission to skip remaining gates.
Smoke-only, PRISM-only, Signal-only, all-strict, analysis-only and advanced-only
commands are intentionally unavailable: their runners are not implemented.
Claiming runnable commands for unbuilt stages would misrepresent reproducibility.
Complete the protocol with exact commands before any full execution; append dated
amendments if anything changes after a full run begins.

## Continuation amendment v2 — 2026-09-26, before any live call

The user requested commit, push, then continuation. Commit 3004bf10ba was pushed
to origin/codex/exp22-evox-comparison. Execution now uses the EXP22-owned detached
worktree at exactly refs/heads/recursive_opt (846580defe935195c1f5f39d6336079ef6ff1e10).
The original dirty checkout remains untouched; no later HEAD is substituted.
The explicit --isolated preflight option accepts only a clean tracked tree at
that exact reference. The historical S0 stop and evidence remain archived.

An EXP22 .venv contains editable Trace and SkyDiscover with their declared
dependencies and SciPy. Isolated Python (-I) prevents this workspace's unrelated
Trace code from shadowing the chosen version. Ten golden/parity repetitions
passed for each task, using the stock evaluator itself inside the Trace adapter.
Stock PRISM defaults cascade on but has no stage1 and falls back to full evaluate;
Signal uses the stock stage1 threshold and stage2 merge. No subprocess adapter
is used. Source and provider-catalog inspections made zero completion calls.

S2 begins with the direct exact-routing HTTP request. The tiny transport smoke
uses max_tokens=32 and prompt 'Reply with OK.'; optimization remains 32000.
Temperature remains stock 0.7, timeout 600 seconds, no seed. A failure stops
before framework smokes or implementation spending; this is not a passed S2.
Completion content is not required to establish transport identity. Actual
model and serving provider must be present in the response or generation metadata.

## Implementation clarification v3 — before S4 optimizer calls

Both CP variants send identical complete mocked HTTP bodies; the real native
OptoPrimeV2 parser formats Python using Black before activation. Mock comparisons
use the actual formatted policy hash. CP-B is selected provisionally after its
162 relevant runtime tests pass; optional GEPA/notebook tests and two existing
provenance-lock failures are recorded separately, without changing old locks.
The narrow CP-B patch also routes the mandatory startup probe, which otherwise
would omit provider/session. All four live transport clients passed.

Trace declares an EXP22 versioned module/engine/evaluator/dataset and replaces
only the inherited controller's meta proposer. The trainable value contains the
complete database source and a real OptoPrimeV2 backward/step observes the measured
window. Stock validation and migration remain inherited. Budget horizon is in
solution attempts: the final stock retry count is capped by remaining attempts
on both paths, avoiding an overshoot of the requested 100. Invalid retry attempts
retain the preceding best score in the quality curve. Stock strategy scoring is
unchanged. A no-op Trace source update sets the shutdown event and fails the run.

S4 uses one solution generation per task/framework; S5 uses five per arm. These
pilot results are separate from immutable strict run directories and never reused
as starting programs. No strict run is enabled until all gate booleans pass.

## Transient provider diagnostic — 2026-09-26, before retry

The first S4 PRISM/SD-EVOX attempt received three HTTP 429 engine_overloaded
responses (startup guide/meta probes and solution request), with zero candidate
generations. Preserve that attempt and retry the same one-generation smoke once
after at least 30 seconds in a new directory. If the provider rejects it again,
stop as STOPPED_PROVIDER. No alternate provider, model, session or token settings
are permitted. A failed availability probe must not qualify a fixed-policy fallback
as a successful EvoX smoke. Neither attempt is a quality comparison.

## Current commands and stop record — continuation amendment v4

Both delayed S4 attempts were refused. Stop as STOPPED_PROVIDER; no further paid
calls are authorized by a passed stage in this record. This is the protocol's
mandatory gate failure, not a negative quality conclusion. RESULTS.md and
artifacts/diagnostic_summary.json contain observed counts and exact failures.
The initial implementation's broad secret regex matched dependency CSS identifiers,
package documentation and wheel hashes. Final scanning uses token boundaries and
provider key lengths, still traversing every file. Provider account IDs in failed
SDK error logs were redacted before sealing/publication; no quality/error code changed.

All commands below assume the repository working directory and an environment
OPENROUTER_API_KEY loaded without printing it. Never put the key on a command line.
Owned worktrees and the environment are ignored by Git; the patch and package
versions are committed. To reconstruct them from a fresh experiment directory:

```bash
git -C /home/xav/code/Trace worktree add --detach "$PWD/experiments/recursive_opt/EXP22/worktrees/trace" 846580defe935195c1f5f39d6336079ef6ff1e10
git -C /home/xav/code/Trace worktree add --detach "$PWD/experiments/recursive_opt/EXP22/worktrees/trace_cp_b" 846580defe935195c1f5f39d6336079ef6ff1e10
git -C experiments/recursive_opt/EXP22/worktrees/trace_cp_b apply --unidiff-zero "$PWD/experiments/recursive_opt/EXP22/artifacts/trace_cp_b.patch"
python3 -m venv experiments/recursive_opt/EXP22/.venv
experiments/recursive_opt/EXP22/.venv/bin/python -m pip install -e experiments/recursive_opt/EXP22/worktrees/trace -e /home/xav/code/evo-compare/repos/skydiscover scipy
```

For exact environment replay, constrain every package to the versions recorded in
artifacts/environment.json; the install command alone is not a version lock.

Preflight and mocked parity:

```bash
python3 experiments/recursive_opt/EXP22/scripts/preflight.py --trace experiments/recursive_opt/EXP22/worktrees/trace --isolated
experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/scripts/evaluator_preflight.py
experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/scripts/policy_fixture.py --variant CP-A
experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/scripts/policy_fixture.py --variant CP-B
```

Resume S4 (one fresh directory per command), only after provider availability:

```bash
experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/scripts/run_stage.py --stage one --task prism --arm SD-EVOX
experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/scripts/run_stage.py --stage one --task prism --arm TRACE-RECURSIVE
experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/scripts/run_stage.py --stage one --task signal_processing --arm SD-EVOX
experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/scripts/run_stage.py --stage one --task signal_processing --arm TRACE-RECURSIVE
```

`--stage pilot` selects five solution attempts. Run all four arms per task only
after S4, then assess S5 and remaining full-run checks. The per-task strict commands
use `--stage strict --task prism` or `--task signal_processing`, with each arm
in the preregistered order; they intentionally reject the current gate file.
No all-strict orchestration or advanced-phase runner is claimed as finished.
There is no basis for running an advanced phase. Analysis-only for the current
provider stop is reproducible without network:

```bash
python3 experiments/recursive_opt/EXP22/scripts/analyze.py
```

This diagnostic analyzer refuses evidence that no longer matches the two recorded
failed attempts. Extend it explicitly if resumed execution produces new evidence.
The remaining full-run metering/checkpoint/analysis requirements still need
validation before any strict gate can be marked passed.

## Provider amendment v5 — 2026-09-26, before Novita paid execution

The user explicitly requested replacing the upstream provider with `novita` and
continuing. This supersedes the previous DeepInfra-only routing requirement and
stop. Every new request uses `provider.only=["novita"]`; actual serving metadata
must identify `Novita`. Model, session ID (including PRIMS), temperature, token
limit, timeout, sources, evaluator and arm budgets are unchanged. No fallback
provider is authorized. The public OpenRouter endpoint catalog confirms this route.

DeepInfra transport/diagnostic/config evidence is archived under
`artifacts/deepinfra_pre_novita/`; its immutable run directories remain intact.
No DeepInfra result is pooled with the Novita series. Repeat S2, compile all S3
specs, and repeat CP-A/CP-B mocked equivalence for Novita before S4. S0/S1 source
and evaluator evidence remain applicable because neither changed. S4 and S5
remain mandatory before strict execution; the bounded transient-retry rule still
applies. Existing analysis commands must be updated to select provider evidence
explicitly before they may write the current report.

### CP-A fallback selected after Novita S2, before S4

Direct, SkyDiscover and CP-A each completed one correctly routed live request.
CP-B made three successful HTTP requests: its stock empty-response recovery
changed max_tokens from 32 to 64 to 128, violating single-call smoke semantics.
Source inspection shows the same recovery can escalate a 32000-token generation
to 32768. Therefore CP-B is excluded from live benchmark execution. Use the
pre-authorized CP-A fallback (`test_mode=true`, explicit llm_factory), labeled
non-portable/non-promotable. Successful-response mocked equivalence still passes,
but does not establish empty-response equivalence. All six Novita HTTP requests
remain in transport evidence and cost accounting. S2 passes for the active
direct/Sky/CP-A paths only; CP-B's failed diagnostic remains visible.

### Bounded empty-generation diagnosis — v5.1, before second Novita S4

The first Novita PRISM S4 returned HTTP 200 with 32000 completion tokens
(31997 reasoning tokens), but stock SkyDiscover reported None content and no
candidate. Preserve this failure. Repeat the same one-generation smoke once,
without changing max_tokens, timeout, prompts, temperature, provider or model.
If empty generation recurs, stop as STOPPED_PROVIDER: the required generation
gate cannot be established at the fixed request settings. A diagnostic success
would still require every remaining S4/S5 gate; no failed attempt is discarded
or treated as a quality observation. The HTTP observer now also persists finish
reasons and content lengths, without altering requests or returned responses.

### Novita diagnostic disposition — 2026-09-26

Both unchanged S4 attempts ended with empty solution content at 32000 completion
tokens. Stop as STOPPED_PROVIDER (generation failure; all HTTP routes passed),
with S4/S5 false. No further paid optimizer run is enabled. RESULTS.md and the
current provider-filtered analyzer retain every attempt and all returned costs.
The strict and advanced phases remain unrun; no quality conclusion is supported.
