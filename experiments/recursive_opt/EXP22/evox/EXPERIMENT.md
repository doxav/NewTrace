# EXP22 protocol v1 — 2026-09-26

Current status: amendments v8/v9 run the four-worker Trace comparison with repaired accounting and lossless meta feedback. All S0–S5 gates passed; see RESULTS.md for execution status.
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

## Reasoning amendment v6 — 2026-09-26, before low-effort paid calls

The user requested testing the failing PRISM cases with reasoning effort `low`
and continuing if this resolves generation. Add the exact OpenRouter parameter
`reasoning_effort="low"` to every active client, including guide/meta calls.
No other request parameter changes: Novita only, same model/session, 0.7
temperature, 32000 generation tokens and 600-second timeout. OpenRouter documents
this field at https://openrouter.ai/docs/api_reference/parameters#reasoning-effort.
The endpoint catalog already advertises support; outgoing HTTP bodies are checked.
Acceptance of a field is not proof that the provider honors it; judge the observed
completion/finish reason and stock candidate evaluation.

Archive previous Novita/default-effort report and mutable evidence under
artifacts/novita_default_reasoning; retain original runs untouched. Revalidate
active S2 clients, compile S3 specs and verify mocked CP equivalence. Keep CP-A
primary (non-portable) because CP-B's token-escalation behavior is unchanged; do
not spend more live calls rechecking the excluded variant. Then run two sequential
PRISM/SD-EVOX one-generation diagnostics with low effort. If both produce changed,
valid stock-evaluated candidates, continue the remaining S4/S5 checks and eligible
strict work with low effort consistently across all arms. If generation stays
empty or invalid, preserve both attempts and stop before full execution.
Analyze low-effort runs separately from default-effort or DeepInfra history.

### S4 stochastic diff-format retry clarification — before retry

Both requested low-effort PRISM repetitions and Trace/PRISM passed. Signal/SD
passed, but Signal/Trace returned a nonempty diff whose SEARCH blocks did not
match its stock parent. This is ordinary candidate-format failure, not evaluator
or transport divergence. Stock generation permits up to three attempts; S4
intentionally caps each isolated run at one. Allow at most three such isolated
one-attempt smokes per task/framework for diff-format failures, preserving all
failures and costs. Stop if no valid candidate is obtained within that bound.
This gate establishes executable plumbing only; smoke scores are not comparative
quality estimates, and all later pilots/strict runs still start from stock source.

### Population audit correction before Trace pilot

The existing population-preservation check raised on lost/mutated programs, but
stock EvoX catches meta-evolution exceptions and continues. Make that same audit
also set the experiment failure and shutdown flags. This implements the existing
mandatory stop rule without changing successful policy proposal/migration behavior.
The regression fixture simulates a missing retained program and requires shutdown.

### Full-run audit completion — before any strict run

Before strict execution, lock the runtime Python files in
artifacts/strict_source_hashes.json and reject source drift at each strict launch.
Every actual HTTP attempt, including transport exceptions, is counted and timed;
Trace optimizer requests are labeled meta. Retain unknown usage as missing, never
as known zero cost. Per-attempt curves include cumulative call/token/cost counts,
valid/invalid counts, current fitness, population size and policy hashes. A retry
curve point includes only HTTP calls through that solution attempt, not later
retries already completed by stock generation. Snapshot structural checkpoints at
10/30/60/100; persist the kernel result even when canonical Trace execution fails.

Reject a mismatch between solution HTTP calls and consumed attempts before the
next generation. Transport/identity failures, non-finite fitness, population loss
and disconnected feedback set shutdown/failure flags; the inherited meta fallback
cannot hide them. TRACE-RECURSIVE additionally stops after ten consecutive invalid
solution attempts, or at checkpoint 30 if meta proposals have failed and none has
been deployed. Flat valid scores alone do not stop execution. This checkpoint
interpretation makes the existing all-rejected-policy condition concrete before
full execution. The earlier pilot files remain immutable and may lack these added
audit fields; they are not used as strict quality results.

### Strict execution preregistration completion — before the first full run

All eight five-generation pilots passed, with live Trace policies deployed on
PRISM and populations preserved. The pilot data remain diagnostic; every strict
arm starts fresh from stock source. The SD path now uses the same canonical
Python/NumPy RNG scope (`_seed_scope(42)`) that Trace already uses. Database/pool
RNGs remain seeded 42. No LLM request seed is added. This closes the global-RNG
asymmetry before any confirmatory run; pilot scores are not treated as effects.

Persist each meta proposal (including rejected source), its stock validation and
every scored search window. Valid proposal count and validation-failure rate use
these proposal records; parser/transport failures are labeled separately. Preserve
the final stock-scored window too. Expose any missing billing metadata separately.

Quality AUC is the discrete sum of the best score after attempts 1 through 100;
normalized AUC is that sum divided by 100. First-improvement/best iteration uses
1e-12 tolerance, with iteration 0 denoting the unchanged initial best. Absolute
gain is final minus initial; relative gain divides by the nonzero initial score.
Window gain is stock window end minus start; improving-window fraction uses the
same tolerance. Never use execution time as a quality metric. Report PRISM
success_rate beside inverse-pressure score: its native objective can reward a
program that fails some cases, so combined score alone does not imply uniformly
better valid placements. Signal native component metrics remain visible.

Commands below assume the EXP22 credential is already loaded securely in the
environment. Single strict arms use the same runner as the pilots. Full runs must
pass all gate files and the exact runtime source lock before any paid call.

```bash
# PRISM only, in preregistered order; stop on a failed run.
for arm in SD-EVOX TRACE-RECURSIVE TRACE-FIXED SD-FIXED; do
  experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/scripts/run_stage.py --stage strict --task prism --arm "$arm" || break
done
# Signal only; execute after the PRISM suite succeeds.
for arm in TRACE-RECURSIVE SD-EVOX SD-FIXED TRACE-FIXED; do
  experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/scripts/run_stage.py --stage strict --task signal_processing --arm "$arm" || break
done
# Analysis only (no network).
python3 experiments/recursive_opt/EXP22/scripts/analyze.py
```

The all-strict execution is these two ordered task suites, aborting before Signal
if any PRISM arm fails. Full artifacts/checkpoints provide the reviewable result;
advanced eligibility is assessed only after both suites complete without an
unresolved provider/evaluator/control-plane confound. No advanced phase is
authorized by the pilots alone. Plotting uses the already-installed system
Matplotlib interpreter, separate from the unchanged benchmark environment.

## Amendment v7 — 2026-09-26: enforceable evaluator termination

The user requested a definitive repair of the surviving timed-out worker and
resumption. The failure is local: stock `asyncio.wait_for(run_in_executor(...))`
does not terminate its executing thread, and PRISM's nested executor context
waits for a timed-out worker. Novita did not cause this execution leak.

Both solution and policy evaluators now use EXP22 `ProcessEvaluator`, which
inherits the unchanged stock evaluator's cascade, threshold, retry and metric
normalization logic. Only `_run_stage` execution is replaced: each stock function
runs in a fresh Python process/session, under the original per-stage timeout.
The parent unconditionally SIGKILLs the owned process group and waits for its
worker before returning, including timeout, cancellation and successful exit.
Timeout metrics and Signal stage-2 fallback are still generated by stock code.
This is process lifecycle containment, not a security sandbox for hostile code.
It handles the observed hanging threads and ordinary descendants; processes
that deliberately escape the group are outside this benchmark's threat model.

Each stage starts Python/NumPy RNG state at seed 42, consistently in all arms.
Fresh module state and process startup time are deliberate runtime differences
from v6; startup counts against the original timeout. Candidate RNG changes no
longer alter the parent's search RNG. Golden and invalid-case parity establish
agreement for tested sources, not universal equivalence of stateful candidates.
No evaluator objective, benchmark data, cascade threshold, timeout, policy
validator, LLM parameters, provider or optimization budget changes.

The original v6 runs remain immutable; their report and artifacts are copied to
`artifacts/thread_evaluator_low/`. They are not pooled with `process-stage-v1`.
All eight strict arms restart from stock initial programs after affected gates
pass. S0 provenance and S2 routing evidence may be reused after verification;
S1 process parity, S3 compilation/mocked validation, S4 and S5 are revalidated.
Full evaluated candidate sources and per-stage PID/status/reaping audits are
saved under each run's `evaluations/`, alongside separate invocation counters.
An ordinary cleaned-up timeout remains an invalid candidate, not an experiment
stop. The sequential-execution invariant remains enforced; disabling it would
permit overlapping workers and invalidate the experiment.

Additional verification: `.venv/bin/python -I -m unittest discover -s tests -p
 test_process_evaluation.py -v` from EXP22; the ordinary evaluator preflight,
control-plane fixtures, smoke, pilot and strict commands above remain applicable.

## Amendment v8 — 2026-09-27: four concurrent Trace workers

The user explicitly requested four parallel workers for TRACE-RECURSIVE and
TRACE-FIXED on PRISM and Signal Processing. This supersedes the global sequential
run order for this batch. Each independent process starts from the stock solution
and policy, has a separate persistent population and immutable run directory, and
consumes up to 100 solution HTTP attempts. Candidate generation and evaluation
remain sequential within each worker; evaluator process termination remains
mandatory. One worker's diagnostic stop does not cancel the other workers.

Novita, GLM-5.3-flash, low reasoning effort, session identity, generation parameters,
seed 42, CP-A, process-stage-v1, validation and existing scientific stop rules are
unchanged. No SD arms or advanced extension are included in this requested batch.
Trace continues editing the currently active policy, with the inherited stagnation
trigger; no new parent selection or optimizer memory behavior is introduced.

Before launch, repair the diagnosed accounting error in the experiment adapter:
a failed solution result that consumed HTTP requests must not take stock EvoX's
pre-generation database fallback solely because its prompt field is absent.
Use an explicit empty prompt marker for that case, preserving the original error
and normal stock attempt recording. Genuine database failures before any solution
HTTP request retain stock fallback behavior. Provider/identity failures still
stop the affected worker after accounting; they are not silently retried beyond
the existing bounded generation attempts. Offline regression tests exercise both
paths before the runtime source lock is renewed.

Reuse the unchanged golden evaluator and S0–S5 evidence, verify frozen sources,
run the full EXP22 offline tests, verify golden results under four concurrent
processes, and check the frozen serving route immediately before launch. Preserve
the previous report and source lock in this batch's artifacts. Record worker PIDs,
commands, output paths, outcomes, hashes, and validation in a batch manifest.

The four runs form a new concurrency series and are not pooled with earlier
sequential or thread-evaluator runs. Shared CPU/memory and provider capacity can
affect latency, timeout frequency and cache behavior; timing is not a quality
metric. Recheck saved best sources sequentially after completion. Compare quality
only for completed 100-attempt pairs, report all stops and policy-window counts,
and treat each arm as one stochastic trajectory rather than statistical evidence
of a general advantage. More solution attempts alone do not guarantee more policy
updates because the stagnation threshold scales with the configured horizon.

Batch analysis is explicitly scoped rather than using the historical all-run
analyzer. For the first v8 batch:

```bash
python3 experiments/recursive_opt/EXP22/scripts/analyze_parallel.py --manifest experiments/recursive_opt/EXP22/artifacts/parallel_trace_20260927T194336Z/manifest.json
python3 experiments/recursive_opt/EXP22/scripts/plot_results.py --parallel-manifest experiments/recursive_opt/EXP22/artifacts/parallel_trace_20260927T194336Z/manifest.json
```

Analysis supports live snapshots; plots require sealed results from all workers.
The manifest records each worker's exact launch command. Launch those commands in
four separate processes with the EXP22 attachment credential loaded only into
their environment; the repository's Experiment-0 credential is a different key.

## Amendment v9 — 2026-09-27: lossless Trace observation references

The v8 Signal recursive worker stopped after 52 recorded solution attempts:
Novita rejected the fourth meta request with HTTP 400, stating that its input
exceeded the model context. The feedback contained repeated full solution
populations in each archived policy's start/end database statistics. There were
181 program occurrences but only 47 distinct program records. The raw request
and stopped run remain immutable. This is a prompt-construction failure, not a
negative quality result about meta-optimization.

For fresh replacement recursive runs, normalize observations with stock
`make_json_serializable`, then encode repeated JSON values using deterministic
content-addressed references. Preserve all sources, IDs, measurements, metadata
versions and ordered population membership. Retain raw normalized observations
as artifacts and verify lossless reconstruction in tests. Supply the complete
encoded observation once as optimizer feedback; the graph observation contains
its digest, active policy hash and measured window metrics. This removes the
two redundant, already-truncated Input/Output copies of the full observation.
Neither the policy parameter nor the measured feedback is silently truncated.

This changes prompt representation and therefore requires fresh recursive runs
from the stock initial policy and solution, not splicing a continuation onto the
stopped curve. It does not change the model, request parameters, solution budget,
policy-parent choice, trigger, evaluator or validation. Unmodified fixed controls
remain applicable: they never construct a meta observation. Their v8 launch
provenance remains explicit. Reuse them without spending another 200 solution
calls; do not pool or average old and replacement recursive trajectories.

Prepare a separately copied runtime for the replacements, with its own source
lock and tests. Keep existing workers on their unchanged runtime and fill free
worker slots with replacements, never exceeding four experiment workers. Record
replacement ancestry, separate run costs, actual concurrency and final matched
pairs in a new manifest/report. Concurrency varies as workers finish, so latency
and timeout caveats from v8 still apply. No model context capacity guarantee is
claimed for arbitrary future sources: unique information still consumes context.
