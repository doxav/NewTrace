# EXP22 protocol v1 — 2026-09-26

Status: preregistered intent; stopped during S0 source identity audit. No full
benchmark has begun. This document does not certify an implemented benchmark.
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
