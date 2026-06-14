# recursive_opt — recursive (meta) optimization on Trace / Trace-Bench

A small, robust layer that lives at **`opto/features/recursive_opt/`** and turns
Trace into a *recursive* optimizer: it optimizes a task artifact, then optimizes
**how that optimization is done**, then optimizes *that* across families of
problems — reusing the existing `opto` trainer/optimizer/guide machinery
unchanged.

---

## 1. The goal — what "meta / recursive optimization" means here

Ordinary Trace optimizes a **task artifact** (a prompt or a code block) so an
agent solves a task better. Meta-optimization optimizes **the thing that does the
optimizing** (the trainer, batch design, trace representation, guide, memory
policy). Recursive optimization stacks these levels:

| Level | Optimizes | Trainable object |
|---|---|---|
| **O0** | a task artifact | the artifact (prompt / code / harness / SKILL.md) |
| **O1** | *how* O0 is optimized | the **A-list**: starting artifact, batch size/design, trace type/horizon, memory, optimizer + tools, guide, trainer |
| **O2** | the O1 policy per problem family | which O1 setup to use per family |
| **O3** | transferable priors across families | a default that works on an unseen family |

Why bother: the March-2026 "hidden choices" result says there is **no universal
default** for starting artifact / credit horizon / batch size — the best choice is
task-dependent. Recursive optimization *learns* those choices per family instead
of hand-guessing them, and induces priors that transfer.

---

## 2. The implementation — one idea, two surfaces

**One idea.** A recursion level is itself a `trace.Module` whose `forward()` runs
the optimization of the level below and returns its held-out score; its trainable
parameters are whatever defines the level below. Because every level is "just a
Module", the *same* `opto.trainer.train` + `opto.optimizers` optimizer + `Guide`
optimize all of them. No new core machinery — that is what makes it robust and
simple.

**Two trainable surfaces** (you pick per goal — this is the key design point):

| Surface | Class | Trainable parameter | Use it to… | Example |
|---|---|---|---|---|
| **Selection / config** | `LevelConfig` + `MetaLevel` | a small config (text) over existing components | *select & configure* existing trainers/batch designs/traces | **A** |
| **Code / implementation** | `ComponentSpec` + `CodeArtifactLevel` (via `@bundle(trainable=True)`) | the **source code** of a component | *rewrite / invent* a Trainer hot-path, batch sampler, trace summarizer — including components that don't exist yet | **B** |

> This answers "can we optimize the actual Trainer / batch-design / Trace classes
> *and* future ones?": **yes** — the config surface selects among existing ones;
> the code surface rewrites their source, so the optimizer can produce genuinely
> new components, not enum picks.

### Package layout
```
opto/features/recursive_opt/
  levels.py        # spine: LevelConfig, ArtifactLevel(O0), MetaLevel(O1+),
                   #        ComponentSpec + CodeArtifactLevel (code surface), RecursiveGuide
  memory.py        # MemoryLite tiers M0–M3 + thin retrieval (active knowledge building, C.1)
  progress.py      # per-level events.jsonl + summary.json progress tracking
  traces.py        # optional GraphAdapter (B.3), OTEL (B.4), multi-trace TGJ (B.5)
  capabilities.py  # AgenticOptimizer (C.2), TinkerEnvAdapter (C.3), HITLGate (C.4)
  tracebench.py    # turns real Trace-Bench task ids into inner_runner / code- and
                   #   multi-objective evaluators (section D)
examples/          # A/B/C/D, each documented, runnable offline
examples/recursive_opt_demo.ipynb    # Colab/local walkthrough
REPORT.md          # the 3-approach comparison, conformity tables, convergence
```

---

## 3. The initial test types (what each example proves)

| Example | Question it answers | Surface | Elements | Trace-Bench problems |
|---|---|---|---|---|
| **A** `recursive_opt_example_A_learn_setup.py` | What is the best *setup* for a family? | selection/config | A.2 batch, A.4 memory, A.7 trainer | eval-only: `llm4ad:online_bin_packing_local`, `internal:multi_param`; live: `internal:multi_param` |
| **B** `recursive_opt_example_B_improve_component.py` | Can we *rewrite a component's code* to a better one? | code/implementation | B.2 trainer sampling, B.5 trace repr | `llm4ad:online_bin_packing_local`, `internal:code_param` |
| **C** `recursive_opt_example_C_learn_capability.py` | Can we *learn a new capability* from a spec under multiple objectives? | prompt artifact + multi-objective | C.1 + C.2 + C.4 (C.3 stub) | `internal:multiobjective_gsm8k` |
| **D** `recursive_opt_example_D_cross_family.py` | Do the best choices *transfer across families*? | full stack O0→O3 | A/B/C combined | `{online_bin_packing_local, circle_packing}`, `{internal:multiobjective_gsm8k, internal:multi_param}` |

A/C/D non-live runs now register a bounded **real Trace-Bench eval-only adapter**
when Trace-Bench is installed: one real example, no nested trainer, and no
optimizer LLM. That checks wiring without synthetic task scores, but it is not a
full efficacy benchmark. B still uses a deterministic code validator because it
tests source-code rewriting mechanics rather than external benchmark scoring.

---

## 4. HowTo

### 4.1 Run the non-live demos
```bash
# from a checkout of OpenTrace with recursive_opt placed under opto/features/
export PYTHONPATH=/path/to/OpenTrace
python examples/recursive_opt_example_A_learn_setup.py
python examples/recursive_opt_example_B_improve_component.py
python examples/recursive_opt_example_C_learn_capability.py
python examples/recursive_opt_example_D_cross_family.py
```

### 4.2 Install the full stack (graph telemetry + Trace-Bench)
```bash
git clone https://github.com/AgentOpt/OpenTrace && cd OpenTrace
pip install opentelemetry-api opentelemetry-sdk            # graph/OTEL backends
# PYTHONPATH works for local development if you are not using an editable install.
# place this package at opto/features/recursive_opt/

git clone https://github.com/AgentOpt/Trace-Bench && cd Trace-Bench
pip install -e ".[hf]"        # HotpotQA / BBEH / GSM8K ; add ".[dspy]" etc. as needed
```
With Trace-Bench installed, non-live examples register a bounded eval-only
adapter and live mode registers the default bundle adapter. You can also
register a custom adapter with `register_task_adapter(...)`. `traces.py` emits
real OTEL/Sysmon spans merged into TGJ when the optional graph/telemetry modules
are present.

### 4.3 Run with the real LLM optimizer
```bash
export OPENAI_API_KEY=...          # NEVER hard-code; read from env / a secret manager
export TRACE_LITELLM_MODEL=gpt-5.4-nano  # optional backend choice
export RECURSIVE_OPT_BUDGET_PRESET=demo  # optional global safety envelope
python examples/recursive_opt_example_A_learn_setup.py --live
python examples/recursive_opt_example_B_improve_component.py --live   # OptoPrime rewrites the code
python examples/recursive_opt_example_C_learn_capability.py --live    # OptoPrimeMulti, multi-objective
```
Live mode replaces the hand-driven loops with `opto.trainer.train` / `OptoPrime` /
`OptoPrimeMulti`, so the LLM optimizer proposes configs, rewrites component code,
and trades off objectives itself.

`RECURSIVE_OPT_ITERATIONS` and `RECURSIVE_OPT_NUM_CANDIDATES` are per-call
recursive optimizer settings. The optional global budget is a separate safety
envelope across levels:

```bash
export RECURSIVE_OPT_MAX_OPTIMIZER_LLM_CALLS=64   # proposal LLM calls
export RECURSIVE_OPT_MAX_EVAL_LLM_CALLS=80        # known task-eval LLM calls
export RECURSIVE_OPT_MAX_CANDIDATES=16            # planned outer candidates
export RECURSIVE_OPT_MAX_WALL_TIME_SECONDS=300
export RECURSIVE_OPT_BUDGET_STOP_POLICY=return_best  # or: raise
```

Unset, `none`, `null`, `unlimited`, `off`, or `-1` means no global limit for that
resource. `0` is not unlimited: it allows zero units and is useful for verifying
that live optimizer calls fail early. Local loop limits still prevent runaway
runs when the global envelope is disabled.

Every `run_spec(...)` run writes progress metadata into the memory root:

* `events.jsonl` is an append-only per-level ledger (`level_start`,
  `trainer_metric`, `level_end`) with `level_step`, `global_step`, budget usage,
  task ids, and separate `problem_score` / `objective_score` fields.
* `summary.json` is the compact report for notebooks: planned/executed steps,
  where the best problem score and best configured objective appeared, and the
  final artifact id.
* `artifacts.jsonl` remains backward-compatible; new artifacts add the same
  progress summary under `metrics["progress"]`.

Use `artifact_version` for lineage and `best_*_at.level_step` for optimization
progress. They are intentionally different: the best artifact version can be
`0` even when a level ran multiple trainer steps but did not improve over the
first persisted artifact.

On startup, live mode preflights the configured LiteLLM model and registers the
Trace-Bench adapter. If the model is inaccessible or Trace-Bench cannot be
initialized, the run fails early instead of silently reporting synthetic scores.

### 4.4 Notebook (Colab or local)
Open `examples/recursive_opt_demo.ipynb`. The setup cell clones OpenTrace,
installs OpenTelemetry, ensures `recursive_opt` is under `opto/features/`, and
runs A/B/C/D. A final cell reads `OPENAI_API_KEY` via `getpass` for the live pass.

---

## 5. Verified API

`opto.trace`: `node`, `bundle(trainable=True)` (function **source** = a trainable
param), `model`, `Module`, `ParameterNode` · `opto.trainer.train(model=,
train_dataset=, algorithm=, optimizer=, guide=)` · `opto.trainer.algorithms`:
`Minibatch`, `MinibatchAlgorithm`, `BeamsearchAlgorithm`, `UCBSearchAlgorithm` ·
`opto.optimizers`: `OptoPrime`, `OptoPrimeMulti` · `opto.trainer.objectives`:
`pareto_rank(candidates, metrics=)`, `select_best(candidates, ObjectiveConfig)`,
`ObjectiveConfig(mode="pareto", minimize={...})` · `opto.features.graph`:
`GraphAdapter`, `LangGraphAdapter`, `GraphModule` · `opto.trace.io`:
`instrument_graph`, `TelemetrySession`, `make_dict_binding`, `merge_tgj`,
`otlp_traces_to_trace_json`.

**Trainer class names are case-sensitive** in the `train()` facade — use
`"BeamsearchAlgorithm"` (not `"BeamSearchAlgorithm"`).
