# `opto/features/recursive_opt` — a deep assessment

*Analysis 2026-08-29/30. Branch `recursive_opt` @ `80777032`. 16 modules, 8,850 runtime lines as
found, ~1,000 branch commits, 483 unit tests as found. Static analysis, offline execution, and nine
live probes against OpenRouter (~$0.10 total).*

> **Status: CURRENT — this file supersedes all earlier results.**
> `artifacts/RESULTS_INDEX.md` lists what every other results file is now worth.
> Superseded: `examples/recursive_opt_use_cases_CURRENT_LIMITS.MD`, everything under
> `examples/notebook_outputs/`, and `experiments/recursive_opt/`. Raw data for §7 and
> §11–§14 is in `artifacts/probe_2026/`.

### How to read this

Each layer assumes only the ones before it, so you can stop at any depth.

| | Layer | Answers | Read it if |
|---|---|---|---|
| §0 | **Verdict** | what is real, what is not, what to do | you read nothing else |
| §1 | **The idea** | what "recursive optimization" means here and why the design trick is right | you want the concept |
| §2 | **The implementation** | 16 modules, keep / cut / rewrite | you want to prune |
| §3 | **How a score is produced** | which knobs are causal, which are proxies, which are inert | ← *the most useful section for daily use* |
| §4 | **The v2 control plane** | what the newest layer does well, and what it cannot express | you are deciding its fate |
| §5 | **The evidence** | the UC4 teardown, and a second experiment that was scoring its own crashes | ← *read this one carefully* |
| §6 | **Errors** | 14 defects, all fixed here except one structural | you are reviewing the diff |
| §7 | **The live probe** | fresh measurements: is there any signal at all? | you want new data, not archaeology |
| §8 | **Correct usage** | what it is good for today + the rules that make numbers mean something | you are about to run it |
| §9 | **Prescription** | the ~150-line fix, the one experiment worth running, what to delete | you are deciding what happens next |
| §11 | **Why the surface is flat** | the objective is a token counter with a rare error term; what that costs the next experiment | ← *read before launching anything* |
| §12 | **The task pool** | why five healthy tasks looked broken, and the four certified surfaces (three with zero noise) | you are choosing what to run on |
| §13 | **Optimization measured properly** | an effect above the resolution limit — and why half of it is brevity | context for §14 |
| §14 | **Retractions and final state** | why both live results are withdrawn, and the three surfaces that survive | ← *the current bottom line* |

If you have five minutes: §0, then §5.2, then §8.2.

---

## 0. Executive verdict

Three questions, three answers.

**Is the idea new and viable?** The *idea* is sound and unusually economical: a recursion level is
itself a `trace.Module`, so `opto.trainer.train` optimizes O1/O2/O3 with no core changes. That is the
right trick, it is correctly implemented, and `effects.py` — which asks "does this knob have an
*active causal path* to artifact / trajectory / feedback / trace?" rather than "does it move the
score?" — is a genuinely good contribution that deserves to survive independently of everything else.

**Is it demonstrated to work?** No. Not once, anywhere in the repository. The single robust positive
result in the whole programme (UC4) is an **arithmetic artifact of comparing two different
measurements**, and its winning and losing arms produced **byte-identical artifacts**. §5.2 proves
this from the saved run data — the reported `+0.163` equals `(qasper − gsm8k)/2` to four decimals.
Everything else in the user's own review is 3 real losses and 5 ties.

A live re-run settles it (§7.2). Scoring **both** arms on the same held-out family — the only change —
turns the `+0.163` into **−0.018**, with a paired delta of **−0.006** against a measured noise floor of
**±0.039**. The gain does not shrink; it disappears. And on fresh data the underlying problem
reappears: eight of nine arm-seeds finished with the *unchanged default* artifact, while one pair moved
`+0.055 → +0.196` — a swing nearly the size of the entire historical claim — with a byte-identical
artifact. On this benchmark, run-to-run variance is comparable to every effect anyone has reported
(§7.1: signal/noise **0.96** on gsm8k, **0.74** on qasper).

**Is it overly complex, or is the complexity required?** Both, in different places. The recursion
spine (`levels.py`, 915 lines) is close to minimal and should be kept. The v2 control plane
(`spec.py`, 2,732 lines) is well-engineered infrastructure that solves reproducibility problems the
project did not yet have, while **structurally excluding the feature it exists to serve**: the only
fully portable module it can execute is `score = 1.0 if actual == expected else 0.0` over trainable
strings. Recursion is reachable only through a legacy compatibility shim.

**And — found while verifying that claim — the v2 migration silently broke the recursive path
itself.** A multi-level run through the v2 engine died with
`TypeError: optimize() got multiple values for keyword argument 'num_candidates'` whenever the level
carried a per-level budget in `trainer_kwargs` — which is exactly what the standard budget allocator
`allocate_levels` writes, so it hit every equal-budget three-way arm. The migration added a positional
`num_candidates=` beside a `**trainer_kwargs` splat that already carried it (**D14**, a regression at
`c92f0af4`). Separately, a recursive level could be stamped
`portable=True, promotable=True` while its score came from a code path that ignored the declared
evaluator entirely (**D11**). Neither was caught, because **no experiment has been run since the
migration** — the current one is still `experiment_results: not_started`. Both are now fixed, and the
first genuinely multi-level run since the migration happens in §7.2.

### The one-line summary

> The mechanism is real and mostly well built. The *evidence* is not — the flagship result is a
> subtraction, not an effect — and the newest layer is elaborate infrastructure that, until this
> pass, could not execute the feature it was built for.

### What to do

| | Action |
|---|---|
| **Keep** | `levels.py`, `effects.py`, `budget.py`, `memory.py`, `optimize.py` — the actual feature |
| **Keep, but re-scope** | `spec.py` v2 control plane — needs one portable module that can carry a recursive level, otherwise it is scaffolding around a tautology |
| **Do not trust** | every score from a cross-level arm comparison (§5.2), and every example-C capability score produced without a live backend (§5.4 — they are failed evaluations recorded as −0.5) |
| **Delete/replace** | `make_code_evaluator`'s synthetic validators; the line-count footprint gate |
| **Fixed here** | 15 defects (§6), five of them severe: a −1e9 sentinel reaching promoted priors; an unguarded cross-metric comparison; an unsound portability flag; a regression that broke every multi-level run; and an evaluator that scored its own crashes |
| **Do first** | the honest experiment in §9.2 — it is ~2 hours of compute and it decides whether any of this is worth continuing |

---

## 1. Layer 1 — the idea

Ordinary Trace optimizes a **task artifact** (a prompt, a code block) so an agent solves a task
better. This package stacks optimizations:

| Level | Optimizes | Trainable object |
|---|---|---|
| **O0** | a task artifact | the prompt / code / harness itself |
| **O1** | *how* O0 is optimized | starting artifact, batch size & design, trace type, memory policy, optimizer, guide, trainer |
| **O2** | the O1 policy *per problem family* | one config line per family |
| **O3** | transferable priors *across* families | one shared config, scored on held-out families |

The motivating claim is reasonable: there is no universal best choice of starting artifact / credit
horizon / batch size — the best choice is task-dependent — so *learn* it instead of guessing.

### The design trick, and why it is good

```python
@trace.model
class MetaLevel(Module):
    def forward(self, family):
        return self._run_inner(self._cfg_node, family)   # runs the level BELOW
```

A recursion level is a `trace.Module` whose `forward()` runs the optimization beneath it and returns
that level's score, and whose trainable parameter is whatever *defines* the level below. Because every
level is "just a Module", the same `opto.trainer.train` + `opto.optimizers` + `Guide` drive all of
them. **No new core machinery.** That is the whole idea, and it is correct.

The trainable parameter for O1 is a single text node holding the whole config:

```
batch_size: 4
batch_design: random
trainer: PrioritySearch
```

One node, not seven — deliberately low-dimensional, which is the right answer to the standard
credit-assignment objection against meta-optimization. Both the score **and the textual feedback**
flow back through that node, so an LLM optimizer can read *why* a config underperformed.

### Two trainable surfaces

| Surface | Class | Trainable thing | Use it to |
|---|---|---|---|
| **Selection / config** | `LevelConfig` + `MetaLevel` | a small config text | *select and configure* existing components |
| **Code / implementation** | `ComponentSpec` + `CodeArtifactLevel` | the component's **source code** (via `@bundle(trainable=True)`) | *rewrite or invent* a component |

The code surface is the more ambitious claim — "we can rewrite a Trainer's hot path" — and the
mechanism genuinely works: `CodeArtifactLevel` seeds a trainable bundle from `spec.baseline`, and the
optimizer rewrites its source. Two real subtleties are handled correctly and are worth noting because
they are the kind of thing that silently returns 0.0 forever if missed:

- `_canonicalize_def_name` renames the optimizer's emitted `def` to the bundle's expected `_fun_name`,
  otherwise the recompiled namespace would not contain the function and every candidate would be lost.
- `_attach_eval(self._last_node, payload)` re-anchors the returned payload to the last traced output
  of the trainable code, so `backward()` actually reaches the code parameter. `forward()` raises if
  the evaluator never invoked the candidate — a disconnected node cannot be optimized, and the package
  correctly refuses to pretend otherwise.

**Verdict on Layer 1: the idea is sound, and the spine implementing it is good code.**

---

## 2. Layer 2 — the implementation, module by module

8,850 lines across 16 modules. Verdict per module:

| Module | Lines | What it is | Verdict |
|---|---:|---|---|
| `levels.py` | 915 | The recursion spine: `LevelConfig`, `ArtifactLevel`, `MetaLevel`, `FamilyPolicyLevel`, `PriorInductionLevel`, `CodeArtifactLevel`, `RecursiveGuide` | **Keep.** This is the feature. Close to minimal. |
| `effects.py` | 137 | Causal-effect contract for trainable fields | **Keep — best idea in the package.** See §3.3. |
| `budget.py` | 403 | Global LLM/candidate/wall-time envelope, spec↔dataclass mapping | **Keep.** Small, correct, genuinely needed for recursive runs. |
| `memory.py` | 531 | Tiered `MemoryLite` (episodes → artifacts → family priors) + retrieval | **Keep.** JSONL + dataclasses; appropriately tiny for the job. |
| `optimize.py` | 397 | One DRY entry point onto a real Trainer, plus `restore_best_validated` | **Keep.** The docstring on `restore_best_validated` documents two real, subtle bugs it fixes. |
| `progress.py` | 184 | Per-level `events.jsonl` / `summary.json` ledger | **Keep.** |
| `tracebench.py` | 1,327 | Trace-Bench adapter + evaluators | **Keep the adapter; cut the synthetic evaluators.** See §3.2. |
| `spec.py` | 2,732 | Legacy spec API **and** the entire v2 control plane | **Split.** Two unrelated systems in one file; see §4. |
| `capabilities.py` | 404 | `AgenticOptimizer` (tool-using optimizer), `HITLGate`, `TinkerEnvAdapter`, learnable search policy | **Trim.** `AgenticOptimizer` earns its place (9 references incl. `spec.py` and tests). `TinkerEnvAdapter` has **zero** references outside its own definition. `HITLGate` is used once, in example C, with the `auto_allow` stub approver and no test — it is demonstrated, not integrated: the trainer's real accept/reject path never consults it. |
| `decisions.py` | 444 | Guarded decision parsing/scoring for policy artifacts | **Keep if the policy use cases stay**, else cut — it exists to serve UC8/UC10/UC14, which are the losing arms. |
| `numeric_optimizers.py` | 243 | Optuna / least-squares optimizers for numeric config fields | **Keep.** The clearest, most defensible sub-result: LLM optimizers are bad at low-dimensional numeric search, so route it elsewhere. |
| `experiments.py` | 249 | Multi-seed re-runs with real RNG seeding + aggregation | **Keep.** Its docstring correctly identifies that a prior "multi-seed" loop only varied the memory-root suffix and reported memory-isolation variance as seed variance. |
| `traces.py` | 195 | Optional OTEL / sysmon trace merge | **Optional.** Guarded imports; degrades cleanly. |
| `inspect_utils.py` | 116 | Notebook diff/snapshot helpers | **Keep.** Harmless. |
| `runmode.py` | 328 | Live-vs-offline resolution, model preflight, secret redaction | **Keep.** Its "make the execution mode impossible to misread" goal is right. |

### The shape of the complexity

The complexity is not uniformly unjustified — but it is unevenly distributed, and `spec.py` alone is
31% of the package. More telling is *how* it is written:

| File | Lines | Blank-line share |
|---|---:|---:|
| `spec.py` | 2,732 | **6.3%** |
| `levels.py` | 915 | 15.2% |
| `budget.py` | 403 | 14.9% |
| `opto/trainer/objectives.py` (repo norm) | 555 | 14.8% |
| `opto/optimizers/optoprime.py` (repo norm) | 943 | 14.7% |

`spec.py` has less than half the whitespace of everything around it, nearly every function body
collapsed onto one line, and dict literals over 400 characters. That is not a style preference — it is
the direct consequence of a **line-count gate baked into the test suite**:

```python
assert evidence["total_lines"] == sum(actual.values()) <= 8850     # test_34, before the fix
```

The ADR reasons explicitly about staying under this number ("recursive-opt runtime is 8,750 physical
lines versus 8,803 (-53) … No footprint exception is required"). The gate did not constrain
complexity; it constrained *formatting*, and made the same logic markedly harder to read. This is
defect **D8**, and it is fixed.

Concretely: the 14 fixes in §6 take the package from 8,850 to **8,971** lines, most of it comments
explaining *why* each guard exists. Under the old gate that would have been a hard failure requiring
an "itemized footprint exception". Catching a severe regression is worth 121 lines of explanation, and
a gate that says otherwise is measuring the wrong thing.

`spec.py` is left compressed (6.2% blank lines). Reformatting it to the repo's normal spacing touches
every line and should be its own reviewable commit — it is recommended, not done here.

---

## 3. Layer 3 — how a score is actually produced

This is where the design meets reality. Follow one number from `LevelConfig` to a reported score.

```
LevelConfig (text node)
  → decode_cfg / validate            levels.py
  → inner_runner                     spec.make_scored_task_runner
  → TraceBenchTaskAdapter.run_task   tracebench.py
      ├── _apply_starting_artifact   seeds the bundle's trainable param
      ├── _train_bundle              nested opto.trainer.train  (only if inner_steps > 0)
      └── _score_bundle              REAL Trace-Bench evaluation
  → scoring normalization (clip / relative_delta)
  → MetaLevel score + feedback → the outer optimizer
```

The good news: **the task score is real**. `_score_bundle` calls Trace-Bench's public evaluator on
real examples. There is no synthetic fallback, and `_require_adapter` raises rather than inventing a
number — the module is explicit and repeatedly honest about this. `real_mode_status()` will tell you
plainly that scoring "will RAISE until `register_task_adapter(...)` is called."

### 3.1 Which knobs are actually causal

The `LevelConfig` surface has 12 fields. They are **not** equally real, and the adapter's own
`field_effects()` contract says so:

| Field | Causal path | Reality check |
|---|---|---|
| `starting_artifact` | artifact → score | **Real.** Seeded into the bundle's trainable param. The strongest knob. |
| `initial_knowledge` | artifact → score | **Real**, and thoughtfully done: attached to the node's `description` (which OptoPrime reads as documentation) rather than concatenated into the prompt being optimized. |
| `trainer`, `optimizer` | optimization → score | Real **only when `inner_steps > 0`** — i.e. only when a nested trainer actually runs. |
| `batch_size`, `num_threads` | optimization / budget | Same condition. |
| `batch_design` | optimization | Real but **proxied**: the "difficulty" ordering is `len(str(text))`. Longest-first is called `failure_balanced`, shortest-first `curriculum`. Model-free by design (no LLM cost), but it means `batch_design` is ordering by *string length*, not by difficulty. |
| `credit_horizon` | feedback | Real but **feedback-only**: it controls how many per-example feedback strings are concatenated for the optimizer (top-3 / all / first / indexed). It cannot move the score directly, only via better proposals. |
| `trace_type` | trace / feedback | Same: changes the trace evidence, explicitly "NOT score-plumbed". |
| `memory_policy` | — | **Declared `active=False` by the adapter itself.** It is an inert knob. |
| `guide` | — | Not in the adapter's contract at all. |

This is the honest part of the package and it deserves credit: it *tells you* which knobs are dead.
But read the consequence. `MetaLevel`'s default `trainable_fields` are
`['batch_size', 'batch_design', 'trace_type', 'memory_policy', 'optimizer', 'guide', 'trainer']`.
Queried against the adapter's own contract at `inner_steps=0`:

```
inactive or undeclared : ['batch_size', 'batch_design', 'memory_policy',
                          'optimizer', 'guide', 'trainer']       <- 6 of 7
active but no score effect : ['trace_type']                      <- the 7th
```

**Zero of the seven default fields have any declared path to the score.** The default O1 search
surface is flat by construction, and `starting_artifact` — the one field that does have such a path —
is *not* in the default set. At `inner_steps=1` five of them come alive; `memory_policy` never does.

This single fact explains most of §5: experiments that searched the default field set were searching
a surface the adapter itself declares inert, and nothing stopped them, because the gate that would
have raised was not on the execution path (**D13**).

### 3.2 Where the evaluators are synthetic

The *task* evaluators are real. Two *component* evaluators are not. They are the **default** in
`run_code_experiment` (`evaluate or make_code_evaluator(task_id, name)`), which is how the
`uc1_batch_design_*` and `uc1_trace_summarizer_*` runs in `notebook_outputs/` were scored, and they
back example B. (Credit where due: the later code use cases — UC1's BBEH solver, UC5, UC8–UC11 — were
moved onto real evaluators such as `make_tracebench_direct_answer_evaluator` and
`evaluate_optimizer_tool_policy`, and UC14's `CURRENT_LIMITS` entry records catching and fixing a
mistaken use of `make_code_evaluator` there.)

- **`make_code_evaluator("batch_design")`** scores against a closed-form target: hard items are
  `[i for i in range(n) if i % 3 == 0]`. A candidate that returns `[0, 3, 6, 9]` scores 1.0. The
  feedback string even names the rule (`"defined by idx % 3 == 0"`). This measures whether an LLM can
  read an instruction, not whether it can design a batch sampler.
- **`make_code_evaluator("trace_summarizer")`** scores `"AssertionError" in summary` plus a length
  ratio.

The file says so plainly ("deterministic local validators … they do not replace task scoring"), and
that honesty is real. But downstream reports do not carry the caveat, and a "code rewrite improved
0.36 → 0.52" line reads as a benchmark result when it is a keyword check.

Similarly, in the multi-objective evaluator, `cost` is `min(len(text)/600, 1.0)` — a **length proxy,
not tokens** — and the scalarization is a hard-coded `accuracy − 0.5·cost + 0.5·compliance`.

### 3.3 The one genuinely novel contribution

`effects.py` reframes knob validation:

> A field does **not** need to change the final score to be a valid recursive knob; it needs a real
> causal path to at least one relevant intermediate object:
> `knob → artifact / trajectory / feedback / trace / memory / budget → candidate → (maybe) score`.

So validation asks *"does this field have an ACTIVE declared effect under the current run mode?"*,
never *"does it move the score?"*. Fields are declared with their activating condition, so the error
message **is** the documentation:

```
InactiveFieldError: trainable fields with no ACTIVE causal path under the current run mode:
  'memory_policy' (inactive until retrieval/promotion feeds optimizer input or warm-start)
```

This is a real idea, correctly generalized (`required_effects` lets an experiment declare which effect
kinds count), and it is the piece most worth extracting and keeping regardless of the fate of the rest.

---

## 4. Layer 4 — the v2 declarative control plane

This is the newest and largest layer (`spec.py` + `artifacts/control_plane_v2/`), motivated by wanting
to detach from the unclear notebook. The stated pipeline:

```
raw mapping → migrate legacy → normalize/materialize defaults → structural validation
→ semantic validation → resolve versioned refs → expand arms/seeds/matrix
→ immutable ExecutionPlan → ordered level plans → one canonical runner → canonical RunResult
```

### 4.1 What it does genuinely well

This is competent infrastructure, and the list is long:

- **Immutable, JSON-serializable normalized specs** with SHA-256 fingerprints over canonical JSON.
- **Unknown keys fail** — typos become compile-time errors, not silent no-ops.
- **No callables, no secrets in specs**; credentials only as `env:NAME` references.
- **Holdout is capability-gated**: `DatasetAccess.read(split, phase=...)` raises `PermissionError` if
  anything touches `holdout` during fit / proposal / induction / candidate selection. The fit context
  physically does not contain the holdout split. This is the right design and it is enforced.
- **Per-role LLM accounting** (`forward` / `optimizer` / `feedback` / `judge`) charged at the wrapped
  client, with deterministic ordered fallbacks.
- **A budget guard that charges before the operation**, not after.
- **Cross-process resume** keyed on a full identity: spec fingerprint, level fingerprint, engine,
  module ref, dataset refs, *and* a hash of the recursive-opt source tree and resolved registry — so
  editing the source correctly invalidates a resumed result. (It invalidated mine, correctly, when I
  applied the fixes.)
- **Reproducible mode rejects moving `latest` model aliases.**
- **A serious test suite**: 42 control-plane tests + 12 hardening tests, most of which check
  *causality* (`test_08_engine_config_is_causal`, `test_09_module_artifact_is_causal`,
  `test_06_upstream_output_counterfactual`) rather than just shape.

None of that is fake. If you need a reproducible experiment harness, this is a good one.

The one caveat, which connects to §5.4: those causality tests establish causality *of the plumbing*,
over string-equality fixtures and scripted optimizers. `test_08_engine_config_is_causal` proves that
raising `iterations` raises the candidate count; it does not run a recursive level. That is why 490
green tests coexisted with a `TypeError` on every multi-level run (D14) — the green suite never
executed the feature.

### 4.2 The structural problem

**The portable canonical path cannot express recursive optimization.**

The complete v2 registry is:

```
MODULES    : recursive_opt.module.graph@1             (needs an injected GraphExecutor)
             recursive_opt.module.legacy_level@1      (capability: 'legacy')
             recursive_opt.module.reasoning_workflow@1
ENGINES    : fixed | trace | gepa_optimize_anything
EVALUATORS : recursive_opt.evaluator.module_output@1  (mode: output)
             recursive_opt.evaluator.reasoning@1      (mode: output)
             recursive_opt.evaluator.legacy_level@1   (mode: legacy_module)
DATASETS   : recursive_opt.dataset.legacy_level@1     (legacy only)
```

Now trace the constraints:

1. `compile_plan` requires an `output`-mode evaluator unless the module declares the `legacy`
   capability or `runtime.test_mode` is set.
2. `execute_plan` computes `portable = not overrides and evaluator.mode == 'output'`, and
   `promotable = portable and valid`.
3. `_validate_runtime_resources` classifies `graph_executors`, `optimizer`, `trainer`, `evaluator`,
   `memory`, `legacy_levels` as **behavioral** resources, which require `runtime.test_mode=true`,
   which forces `portable=False`.

Therefore:

- `legacy_level@1` — the *only* module that builds `MetaLevel` / `FamilyPolicyLevel` /
  `PriorInductionLevel` / `CodeArtifactLevel` — must pair with the legacy evaluator, which is
  `legacy_module` mode, which makes its results **`portable=False, promotable=False`**.
- `graph@1` requires the behavioral `graph_executors` resource → also non-portable in practice.
- Which leaves exactly one fully portable module: `reasoning_workflow@1`, a bag of trainable strings,
  whose only task semantics are:

```python
score = 1.0 if actual == expected else 0.0        # spec._reasoning_evaluator
```

**The entire portable v2 surface is string equality.**

### 4.2.1 …and the portability flag was not sound

While verifying the above I tried the obvious escape hatch: declare the *portable* output-mode
evaluator on a legacy (recursive) level. It compiled and ran. The result:

```
declared evaluator          : probe.evaluator.tripwire@1 (mode=output, portable)
times that evaluator ran    : 0            <- IGNORED
reported score              : {'score': 0.25}   (from the level's own _final_eval;
                                                 the tripwire would have said 0.99)
RunResult.portable          : True
RunResult.promotable        : True
```

`_run_legacy_trace_engine` scores through the level's own `_final_eval` and never consults
`objective.evaluator_ref`, but `execute_plan` computes portability from the evaluator's *declared*
mode. So a recursive level could be stamped `portable=True, promotable=True` while its score came from
an entirely different, undeclared code path.

That is worse than the exclusion it seemed to be: the single guarantee the control plane sells —
"portable means this result is reproducible from the declared spec" — was **unsound on exactly the
path that runs the actual feature**. It is the same "assert the claim, not the property" pattern as
§5.4. Fixed as **D11**: a legacy-capability module now must declare the matching legacy evaluator, and
declaring an output evaluator fails at compile time with an explicit reason.

So the corrected statement is: recursive optimization is reachable only through a compatibility shim
whose results are — now genuinely, rather than nominally — neither portable nor promotable.

### 4.3 What the notebook now shows

The old use-case notebook (46 cells, UC1–UC14, three-way harness) was replaced by a 5-cell smoke
notebook that runs two golden fixtures. The "positive" one is:

```json
"module": {"config": {"components": {"transfer_policy": "reuse_promoted_cross_family_prior"}}},
"engine": {"name": "fixed"},
"datasets": {"holdout": [{"component": "transfer_policy",
                          "expected": "reuse_promoted_cross_family_prior"}]}
```

A `fixed` engine (no optimization at all) scoring a hard-coded string against a holdout expecting that
same string. Score 1.0 by construction. To the project's credit the notebook says so explicitly —
"deterministic offline fixtures, not historical or scientific recursive-optimization results" — but it
means the flagship artifact of the v2 work demonstrates **plumbing only**.

`migration_report.md` completes the picture: of 85 historical specs, `execution_replayable: **0**`.
Nothing from the previous year of experiments can be re-run under the new control plane.

### 4.4 The current experiment

> **Updated 2026-08-30 after merging `origin/recursive_opt`.** Parallel work on the remote has
> moved this on considerably: the GEPA reflection adapter and the optimizer's empty-text
> handling were both fixed there, transport was hardened, and a frozen Experiment-0 main run
> was authorised and **started** on 2026-08-24. It stopped on its first unit (seed 0, budget 6,
> arm A) with `hard constraints not satisfied` — accuracy 22/24, invalid rate 1/24, zero
> optimizer calls. So the experiment is no longer "never started"; it is "started and stopped
> at the first hard-constraint check". The paragraphs below describe the state as found at
> `80777032` and are kept as the record of that point.

`experiments/recursive_opt/multiobjective_reasoning/` contains the latest attempt. Its own report:

```
Status: BLOCKED — RETURN_TO_CONTROL_PLANE
experiment_results: not_started
provider_calls: 0
monetary_cost_usd: 0
```

The named blocker — the production GEPA evaluator returned a 3-tuple where `gepa 0.1.4`'s public
`EvaluatorWrapper` expects `(score, side_info)` — **was fixed** in `52a7b0bd`, with a green CI run. But
the experiment reports were never updated, so the tree still asserts a blocked state under stale
digests (defect **D10**, now superseded).

The root cause analysis in that report is excellent and worth preserving verbatim, because it is the
same pattern as everything else here:

> `test_22_gepa_externalizes_holdout` injects a fake `gepa_optimize` … it does not execute installed
> `optimize_anything`. `test_22b_...` invokes `OptimizeAnythingAdapter.evaluate` directly … bypassing
> the top-level `EvaluatorWrapper` that enforces the public two-item contract.

**Green tests that test a mock of the boundary rather than the boundary.** Hold that thought for §5.

---

## 5. Layer 5 — the evidence

This is the decisive section. The question is not "is the code good" but "did it ever work".

### 5.1 The user's own review

`examples/recursive_opt_use_cases_CURRENT_LIMITS.MD` reviews the last full live run
(`three_way_stage2_20260624_150102`) with an explicit reliability gate:

| UC | Standard | Recursive | Δ | Seed-δ std | Reliability | Root status |
|---|---:|---:|---:|---:|---|---|
| UC1 code BBEH solver | 1.000 | 1.000 | 0.000 | 0.000 | tie | score saturated |
| UC2 QASPER config | 0.292 | 0.265 | −0.027 | 0.050 | tie | noisy, no gain |
| UC5 tool policy code | 0.750 | 0.792 | +0.042 | 0.629 | tie | variance ≫ effect |
| UC8 campaign policy | 0.517 | 0.445 | −0.072 | 0.063 | **standard wins** | real loss |
| UC9 agentic policy | 0.760 | 0.843 | +0.083 | 0.112 | tie | promoted on a tie |
| UC10 promotion policy | 0.342 | 0.398 | +0.057 | 0.049 | recursive | not LCB-safe |
| UC11 prompt emitter | 0.499 | 0.381 | −0.118 | 0.085 | **standard wins** | real loss |
| UC13 numeric | −0.158 | −0.160 | −0.002 | 0.004 | tie | flat objective |
| UC14 code transfer | 0.643 | 0.573 | −0.070 | 0.000 | **standard wins** | transfer fails |
| **UC4 family prior** | **0.013** | **0.176** | **+0.163** | — | **promote** | **the flagship** |

Nine use cases: 3 real losses, 5 ties, 1 win. The review's own conclusion — *"same-level warm code
restart is not a robust general recursive pattern"* — is correct and well argued.

That leaves UC4 carrying the entire programme. So examine UC4.

### 5.2 UC4 is an arithmetic identity, not a result

**Finding 1 — the two arms are scored on different task sets.**

From the pre-v2 notebook (`git show 2ad8eb09:examples/recursive_opt_use_cases.ipynb`, cell 37):

```python
tw_uc4 = benchmark_uc(
    "UC4_family_policy_prior",
    standard  = family_policy_spec("o2", warm=False, ...),
    recursive = family_policy_spec("o3", warm=True,  ...),
    primary_level={"initial": "o2_policy", "standard": "o2_policy",
                   "recursive": "o3_prior"},          # <-- different levels
)
```

with `fams = {"gsm8k": ["internal:multiobjective_gsm8k"], "qasper": ["hf:qasper"]}`. And the two
levels do not compute the same quantity:

- `FamilyPolicyLevel._run_policy` → `_mean(per_family.values())` over **every** family.
- `PriorInductionLevel._run_prior` → mean over **holdout families only**, where `compile_level` sets
  `train = names[0]`, `holdout = names[1:]`.

So the standard arm reports `mean(gsm8k, qasper)` and the recursive arm reports `qasper`. Confirmed
mechanically with the helper added in this pass:

```
o2_policy scores: ['internal:multiobjective_gsm8k', 'hf:qasper']
o3_prior  scores: ['hf:qasper']
```

**Finding 2 — the delta is a closed form, exact to four decimals.**

Every `episodes.jsonl` line in the UC4 run carries the per-family breakdown. Pooling all **65**
evaluations across all arms and seeds:

| family | mean | sd | range |
|---|---:|---:|---|
| `gsm8k` | **−0.1601** | **0.0035** | [−0.168, −0.151] |
| `qasper` | +0.1417 | 0.0243 | [+0.093, +0.202] |

`gsm8k` is effectively a **constant**: sd 0.0035 on a score of −0.16, never moving more than 0.017
across 65 evaluations and every configuration tried. Now compute what the two levels report:

```
o2 = mean(gsm8k, qasper) = -0.0092      (what the STANDARD arm is scored on)
o3 = qasper only         = +0.1417      (what the RECURSIVE arm is scored on)

o3 - o2                  = +0.1509
(qasper - gsm8k) / 2     = +0.1509      <-- identical, to four decimals
```

The two are the same number **algebraically**: if the standard arm reports `(a+b)/2` and the recursive
arm reports `b`, their difference is exactly `(b−a)/2` — half the constant offset between the two
tasks. Reported UC4 delta: **+0.163**, which is `+0.1509` plus qasper sampling noise.

The "recursive gain" is not an effect. It is a subtraction. **It would appear with any optimizer,
with a broken optimizer, or with no optimizer at all** — as Finding 3 shows it in fact did.

**Finding 3 — nothing was learned in either arm.**

The final artifacts, from `three_way_report.json`:

| arm | seed | score | artifact |
|---|---|---:|---|
| initial | 0 | −0.0037 | `gsm8k => batch_design=random, batch_size=4 \| qasper => batch_design=random, batch_size=4` |
| standard | 0 | +0.0212 | `gsm8k => batch_design=random, batch_size=4 \| qasper => batch_design=random, batch_size=4` |
| recursive | 0 | +0.1823 | `batch_design: random \| batch_size: 4` |
| initial | 1 | −0.0235 | *(identical)* |
| standard | 1 | +0.0062 | *(identical)* |
| recursive | 1 | +0.1780 | *(identical)* |
| initial | 2 | −0.0251 | *(identical)* |
| standard | 2 | +0.0115 | *(identical)* |
| recursive | 2 | +0.1675 | *(identical)* |

Every arm, every seed, ends at `batch_design=random, batch_size=4` — **the stock `LevelConfig`
defaults**. No configuration was ever changed. The +0.025 "initial → standard" movement on a
byte-identical artifact is therefore pure evaluator noise, and the +0.163 "standard → recursive"
movement is the task-set offset.

**UC4 measured nothing. The programme's only robust win is an artifact of the harness.**

### 5.3 Why the promotion gate passed it

`promotion_decision` computed a lower confidence bound and promoted, because the LCB was over the
*recursive arm's* variance — and the recursive arm's variance is genuinely small (it is one stable
task). No gate anywhere asked whether the two arms measured the same thing. That is defect **D3**,
now fixed: replaying the historical UC4 report through the patched gate yields

```
new action  : invalid_comparison
new promote : False
reason      : arms are scored on DIFFERENT task sets, so their difference is not a
              measure of optimization quality: it also contains the offset between
              the two task sets
```

### 5.4 A second silently-dead experiment: example C

UC4 is not the only case where a green result rested on a failure. Removing the D2 swallow
immediately turned a passing test red, and the traceback is unambiguous:

```
tracebench.py:1317  in evaluate        -> _evaluate_real(...)
tracebench.py:1260  in _evaluate_real  -> response = param.forward(inputs[i])
predefined_agents/learner.py:81        -> self.model(self.system_prompt, ...)
...
litellm/llms/openai/openai.py:377      -> raise OpenAIError(   # no api_key
```

Example C's capability evaluation makes a **real provider call**. With no key — i.e. in every offline
run and in CI, where sockets are deliberately blocked — it raises. Before the fix, that exception was
converted to `({"accuracy": 0.0, "cost": 1.0}, ..., −0.5)` for *every* candidate, so:

- the offline "Pareto-best capability" was selected among a set of identically-failed evaluations;
- the printed `acc=0.00 cost=1.00` lines were failure reports, not measurements;
- `test_review_regression_returns_rows` passed because its only quantitative assertion is
  `episodes >= 1`, and `CapabilityArtifact` records to memory whether or not evaluation succeeded.

So a second use case has been producing publication-shaped output from a dead evaluation, for as long
as the swallow has been there. The test now skips without a backend, and a new test asserts that the
failure path raises rather than scores.

### 5.5 The pattern

Seven independent instances of the same failure mode, in the project's own words, data and code:

0. **Example C**: a green test over an evaluation that had failed on every candidate (§5.4).
1. **UC4**: a green promotion over an invalid comparison (§5.2).
2. **GEPA**: green tests that exercised a *mock* of the boundary, so the real contract mismatch
   survived to production (the project's own report diagnoses this, excellently).
3. **The footprint gate**: a green assertion (`total_lines <= 8850`) satisfied by reformatting rather
   than simplifying (D8).
4. **Portability**: `portable=True` asserted from the *declared* evaluator while the executed path
   ignored it (D11, §4.2.1).
5. **Readiness**: a test asserting `required_gepa_ci is True` — the answer — rather than
   `ready ⇔ gates ∧ ¬blockers` — the property. Honestly withdrawing readiness broke the suite, which
   is precisely the pressure that produces a false green.
6. **The effects gate**: the package's own guard against searching a dead knob existed, was correct,
   and **was never called on the execution path** (D13).

The common root is **verifying the claim instead of the property**. The infrastructure is
sophisticated and the tests are numerous; the gates measure the wrong things. That is also why 490
green tests coexisted with a `TypeError` on every multi-level run (D14): nothing green was actually
executing a recursive spec.

---

## 6. Layer 6 — errors found, and what was done

Fifteen defects. All fixed in this pass except the structural one (G1), which is a design decision.
D1–D10 came from static analysis; **D11–D15 were found only by trying to run the thing** — which is
itself the point of §5.4.

| # | Defect | Where | Impact | Status |
|---|---|---|---|---|
| **D1** | The `-1e9` invalidity sentinel is emitted as a *score* when `scoring.clip` is unset. It then poisons every mean, promotion gate and persisted family prior. | `tracebench._trainer_budget_feedback`, `_trace_backend_failure` → `spec.make_scored_task_runner` | **Severe.** Verified: `mean(0.2, invalid)` was `-499999999.9`; the value was persisted as a *promoted* family prior. This is the same "−666,666,666" bug `levels.invalid_result` claims to have fixed — it fixed only the decode path. | **Fixed** |
| **D2** | The multi-objective evaluator converted *any* exception into `(accuracy 0.0, cost 1.0, scalar −0.5)`. | `tracebench.make_multiobjective_evaluator` | **Severe — and it was actively firing.** Removing the swallow immediately broke `test_review_regression_returns_rows`, which had been green. Root cause: example C's capability evaluation calls `param.forward(...)`, a real provider call, which raises `OpenAIError` with no key. So in every offline/CI run **every candidate evaluation failed and was scored `−0.5`**, and example C's "Pareto-best capability" was selected among identically-zero results. `episodes >= 1` held because the artifact records to memory regardless. See §5.4. | **Fixed** (re-raises; the test now skips without a backend and a new test asserts the failure path raises) |
| **D3** | Nothing checked that compared arms are scored on the same task set. | `examples/recursive_opt_three_way.py` | **Severe — root cause of the UC4 illusion.** | **Fixed** (new `scored_task_ids` + hard `invalid_comparison` gate) |
| **D4** | The `prior` surface silently degenerates to train-on-test with a single family (`holdout = names[1:] or names[:1]`). | `spec.compile_level` | A "held-out transfer" score measured on the training family, with no warning. | **Fixed** (raises; explicit opt-in) |
| **D5** | Loop variable shadowed the `index` parameter, so constraint errors reported the wrong level. | `spec._validate_objective_semantics` | Misleading diagnostics. | **Fixed** |
| **D6** | A level with all three splits empty still reported `status='success', valid=True`, scoring the module *inputs*. | `spec._evaluate_dataset` | Verified: an empty spec returned score 1.0. | **Fixed** (raises in final evaluation) |
| **D7** | `surface.kind` is decorative in the canonical path (dispatch is on `module.ref` alone), contradicting the ADR's own invariant 3. | `spec._normalize_level`, ADR table | Misleading contract. | **Fixed** (reclassified + documented) |
| **D8** | `test_34` asserted `total_lines <= 8850`, driving compression of `spec.py` instead of simplification. | test suite | Actively degraded readability. | **Fixed** (inventory kept, gate removed) |
| **D9** | `_INVALID_FLOOR = -1.0` duplicated "in sync by comment". | `experiments.py` | Drift risk. | **Fixed** (imported) |
| **D10** | Experiment reports still asserted `BLOCKED / RETURN_TO_CONTROL_PLANE` after the blocker was fixed. | `experiments/.../offline_contract_report.md`, `reports/control_plane_blocker.json` | Anyone reading the tree sees a false blocked state. | **Fixed** (superseded) |
| **D11** | The legacy (recursive) engine path ignores `objective.evaluator_ref` and scores via its own `_final_eval`, yet `portable`/`promotable` were computed from the *declared* evaluator mode. | `spec._run_legacy_trace_engine` / `execute_plan` | **Severe.** Verified: a tripwire evaluator ran **0** times while the result was stamped `portable=True, promotable=True`. The control plane's central guarantee was unsound on the feature's own path. | **Fixed** (compile-time rejection) |
| **D12** | When a level failed, the legacy return shape came back *without* `results`, so callers raised `KeyError: 'results'` and the real cause was lost. | `spec.run_spec` | Found the hard way: it masked an `InactiveFieldError` **and** D14 below. | **Fixed** (documented keys always present + `errors`) |
| **D13** | `effects.check_field_effects` — the package's best idea — was only reachable via `validate_spec`, which **`run_spec` never calls**. | `spec.compile_level` | Optimizing a dead knob ran happily to completion and returned a flat surface instead of an error naming the dead field. | **Fixed** (wired into `compile_level`; inert with no adapter) |
| **D14** | `iterations`/`num_candidates` carried in `trainer_kwargs` (which is exactly how per-level budgets are allocated) were also passed positionally → `TypeError: got multiple values for keyword argument 'num_candidates'`. | `spec._run_legacy_trace_engine`, `_run_module_engine` | **Severe: this broke every multi-level — i.e. genuinely recursive — run.** A **regression introduced by the v2 migration**: the pre-v2 call at `21a0ad3d` was `optimize(..., iterations=iterations, **opt_kwargs)` and passed `num_candidates` only through the splat; `c92f0af4` added `num_candidates=num_candidates` beside it. Also charged the budget at the engine default instead of the value actually used. | **Fixed** (`_resolve_search_size`, resolved before the budget charge) |
| **D15** | `OptunaOptimizer` never seeded its TPE sampler, so a search result depended on whatever ambient RNG state preceded it. | `numeric_optimizers.OptunaOptimizer` | Surfaced as an order-dependent test failure: `test_optuna_learns_categorical_and_integer_optimum` asserts an *exact* optimum and passed or failed according to what ran before it. Indefensible in a package about reliable measurement. | **Fixed** (`seed=0` default, settable for sweeps; verified identical across three different ambient RNG states) |
| **G1** | The portable v2 path cannot express a recursive level (§4.2). | `spec.py` registries | Structural. | **Not fixed** — see §9.1 |

An additional honesty correction was required: the source-digest gate correctly invalidated
`prompt18_readiness.json` when these fixes changed the runtime tree. Rather than re-pin the digest and
keep the green, readiness is now `ready_for_prompt_18: false` with explicit blockers, and
`test_readiness_uses_source_digests_without_sha_environment` was changed from asserting
`required_gepa_ci is True` (a *claim*) to asserting `ready ⇔ all gates pass and no blockers` (a
*property*) — so honestly withdrawing readiness no longer breaks the suite, which is what pressures
the record toward a false green in the first place.

**Test status.** Baseline before changes: **2 failed, 478 passed, 3 skipped**. After: the **same 2
pre-existing failures** (`gepa.optimize_anything` is not importable in this environment; CI installs
`.[gepa]`), **554 passed, 4 skipped** — including 22 new regression tests in
`tests/unit_tests/test_recursive_opt_validity_guards.py`, 2 in the reworked
`test_recursive_opt_review_regression.py`, and 54 in
`tests/unit_tests/test_recursive_measurement.py` (§11–§13). Baseline 478 passed →
554 passed, with the same two pre-existing failures throughout.

The fourth skip is new and deliberate: `test_review_regression_returns_rows` now skips without a
backend key instead of asserting numbers produced by a failed evaluation (§5.4).

Worth noting how the source-digest gate behaved during this work: it invalidated
`prompt18_readiness.json` **three times**, once after each round of source edits, exactly as designed.
That mechanism is sound and was left intact — the readiness record now carries
`ready_for_prompt_18: false` with explicit blockers rather than a re-pinned green.

---

## 7. Layer 7 — the live probe

Two live probes were run against OpenRouter. `.env` supplied only `OPENROUTER_API_KEY` and no model,
so the model used is the package's own OpenRouter default, `deepseek/deepseek-v4-flash-0731`
(resolved as `openrouter/deepseek/deepseek-v4-flash-0731`). Raw data:
`artifacts/probe_2026/probe_a_results.json`, `artifacts/probe_2026/probe_b.log`.

### 7.1 Probe A — is there any signal to optimize?

Everything in this package presupposes that changing `starting_artifact` moves the benchmark score by
more than the evaluator's own run-to-run variability. That had never been measured. So: three prompts
(the same three `score_spread` uses by default) × 3 repeats × 2 tasks, `inner_steps=0` (no nested
trainer, no optimizer LLM calls), `max_examples=2`. This separates

- **signal** = spread of the per-prompt means (what optimization could exploit), from
- **noise floor** = stdev of re-running the *identical* prompt.

**`internal:multiobjective_gsm8k`**

| prompt | mean | sd | runs |
|---|---:|---:|---|
| *(bundle default)* | −0.2117 | 0.0386 | −0.164, −0.212, −0.259 |
| `Answer directly.` | −0.1958 | 0.0324 | −0.154, −0.233, −0.200 |
| `Plan step by step, then verify…` | −0.2293 | 0.0341 | −0.191, −0.273, −0.224 |

```
between-prompt spread (signal) : 0.0335
within-prompt  noise  (floor)  : 0.0350
signal / noise                 : 0.96
```

**The effect of changing the prompt is smaller than the effect of running the same prompt twice.**
Also note the ordering: "Plan step by step, then verify" scores *worst*. There is nothing here to
climb.

**`hf:qasper`**

| prompt | mean | sd | runs |
|---|---:|---:|---|
| *(bundle default)* | +0.2478 | **0.2129** | +0.088, **+0.549**, +0.107 |
| `Answer directly.` | +0.0867 | 0.0384 | +0.049, +0.071, +0.139 |
| `Plan step by step, then verify…` | +0.1157 | 0.0399 | +0.067, +0.115, +0.165 |

Raw signal/noise looks better (1.66) — but the entire spread comes from **one outlier run** (+0.549)
in the bundle-default arm, whose own sd is 0.21. Between the two arms that actually set an artifact:

```
spread between the two seeded prompts : 0.0290
their mean within-prompt noise        : 0.0391
signal / noise                        : 0.74
```

**Also below 1.**

**Conclusion.** On both tasks the config→score surface is at or below its own measurement noise at
this budget. `spec.score_spread` exists precisely to detect this ("prove the config→score surface is
non-flat", returns `flat: True`), and it appears never to have gated the experiments that assumed a
non-flat surface.

*Honest caveat, in both directions.* This probe used `max_examples=2`, which inflates the noise floor;
the historical runs used 4–8 and measured a much tighter floor. But that cuts against the surface too,
not for it: at `max_examples=4` the historical UC4 data has `gsm8k` at **−0.1601 ± 0.0035 across 65
evaluations and every configuration tried** — that is not a low noise floor with a signal on top, it
is a task whose score does not respond to the configuration at all. Either way, the effect being
searched for has not been shown to exist above measurement error.

### 7.2 Probe B — the corrected UC4

The historical UC4 compared `o2_policy` (mean over both families) against `o3_prior` (held-out family
only). The corrected version scores **every arm on the same level**, `o3_prior`, and therefore on the
same held-out family:

```
initial   : one cold `prior` level, no optimizer updates
standard  : one cold `prior` level, optimized                     } identical structure,
recursive : `family_policy` -> warm `prior` transfer, optimized    } equal candidate budget
```

3 seeds, `starting_artifact` as the target (Probe A's finding: it is the only knob with an
unconditional path to the score), 8 candidates per arm, `max_examples=2`, ~33 minutes of wall time.

**This is the first genuinely multi-level recursive run through the v2 engine since the migration.**
It was not possible before this pass: the recursive arm died on **D14** (`TypeError: optimize() got
multiple values for keyword argument 'num_candidates'`), and the real error was masked by **D12** as a
bare `KeyError: 'results'`.

**Result**

```
comparability : {"known": true, "comparable": true,
                 "standard_tasks": ["hf:qasper"],
                 "recursive_tasks": ["hf:qasper"],
                 "reason": "both arms scored on the same task set"}
```

| arm | mean final | n valid | errors |
|---|---:|---:|---:|
| initial | +0.072 | 3 | 0 |
| standard | +0.184 | 2 | 1 |
| recursive | +0.167 | 1 | 2 |

```
recursive - standard          : -0.018
paired delta (seed 1, the only seed where BOTH arms completed) : -0.0060
qasper noise floor (Probe A)  :  0.039
promotion                     : retest (support below min_support=3)
```

**The +0.163 becomes −0.018 once both arms are scored on the same thing.** More precisely it becomes
*nothing*: −0.018 and the paired −0.006 are both well inside the ±0.039 noise floor measured in
Probe A. The correct reading is "no detectable difference", not "recursive is worse".

**And the same artifact problem recurs.** Of the nine arm-seeds, **eight ended with the empty
(default) `starting_artifact`** — including the recursive arm that scored +0.167. Exactly one arm
(`standard`, seed 1) actually changed its artifact:

```
standard seed1 : "Use the provided context as evidence, reason briefly, then answer exactly."
every other arm: "" (unchanged default)
```

Yet `initial seed0 = +0.055` and `standard seed0 = +0.196` with a byte-identical artifact — a +0.14
"improvement" from changing nothing, which is nearly the whole size of the historical UC4 claim. This
is the §5.2 finding reproduced live on fresh data: **on this benchmark the run-to-run variance is
comparable to every effect anyone has reported.**

**Honest limitations of this probe.** Three of nine arms failed with
`TypeError: argument of type 'NoneType' is not iterable` inside the `o3_prior` level's optimization,
under heavy provider throttling (up to 97 s per proposal). I could not reproduce it offline with
malformed node contents, so I cannot attribute it definitively — the most likely cause is the provider
returning an unexpected payload that the optimizer does not defend against. It reduced the sample to
n=2/n=1, so this probe **cannot** be read as a measurement of the transfer hypothesis; it is only
strong evidence that the specific historical +0.163 does not survive a same-metric comparison. It is
also worth noting that these failures were *visible at all* only because of D12 — before that fix they
surfaced as `KeyError: 'results'`.

The properly powered version is §9.2: ≥4 families, n≥5 seeds, noise floor reported alongside the
effect, and an artifact-change assertion so a "gain" on an unchanged artifact is void by construction.

---

## 8. Layer 8 — how to use it properly

Given everything above, here is what this package is legitimately good for **today**, and the rules
that make its numbers mean something.

### 8.1 The three things it actually does well

**(a) Optimize a starting artifact against a real benchmark (O0/O1, `starting_artifact` only).**

This is the one knob with a direct, unconditional artifact → score path. Use the `config` surface with
a single target and a real Trace-Bench adapter:

```python
from opto.features.recursive_opt import run_spec

spec = {
    "families": {"reasoning": ["internal:multiobjective_gsm8k"]},
    "memory_root": "./mem_run",
    "scoring": {"clip": [-1.0, 1.0]},               # ALWAYS set this (see 8.2)
    "tracebench": {"max_examples": 8, "inner_steps": 0, "timeout_seconds": 120},
    "budget": {"optimizer_llm_calls": 40, "eval_llm_calls": 120,
               "candidates": 16, "wall_time_s": 1800, "on_exceed": "return_best"},
    "levels": [{
        "id": "o1_setup", "surface": "config", "family": "reasoning",
        "targets": ["starting_artifact"],           # the only unconditionally causal knob
        "constraints": {"starting_artifact": ["", "Answer directly.",
                                              "Plan step by step, then answer."]},
        "fixed": {"optimizer": "OptoPrimeV2", "trainer": "PrioritySearch"},
        "iterations": 4,
    }],
}
out = run_spec(spec)
```

*(Verified: this shape passes `validate_spec`, compiles, and `scored_task_ids` reports exactly
`['internal:multiobjective_gsm8k']`. Note that supplying a `tracebench` block makes `compile_plan`
install a **real** Trace-Bench adapter with `require=True`, replacing any adapter you registered
yourself — omit the block if you want to keep a test adapter in place.)*

**(b) Rewrite a component's source code against a real evaluator (`code` surface).**

The mechanism is sound. The requirement is that you supply a **real** evaluator — not
`make_code_evaluator`, whose targets are closed-form. `make_tracebench_direct_answer_evaluator` is the
honest one: it runs the candidate against real bundle examples with reference answers.

**(c) Route low-dimensional numeric/categorical search away from the LLM (`numeric_optimizers.py`).**

The most defensible sub-result in the package: for `batch_size`, `batch_design`, `num_threads`, an
LLM optimizer is the wrong tool, and Optuna/TPE reaches the same score with **zero optimizer LLM
calls** and ~4× less wall time (UC13). Use `optimize_config_numeric`.

### 8.2 Non-negotiable rules for any comparison

These are what the evidence in §5 costs you if you skip them.

1. **Score every arm on the same task set.** With the D3 fix, `promotion_decision` now returns
   `invalid_comparison` if you don't. Never set `primary_level={"standard": "o2_policy",
   "recursive": "o3_prior"}` or any equivalent cross-level comparison.
2. **Always set `scoring.clip`.** Without it, one rejected candidate used to emit `-1e9` into your
   means. D1 now floors it, but a clip makes the intent explicit.
3. **Diff the artifacts before believing a delta.** UC4's arms were byte-identical. Add
   `assert initial_artifact != final_artifact` before reporting any "gain" — if the artifact did not
   change, the delta is noise by definition.
4. **Measure the noise floor first.** Run the *same* config n times and take the stdev. Any effect
   smaller than that is not an effect. `spec.score_spread` is the built-in diagnostic and it reports
   `flat: True` when the surface has no signal — check it before spending on a search.
5. **Check `effects.check_field_effects` before choosing targets.** With `inner_steps=0`, most of
   `LevelConfig` is inactive; the contract will tell you, and raise with the activating condition.
6. **Use paired per-seed deltas, n ≥ 5.** The recursive arm's marginal variance is not the uncertainty
   of the comparison; `summary["paired_delta"]` is.
7. **For the `prior` surface, supply ≥ 2 families.** D4 now raises instead of silently
   training on the holdout.

### 8.3 What not to claim

- Do not report `make_code_evaluator` scores as benchmark results.
- Do not read `cost` in the multi-objective evaluator as token cost — it is `len(text)/600`.
- Do not describe a `credit_horizon` or `trace_type` change as a score improvement; they are
  feedback-plumbed, and the adapter says so.
- Do not treat a v2 `RunResult` with `portable=False` as a reproducible artifact — that flag is the
  control plane telling you the result depended on injected behavior.

---

## 9. Layer 9 — prescription

### 9.1 Make recursion expressible in portable v2 (the minimum viable fix for G1)

The gap is narrow and specific: v2 needs **one portable module that carries a recursive level**, plus
an `output`-mode evaluator for it. Concretely:

1. **Register `recursive_opt.module.config_level@1`** — builds a `MetaLevel` from
   `module.config = {family, targets, constraints, fixed}`, with `snapshot`/`restore` over the config
   text (it is already a plain string, so the artifact is trivially JSON).
2. **Register `recursive_opt.evaluator.task_score@1` in `output` mode** — takes the module's
   `{"score", "feedback"}` output and returns an `EvaluationResult`. `_default_module_evaluator`
   almost does this already.
3. **Register `recursive_opt.dataset.tracebench@1`** — resolves `{ref, split, config}` into task ids,
   so datasets stop being legacy-only.
4. Only then is the `trace` engine running a *real* recursive level on a portable, promotable spec.

That is roughly 150 lines against a 2,835-line file, and it converts the control plane from
scaffolding into the thing it was built for. Everything else (fingerprints, budgets, holdout gating,
resume) already works and would immediately apply.

**This was checked, not assumed.** The three things that would make it hard are all already true:

```
MetaLevel parameters   : [('level_config:0', 'str')]      -> trivially JSON-snapshottable
forward() output keys  : ['feedback', 'score']            -> an output-mode evaluator
                                                             can consume it directly
registry surface       : register_module / register_evaluator / register_dataset /
                         ModuleRegistryEntry all present
```

So `snapshot` is `{name: node.data}`, `restore` is the inverse, and the evaluator is
`_default_module_evaluator` with a `{"score", "feedback"}` shape it already understands. There is no
architectural obstacle — the modules simply were never registered.

### 9.2 The smallest experiment that proves or kills the idea

**Step 0, before anything else: pick a benchmark with a live surface.** §7.1 measured signal/noise of
0.96 on `internal:multiobjective_gsm8k` and 0.74 on `hf:qasper` — on both, the prompt effect is inside
the run-to-run noise, and the historical data shows gsm8k pinned at −0.1601 ± 0.0035 across 65
evaluations and every configuration tried. **No optimizer can demonstrate a gain on a surface that
flat**, so running the experiment below on these two tasks would waste the budget whatever the answer.
Screen candidate tasks with `spec.score_spread` (it reports `flat: True`) at the *same* `max_examples`
you intend to run at, and only keep tasks where the probe spread exceeds the repeat noise by a clear
margin. This is the cheapest step and the one most likely to change the outcome.

Then, do **not** restart the 14-use-case programme. Run one experiment with a pre-registered
hypothesis:

> **H:** For a family of tasks, an O1 config learned on family members *transfers* to a held-out
> member better than a config tuned directly on the held-out member at equal budget.

Design:

- **≥ 4 families**, so the holdout is not a single task (D4's minimum of 2 is a floor, not a target).
- **Both arms scored on the identical held-out set** — enforced now by the D3 gate.
- **Targets = `starting_artifact` only** for arm 1 (the one knob with a real path), and
  `starting_artifact + numeric` for arm 2. Do not include `memory_policy` or `guide`.
- **Noise floor measured first** (§8.2 rule 4) and reported alongside the effect.
- **n ≥ 5 seeds**, paired deltas, `promotion_decision` with the comparability gate.
- **Artifact-change assertion**: if the winning arm's artifact equals the initial artifact, the run is
  void regardless of the score.

If the transfer gain does not exceed the noise floor under those conditions, the honest conclusion is
that recursive optimization does not pay for itself on this benchmark family, and the package should
be reduced to §8.1(a)–(c), which stand on their own.

### 9.3 What to delete

| Target | Why |
|---|---|
| `make_code_evaluator` (both components) | closed-form targets; produces numbers that read as benchmark results |
| `TinkerEnvAdapter` | zero references outside its own definition; no test with a real client |
| `HITLGate` | **keep or integrate, don't leave as-is**: its only caller is example C with the `auto_allow` stub, so nothing ever exercises a real approval. Either wire it into the trainer's accept/reject path or drop it. |
| The line-count footprint gate | done (D8) — it made the code worse |
| `examples/XP_1stattempt/`, the ~90 stale `notebook_outputs/` run dirs | untracked byproducts that make the tree unreadable; keep `three_way_stage2_20260624_150102` as the evidence for §5 |

### 9.4 Restore the empirical harness

The v2 migration deleted the only apparatus that ever produced an efficacy measurement — the 46-cell
notebook — and replaced it with two tautological fixtures. `examples/recursive_opt_three_way.py`
(969 lines) survives intact and is good: equal-budget arms, learning curves, paired deltas, artifact
diffs, and now a comparability gate. The notebook cells are recoverable:

```bash
git show 2ad8eb09:examples/recursive_opt_use_cases.ipynb   # cell 10 = spec builders, 37 = three-way
```

Rebuild §9.2 on top of that harness rather than on the v2 golden fixtures.

---

## 10. Closing assessment

**Is it a viable new meta/recursive optimization system for Trace?**

The *mechanism* is viable and genuinely reusable — it works with GEPA and any other optimizer, because
it never touches the optimizer contract, only wraps levels as `trace.Module`s. Two components
(`effects.py`'s causal-effect contract, and the numeric-optimizer routing) are worth keeping on their
own merits regardless of what happens to recursion.

**Does it hide a bullshit mechanism?** Not in the code — the code is more honest than most: it refuses
synthetic fallbacks, raises when an adapter is missing, declares which knobs are inert, and documents
its own subtle bugs. The problem is one level up, **in the measurement**. The single result that
justified a year of work compares two different quantities on two different task sets, and both arms
ended with the untouched default configuration. That was not deception; it was an unguarded harness
plus a promotion gate that never asked whether the two numbers were comparable. It is now guarded.

**Is it overly complex?** The recursion spine is not. The control plane is well built but currently
serves an empty set — and its complexity was actively worsened by a line-count gate that rewarded
compression over simplification.

**Where did the effort go, then?** Overwhelmingly into *infrastructure that verifies its own
declarations*. That is the honest lesson of the ~1,000 commits: fingerprints, immutability, registries,
resume identity, capability gates and 490 tests were built to a high standard, while the two questions
that actually decide the project — "do these two numbers measure the same thing?" and "is this effect
bigger than the noise?" — had no gate at all. The result is a system that could not run its own
feature (D14) yet reported green, and that promoted a subtraction as a result (§5.2). Both of those
are now guarded, and the guards are cheap: `scored_task_ids` is 20 lines.

**What is genuinely required?** About 150 lines to make v2 able to express a recursive level (§9.1),
and one honest experiment (§9.2). Until that experiment runs, the correct status of "recursive
optimization improves on standard Trace optimization" is **unproven**, not *disproven* — the existing
evidence does not support it, but it never actually tested it, and the corrected live re-run in §7.2
was too underpowered (n=2/n=1 after provider failures) to settle it either. The difference matters:
the idea has not failed, it has not yet been given a fair trial.

There is also a prior question §7.1 raises and §9.2 must answer first: **on these two benchmarks the
config→score surface is not measurably non-flat.** If a properly powered noise-floor measurement
confirms that, then no optimizer — recursive, standard, or otherwise — can show a real gain there, and
the right response is to change the benchmark, not the algorithm. `spec.score_spread` was built
exactly to answer this and returns `flat: True` when the surface is dead; run it before spending
anything else.

What this pass changes is that a fair trial is now possible (D14 unblocked multi-level runs; D13 wired
in the dead-knob guard) and a rigged one now fails loudly (D3 rejects cross-metric comparisons; D11
rejects false-portable results).


---

## 11. Probe C — why the surface is flat (and what it costs the next experiment)

§7.1 established that the config→score surface is inside its noise floor. Probe C asks *why*, because
"dead task" and "broken instrument" have opposite remedies. Answer: **broken instrument.**

### 11.1 The objective is a blend of a rare discrete term and a continuous one

`internal:multiobjective_gsm8k` carries its own `ObjectiveConfig`:

```python
ObjectiveConfig(mode='weighted',
                weights={'error': 1.0, 'tokens_in': 0.001, 'tokens_out': 0.001},
                minimize={'error', 'tokens_in', 'tokens_out'})
```

so `score = −(1.0 × error) − (0.001 × tokens)`. Measured over 6 examples per prompt:

| prompt | error rate | mean tokens | error term | token term | score |
|---|---:|---:|---:|---:|---:|
| `Answer directly.` | 0.167 | 306 | −0.167 | −0.306 | **−0.472** |
| `Plan step by step, then verify…` | 0.000 | 397 | 0.000 | −0.397 | **−0.397** |

Decomposing the prompt effect of **+0.075**:

```
from ERROR  : +0.1667      <- the real quality signal
from TOKENS : -0.0915      <- cancels 55% of it
```

**The cost term eats more than half the accuracy gain**, and at the `max_examples=2..4` used by the
historical experiments the error term almost never fires at all — which is exactly why gsm8k reads as
a deterministic constant (−0.1601 ± 0.0035 over 65 evaluations, §5.2). At those budgets the objective
*is* a token counter, so "optimization" was rewarding brevity.

### 11.2 The accuracy signal is real but far too thinly sampled

Across all 12 gsm8k evaluations there was **exactly one wrong answer**. So the `+0.167` above rests on
a single event and is not a usable effect estimate. Sampling an ~8% error rate to ±0.025 needs on the
order of **n ≈ 120 unpaired examples**; the experiments ran at 2–8. Pairing (every arm scored on the
*same* fixed examples) cancels example-difficulty variance and is the cheap way out.

### 11.3 A third silently-dead evaluator

`internal:multiobjective_bbeh` returned `accuracy: 0.0` on every example for both prompts, with
`execution_time_s ≈ 6×10⁻⁶` — **six microseconds, i.e. no inference happened at all.** It is not a hard
task, it is a broken one, and it has been sitting in the task pool alongside gsm8k. That is the third
component found scoring without evaluating (after example C, §5.4, and the `make_code_evaluator`
closed-form targets, §3.2).

### 11.4 What this predicts for a knowledge-transfer experiment

A typed knowledge card of ~200 tokens costs `0.001 × 200 = −0.20` on this objective. The **entire
measured prompt effect is +0.075**. So every knowledge-injecting arm starts roughly 2.7× the effect
size in the hole, purely for existing:

> On the current task pool, `M0 no_knowledge` beats `M4 typed_knowledge_cards` **by arithmetic**,
> before any question of whether the knowledge is any good.

Running the experiment as specified would therefore produce a confident, expensive **false negative**
for its own central hypothesis. The fix is not a bigger n or a different model — it is to stop charging
the injected context to the score, and to score accuracy on paired examples.

---

## 12. The task pool: why tasks looked broken, and what each one actually needs

§11 ended by recommending a certification screen. Building it (`measurement.py`, 54 tests) and
running it produced a result I had to retract and re-derive twice. Both retractions are the point of
this section, so they are recorded rather than tidied away.

### 12.1 First sweep: "1 of 8 usable" — and it was my harness, not the tasks

The first certification injected the prose probe `"Answer directly."` into every task. Six of eight
came back broken or degenerate, and I reported that examples A and B "run on tasks that do not work".
**That was wrong.** Mapping the verdict against each task's actual trainable surface shows a perfect
pattern:

| task | trainable surface | evaluation calls a model | first verdict |
|---|---|---|---|
| `internal:multiobjective_gsm8k` | prose | yes | certified |
| `hf:drop` | prose | yes | too_noisy |
| `internal:code_param` | **Python code** | **no** | broken (0.007 s) |
| `internal:multi_param` | **float** | **no** | broken (`ExecutionError`) |
| `internal:multiobjective_bbeh` | prose | **no** | broken (0.017 s) |
| `llm4ad:…online_bin_packing` | **code** | **no** | broken (`-1e6`) |
| `veribench:binary_search` | **Lean 4** | **no** | degenerate (0.0) |

Every "broken" task had a non-prose surface and an LLM-free evaluator. Writing `"Answer directly."`
into a float parameter gives `float("Answer directly.")` → `ExecutionError`. Writing it into a code
parameter gives a program that does not run → the evaluator's failure score. **This is the same class
of error as §5.2**: an instrument mis-applied, producing confident output about the wrong thing.

The second, worse assumption was baked into the verdict logic: "returned in 0.007 s → no model was
called → broken". For an LLM-free evaluator, returning instantly is *correct*. Worse, it is
**desirable** — a deterministic evaluator has no sampling noise at all.

### 12.2 The liveness probe: all five are alive

The right question is not "did it call a model" but "does the score respond to a valid change in its
own trainable parameter". Perturbing each parameter in a way appropriate to its surface:

| task | surface | baseline | perturbed | spread | live |
|---|---|---:|---:|---:|:--:|
| `internal:code_param` | code | 1.0 | 0.0 | 1.00 | ✅ |
| `internal:multi_param` | numeric | −1.0 | −2.0 / −5.5 | 4.50 | ✅ |
| `internal:multiobjective_bbeh` | prose | 1.0 | ~0.0 | 1.00 | ✅ |
| `llm4ad:…online_bin_packing` | code | **−2091.8** | −1e6 | 9.98e5 | ✅ |
| `veribench:binary_search` | Lean 4 | 0.0 | 0.1 | 0.10 | ✅ |

**All five respond. None was ever broken.**

### 12.3 Second retraction: a real cost is not a sentinel

The corrected certifier then flagged `llm4ad:online_bin_packing` broken because `-2091.8` exceeded a
magnitude threshold I had chosen for sentinel detection. But `-2091.8` is its **real objective** — a
bin-packing cost — and `-1e6` is its failure code. A magnitude threshold cannot tell them apart. The
check now tests the **exact** sentinel magnitudes (1e6, 1e9, 1e12, non-finite) and nothing else.

### 12.4 Corrected certification, and what each task needs

| task | verdict | diagnosis | what it needs |
|---|---|---|---|
| `internal:multi_param` | **certified** | numeric, deterministic, **noise 0.0**, headroom 4.5 | nothing — use as-is |
| `llm4ad:…online_bin_packing` | **certified** | code, deterministic, **noise 0.0**, real cost objective | nothing — use as-is |
| `veribench:binary_search` | **certified** | Lean 4, deterministic, **noise 0.0**, baseline is a placeholder at 0.0 | nothing — maximum headroom |
| `internal:multiobjective_gsm8k` | **certified** | prose + live model, sd 0.033, resolves 0.041 at n=5 | keep the evaluation bounds pinned |
| `internal:code_param` | saturated | baseline **already scores 1.0** | seed a deliberately weak baseline (which `run_code_experiment` already does) |
| `internal:multiobjective_bbeh` | saturated | baseline **already scores 1.0** | a harder BBEH subset, or a weak seed |
| `hf:qasper` | degenerate | scored 0.0 three times after **204 s** of real compute | raise `max_tokens` — 512 truncates long-form answers; re-certify |
| `hf:drop` | too_noisy | sd 0.118 → needs n=44 for a 0.05 effect | more examples per evaluation, or pin temperature; re-certify |

**Four certified tasks, not one.** Three of them are **deterministic with exactly zero sampling
noise**, which means they resolve *any* effect size at n=1. They were the best measurement surfaces
in the repository the entire time, and the harness had been reporting them as dead.

### 12.5 Why this matters more than the noise fix

§11's temperature fix bought a 14.6× resolution improvement on a noisy prose task. This buys
something categorically better: on `multi_param`, `online_bin_packing` and `veribench`, **the noise
floor is zero**. Any measured difference between two arms on those tasks is real by construction —
no seeds, no confidence intervals, no power analysis needed.

The practical consequence for the research programme: questions about *optimizer* behaviour (does
recursion beat standard search? does a warm prior help? is the numeric route better than the
generative one?) should be answered **first** on the deterministic surfaces, where the answer is
unambiguous and nearly free, and only then re-tested on the noisy LLM-scored tasks where the effect
must additionally clear a measured noise floor.

---

## 13. Deliverable B — optimization on a certified instrument

B was specified as the corrected UC4 (≥4 families, both arms on the same holdout). §12's
certification made that impossible at the time: only one *LLM-scored* task was certified, and a
`prior` level needs ≥2 families (guard D4). Running it on the mis-certified tasks would have repeated
the exact failure this document is about, so B was reduced to the strongest question the certified
instrument could answer:

> Does standard Trace optimization move a certified surface by more than the instrument's own
> resolution limit (0.041 at n=5)?

`internal:multiobjective_gsm8k`, 5 seeds, paired, bounded evaluation (temp 0.2 / 512 tokens / 60 s).

### 13.1 Result

```
initial  mean : -0.5694
standard mean : -0.1420
paired deltas : [+0.1195, +0.0903, +0.4403]
paired mean   : +0.2167          (5.3x the resolution limit of 0.041)
```

Two things here are new in this project's history. The **artifacts actually changed** — every prior
run ended on the stock defaults (§5.2, §7.2):

```
""                                                                    (initial)
"Let's think step by step"
"Solve the math problem. Do not show work. Output only the final numeric answer."
```

and the effect exceeds the instrument's *measured* resolution rather than sitting below it.

### 13.2 …but it is not established, and the reason is instructive

Two of five seeds died on the recurring `NoneType` optimizer crash under provider load, leaving n=3:

```
paired sd              : 0.1585
standard error         : 0.0915
one-sided 95% LCB (t)  : -0.0506      <- crosses zero
sign test, 3/3 positive: p = 0.125
```

All three deltas are positive and the mean is large, but n=3 at that spread does not clear 95%.

### 13.3 Decomposition: half the gain is brevity, and the *choice* is all brevity

The winning artifact — *"Do not show work. Output only the final numeric answer."* — is what a token
minimiser produces against `-(1.0 x error) - (0.001 x tokens)`. Decomposing (n=12 examples per arm):

| arm | score | error rate | tokens | error term | token term |
|---|---:|---:|---:|---:|---:|
| initial (empty) | −0.9417 | 0.083 | 273.5 | −0.0833 | −0.2735 |
| optimizer best (terse) | −0.1682 | 0.000 | 162.7 | 0.0000 | −0.1627 |
| optimizer alt (CoT) | −0.1932 | 0.000 | 200.2 | 0.0000 | −0.2003 |

```
terse vs initial :  from ERROR +0.0833 | from TOKENS +0.1108  ->  57% of the gain is TOKENS
CoT   vs initial :  from ERROR +0.0833 | from TOKENS +0.0733  ->  47% of the gain is TOKENS
```

So the honest reading is threefold, and the middle point is the one that matters:

1. **There is a real accuracy component** — error fell from 8.3% to 0% for *both* optimized prompts.
   But it rests on one error disappearing out of twelve, and the Wilson intervals overlap
   (initial `(0.015, 0.354)` vs optimized `(0.000, 0.243)`), so **the accuracy half is not
   established either.**
2. **The token component is solid**: 273.5 → 162.7 tokens is large and reproducible.
3. **The ranking between the two candidates was decided purely by brevity.** Both achieve error
   0.000; they differ only in length, and the objective preferred the shorter one. §11.4 predicted
   exactly this.

### 13.4 What B actually establishes

- The optimizer **works**: it changed the artifact and improved the stated objective by more than the
  instrument's resolution — the first time either has been true here.
- It **optimized the objective it was given**, and that objective pays for silence on a task the model
  already mostly solves. The optimizer is not at fault; the metric is.
- Therefore **A2 (splitting accuracy from cost) is now the blocking item**, not the optimizer and not
  the recursion. With the two reported separately, this run would have read: *"accuracy ties at 0
  errors; cost differs by 111 tokens"* — an unambiguous, immediately interpretable result, instead of
  a single number that conflates them and hides that the choice was made on length.
- B's ≥4-family design is now **achievable but was not run**: §12 unlocked three deterministic
  surfaces with a **zero** noise floor, where n=1 suffices and no seed can be lost to provider
  failures. That is where the recursive comparison should be run next.

---

## 14. Probe K, and two retractions

### 14.1 What Probe K appeared to show

Run on `llm4ad:optimization/online_bin_packing`, whose evaluator *runs the candidate
heuristic* — certified at the time with a **zero** noise floor:

```
baseline  : -2091.8 on every seed (identical)
optimized : -2087.0 on 2 of 3 seeds (identical)
delta     : +4.8, reproduced exactly across independent seeds
```

I reported that as the first unambiguous result in this project. **It is not.**

### 14.2 The tell, and the fourth certifier flaw

The reported artifact was *empty* — the same as the baseline's — which cannot produce a
different score. The recorded candidates then showed **identical configurations**
(`artifact=''`, `batch=4`, same trace and horizon) scoring anywhere from −2097.0 to
−2087.0, while a direct check of the same task gave exactly −2091.8 twelve times.

The difference is **concurrency**. The evaluator runs the candidate under a time budget;
under CPU contention fewer instances complete:

| condition | noise sd | score range |
|---|---:|---:|
| sequential | 0.0000 | 0.00 |
| 8 concurrent | 3.15 – 4.41 | 10.80 |

Certification measured the task **alone**; the experiment runs it **under load**. A noise
floor measured under conditions the experiment never reproduces is not a noise floor —
and this one turned a 4.4-sd surface into a reported zero.

`certify_task` now takes an explicit `concurrency` that must match the experiment, records
it on the certificate, and refuses to present a sequentially-measured zero as usable.

### 14.3 Both live results are retracted

Re-certifying at concurrency 8 with more repeats:

```
Probe K  +4.8   on online_bin_packing   noise sd 4.41 -> resolvable@n=5 = 5.52   NOT established
Probe F  +0.217 on gsm8k (n=4)          noise sd 0.32 -> resolvable@n=5 = 0.405  NOT established
```

The gsm8k floor had been estimated at **0.033** from three repeats; six repeats put it at
**0.32–0.53**, ten times larger, because the rare error events only appear with more
samples. Three repeats was not enough to characterise a Bernoulli term — the same mistake,
in my own measurement code, that §11.2 identified in the experiments being audited.

**No positive efficacy result survives.** The optimizer demonstrably changes artifacts and
moves scores; whether it *improves* anything remains unproven.

### 14.4 Certified pool, at realistic concurrency

| task | sequential sd | sd @ concurrency 8 | verdict |
|---|---:|---:|---|
| `internal:multi_param` | 0.0 | **0.0** | **certified** |
| `llm4ad:…admissible_set` | 0.0 | **0.0** | **certified** |
| `llm4ad:…online_bin_packing` | 0.0 | **4.41** | certified (real floor) |
| `internal:multiobjective_gsm8k` | 0.53 | 0.32 | **too_noisy** |
| `internal:code_param` | 0.0 | 0.0 | saturated |
| `veribench:*` | 0.0 | 0.0 | needs a liveness check (disabled in this sweep) |

Two surfaces hold a genuine zero floor under load — `multi_param` and `admissible_set`.
Those are where the next experiment belongs.

### 14.5 The honest summary of this whole pass

Five certifier flaws were found, every one a variant of **reading a constant as "quiet,
therefore good"**: a `-1e6` failure sentinel, prose injected into code and numeric
parameters, an absolute effect target applied to a large-scale objective, a noise floor
measured under the wrong concurrency, and a Bernoulli term estimated from three samples.

All five were in code written *during* this analysis, to audit exactly that error class.
That is the strongest available evidence for the assessment's central claim: the
difficulty here has never been the optimizer, it is that measuring an LLM optimization
system correctly is genuinely hard, and every unexamined shortcut defaults to a
false positive.

---

## 15. Experiment-0's first unit, re-run diagnostically

The frozen main run of 2026-08-24 stopped on its first unit (seed 0, budget 6, arm A)
against the hard constraint `invalid_rate <= 0`: one of 24 holdout samples
(`gsm8k:test:915`, expected 23) produced an empty deterministic extraction. The run's own
report had to record that *"the raw provider response text is not persisted, so the exact
upstream formatting cause is unknown and is not inferred"* — an unanalysable stop.

`evaluator.py` now persists a bounded excerpt of the raw text **when extraction fails**,
so the next such stop is diagnosable. That changes the Experiment-0 source hash, and
`_load_main_lock` therefore refuses to continue the frozen matrix — correctly. What
follows is a **diagnostic replication, not a resumption**: same model profile
(temperature 0, `max_tokens` 384, reasoning disabled), same 24-sample v2 holdout, run
8-way parallel.

```
samples=24  evaluated=24  errors=0
accuracy     : 24/24        (frozen run: 22/24)
invalid_rate : 0/24         (frozen run: 1/24; hard constraint requires 0)
truncated at max_tokens=384 : 0/24
wall=31s     (the frozen sequential run took 199s)
```

`gsm8k:test:915` extracted correctly this time — `FINAL: 23 hours.`, 8 completion tokens.

### 15.1 What this does and does not establish

**My truncation hypothesis was wrong, and the data says so cleanly.** The longest answer
across all 24 samples used **177** completion tokens against a 384 cap, and nothing hit
`finish_reason='length'`. The 384-token budget was not the cause.

**It does not show the failure is fixed.** 0/24 has a Wilson 95% interval of
**(0.000, 0.138)**, and the frozen run's observed 1/24 = 0.042 sits inside it. One clean
replication of a ~4% event is exactly what you would expect to see whether or not
anything changed. Claiming the constraint now passes would repeat the error this whole
document is about.

**It is also not an exact code-path replication.** The probe drives the two-call workflow
directly rather than through `CompoundReasoningModule`, so it reproduces the *prompts and
model profile* but not the full harness.

### 15.2 What would actually settle it

The remaining candidate cause is a transient empty provider response — the same class as
D11/D14 and the one the remote independently hardened. To distinguish "fixed" from "got
lucky" you need enough samples to resolve a ~4% rate: at n=24 the interval spans 0-14%,
so roughly **n ≈ 200 forward calls** for a ±2.5% estimate. That is cheap here (31 s per
24 samples at 8-way parallelism, ~$0.002), and it is the measurement to run before
re-freezing the lock and restarting the frozen matrix.

---

## 16. Experiment-0: the measurement, and what it says about the design

§15 left one question open — is the invalid extraction that stopped the frozen run a real
defect, or a rare event? Probe N measured it: **246 evaluations, 492 forward calls, 818 s
at 8-way parallelism, ~$0.01**, split to separate two causes that need different fixes.

```
                invalid            95% CI              accuracy
A_holdout       0/96   = 0.0000    (0.0000, 0.0385)    95/96
  (24 frozen holdout x 4 repeats)
B_fresh         0/150  = 0.0000    (0.0000, 0.0250)    147/150
  (150 fresh GSM8K test samples, disjoint from every frozen pool)
POOLED          0/246  = 0.0000    (0.0000, 0.0154)    242/246 = 0.9837
transport errors: 0
```

Part A's repeat structure found **0 samples always invalid** and **0 samples sometimes
invalid** — so the failure is neither reliably sample-specific nor visibly transient at
this rate.

### 16.1 The stop was a rare event, not a defect

The invalid rate is bounded below **1.54%**. The original 1-in-24 remains entirely
consistent with that:

| true rate | P(≥1 invalid in a 24-sample holdout) |
|---:|---:|
| 0.0154 | 0.311 |
| 0.0100 | 0.214 |
| 0.0050 | 0.113 |

A single event in 24 was never evidence of a defect. It is what a low-but-nonzero rate
looks like.

### 16.2 …which makes the hard constraint unsatisfiable by construction

`invalid_rate <= 0` is evaluated per unit, on 24 samples, across
**5 seeds × 2 budgets × 4 arms = 40 units**:

| true rate | P(one unit passes) | **P(all 40 units pass)** |
|---:|---:|---:|
| 0.0154 | 0.689 | **0.0000** |
| 0.0100 | 0.786 | **0.0001** |
| 0.0050 | 0.887 | **0.0081** |
| 0.0020 | 0.953 | **0.1463** |

Even at a rate of 0.2% — seven times better than the measured upper bound — the frozen
matrix completes only 15% of the time. **The experiment was overwhelmingly likely to stop
on a hard-constraint violation whatever the optimizer did.** That is a design property,
not a result, and it is the actual reason the run halted.

### 16.3 And the primary metric has almost no headroom

Arm A — the *fixed baseline*, no optimizer — scores **242/246 = 98.4%** on GSM8K with this
model. Total room for any optimizer to improve on: **1.6 percentage points**.

Samples per arm needed to detect a gain, at 80% power:

| gain | n per arm |
|---:|---:|
| +0.5 pp | 8,521 |
| +1.0 pp | 1,747 |
| +1.6 pp (to a perfect score) | 502 |

The frozen design provides **120** observations per arm (24 holdout × 5 paired seeds) —
roughly **15× short** for a +1 pp effect, and it cannot resolve anything smaller at all.

### 16.4 What to change

1. **Drop `invalid_rate <= 0`** for a rate the instrument can actually meet, e.g.
   `invalid_rate <= 0.02` measured over the pooled holdout rather than per unit. A
   zero-tolerance constraint on 24 samples is a coin flip dressed as a gate.
2. **Change the task.** GSM8K at 98.4% is saturated for this model — the §12 finding
   again. Pick a task where the baseline leaves room, and certify it first
   (`measurement.certify_task`).
3. **Size the holdout to the effect.** 500+ samples per arm to see a 1.6 pp gain, or
   accept that only large effects are detectable and say so in the preregistration.
4. Only then re-freeze the lock and restart the matrix.

None of this reflects badly on the experiment's engineering, which is careful: the
provenance locks, the frozen preregistration and the watchdog all did their jobs, and the
stop-decision report is honest about what it could not determine. The gap is that no step
asked whether the instrument could satisfy its own constraint or resolve its own effect —
which is the same gap §5 found in UC4 and §11 found in the objective.

---

## §17 — Iteration 3: recursive vs standard on the zero-noise surfaces

**Verdict: VOID.** Not "recursion does not help" — *no result*. The instrument could not have
detected an effect, for four independent reasons. Raw numbers:
`artifacts/probe_2026/iteration3_analysis.json`.

Design: 3 hypotheses × 3 seeds × 2 arms, run as three concurrent processes. Both arms scored on
the **same** level (`o3_prior`) — the fix for the D3 cross-metric defect that killed UC4. Standard =
2 iterations at `o3`; recursive = 1 at `o2` + 1 at `o3`, warm-started (`reuse_priors`).

| hypothesis | scored task | paired Δ | n | artifacts differ | verdict |
|---|---|---|---|---|---|
| numeric | `internal:code_param` | **0.0** | 3 | no | surface flat + saturated |
| packing | `llm4ad:.../online_bin_packing` | **−3.0** | 3 | no | inside noise |
| mixed | `llm4ad:.../admissible_set` | **0.0** | 3 | no | surface flat |

### 17.1 The in-run replicate control — the one thing worth keeping

The run produced a noise control for free that is stronger than anything measured so far. Every
arm re-scored the *identical* default artifact (content `starting_artifact:`, empty) many times:

| hypothesis | arm | replicates | distinct values | range | sd |
|---|---|---|---|---|---|
| packing | standard | 29 | 10 | **8.00** | 2.61 |
| packing | recursive | 15 | 5 | 6.40 | 1.56 |
| numeric | standard | 27 | **1** | 0.00 | 0.00 |
| mixed | standard | 27 | **1** | 0.00 | 0.00 |

The same bytes scored 29 times on `online_bin_packing` span **8.00 points**. The measured effect
was **−3.0**. The effect is roughly one third of the spread the null produces on its own — and it
is *negative*. This is an internal, same-run, same-concurrency control, and it independently
confirms Probe L's external estimate (sd 4.41 at concurrency 8), which had forced the retraction
of Probe K's +4.8.

### 17.2 Why the other two hypotheses could never have shown anything

`numeric` and `mixed` returned **one distinct value across 27 replicates**. That is not a quiet
surface, it is a *flat* one: every candidate in the menu scores identically, so no optimizer of any
kind — recursive or standard — can express a preference. A 0.0 delta here is arithmetic, not evidence.

The two cases fail for *different* reasons, and the difference matters:

- **`numeric` (`internal:code_param`) — process failure, mine.** Probe L already certified this task
  `saturated` at both concurrency 1 and 8, and §12.4 already recorded it. I had a certified
  instrument that rejects this surface and I did not run it before spending the compute.
- **`mixed` (`llm4ad:.../admissible_set`) — instrument gap, D18.** Probe L certified this task
  **`certified`** (live, quiet) at both concurrencies. It is nevertheless perfectly flat here.

D18 is the more important of the two. `probe_liveness` establishes headroom by perturbing the task
input and checking the score moves. The experiment perturbs something else entirely — the
`starting_artifact` config knob — and with respect to *that* knob the surface is flat. **Certification
is not menu-conditional: a task can be certified live and still have zero headroom for the specific
knob the experiment varies.** Every "certified" verdict issued so far therefore carries an unstated
qualifier, and none of them licenses an experiment that turns a different knob.

### 17.3 The fairness flaw, and the real finding

"Equal total budget" (`candidates: 4`) gave the standard arm **2 iterations at the scored level**
and the recursive arm **1** — the recursive arm spent its other iteration on `o2`. The candidate
counts confirm it: standard evaluated 12 prior-level candidates, recursive 5. This is a flaw in my
probe design, and it is the genuinely interesting output of Iteration 3:

> **You cannot hold total compute and scored-level search equal at the same time.** Equalise total
> compute and the recursive arm is starved where it is measured; equalise scored-level search and
> the recursive arm consumes strictly more compute, so any win is confounded with spend.

Every future recursive-vs-standard comparison must state which of the two it equalises, and report
the other. Neither UC4, Probe F, Probe K, nor this iteration did so.

### 17.4 Defect reproduced live: invalid-config sentinel as a score

12 candidates (6 in `packing`, 6 in `mixed`) were recorded with score **−1,000,000.0**. The
`ART_MENU` entry `"The prior should generalise to held-out families…"` is prose, and prose is not
decodable as config text on the prior surface, so it scores the invalid sentinel every time. Two
consequences: (a) **one third of the standard arm's search budget was spent on a candidate that is
invalid by construction**, and (b) the sentinel reaches `artifacts.jsonl` as a *score*. Selection
took the max so it did not propagate here, but any mean over candidates would be destroyed by it —
this is D1's failure mode surviving on a path the D1 fix did not cover. It is also the same error
class as the prose-injection certifier bug of §14: putting prose on a non-prose surface.

### 17.5 Against the previous results

| result | claimed | correctly-measured noise | status |
|---|---|---|---|
| UC4 `o2→o3` | +0.163 | — (cross-metric identity) | **retracted** — `qasper − mean(gsm8k, qasper)` |
| Probe F (gsm8k) | +0.217 | sd 0.32–0.53 (n=6) | **retracted** — inside noise |
| Probe K (packing) | +4.8 | sd 3.15–4.41 @ concurrency 8 | **retracted** — inside noise |
| Iteration 2 | — | — | killed at pre-flight |
| **Iteration 3 (packing)** | **−3.0** | **range 8.00, sd 2.61 (in-run, n=29)** | **void** — inside noise, wrong sign, flat co-surfaces, unequal scored-level search |

The pattern is now four for four: **every recursive gain measured so far has been smaller than the
correctly-measured noise floor of its own surface**, and each was believed only because the noise
floor was measured wrongly (or not at all) at the time. Iteration 3 is the first where the noise
control came from *inside the same run*, which is why it was caught before it was claimed.

**Standing status: there is still no measured evidence that recursive optimisation beats standard
optimisation on any surface.** There is also none against it — the instrument has never yet been
pointed at a surface that is simultaneously live, unsaturated, and fairly budgeted.

### 17.6 What Iteration 4 must satisfy

1. **Fix D18 — make headroom menu-conditional.** `certify_task` must probe liveness by varying the
   *same knob the experiment will vary* (the candidate menu), not a generic input perturbation, and
   reject any surface whose menu yields one distinct score. Until this lands, a `certified` verdict
   does not license an experiment that turns a different knob. Re-certify against their menus before
   reuse; `internal:code_param` was already `saturated` and should simply have been excluded.
2. **Menu/surface type check** — refuse prose candidates on non-prose surfaces (reuse
   `detect_surface`); no candidate may be invalid by construction.
3. **Declare the equalisation** — pick scored-level search parity, and report the compute ratio.
4. **In-run replicate control** — always re-score the initial artifact n≥10 times per arm and
   publish the range next to the effect. Cheap, and it caught this one.

---

## §18 — When *can* recursion beat standard? Two win conditions

On a held-out task, O3's output is a single config; standard's output is also a single config.
The only difference is which data found it — O3 used *other* tasks, standard used the target
itself. Standard has direct access to the objective, so **standard wins by default** and
recursion needs a specific structural reason. There are exactly two, and they have *opposite*
measurement requirements.

**W1 — variance / bias-variance. REQUIRES NOISE.**
Direct per-task optimisation fits the target's *noise* when the per-task budget is small and the
score is noisy. A prior averaged over sibling tasks has lower variance. Recursion then wins on
**quality at equal per-task budget**.
→ On a deterministic, zero-noise task, direct optimisation *cannot overfit*, so **W1 is
structurally impossible there**.

**W2 — amortisation / compute. WORKS AT ZERO NOISE.**
Even with zero noise: if standard needs 20 candidates to reach the optimum and a transferred
prior arrives within 2, recursion wins on **total compute to reach quality Q across K tasks**.

    standard = K·c_std      recursive = c_meta + K·c_rec      K* = c_meta / (c_std − c_rec)

Requires a shared optimum and non-trivial search cost. Noise is *not* required.

### 18.1 The trap this explains

I had been selecting zero-noise surfaces to buy statistical power. **Those are exactly the
surfaces on which W1 cannot exist.** Every null so far is consistent with having measured W1 where
W1 is impossible by construction. The two requirements pull in opposite directions:

| noise | measurement power | W1 possible? |
|---|---|---|
| near zero | high | **no** — nothing to overfit |
| moderate/high | low (n ≥ 115) | yes |

**Rule:** every experiment must state which win condition it tests. Testing W1 on a deterministic
surface is a category error — and that is precisely what Iteration 3 did. Deterministic families
should target **W2** and report break-even K*; noisy families target **W1** and must budget for
the n the noise floor demands.

---

## §19 — Probe R: the menu is the instrument (root cause of every null so far)

**This supersedes the "flat surface" diagnosis of §17.2, and it is the most consequential finding
in this document.** Raw data: `artifacts/probe_2026/probe_r_results.json`.

`TraceBenchTaskAdapter._apply_starting_artifact` (`tracebench.py:642`) writes the candidate text
straight into the trainable node's `_data` — `param.system_prompt._data = text`, else
`plist[0]._data = text`, else `param._data = text` — with **no surface check at all**, and
returns `True`. `measurement.detect_surface` already computes the surface type correctly; the
production path simply never consults it.

Measured consequence on the two tasks Iteration 3 used:

| task | surface | original param | prose candidate → |
|---|---|---|---|
| `online_bin_packing` | `code`, `calls_llm=False` | 366-char `priority()` | overwritten, **2/2**, score `−1e6` |
| `admissible_set` | `code`, `calls_llm=False` | 353-char `priority()` | overwritten, **2/2**, score `−1e6` |
| `internal:multi_param` | `numeric` | float `1.0` | overwritten by prose |
| `internal:code_param` | `code` | `def f(x): return x` | overwritten by prose |

The Iteration 3 menu was `["", "Answer directly.", "Plan step by step…"]`. The two prose entries
destroyed the program and scored the sentinel; only the empty entry produced a valid score.
**The effective menu size was 1.** Both arms searched a singleton space. No optimiser of any kind
— recursive or standard — could have expressed a preference, and the "flat surfaces" were never
flat.

### 19.1 A menu collapses to size 1 in two independent ways

**(a) Type-incompatible** — prose on a code/numeric surface; every candidate invalid. This is the
same defect as the §14 certifier incident, where I injected prose into code/float/Lean parameters
and made 5 healthy tasks look broken. I fixed it *in the certifier* via `detect_surface` and
**never fixed it in the production path**. The bug was diagnosed and then half-fixed.

**(b) Ranking-equivalent** — candidates that are monotone transforms of one another. My own first
re-test fell into this: for `online_bin_packing` only the *argmax* matters, so `item - bins`,
`-(bins - item)`, `1/(gap+eps)` and `-(gap**2)` are **the same heuristic** and scored identically
(all −2091.8), which I briefly misread as the task being insensitive to code.

### 19.2 With a valid menu, both surfaces are richly live

Executability controls first (proving the candidate reaches the benchmark): `raises`,
`syntax_err` and `wrong_name` all score `−1e6`; the original scores `−2091.8`.

| `online_bin_packing` | score | | `admissible_set` | score |
|---|---|---|---|---|
| best_fit (tightest) | **−2091.8** | | baseline `0.0` | **−1161.0** |
| exact_then_best | −2091.8 | | count_nonzero | −1161.0 |
| ratio | −2091.8 | | mod3 | −1161.0 |
| first/last_fit | −2099.4 | | sum | −1263.0 |
| almost_worst | −3477.8 | | weighted_idx | −1500.0 |
| worst_fit (loosest) | −5000.0 | | neg_sum | −1551.0 |
| **distinct 4, range 2908.2** | | | **distinct 4, range 390.0** | |

Both tasks are deterministic (`calls_llm=False`) and sequentially noise-free, giving effect sizes
of 2908 and 390 against a noise floor of 0. That is enormous statistical power — these are
excellent instruments, and they were being read through a broken menu.

### 19.3 What this retracts and what it re-opens

- **Retracted:** "`admissible_set` and `internal:code_param` are flat surfaces" (§17.2). They are
  not. The menu was.
- **Retracted:** D18 as primarily "certification is not menu-conditional". That remains true and
  worth fixing, but it is *secondary*. The primary defect is that the menu was type-incompatible
  with the surface and nothing checked.
- **Re-opened:** every null result on a code surface. Iteration 3, Probe K, and any UC4 arm scored
  on a code surface were all run through an effective menu of size 1.
- **Unchanged:** UC4's `+0.163` remains an arithmetic identity (a different defect, §5.2).

### 19.4 The open structural question this exposes

On a *code* surface the artifact is a function with a task-specific signature —
`priority(item, bins)` for bin packing, `priority(el, n, w)` for admissible set. **Such an
artifact cannot transfer across tasks with different signatures.** So O3's "one transferable
config" can only be the `LevelConfig` knobs (`batch_design`, `batch_size`, `credit_horizon`, …),
not the artifact itself — and those are precisely the knobs whose causal effect `effects.py` was
built to interrogate. Whether *any* of them carries transferable signal is now the pivotal
question for the whole recursive program.
