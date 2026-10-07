# EXP00 — protocols of the pre-numbered campaigns (June–August 2026)

These campaigns ran before the numbered EXP series, from notebooks now archived in [`notebooks/`](notebooks/)
(moved from `examples/`) and from the `multiobjective_reasoning` package. None was pre-registered as an EXP. The protocols below are
**reconstructed from the executed notebooks, the persisted run outputs and the commit history**: what was
actually run, not what was planned. Sub-studies are labelled A–E in chronological order.

Common elements of A–D:
- Branch `recursive_opt` (fork `doxav/NewTrace`).
- Live model `gpt-5.4-nano` through LiteLLM/OpenAI.
- Tasks from Trace-Bench (`llm4ad:*`, `internal:*`, `hf:*` bundles).
- Package `opto/features/recursive_opt` (levels O0–O3, `MemoryLite`, declarative `run_spec`).
- Default inner trainer `PrioritySearch` with `OptoPrimeV2`.

Level vocabulary used by these notebooks:

| Level | What is optimized |
|---|---|
| O0 | a task artifact (prompt/code) |
| O1 | the setup used to optimize O0 (`LevelConfig`/`MetaLevel`: batch size/design, memory policy, trainer, trace type) |
| O2a | a component's source code (`CodeArtifactLevel`) |
| O2b | a per-family setup policy (`FamilyPolicyLevel`) |
| O3 | a transferable prior scored on held-out families (`PriorInductionLevel`) |

---

## EXP00-A — [`notebooks/recursive_opt_demo.ipynb`](notebooks/recursive_opt_demo.ipynb) (formerly `examples/`): the four meta-optimization types (7–11 June 2026)

**Question:** can each recursion type be expressed and executed on Trace, and does it improve its target?

**Notebook history:** commits `56a27a0514`, `5673f8399a`, `1ebbc575c3`, `ac73001be7`, `51c0bc3c97`, `c32649889c`, `58a5c0f8ef`. The executed version is `58a5c0f8ef` (11 June).

| Arm | Optimized surface | Task | Procedure |
|---|---|---|---|
| A — setup | O1 config: `batch_size`, `batch_design`, `memory_policy`, `trainer` | `llm4ad:online_bin_packing_local`, `internal:multi_param` | 3 hand-listed configs scored offline. Live: `examples/recursive_opt_example_A_learn_setup.py --live` |
| B — component code | O2a `batch_design` source (`@bundle(trainable=True)`) | local hard-item validator (`idx % 3 == 0` items are "hard") | Offline: hand-written improvement. Live: `PrioritySearch` + `OptoPrimeV2`, 4 iterations, 1 candidate |
| C — capability | capability prompt, objectives accuracy↑ and cost↓ | `internal:multiobjective_gsm8k` (2 examples) | 4 hand-written candidates; Pareto/scalar selection |
| D — cross-family | O2 family policy and O3 held-out prior | families {online_bin_packing, admissible_set} and {multi_param, numeric_param} | `relative_delta` scoring against the default config, clipped to [−1, 1] |
| E — declarative spec | O1→O2→O3 chained by `run_spec` | same families | `max_examples=2`, `inner_steps=1`; prior promotion with `min_support=2`; warm-start reuse |

**Budget (live):** at most 64 optimizer calls, 80 eval calls, 16 candidates, 300 s;
`max_examples=4`, `inner_steps=2`.

**Endpoints:** score before → after for each arm, and the initial → final diff of the trained variable.

---

## EXP00-B — [`notebooks/recursive_opt_phases.ipynb`](notebooks/recursive_opt_phases.ipynb) (formerly `examples/`): the Phase 0→7 campaign (10–13 June 2026)

**Question:** which O1 choices should be adopted (trainer, trace type, warm priors, optimizer tools,
threads, distilled skills), decided phase by phase?

**Notebook history:** commits `c32649889c` → `611d31eecf`. The executed version is `611d31eecf` (13 June).

**Decision rule (stated in the notebook):**
- ADOPT if the paired same-seed Δ exceeds one pooled std on at least 2 families at equal budget.
- REJECT if Δ < 0.
- Otherwise PARK.

| Phase | Factor | Arms | Surface / task | Seeds × iterations |
|---|---|---|---|---|
| 0 | gates | unit tests, live preflight, real adapter, score-spread probe | panel of 6 tasks | — |
| 1 | inner trainer | `MinibatchAlgorithm`, `PrioritySearch` | code: `_weak_batch_design` validator | 3 × 8 |
| 2 | trace type (feedback channel) | `internal`, `otel`, `hybrid` | prompt: `internal:multiobjective_gsm8k`, `inner_steps=0` | 3 × 4 |
| 3 | prior reuse at start-up | cold, warm (namespaced campaign memory) | prompt config: `starting_artifact` menu + `batch_size` | 3 × 4 |
| 4 | optimizer-side tools | plain, `trace_search` + `note` (gated on failure episodes in memory) | prompt | 3 × 4 |
| 5 | threads | 1, 8 | code validator | 1 × 6 |
| 6 | distilled `skills.md` | plain, skill (distilled from the best positive-score artifact) | prompt | 3 × 4 |
| 7 | Terminal-Bench 2 onboarding | adapter template only | — | not run |
| Confirm | Phases 2, 3, 6 again at `max_examples=8` | as above | prompt | 3 × 4 |

The campaign closes with a causal-effect contract demo: which `LevelConfig` fields actually reach the score at
`inner_steps=0` and at `inner_steps=2`.

---

## EXP00-C — [`notebooks/recursive_opt_phases_V2.ipynb`](notebooks/recursive_opt_phases_V2.ipynb) (formerly `examples/`): level-by-level evidence notebook (12 June; re-executed 30 September 2026)

**Question:** for each level (O0–O3), what has the branch demonstrated, with numbers?

**Notebook history:** committed in `ad560d81df` (12 June). The saved outputs come from a re-execution
committed in `0f6786f127` (30 September); they reference post-restructuring paths.

**Procedure:**
- Re-runs examples A–D offline with the bounded real Trace-Bench eval-only adapter (no optimizer LLM).
- Builds a positive/negative claims table.
- Writes three recommended specs (HF QA family, internal multi-objective family, one `llm4ad` family) for O1, O2
  and O3. These specs are not executed (`RUN_RECOMMENDED=False`).

| Example | What is compared |
|---|---|
| A | baseline config vs 4 candidate configs on `online_bin_packing_local` and `multi_param` |
| B | hand-written baseline vs improved code for `batch_design` and `trace_summarizer` |
| C | 4 capability prompts on accuracy vs cost |
| D | 2 O2 policies and 2 O3 priors |

---

## EXP00-D — use-case suite ([`notebooks/recursive_opt_use_cases_executed_5a148ddba9.ipynb`](notebooks/recursive_opt_use_cases_executed_5a148ddba9.ipynb), archived from commit `5a148ddba9`): use-case suite UC1–UC14 (15 June – 5 July 2026)

**Question:** in which concrete use cases does recursive optimization beat standard (one-level) Trace at
equal budget?

**Notebook history:** commits `cfc3a1daab` (around 15 June) … `5a148ddba9` (1 July, last version with
executed outputs, 72 cells) → `a8fab200a1` (5 July, restructured, no outputs). The notebook was later
replaced by a 5-cell smoke notebook (`21a0ad3d2f`, `c92f0af4af`, `0f6786f127`).

Persisted outputs: `examples/notebook_outputs/recursive_opt_use_cases/` (about 90 run directories). The
author's analysis is `examples/recursive_opt_use_cases_CURRENT_LIMITS.MD`.

### Single-arm use cases (3 experiments per use case, n = 3–5 seeds)

| UC | Surface | What is optimized | Task / evaluator |
|---|---|---|---|
| UC1 | code | `batch_design`, `trace_summarizer` (default/strict), BBEH direct solver | local validators; `internal:multiobjective_bbeh` direct answers |
| UC2 | config | prompt menu, `initial_knowledge` / warm prior, DROP vs QASPER, mixed GSM8K+QASPER; causal numeric arm (`batch_design`, `batch_size`, `inner_steps=2`) | Trace-Bench GSM8K, DROP, QASPER |
| UC3 | capability | 3 seed framings (weak, terse, verify); Pareto over accuracy and cost | `internal:multiobjective_gsm8k` |
| UC4 | O2 / O3 | O2 family policy; O2 causal numeric policy; O2 with Optuna; O3 cold and warm prior | families {gsm8k, qasper} |
| UC5 | code vs tools | helper code (take_last, stride baselines); optimizer-side tools; tool-policy code | `internal:batch_design`, tool-policy cases |
| UC6 | feedback channel | `trace_type` ∈ {internal, otel, hybrid}, `credit_horizon=step`; plus a causal numeric arm | QASPER |
| UC7 | graph | routing to a SciPy sub-optimizer node (unconditional, then conditional with tool cost) | `examples/recursive_opt_abc_probe.py` graphs |
| UC8 | code | meta-campaign policy (stop / control / switch / split) | `internal:campaign_policy` decision cases |
| UC9 | code | agentic tool + hint policy | `internal:agentic_trace_policy` |
| UC10 | code | artifact promotion gate (`GuardedDecisionEvaluator`) | `internal:artifact_promotion_policy` |
| UC11 | code | QASPER prompt-emitter function scored through Trace-Bench injection | `hf:qasper` |
| UC12 | API | six promoted primitives (budget dict, multi-seed, numeric routing, active fields, …) | deterministic checks |
| UC13 | numeric | Optuna vs LLM config search on causal fields | offline preflight; live QASPER/GSM8K, 8-trial budget |
| UC14 | code transfer | source-policy prior → held-out target adaptation | code-policy pool |

### Three-way benchmark (`examples/recursive_opt_three_way.py`)

- **Arms:** `initial` (no optimization), `standard` (one-level Trace) and `recursive` (prior carry, extra level,
  or numeric route), all at equal total candidate budget N.
- **Reported:** learning curves and artifact diffs.
- **Recursive verdict:** credited for a higher final or best score, for fewer candidates to reach standard's best,
  or for fewer optimizer calls.
- **Stage 2** (24 June) added:
  - code transfer to a held-out task (MOD 1);
  - a budget sweep at N = 6 and 12 (MOD 2);
  - a lower-confidence-bound promotion gate (MOD 3).
- **Tier follow-ups T1.1–T3.7** (25–26 June): gate check, a new UC4-shaped family, fair code transfer, UC9 at
  n = 5, numeric landscape, tool-call count, solver + critic.

### Outer-optimizer comparisons (1–4 July)

UC4 and UC14 were re-run with different outer optimizers, all at N = 24 with seed 0:
- `OptoPrimeV2`;
- `OptoPrimeMultiV2` with multi-expert or multi-LLM rolling generation;
- `PrioritySearchMulti`.

Run directories: `uc4_*`, `uc14_*`.

---

## EXP00-E — "Experiment 0" (`experiments/recursive_opt/multiobjective_reasoning/`): pre-registered GSM8K engine comparison (23–31 August 2026)

**Question:** on a two-stage reasoning program, does Trace optimization or GEPA improve a fixed baseline on
accuracy and token cost, and does Trace's validation gate matter?

**Pre-registration:** `manifests/preregistration_frozen.json` (`experiment-0-v2`, frozen 2026-08-24 13:39 UTC).

- **Program:** two trainable instructions, an analysis stage and an answer stage, with the initial texts frozen in
  the manifest.
- **Data:** GSM8K, manifest order. 16 train items, 12 validation items, 24 hold-out items from the test split.
- **Objective:** weighted accuracy (weight 1.0, maximized) and forward-token ratio (weight 0.1, minimized). Hard
  constraint: `invalid_rate ≤ 0`.
- **Model:** `deepseek/deepseek-v4-flash-0731` via OpenRouter.
  - Forward: temperature 0, 384 max tokens, reasoning off.
  - Optimizer: temperature 0, 8,192 max tokens, reasoning effort low.
- **Arms:**
  - A, fixed baseline;
  - B, Trace `OptoPrimeV2`;
  - C, GEPA `optimize_anything`;
  - D, Trace without the validation gate.
- **Matrix:** 5 seeds × budgets {6, 12} × 4 arms = 40 canonical units, all executed through the control-plane
  runtime.
- **Statistics:** paired deltas against A, with 95% CIs. Success criterion frozen in the manifest (quality and
  efficiency).
- **Execution:** `main_experiment.py`. Stop and amendment decisions are in `reports/prompt18_*.md`. Run data in
  `outputs/recursive_opt/experiment_0/experiment-0-v2/`.

---

## Later re-designs and equivalents

Most EXP00 questions were re-asked later with a corrected instrument, pre-registration, the control plane or a
different implementation. **No EXP00 run was replayed as-is.** The control-plane v2 migration (21–22 August,
[`_shared/control_plane_v2/migration_report.md`](../_shared/control_plane_v2/migration_report.md)) classified the
85 tracked use-case specs as:
- 0 execution-replayable;
- 10 normalized-only;
- 23 missing a dependency (unpinned task, evaluator or provider);
- 46 historical-only;
- 6 local and non-portable.

| EXP00 element | Later equivalent | Same or different |
|---|---|---|
| A/C/D task and score surfaces (Trace-Bench eval-only adapter) | [EXP01](../EXP01/README.md) prompt signal vs noise, [EXP05](../EXP05/README.md) concurrency noise, [EXP10](../EXP10/README.md) knob liveness | **Re-measured the instrument** that A–D relied on: noise floors and whether knobs reach the score |
| A setup search (O1 config: batch size/design, memory policy, trainer) | [EXP10](../EXP10/README.md), [EXP11](../EXP11/README.md) knobs; [EXP21](../EXP21/README.md) development axes; control-plane `fixed`/`trace` engines | Different: liveness checked first; config axes run through declarative specs |
| A-B / UC1 component code (`CodeArtifactLevel`, hard-item validator, BBEH solver) | [EXP04](../EXP04/README.md) packing code search; [EXP06](../EXP06/README.md) code menus; [EXP15–EXP18](../EXP15/README.md) optimizer-program discovery; [EXP23–EXP24](../EXP24/README.md) native coevolution on PRISM; [EXP28](../EXP28/README.md) `VariationSearch` | Same idea (LLM rewrites code against an evaluator), scaled to real benchmarks with fixed and independent-generation controls |
| A-C / UC3 capability under accuracy + cost | **EXP00-E Experiment 0** (section above) (weighted accuracy + token ratio, hard invalid-rate constraint); [`docs/multi_objective_scores.md`](../../../docs/multi_objective_scores.md) | Direct successor: same multi-objective GSM8K question, pre-registered, with GEPA and a no-validation-gate control |
| A-D / UC4 family policy O2 and prior O3 (`FamilyPolicyLevel`, `PriorInductionLevel`) | [EXP02](../EXP02/README.md) corrected UC4; [EXP07](../EXP07/README.md)/[EXP08](../EXP08/README.md) routing-menu optimum and prior amortization; [EXP22-QA](../EXP22/qa/README.md) learned O1 instruction/selector; control-plane golden specs `uc4_positive` / `uc14_negative` (deterministic contracts, `historical_replay=false`) | EXP02 re-scored UC4 on the same task set (−0.006). EXP08 is the first clean prior-transfer test (on a finite menu) |
| A-E declarative spec (`run_spec`, budgets, prior promotion, warm-start) | Control plane v2 ([`_shared/control_plane_v2/`](../_shared/control_plane_v2/README.md), `opto/features/recursive_opt/spec.py`), used by EXP19–EXP21 and by the coevolution engine in EXP23–EXP28 | Re-implemented: v2alpha schema, typed outcomes, provenance and budget guards; zero historical replays certified |
| B Phase 1 trainer choice / D three-way "standard vs recursive" at equal budget | [EXP24](../EXP24/README.md) fixed vs `llm_rewrite` vs Trace at 100 calls; [EXP22-EvoX](../EXP22/evox/README.md) recursive vs fixed; [EXP28](../EXP28/README.md) trainer arms | Same comparison, now with a fixed-policy control, the same evaluator for every arm, and a known optimum |
| B Phase 2 / UC6 trace type (internal/otel/hybrid) | [EXP16](../EXP16/README.md) rich vs compact feedback; [EXP19](../EXP19/README.md) Trace capture | Different: feedback richness tested on code-discovery surfaces |
| B Phase 3 warm priors, Phase 6 skills, UC2 `initial_knowledge` | [EXP18](../EXP18/README.md) archive memory; [EXP19](../EXP19/README.md) S3 recursive preparation; [EXP20](../EXP20/README.md) curriculum | Different surfaces; memory and curriculum remain unresolved |
| B Phase 4 / UC5 / UC9 optimizer-side tools and agentic policies | `opto/features/recursive_opt/capabilities.py` (`AgenticOptimizer`) | **No later experiment.** UC5/UC9 policies were never executed (0 tool calls) |
| B Phase 5 threads | [EXP05](../EXP05/README.md) concurrency-dependent evaluation noise | EXP05 showed parallel evaluation changes the noise (packing SD 0 serial vs 4.41 at 8 workers) |
| B Phase 7 Terminal-Bench 2 | — | Never implemented |
| D UC2 / UC6 / UC11 QASPER prompt and config | [EXP12](../EXP12/README.md) paired QASPER smoke; [EXP13](../EXP13/README.md) found prompt vs noise; [EXP03](../EXP03/README.md) certified GSM8K prompt search | Same surfaces with paired design and certified noise |
| D UC7 graph routing / UC14 code transfer | [EXP07](../EXP07/README.md)–[EXP09](../EXP09/README.md) routing menus, prior and generated routing code (0/22 transfers executable) | Different tasks (routing); transfer still unshown |
| D UC8 / UC10 guarded campaign and promotion policies | `opto/features/recursive_opt/decisions.py` (`GuardedDecisionEvaluator`, `ConfidenceGate`) | Kept as library code; no later experiment |
| D UC12 promoted primitives | Control plane v2 (budget dict, multi-seed, numeric routing, effect contract) | Absorbed into the runtime |
| D UC13 numeric config search | `opto/features/recursive_opt/numeric_optimizers.py`; [EXP10](../EXP10/README.md)/[EXP11](../EXP11/README.md) knobs | Library kept; no live gain was ever shown (flat objective) |
| D use-case notebook itself | `examples/recursive_opt_use_cases.ipynb` is now a 5-cell **control-plane v2 smoke notebook** that normalizes, explains and runs the UC4/UC14 golden specs | Different: offline contract checks, not the historical runs |
| D audit | [EXP01–EXP14](../EXP01/README.md) (probes A–L, `_history/probe_2026/`), assessment of 29–30 August | The corrected measurements that supersede A–D |
| E Experiment 0 (Trace vs GEPA vs fixed, frozen manifests) | Control-plane `gepa_optimize_anything` engine; candidate-trajectory persistence (`6aa9da0418`); pre-registration and frozen-manifest discipline of EXP15–EXP28; [`o1_qa/`](../o1_qa) (imports it) for EXP20–EXP22-QA | Infrastructure reused; the engine comparison was not repeated on GSM8K |
