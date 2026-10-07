# EXP00 — protocols of the pre-numbered campaigns (June–August 2026)

These campaigns ran before the numbered EXP series, from notebooks in `examples/` and from the
`multiobjective_reasoning` package. None was pre-registered as an EXP. The protocols below are
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

## EXP00-A — `examples/recursive_opt_demo.ipynb`: the four meta-optimization types (7–11 June 2026)

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

## EXP00-B — `examples/recursive_opt_phases.ipynb`: the Phase 0→7 campaign (10–13 June 2026)

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

## EXP00-C — `examples/recursive_opt_phases_V2.ipynb`: level-by-level evidence notebook (12 June; re-executed 30 September 2026)

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

## EXP00-D — `examples/recursive_opt_use_cases.ipynb`: use-case suite UC1–UC14 (15 June – 5 July 2026)

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
