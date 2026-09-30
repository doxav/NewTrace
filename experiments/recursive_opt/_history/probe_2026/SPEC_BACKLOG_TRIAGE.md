# Unexecuted spec backlog: triage and run queue

Source: `examples/notebook_outputs/recursive_opt_use_cases/use_cases_structured_20260821_225233/`

Generated deterministically — **no LLM calls**. Surfaces come from `measurement.detect_surface` on really-loaded Trace-Bench bundles; menu checks from `measurement.artifact_fits_surface`; sample sizes from `measurement.required_n(sd, delta)`; loadability from the real `spec.validate_spec` / `spec.compile_level` path.

Machine-readable: `spec_backlog_catalogue.json` (90 per-spec records) and `spec_backlog_variants.json` (18 per-variant decision rows).

## 0. Four corrections to the backlog framing

These change the answer, so they come first.

1. **The 90 specs are 18 experiments, not 90.** Directory names end in `_0`..`_4`: 18 distinct variants x 5 seed replicates. Every spec already implies **5 seeds**, and the 5 replicates of a variant are byte-identical apart from `memory_root`. The decision unit is the variant.
2. **Only UC2, UC3, UC4 and UC6 are represented.** The June summary table has no UC3 and no UC6 row, so **10 of 18 variants (50 specs) have no target effect at all** and cannot be placed on a required_n grid.
3. **The menu-collapse check is vacuous on this backlog** (Section 3). All three tasks are `prose` surfaces, and `artifact_fits_surface` returns `None` for every candidate on a prose surface. 0/90 collapse is guaranteed by construction, not measured.
4. **8 of 18 variants are structurally unrunnable** for reasons that no sd value can fix (Section 4). Four cannot even be validated from their stored JSON.

## 1. What the catalogue found

| | |
|---|---|
| spec.json files | 90 |
| distinct variants | 18 |
| seeds per variant | 5 |
| distinct tasks | 3 — `hf:qasper`, `internal:multiobjective_gsm8k`, `hf:drop` |
| tasks that failed to load | **0** (adapter: Trace-Bench, network reachable) |
| distinct starting_artifact menus | **1** (all 70 menu-bearing levels byte-identical, digest `1d7ef3397fef`) |
| total cost as specified | **12,000 LLM calls** |

All three tasks detect identically: `kind=prose`, `calls_llm=True`, unchanged by `inner_steps`. The parameter is `system_prompt` for gsm8k and `meta_instructions:N` for the two `hf:` tasks.

The single shared menu is 5 candidates, one of which is the empty string:

```
''
'Answer directly.'
'Plan step by step, then answer.'
'Plan step by step, then verify the answer before replying.'
'Use the provided context as evidence, reason briefly, then answer exactly.'
```

Every variant that searches `starting_artifact` — across UC2, UC4 and UC6, across three different tasks — searches this same 5-item list. The backlog contains far less independent variation than 90 directories suggests.

## 2. The catalogue, per variant

`llm/seed` = `eval_llm_calls + optimizer_llm_calls`; as-specified cost = `seeds x arms x llm/seed` with **arms = 2** (paired recursive vs standard; the specs do not encode an arm count, so this is an assumption — a three-way run is 1.5x).

| variant | levels | level surface | targets | tasks | task surface | menu | iters | llm/seed | cost |
|---|---|---|---|---|---|---|---|---|---|
| `uc2_drop` | 1 | config | starting_artifact | drop | prose | 5->5 | 2 | 56 | 560 |
| `uc2_gsm8k` | 1 | config | starting_artifact | gsm8k | prose | 5->5 | 2 | 56 | 560 |
| `uc2_gsm8k_warm_knowledge` | 1 | config | starting_artifact,initial_knowledge | gsm8k | prose | 5->5 | 2 | 56 | 560 |
| `uc2_mixed_gsm8k_qasper` | 1 | config | starting_artifact | qasper, gsm8k | prose | 5->5 | 2 | 56 | 560 |
| `uc2_qasper` | 1 | config | starting_artifact | qasper | prose | 5->5 | 2 | 56 | 560 |
| `uc2_qasper_numeric` | 1 | config | batch_design,batch_size | qasper | prose | 5->5 | 2 | 56 | 560 |
| `uc3_decompose` | 1 | capability | — | gsm8k | prose | 0->0 | 2 | 104 | 1040 |
| `uc3_terse` | 1 | capability | — | gsm8k | prose | 0->0 | 2 | 104 | 1040 |
| `uc3_verify` | 1 | capability | — | gsm8k | prose | 0->0 | 2 | 104 | 1040 |
| `uc3_weak` | 1 | capability | — | gsm8k | prose | 0->0 | 2 | 104 | 1040 |
| `uc4_o2_numeric` | 1 | family_policy | batch_design,batch_size | qasper, gsm8k | prose | 0->0 | 2 | 56 | 560 |
| `uc4_o2_policy` | 1 | family_policy | starting_artifact | qasper, gsm8k | prose | 5->5 | 2 | 56 | 560 |
| `uc4_o3_cold` | 2 | family_policy+prior | starting_artifact | qasper, gsm8k | prose | 5/5->5/5 | 2/2 | 56 | 560 |
| `uc4_o3_warm` | 2 | family_policy+prior | starting_artifact | qasper, gsm8k | prose | 5/5->5/5 | 2/2 | 56 | 560 |
| `uc6_trace_hybrid` | 1 | config | starting_artifact | qasper | prose | 5->5 | 2 | 56 | 560 |
| `uc6_trace_internal` | 1 | config | starting_artifact | qasper | prose | 5->5 | 2 | 56 | 560 |
| `uc6_trace_internal_numeric` | 1 | config | batch_design,batch_size | qasper | prose | 0->0 | 2 | 56 | 560 |
| `uc6_trace_otel` | 1 | config | starting_artifact | qasper | prose | 5->5 | 2 | 56 | 560 |

`inner_steps=2` (the "numeric" variants: `uc2_qasper_numeric`, `uc4_o2_numeric`, `uc6_trace_internal_numeric`) turns on inner training, which is what makes their `batch_design`/`batch_size` targets causally live. "numeric" names the optimizer route, not the surface — those surfaces are still prose.

## 3. Menu validity — verified, and vacuous

**Result: 0 of 90 specs collapse.** This independently reproduces the prior finding, and the number is correct. But it is not evidence, and it should not be read as one.

`artifact_fits_surface` opens with:

```python
if not str(text).strip() or surface.kind in ("unknown", "prose"):
    return None
```

Every task in this backlog is `prose`. The guard therefore returns "fits" for every candidate without inspecting it, and the empty-string candidate short-circuits on the first clause regardless of surface. **A collapsed menu on a prose task is undetectable by this check.** Effective size 5/5 is the only answer it can give here.

The repo says the same thing at `tracebench.py:635`:

> `TODO(menu-collapse)`: the surface guard below stops a WRONG-KIND candidate, but not a RANKING-EQUIVALENT one [...] That is invisible here and only shows up as tied scores; see `score_spread()`.

and `score_spread.effective_menu_size` — the check that *would* catch it — is opt-in and was not run (`spec.py:2732`: "this is opt-in, so a run whose menu collapsed still reports a clean null").

What I could check deterministically, and did: across all 70 menu-bearing levels there are **0 exact duplicates and 0 whitespace/case/punctuation-normalised duplicates**. So the menus are not *textually* degenerate. Whether the four non-empty prompts are *ranking*-equivalent on a prose surface is exactly the open question, and answering it needs scored probes — LLM calls, out of scope here.

**No spec collapses. Flagged loudly: no spec in this backlog *could have* collapsed under this check.**

## 4. Structurally unrunnable — 8 variants, 40 specs

Each verdict below is an executed check, not a reading.

| variant | specs | evidence |
|---|---|---|
| `uc3_decompose` | 5 | validate_spec raises TypeError: capability surface requires a callable evaluator; spec.json stores it as the repr string '<function make_multiobjective_evaluator.<locals>.evaluate at 0x...>' |
| `uc3_terse` | 5 | same: evaluator serialised as repr string |
| `uc3_verify` | 5 | same: evaluator serialised as repr string |
| `uc3_weak` | 5 | same: evaluator serialised as repr string |
| `uc4_o3_cold` | 5 | reproduces the June UC4 confound verbatim: level o2_policy scores on [gsm8k, qasper], level o3_prior scores on [qasper] only, so the cross-level delta is the arithmetic identity (qasper-gsm8k)/2, not an effect |
| `uc4_o3_warm` | 5 | same different-task-sets confound as o3_cold (reuse_priors=True) |
| `uc6_trace_hybrid` | 5 | trace_type='hybrid' needs otel/sysmon backends; traces.HAVE_TRACE_IO is False, so _trace_backend_failure returns INVALID_CONFIG_SCORE (-1e9) for every candidate |
| `uc6_trace_otel` | 5 | trace_type='otel' same backend failure (-1e9 for every candidate) |

### 4a. UC3 (20 specs) — the stored JSON does not validate

`capability` levels require a callable `evaluator`. Serialising the spec wrote the function's repr instead:

```
"evaluator": "<function make_multiobjective_evaluator.<locals>.evaluate at 0x71e5d8b16340>"
```

`validate_spec` on all four UC3 variants raises `TypeError: level cap: capability surface requires a callable evaluator`. These specs cannot be re-run from disk at all; the evaluator must be re-supplied in Python. (`compile_level` happens to succeed because `CapabilityArtifact` stores the value without calling it — the failure is merely deferred to scoring time.)

### 4b. UC6 otel + hybrid (10 specs) — the backend is absent

`traces.HAVE_TRACE_IO` is `False` in this environment, so:

```
otel   -> (-1000000000.0, "[real_trace_bench] trace_type='otel' requires graph/telemetry backends, but opto.features.graph/opto.trace.io are not importable.")
hybrid -> (-1000000000.0, ... same ...)
```

Every candidate scores `INVALID_CONFIG_SCORE`. These produce a constant, not a measurement. Installing the telemetry backends would move them to the UC6 group in Section 5.

### 4c. UC4 o3_cold + o3_warm (10 specs) — the June confound, reproduced verbatim

This is the most consequential finding. The stored specs still encode the exact defect the SUPERSEDED banner invalidated:

```
level o2_policy : tasks = ["internal:multiobjective_gsm8k", "hf:qasper"]
level o3_prior  : tasks = ["hf:qasper"]
```

The two levels are scored on **different task sets**, so their difference is the arithmetic identity `(qasper - gsm8k)/2` — which is precisely how `+0.163` was manufactured. Running these as written would reproduce the invalid result, not test it. They need the task sets equalised before they mean anything.

## 5. Variants with no target effect — 2 variants, 10 specs

`uc6_trace_internal` and `uc6_trace_internal_numeric` load and score fine, but the June table has no UC6 row, so there is no effect size to power against. Worse, the contrast UC6 exists to draw — `trace_type` — is declared **not score-plumbed**:

> `trace_type`: `(Effect.TRACE, Effect.FEEDBACK)`, condition="feedback-plumbed (changes optimizer-visible evidence), NOT score-plumbed", notes="a score effect can only appear later, via better update proposals"

With its two comparison partners (otel, hybrid) unrunnable, the surviving pair cannot form the UC6 contrast at all. They are runnable as generic recursive-vs-standard runs on `hf:qasper`; they are not a UC6 experiment.

What 5 seeds could resolve, if you ran them anyway (`resolvable_delta(sd, n)`):

| sd | resolvable at n=5 | at n=20 |
|---|---|---|
| 0.03 | 0.0376 | 0.0188 |
| 0.05 | 0.0626 | 0.0313 |
| 0.1 | 0.1253 | 0.0626 |
| 0.15 | 0.1879 | 0.0940 |
| 0.21 | 0.2631 | 0.1316 |

## 6. The decision table

`required_n(sd, target_effect)`, argument order as in the signature. Cells show required paired seeds; **bold** = already satisfied by the 5 seeds the spec specifies.

| variant | target | prov. | sd=0.03 | sd=0.05 | sd=0.1 | sd=0.15 | sd=0.21 |
|---|---|---|---|---|---|---|---|
| `uc2_drop` | 0.027 | ok | 10 | 27 | 108 | 243 | 475 |
| `uc2_gsm8k` | 0.027 | ok | 10 | 27 | 108 | 243 | 475 |
| `uc2_gsm8k_warm_knowledge` | 0.027 | ok | 10 | 27 | 108 | 243 | 475 |
| `uc2_mixed_gsm8k_qasper` | 0.027 | ok | 10 | 27 | 108 | 243 | 475 |
| `uc2_qasper` | 0.027 | ok | 10 | 27 | 108 | 243 | 475 |
| `uc2_qasper_numeric` | 0.027 | ok | 10 | 27 | 108 | 243 | 475 |
| `uc4_o2_numeric` | 0.163 | **invalid** | **1** | **1** | **3** | 7 | 14 |
| `uc4_o2_policy` | 0.163 | **invalid** | **1** | **1** | **3** | 7 | 14 |
| `uc4_o3_cold` | 0.163 | **invalid** | **1** | **1** | **3** | 7 | 14 |
| `uc4_o3_warm` | 0.163 | **invalid** | **1** | **1** | **3** | 7 | 14 |
| `uc3_decompose` | — | none | — | — | — | — | — |
| `uc3_terse` | — | none | — | — | — | — | — |
| `uc3_verify` | — | none | — | — | — | — | — |
| `uc3_weak` | — | none | — | — | — | — | — |
| `uc6_trace_hybrid` | — | none | — | — | — | — | — |
| `uc6_trace_internal` | — | none | — | — | — | — | — |
| `uc6_trace_internal_numeric` | — | none | — | — | — | — | — |
| `uc6_trace_otel` | — | none | — | — | — | — | — |

Two distinct required_n profiles, because there are only two target effects: `0.027` (UC2) and `0.163` (UC4).

## 7. Ranked by value per call

`value_per_call = target_effect / (required_n(sd, target) x arms x llm_per_seed)` at the mid-grid **sd = 0.1**. Only the 8 variants with a target effect can be ranked; the other 10 have no defined value.

| # | variant | target | req_n | calls to resolve | value/call | bucket |
|---|---|---|---|---|---|---|
| 1 | `uc4_o2_numeric` | 0.163 | 3 | 336 | 4.85e-04 | RUNNABLE NOW |
| 2 | `uc4_o2_policy` | 0.163 | 3 | 336 | 4.85e-04 | RUNNABLE NOW |
| 3 | `uc4_o3_cold` | 0.163 | 3 | 336 | 4.85e-04 | UNRUNNABLE |
| 4 | `uc4_o3_warm` | 0.163 | 3 | 336 | 4.85e-04 | UNRUNNABLE |
| 5 | `uc2_drop` | 0.027 | 108 | 12096 | 2.23e-06 | NOT WORTH RUNNING |
| 6 | `uc2_gsm8k` | 0.027 | 108 | 12096 | 2.23e-06 | NOT WORTH RUNNING |
| 7 | `uc2_gsm8k_warm_knowledge` | 0.027 | 108 | 12096 | 2.23e-06 | NOT WORTH RUNNING |
| 8 | `uc2_mixed_gsm8k_qasper` | 0.027 | 108 | 12096 | 2.23e-06 | NOT WORTH RUNNING |
| 9 | `uc2_qasper` | 0.027 | 108 | 12096 | 2.23e-06 | NOT WORTH RUNNING |
| 10 | `uc2_qasper_numeric` | 0.027 | 108 | 12096 | 2.23e-06 | NOT WORTH RUNNING |
| 11 | `uc3_decompose` | — | — | — | undefined | UNRUNNABLE |
| 12 | `uc3_terse` | — | — | — | undefined | UNRUNNABLE |
| 13 | `uc3_verify` | — | — | — | undefined | UNRUNNABLE |
| 14 | `uc3_weak` | — | — | — | undefined | UNRUNNABLE |
| 15 | `uc6_trace_hybrid` | — | — | — | undefined | UNRUNNABLE |
| 16 | `uc6_trace_internal` | — | — | — | undefined | NO TARGET EFFECT |
| 17 | `uc6_trace_internal_numeric` | — | — | — | undefined | NO TARGET EFFECT |
| 18 | `uc6_trace_otel` | — | — | — | undefined | UNRUNNABLE |

**Read this ranking with care.** The four UC4 rows sit on top only because `0.163` is large — and `0.163` is the invalidated arithmetic identity. Two of the four (`o3_cold`, `o3_warm`) are the very design that produced it and are unrunnable. The honest reading: *the only specs that look cheap to resolve look cheap because they are powered against a number that was never an effect.*

The two genuinely rankable, structurally sound UC4 rows are `o2_policy` and `o2_numeric` — single-level, no cross-level task-set mismatch. They still inherit `0.163` as their target, which nothing justifies. Powered against UC2's `0.027` instead they would need 108 seeds at sd=0.10, i.e. they join **NOT WORTH RUNNING**.

## 8. Buckets and totals, by sd

For the two runnable buckets, cost is the calls needed **to reach required_n**, so it grows with sd. UNRUNNABLE and NO-TARGET rows carry their as-specified cost as a sunk figure, constant across sd (6,400 and 1,120 calls respectively).

"stored specs" counts the `spec.json` files on disk (5 per variant). It is not the number of seeds you would run: at sd=0.03 the two RUNNABLE-NOW variants need only **n=1** each, which is why 10 stored specs cost 224 calls, not 1,120.

**sd = 0.03**

| bucket | variants | stored specs | LLM calls |
|---|---|---|---|
| RUNNABLE NOW | 2 | 10 | 224 |
| RUNNABLE, MORE SEEDS | 6 | 30 | 6,720 |
| NO TARGET EFFECT | 2 | 10 | 1,120 |
| UNRUNNABLE | 8 | 40 | 6,400 |

**sd = 0.05**

| bucket | variants | stored specs | LLM calls |
|---|---|---|---|
| RUNNABLE NOW | 2 | 10 | 224 |
| NOT WORTH RUNNING | 6 | 30 | 18,144 |
| NO TARGET EFFECT | 2 | 10 | 1,120 |
| UNRUNNABLE | 8 | 40 | 6,400 |

**sd = 0.1**

| bucket | variants | stored specs | LLM calls |
|---|---|---|---|
| RUNNABLE NOW | 2 | 10 | 672 |
| NOT WORTH RUNNING | 6 | 30 | 72,576 |
| NO TARGET EFFECT | 2 | 10 | 1,120 |
| UNRUNNABLE | 8 | 40 | 6,400 |

**sd = 0.15**

| bucket | variants | stored specs | LLM calls |
|---|---|---|---|
| RUNNABLE, MORE SEEDS | 2 | 10 | 1,568 |
| NOT WORTH RUNNING | 6 | 30 | 163,296 |
| NO TARGET EFFECT | 2 | 10 | 1,120 |
| UNRUNNABLE | 8 | 40 | 6,400 |

**sd = 0.21**

| bucket | variants | stored specs | LLM calls |
|---|---|---|---|
| RUNNABLE, MORE SEEDS | 2 | 10 | 3,136 |
| NOT WORTH RUNNING | 6 | 30 | 319,200 |
| NO TARGET EFFECT | 2 | 10 | 1,120 |
| UNRUNNABLE | 8 | 40 | 6,400 |

## 9. If sd turns out to be X, run these

The two agents are measuring paired seed-delta sd on `hf:qasper` and `internal:multiobjective_gsm8k` — the two tasks carrying 16 of the 18 variants. When their number lands, read the matching line.

### sd = 0.03

**Run now (2 variants, 10 specs, 224 calls):** `uc4_o2_numeric` (n=1), `uc4_o2_policy` (n=1)

**Run with more seeds (6 variants, 6,720 calls):** `uc2_drop` (n=10, +5 seeds, 1,120 calls), `uc2_gsm8k` (n=10, +5 seeds, 1,120 calls), `uc2_gsm8k_warm_knowledge` (n=10, +5 seeds, 1,120 calls), `uc2_mixed_gsm8k_qasper` (n=10, +5 seeds, 1,120 calls), `uc2_qasper` (n=10, +5 seeds, 1,120 calls), `uc2_qasper_numeric` (n=10, +5 seeds, 1,120 calls)

Best case in the grid. The six UC2 variants become reachable at n=10 — double the seeds, 6,720 calls total. Still nothing is runnable at the 5 seeds already specified except the two UC4 `o2_*` rows, whose target is invalid.

### sd = 0.05

**Run now (2 variants, 10 specs, 224 calls):** `uc4_o2_numeric` (n=1), `uc4_o2_policy` (n=1)

**Run with more seeds: none.**

This is the June UC2 seed-delta sd. At this value **every UC2 variant is already out of reach** (n=27 > 20). The backlog is done.

### sd = 0.1

**Run now (2 variants, 10 specs, 672 calls):** `uc4_o2_numeric` (n=3), `uc4_o2_policy` (n=3)

**Run with more seeds: none.**

UC2 needs 108 seeds. Only the invalid-target UC4 rows survive.

### sd = 0.15

**Run now: none.**

**Run with more seeds (2 variants, 1,568 calls):** `uc4_o2_numeric` (n=7, +2 seeds, 784 calls), `uc4_o2_policy` (n=7, +2 seeds, 784 calls)

UC2 needs 243-475 seeds. Nothing in the backlog resolves anything real.

### sd = 0.21

**Run now: none.**

**Run with more seeds (2 variants, 3,136 calls):** `uc4_o2_numeric` (n=14, +9 seeds, 1,568 calls), `uc4_o2_policy` (n=14, +9 seeds, 1,568 calls)

UC2 needs 243-475 seeds. Nothing in the backlog resolves anything real.

## 10. The sd threshold where the backlog stops being worth running

For the only defensible target in the backlog — UC2's `0.027`, at 5 seeds, arms=2:

| condition | threshold |
|---|---|
| runnable at the specified 5 seeds | **sd <= 0.0215** |
| runnable at the 20-seed ceiling | **sd <= 0.0431** |

> **The backlog stops being worth running at sd > 0.043.**

The June UC2 seed-delta sd was **0.050**. If the two agents come back anywhere near that, the entire UC2 line — the only part of this backlog powered against a legitimate target — is unresolvable at any sane budget, and the answer for all 90 specs is: do not run them as written.

For the sd grid to rescue UC2 it would have to come back below `0.0431`, i.e. the instrument fixes since June (bounded eval sampling, the newline-truncation fix) must have cut paired seed noise by **more than 14%** relative to the June UC2 figure. Below `0.0215` UC2 runs as specified with no extra seeds.

## 11. Recommendation

Do not queue this backlog by sd alone. Ordered by what actually unblocks work:

1. **Fix, do not run, the 4 UC4 `o3_*` and UC3 specs.** Equalise `o2_policy` / `o3_prior` task sets; re-emit UC3 specs with a resolvable evaluator reference instead of a function repr. These are serialisation and design bugs, and no sd value touches them.
2. **Get a target effect for UC3 and UC6, or drop them.** 50 of 90 specs are powered against nothing.
3. **Run `score_spread` with `effective_menu_size` on the one shared 5-item menu before spending anything else.** All 70 menu-bearing levels share one menu, so a single cheap probe settles menu adequacy for the entire backlog — and the surface guard provably cannot settle it.
4. **Treat sd = 0.043 as the go/no-go line for UC2.**

