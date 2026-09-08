# recursive_opt — research log

**The single entry point.** Status board first, evidence second. Absorbs the former
`EXECUTION_PLAN.md` and `RESULTS_INDEX.md`. Long-form evidence stays in
`recursive_opt_assessment.md`, referenced here by section (§n) — it is the audit trail,
not the status.

*Last updated 2026-09-08.*

---

## 1. Hypothesis ledger

The global view: what is settled, what is not, and what has never been tried.

| status | meaning |
|---|---|
| **SUPPORTED** | measured, effect outside the noise floor of its own surface |
| **REFUTED** | measured, and the effect is absent or reversed |
| **VOID** | measured, but the instrument could not have detected an effect — carries no information |
| **UNTESTED** | never run |
| **BLOCKED** | cannot be run in the current task pool; the blocker is named |
| **INCONCLUSIVE** | measured, but the registered uncertainty interval does not resolve the contrast |

| # | hypothesis | status | best evidence | where |
|---|---|---|---|---|
| **H1** | Recursion raises the **ceiling** (max performance) vs standard | **REFUTED** | Both arms reach q=1.000 on the routing family (noise floor 0.0). No post-cutoff experiment shows a positive ceiling delta on any task. | §20.2, EXP-08 |
| **H2** | Recursion reaches the **same ceiling with less search** (W2 amortisation) | **SUPPORTED**, conditional | q=1.000 at b=1 vs b=9; break-even **K\*=2.25** (Q=1.0), 6.0 (Q=0.95). *Only* vs an uninformed baseline and hand-written portable candidates. | §20.2, EXP-08 |
| **H3** | A learned **artifact** (code) transfers across a task family | **REFUTED** | **0 of 22** LLM-generated heuristics execute on a sibling task; all score −1e6. Signature-bound by construction. | §20.3, EXP-09 |
| **H4** | A learned **config knob** transfers across a family | **BLOCKED** | 43/47 tasks expose one training example → batch/ordering knobs structurally inert. The one multi-example deterministic venue is saturated (0.99998) with a replicate floor (4.2e-5) above every knob effect. | §21, EXP-10/11 |
| **H5** | For a family of problems, **one config is best on every member** | **SUPPORTED** | `nearest` is argmax on **4/4** routing tasks; Spearman ρ 0.51–1.00, all six pairs positive. The precondition for any transfer. | §20.2, EXP-07 |
| **H6** | Recursion wins by **variance reduction** (W1) on noisy surfaces | **UNTESTED** | Requires a noisy surface and n sized to its floor. Never run. W1 is *structurally impossible* on the zero-noise surfaces used so far. | §18 |
| **H7** | The flagship **UC4 +0.163** is a real effect | **REFUTED** | Arithmetic identity `(qasper − gsm8k)/2`; arms scored on different task sets; artifacts byte-identical. Corrected run gives **−0.0060**. | §5.2, EXP-02 |
| **H8** | Standard optimisation improves anything on **prose** tasks | **UNRESOLVED** | probe F deltas wildly heterogeneous [0.53, 0.009, 0.035, 0.133]; qasper S/N 0.74, gsm8k 0.96 — changing the prompt moves the score less than re-running it. EXP-13 in flight. | §13, EXP-13 |
| **H9** | The config→score surface is non-flat (optimisation is *possible*) | **SUPPORTED** on code; **REFUTED** on prose | Code: ranges **2908.2** and **390.0** at noise 0. Prose: signal below noise (S/N < 1). | §19.2, EXP-06 |
| **H15-A** | Selected recursive optimizer-code search improves held-out anytime regret over its unchanged seed | **SUPPORTED within EXP-15**, positive signal | A2−A0 = **−0.022899**, paired bootstrap 95% [−0.042456, −0.003341], n=5; lower is better | §11 below; assessment §25 |
| **H15-B** | Iterative training feedback improves held-out anytime regret over equal-proposal independent generation | **INCONCLUSIVE** | A2−A1 = **−0.005068**, paired bootstrap 95% [−0.037728, +0.024551], n=5 | §11 below; assessment §25 |

Historical H1/H2/H3/H5 conclusions above remain unchanged. EXP-15 establishes a new
portable optimizer-program venue and a positive signal over the starting policy;
it does not establish an advantage over independent LLM generation. The historical
signature-bound artifact failure and inert configuration surfaces still stand.

---

## 2. Goals

| goal | state | blocker / next |
|---|---|---|
| **G-A** Decide whether recursive_opt is a viable optimisation layer | **answered, conditionally** | EXP-15 improves over its seed; added value over independent generation remains inconclusive. |
| **G-B** Make the instrument trustworthy | **largely done** | automatic menu evidence now recorded; behavioral certainty requires evaluator signatures (Phase 0, §23). |
| **G-C** Find a venue where recursion *can* win | **portable venue established** | EXP-15 supplies a common artifact/evaluator and a valid comparison; a feedback advantage is not yet established. |
| **G-D** Establish the paired-seed noise floor | **historical noisy-prose work unresolved** | EXP-12/13 remain separate; they do not gate EXP-15's deterministic numerical benchmark. |
| **G-E** Clear or retire the spec backlog | **triaged, not run** | 8 of 18 variants need *fixing*, not running. |

---

## 3. Experiment registry

One row per experiment. `n` is usable paired observations, not runs attempted.

| id | date | question | task | n | result | verdict |
|---|---|---|---|---|---|---|
| EXP-01 | 08-29 | Is there signal to optimise? | gsm8k, qasper | 3 rep | S/N **0.96** / **0.74** | H9 refuted on prose |
| EXP-02 | 08-29 | UC4 with both arms on the same holdout | qasper | **1 pair** | Δ **−0.0060**, promotion refused | H7 refuted; draw |
| EXP-03 | 08-29 | Does optimisation beat the initial artifact? | gsm8k | 4 | deltas [0.53, .009, .035, .133] | H8 unresolved |
| EXP-04 | 08-29 | Deterministic optimisation (probe K) | bin packing | 3 | +4.8 → **retracted**: "optimised" artifact is *empty* | void |
| EXP-05 | 08-30 | Noise vs concurrency | 8 tasks | — | sd 0.0 seq → **3.15–4.41 @ c=8** | forced 2 retractions |
| EXP-06 | 08-31 | Is the menu the instrument? (probe R) | packing, admissible_set | — | ranges **2908.2**, **390.0** | H9 supported on code |
| EXP-07 | 09-01 | Does a family share an optimum? | routing ×4 | 9 cand | `nearest` **4/4**, ρ 0.51–1.00 | **H5 supported** |
| EXP-08 | 09-01 | W2 amortisation sweep | routing ×3 | exact | **K\*=2.25** vs uninformed | **H2 supported** |
| EXP-09 | 09-01 | Does LLM-written code transfer? | routing ×3 | 22 | **0/22 execute** | **H3 refuted** |
| EXP-10 | 09-01 | Are config knobs live? | 13 llm4ad | 13 | 0/13 moved — single-example | H4 blocked |
| EXP-11 | 09-01 | Knobs on a multi-example venue | bbeh | 5 rep | all inside replicate floor | H4 blocked |
| EXP-12 | 09-02 | Paired seed-delta sd | qasper | **2 pairs** | sd 0.254 @ c=2 — *not yet usable* | in flight |
| EXP-13 | 09-02 | Was seed 101's 0.5455 a find or noise? | qasper | 6+6 | *running* — see note | pending |
| EXP-14 | 09-02 | Backlog triage | 90 specs | — | 18 variants; **8 unrunnable** | see §5 |
| EXP-15 | 09-08 | Does training-informed optimizer code search improve held-out regret over seed and independent search? | Sphere / Quadratic / Rosenbrock, d=2/4 | **5 complete pairs** | Mean AUC A0/A1/A2 **0.139579 / 0.121748 / 0.116680**; all 80 slots retained | H15-A positive signal; H15-B inconclusive |

**EXP-13 note — environment, not design.** First attempt left evaluation UNBOUNDED: one item
took 514 s and the full design projected to 10+ hours. That is the same unbounded-sampling defect
that cost 14.6x resolution earlier. Rebuilt to match probe A exactly (`max_examples=2`,
`inner_steps=0`, per-run `SIGALRM`) so the numbers are comparable to probe A's empty-prompt figures
— and the very first bounded evaluation still exceeded probe A's own 150 s timeout. **The provider
endpoint is materially slower today than on 2026-08-29**, with repeated LiteLLM errors. Re-running
at n=6 per condition with a 480 s bound. Any qasper timing measured now is not comparable to
probe A's.

### Retractions
| claim | why it fell |
|---|---|
| UC4 **+0.163** | arithmetic identity across different task sets |
| probe F **+0.217** | inside a noise floor measured with too few repeats |
| probe K **+4.8** | concurrency noise; and the "optimised" artifact was **empty** |
| Iteration 3 (3 rows) | collapsed menu; `artifacts_differ: false` |
| "0 of 106 specs collapse" | **vacuous** — the type check accepts everything on prose |

---

## 4. Defect register

| id | defect | status |
|---|---|---|
| D1 | invalid sentinel escapes as a score | fixed; **recurred** in a 3rd path (`trace_type` → −1.0) |
| D3 | arms compared on different task sets | fixed (comparability gate) |
| D8 | line-count gate caused compression, not simplification | removed |
| D14 | control plane never ran a recursive spec | fixed |
| D16/D17 | config encode/decode truncation; registry pollution | fixed |
| D18 | certification not menu-conditional | open — demoted; see menu-collapse |
| **MC-a** | prose overwrote code/numeric params → menu effective size 1 | **fixed** (`artifact_fits_surface`) |
| **MC-b** | ranking-equivalent candidates collapse a menu invisibly | **resolved on signature-equipped evaluators; narrowed on legacy** — actual behavior evidence, metric-only/unknown scope explicit (§23) |
| **MC-c** | type audit is **vacuous on prose** — accepts everything unread | **fixed** (`menu_check_kind`) |
| **MC-d** | `effective_menu_size` is opt-in, not recorded per run | **fixed for future runs** — automatic actual-evaluation evidence in canonical and legacy public results (§23) |
| EX-1 | example A reported a tie-break as a learned result | fixed (`NO WINNER`) |
| EX-2 | `list_tasks` fabricated a task list when Trace-Bench absent | fixed (raises) |

---

## 5. Instrument & task register

| task | surface | LLM? | noise floor | usable for |
|---|---|---|---|---|
| `llm4ad routing ×4` | code | no | **0.0** seq | **W2**, shared-optimum |
| `online_bin_packing` | code | no | 0.0 seq / **3.15–4.41 @ c=8** | W2, with concurrency declared |
| `admissible_set` | code | no | 0.0 | W2 |
| `internal:code_param` | code | no | saturated at 1.0 | nothing |
| `hf:qasper` | prose | yes | within-prompt **0.0391**; paired-seed *unknown* | W1 candidate |
| `internal:multiobjective_gsm8k` | prose | yes | 0.0327; S/N 0.96 | W1 candidate |
| `internal:multiobjective_bbeh` | prose | no | 4.2e-5, saturated | nothing |
| 43 of 47 tasks | — | — | — | **single training example** → batch knobs inert |

**Spec backlog (EXP-14):** 90 dirs = **18 variants × 5 seed replicates**. 8 variants structurally
unrunnable: UC3 (evaluator serialised as a repr string), UC6 otel/hybrid (`HAVE_TRACE_IO` false →
−1e9), UC4 o3_cold/o3_warm (**still encode the June confound verbatim**). All 70 menu-bearing
levels share one byte-identical 5-item menu. Worth running only if paired-seed **sd ≤ 0.043**.

---

## 6. Standing protocol

Rules, each earned by a retraction. Full critic panel: `CRITIC_PANEL.md`.

1. Both arms on the **same level and same task set** (else you measure an identity).
2. Noise floor measured **at the concurrency you run at**.
3. **In-run replicate control** — re-score the identical artifact n≥10; publish its range beside the effect.
4. **`artifacts_differ` must be true**, or the delta is noise by definition.
5. **Headroom first** — verify effective menu size > 1. On prose use `score_spread`; a type audit there is vacuous (`menu_check_kind`).
6. **Declare which budget is equalised** — total compute or per-task search. You cannot hold both.
7. **State the win condition** — W1 needs noise; W2 works at zero noise. Testing W1 on a deterministic surface is a category error.
8. **No iteration ends without a commit.**

---

## 7. How to update this file

Per experiment: add one **EXP-nn** row (§3), update any hypothesis it touches (§1), add defects
(§4), and record a retraction if it invalidates prior work. Per goal: update §2. Keep the long-form
narrative in `recursive_opt_assessment.md` and reference it by §.

---

## 8. Next, by priority

1. **Finish EXP-12/13** — the paired-seed sd gates every remaining decision. Run arms sequentially or declare shared concurrency; the last attempt confounded two agents on one endpoint.
2. **Fix, don't run, the 8 broken variants** — serialisation and design bugs no sd value touches.
3. **Use MC-b / MC-d evidence** — future runs record actual evaluations; require behavior signatures before claiming behavioral menu headroom.
4. **Build a portable-artifact family** (§21.4 property 5) — the only route to testing H3 honestly.
5. **Test H6/W1** — the one hypothesis never explored, on a prose surface with n sized to its floor.

## 9. Optimizer discovery Phase 0 — 2026-09-08

The portable propose(history, bounds, seed) interface and deterministic evaluator are
ready. Future runs retain actual candidate behavior/evaluation evidence, with explicit
invalidity and unknown equivalence instead of silently certifying a collapsed menu.
This changes instrument validity, not any historical performance conclusion; see
[assessment §23](recursive_opt_assessment.md#23--optimizer-discovery-phase-0-instrument-and-interface).

Frozen OpenRouter DeepSeek calibration: **0/3** open-ended generation responses had
parseable code (all reached 3000 completion tokens). All failures are retained. A
separately preregistered interface-only midpoint request succeeded and executed at
seeds 0/1/2, budget 8 each, value 2.125 throughout. This demonstrates the interface,
not optimizer discovery or improvement. **Phase-1 search execution: NO-GO** under
the failed generation protocol; interface/protocol development can proceed.

EXP-12/13 remain historical/unresolved: model-name agreement does not establish
request comparability; the required ephemeral EXP-13 prompt is absent. Their raw
records and H1/H2/H3/H5 conclusions are unchanged. Full evidence:
[PHASE0_REPORT.md](optimizer_discovery/PHASE0_REPORT.md).

## 10. Optimizer generation readiness — separate calibration, 2026-09-08

After the user authorized prospective configuration calibration, a separately
preregistered 3,000-token control again returned no code: all 3,000 reported output
tokens were reasoning. An 8,000-token/default-reasoning pilot yielded 2/3 valid
history-responsive optimizers and failed its fixed gate. The 8,000-token/low-effort
pilot yielded 3/3 and was selected before fresh confirmation seeds 101–110.

Confirmation: **10/10 valid and history-responsive**, **effective menu size 7** on
actual common fixture trajectories. Each program completed 3 × 8 objective calls.
**Phase 0 generation readiness: GREEN** for the selected exact-model configuration;
proceed to Phase-1 preregistration. This does not establish optimization quality or
population reliability. All original failures and both new invalid requests remain.
The provider mix varied; observed readiness is not an isolated causal estimate of
reasoning effort. No historical hypothesis conclusion changed.

See [current Phase-0 report](optimizer_discovery/PHASE0_REPORT.md),
[selected settings](optimizer_discovery/selected_generation_config.json), and
[assessment §24](recursive_opt_assessment.md#24--optimizer-generation-readiness-calibration).


## 11. EXP-15 — first controlled optimizer-program discovery experiment

Preregistered H15-A tests selected A2 deployment against the unchanged seed. H15-B,
the central contrast, tests A2 against equal-response-budget independent code search.
Both were evaluated under the frozen confirmatory analysis. The Phase-1 pilot used
separate instances and outer seed 701; two of four completed responses produced
eligible programs, and validation retained the seed in both arms. All failures and
the transport retry remain recorded. Pilot results do not enter confirmation.

The frozen protocol (`0643691b`) uses five paired outer seeds, eight completed
DeepSeek responses per generative arm, 32 objective calls per trajectory and balanced
6/6/12 train/validation/holdout splits. All selections and the validation-selected
representative were frozen before any holdout evaluation. All five paired seeds and
80 response slots are retained. Invalid candidates and common deployment fallback
are explicit; no holdout fallback was needed.
See [preregistration](optimizer_discovery/PREREG_EXP15.md) and
[pilot evidence](optimizer_discovery/exp15/PILOT_REPORT.md).

Mean normalized anytime regret, lower better: A0 **0.139579**, A1 **0.121748**,
A2 **0.116680**. H15-A's paired delta is **−0.022899**, 95% bootstrap interval
**[−0.042456, −0.003341]**, a positive signal under the registered rule.
H15-B's delta is **−0.005068**, interval **[−0.037728, +0.024551]**, inconclusive.
A2 loses to A1 in two of five pairs. A1 has better sample mean final regret and
target attainment. Five outer seeds do not justify strong superiority claims.

A1 has 9/40 ineligible candidate slots; A2 has 12/40, including one program rejected
for nondeterministic execution. A2 retains the seed in two replications. All failures
remain, including the single transport timeout. There were 81 attempts for 80
completed responses, 573,191 reported tokens and USD 0.086824 reported cost, excluding
unknown billing on the timeout. Reporting-only latency/path-label corrections and
lossless archive packaging changed no scientific values or decisions. The frozen
protocol and complete aggregate pass the integrity audit; 772 offline tests pass.

The validation-selected representative is outer41/slot5, a Halton/incumbent/quadratic
hybrid, with exact source and lineage preserved. The result and portable evaluator
are ready for discussion with Patrick; the brief has not been sent. See
[full report](optimizer_discovery/EXP15_REPORT.md),
[machine-readable results](optimizer_discovery/exp15_results.json),
[Patrick brief](optimizer_discovery/PATRICK_BRIEF.md) and assessment §25.

This creates a portable-artifact venue. It does not retroactively overturn H3's
signature-bound historical setup, measure amortization, or identify a recursion-
depth effect. Historical H1/H2/H3/H5 conclusions and prior retractions are preserved.
