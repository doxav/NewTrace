# EXP00 — results of the pre-numbered campaigns

Numbers are copied from the executed notebook outputs, the persisted run summaries and the cited reports. The
**validity** column applies the later audit (assessment of 29–30 August 2026, now
`_history/reviews/recursive_opt_assessment.history_20260929.md`, and EXP01–EXP14).

**Instrument caveat for A–D.** The evaluation harness of June–July 2026 had:
- unbounded sampling (no temperature, `max_tokens` or timeout);
- objectives that blended a rare error term with a token count;
- a config surface that truncated multi-line artifacts at the first newline;
- no check that compared arms were scored on the same task set.

Its resolution at the budgets used was about **0.24 at n = 5** (`examples/notebook_outputs/recursive_opt_use_cases/SUPERSEDED.md`).
Score differences below that are unresolved by construction.

## Summary

| Sub-study | Dates | Headline as reported then | Valid reading now |
|---|---|---|---|
| A demo | 7–11 Jun | B rewrites code 0.80 → 1.00; A/C/D "plumbing works" | **Mechanics only.** A, D and E are flat (every setup scores the same); C is saturated (accuracy 1.0). B's gain is real but on a 12-item toy validator whose hard items are told in the feedback |
| B phases | 10–13 Jun | ADOPT warm priors (+0.013), ADOPT skills (+0.029); REJECT tools | **Nothing adoptable.** Warm priors reverse to −0.026 at 8 examples; the skill and thread results are inconsistent between cell output and decision board; all deltas are below resolution |
| C phases V2 | 12 Jun (re-run 30 Sep) | O0/O1/O2 "positive: yes" | **Unsupported.** Its own outputs show no O1 gain (non-baseline setups return the −1e9 invalid sentinel), O2/O3 at 0.0, and only hand-written (not optimizer-found) code gains |
| D use cases | 15 Jun – 5 Jul | UC4 O2→O3 prior +0.163 "promote, robust"; same-level warm code restart not robust | **UC4 invalid** (arithmetic identity of two task sets; corrected −0.018, paired −0.006, EXP02). Other deltas unresolved. Valid: UC1 saturated, UC13 flat, tools were policy-as-text only |
| E Experiment 0 | 23–31 Aug | — | **Neither engine met its pre-registered success criterion.** Trace −13% tokens and GEPA −26% tokens at no significant accuracy change; no validation gate gives the lowest accuracy; stopped for missing candidate provenance |

---

## EXP00-A — demo notebook (executed version `58a5c0f8ef`)

| Arm | Before → after | Evidence |
|---|---|---|
| A — setup (offline, 3 configs) | every config scores −2091.80 on `online_bin_packing_local` | flat surface. The printed "0.509 → −2091.8, regressed" compares two different scales |
| A — live (`internal:multi_param`) | test score −1.0 for every candidate | flat |
| B — `batch_design` code, offline hand-written | 0.800 → **1.000** | validator: picks all 4 hard items (`idx % 3 == 0`) |
| B — live LLM (`PrioritySearch` + `OptoPrimeV2`, 4 iterations) | 0.800 → **1.000** | the LLM wrote a hard-first, stride-diverse selector (diff saved in the notebook) |
| B — robustness | 1.000 ± 0.000 (n = 3) | deterministic validator |
| C — capability (4 prompts) | accuracy 1.00 for all; cost 0.03 / 0.05 / 0.18 / 0.18 | saturated: "Answer directly." wins on cost alone; live best capability score 1.00 |
| D — family/prior (relative_delta) | normalized 0.000 for all 6 family × setup cells; O2 0.000 → 0.000; O3 held-out 0.000 | flat: identical raw scores per family |
| D — live | best policy 0.250 at iteration 0, then 0.000 ± 0.25 noise over 63 iterations; prior 0.000 | noise, no learning |
| E — declarative spec | o1/o2/o3 scores 0.000; warm-start reused the prior | plumbing works; no signal |
| Unit tests | 70 passed | — |

**Lesson the notebook itself recorded:** "A/D/E may legitimately stay near 0.0 … that is a score-surface
diagnosis." The only climbable surface was a deterministic toy validator.

## EXP00-B — Phase 0→7 campaign (executed version `611d31eecf`)

| Phase | Result (cell output) | Decision board (saved `phase*.json`) | Decision then | Valid reading |
|---|---|---|---|---|
| 0 gates | 78 tests passed; adapter real. Spread probe: the 2 `llm4ad` tasks and `multi_param` are "CATASTROPHIC" (−1e6 sentinels); `multiobjective_gsm8k` has valid spread 0.049 | 70 tests | — | only GSM8K usable; spread 0.049 is tiny |
| 1 trainer (code) | Minibatch 0.800 ± 0.000, PrioritySearch 0.800 ± 0.000 (n = 3), although the log shows test score 1.0 at step 7 | both 1.000 ± 0.000 | PARK; tie broken by wall time | measurement inconsistent; a saturating toy surface either way |
| 2 trace type | — | internal +0.014 ± 0.026, otel +0.014 ± 0.021, hybrid +0.017 ± 0.007 | — | tie |
| 2 confirm (8 examples) | hybrid Δ −0.000 | — | REJECT | tie |
| 3 warm priors | cold 0.036 ± 0.007, warm 0.049 ± 0.005, Δ +0.013 | same | ADOPT, on 1 family (the rule required 2) | **reversed at confirmation:** warm 0.013 ± 0.009, Δ **−0.026** at 8 examples |
| 4 optimizer tools | plain +0.012 ± 0.011, tools −0.003 ± 0.006, Δ −0.014 | same | REJECT | tools did not help |
| 5 threads | 1 thread 26.0 s, score 0.8; 8 threads 26.2 s, score 1.0 | 1 thread 11.7 s, score 1.0; 8 threads 0.018 s, score 0.8 | ADOPT if faster at tied score | **inconsistent records**; no conclusion |
| 6 distilled skill | skill 0.007 ± 0.005, "Δ +0.341" | plain 0.011 ± 0.005, skill 0.040 ± 0.011 | ADOPT | **inconsistent records**; confirmation cell not executed |
| 7 Terminal-Bench 2 | template only | — | — | not run |

**Causal-effect contract, the valid output of this campaign:**
- At `inner_steps=0`, only `starting_artifact`, `initial_knowledge` and `trace_type` (feedback only) reach the
  score.
- `trainer`, `optimizer`, `batch_size` and `num_threads` need `inner_steps > 0`.
- `batch_design`, `memory_policy` and `credit_horizon` were inactive in both modes.

Most knobs searched in A/D and Phases 1–6 therefore could not move the score. A fixed O0 bug was also shown:
`agent_fn` now injects the artifact (ALPHA vs BETA outputs differ).

## EXP00-C — phases V2 notebook (outputs from the 30 September re-execution)

| Example | Result |
|---|---|
| A setup | Baseline −2091.8 (bin packing) and −1.0 (`multi_param`). The 3 non-baseline configs score **−1e9** (invalid sentinel); "no positive delta" |
| B code (hand-written) | `batch_design` 0.800 → 1.000 (Δ +0.200); `trace_summarizer` 0.823 → 0.959 (Δ +0.137) |
| C capability | accuracy 1.0 for all 4 prompts; cost 0.027–0.185; best = cheapest = "Answer directly." |
| D O2/O3 | O2 policies 0.0 / 0.0; O3 priors 0.0 / 0.0 (held-out transfer 0.0) |

The notebook's claims table lists O0 "yes", O1 "yes", O2 "yes", O3 "partially". Its own outputs support only
"the code surface can hold a better implementation", and the improvements were written by hand. The three
recommended specs (HF QA, multi-objective, `llm4ad` transfer) were not executed. EXP19–EXP22 later pursued
that direction.

## EXP00-D — use-case suite UC1–UC14

### Live single-arm results (`use_cases_uc13_live_fix2_20260618_000000`, `…uc13only_20260619_000000`)

| UC | Experiment | Initial → best | Δ |
|---|---|---|---|
| UC2 | QASPER LLM config | 0.196 → 0.180 | −0.016 |
| UC2 | QASPER causal numeric config | 0.017 → 0.168 | +0.151 |
| UC2 | mixed GSM8K + QASPER | −0.025 → 0.046 | +0.072 |
| UC2 | DROP LLM config | 0.750 → 1.000 | +0.250 (DROP saturates) |
| UC4 | O2 family policy | 0.007 → 0.029 | +0.022 |
| UC4 | O2 causal numeric policy | 0.025 → 0.025 | 0.000 |
| UC4 | O3 cold prior / warm prior | 0.134 → 0.189 / 0.192 → 0.227 | +0.055 / +0.035 |
| UC6 | trace_type internal / otel / hybrid | 0.157 → 0.201 / 0.131 → 0.188 / 0.124 → 0.186 | +0.045 / +0.056 / +0.063 |
| UC6 | internal + causal numeric config | 0.175 → 0.310 | +0.135 |
| UC13 | offline numeric causal preflight (deterministic) | 0.325 → 1.000 | +0.675 |
| UC13 | live numeric (Optuna) / live LLM config search | −0.166 → −0.159 / −0.163 → −0.161 | +0.007 / +0.002 |

### Three-way benchmark, Stage 2 (`three_way_stage2_20260624_150102`; standard vs recursive at equal budget)

| UC | Standard | Recursive | Δ | Seed-δ std | Reported verdict |
|---|---:|---:|---:|---:|---|
| UC1 code BBEH solver | 1.000 | 1.000 | 0.000 | 0.000 | tie, saturated (initial 0.625); recursive used 2 optimizer calls per seed vs 4 |
| UC2 QASPER config | 0.292 | 0.265 | −0.027 | 0.050 | tie |
| UC5 tool policy code | 0.750 | 0.792 | +0.042 | 0.629 | tie |
| UC8 campaign policy | 0.517 | 0.445 | −0.072 | 0.063 | standard wins |
| UC9 agentic policy | 0.760 | 0.843 | +0.083 | 0.112 | tie (promoted by the gate) |
| UC10 promotion policy | 0.342 | 0.398 | +0.057 | 0.049 | recursive, not LCB-safe |
| UC11 prompt emitter | 0.499 | 0.381 | −0.118 | 0.085 | standard wins |
| UC13 numeric | −0.158 | −0.160 | −0.002 | 0.004 | tie, flat |
| UC14 code transfer | 0.643 | 0.573 | −0.070 | 0.000 | standard wins |
| **UC4 family prior** | 0.013 | 0.176 | **+0.163** | — | **promote; budget sweep N = 6: +0.161, N = 12: +0.161** |

### Other runs

- **Tier follow-ups (25–26 June):**
  - T1.1, the promotion gate check, behaved as designed: a paired tie gives "hold", a clear win gives "promote".
  - T1.2 timed out (7,200 s cell limit).
  - T3.6 counted **0 executed optimizer-tool calls** in the agentic use cases: the learned tool policies were text,
    never executed.
- **UC7 graph routing:** the routing graph scored 1.000 and the conditional graph 0.875, with initial = final, so the
  probe showed nothing learned.
- **Outer-optimizer comparisons on UC4 (N = 24, seed 0, 1–4 July):** recursive − standard
  - +0.164 with `OptoPrimeV2` + `PrioritySearchMulti`;
  - +0.172 with `OptoPrimeMultiV2` multi-expert;
  - +0.166 with multi-LLM.

  These used the same UC4 design and are therefore invalid for the same reason (below).

### Audit (30 August 2026; assessment §5.2, §7.2; EXP02)

- **UC4's +0.163 is an arithmetic identity, not an effect.**
  - The standard arm (`o2_policy`) was scored as the mean over gsm8k and qasper. The recursive arm (`o3_prior`) was
    scored on qasper only. The delta equals (qasper − gsm8k)/2 to four decimals.
  - All arms ended with byte-identical default artifacts: nothing had been optimized.
  - Scored on the same tasks, +0.163 becomes **−0.018**, a paired delta of **−0.006**. EXP02 records the corrected
    re-run.
- **The other deltas** (±0.002 … ±0.12) are below the about 0.24 resolution at n = 5, so they are **unresolved**,
  including the three "standard wins". The conclusion "same-level warm code restart is not robust" is therefore
  unsupported, not refuted.
- **Still valid:**
  - UC1 is saturated.
  - UC13's live objective is flat.
  - The agentic tool policies were never executed.
  - The DROP control saturates.
- **The method of `CURRENT_LIMITS.MD` was sound**: a reliability gate before the promotion gate, and paired seed
  deltas. It became the basis of EXP01–EXP14.

## EXP00-E — Experiment 0 (`reports/prompt18_r3f_main_completion_trajectory_stop.md`)

The 40/40 canonical units completed and every infrastructure gate passed. One unit was interrupted by a local
Internet outage, re-run from zero and its partial attempt excluded.

| Arm | Runs | Accuracy (mean) | Token ratio (mean) | Invalid rate (mean) | Final artifact changed |
|---|---:|---:|---:|---:|---:|
| A fixed baseline | 10 | 0.992 | 1.012 | 0.004 | 0/10 |
| B Trace `OptoPrimeV2` | 10 | 0.979 | 0.879 | 0.004 | 8/10 |
| C GEPA `optimize_anything` | 10 | 0.950 | 0.755 | 0.004 | 9/10 |
| D Trace without validation gate | 10 | **0.888** | 0.629 | 0.004 | 10/10 |

Paired deltas against A, with 95% CIs:

| Arm | Accuracy Δ | Token-ratio Δ |
|---|---|---|
| B (Trace) | −0.0125 [−0.042, +0.017] | **−0.133 [−0.249, −0.023]** |
| C (GEPA) | −0.042 [−0.096, +0.004] | **−0.257 [−0.344, −0.173]** |

- **Neither optimized engine meets the frozen quality-or-efficiency success criterion.**
- **The baseline is near the accuracy ceiling** (0.99 on GSM8K hold-out). The optimizers mainly trade a little
  accuracy for fewer tokens.
- **Safety failed:** 4/40 runs had one invalid output each, one run in every arm.
- **Removing Trace's validation gate (D)** cut tokens the most but cost the most accuracy (−0.10).
- **Final status:** `RETURN_TO_CONTROL_PLANE_FOR_TRAJECTORY_PROVENANCE`. Neither engine persisted per-candidate
  trajectories, so candidate-level analysis was impossible without inference. The fix, persisting
  `candidate_trajectory`, followed on 27 August (`6aa9da0418`).
- **Cost:** token-price proxy $0.414.

## What EXP00 contributed to the later series

1. **Instruments.** The resolution-limit, saturation and flat-surface diagnoses here motivated EXP01–EXP14: signal
   vs noise, corrected UC4, certification, concurrency noise and knob liveness.
2. **Engineering.**
   - The causal-effect contract (which knobs can move a score).
   - Declarative specs, budgets and memory lineage.
   - The three-way equal-budget harness and its promotion gate, later fixed to require the same task set.
   - Experiment 0's pre-registration and frozen-manifest discipline, used by every later EXP.
3. **The pattern later experiments kept finding.**
   - Code rewriting works on surfaces with clear feedback (A/B, UC1; later EXP15–EXP19, EXP24).
   - Setup, family-policy and prior search showed no resolvable gain (A/D, Phases 1–6, UC2/UC4/UC6, UC13;
     later EXP22, EXP24, EXP28).
   - Headline wins came from measurement defects: UC4 here, PRISM refusals and Signal look-ahead later.
