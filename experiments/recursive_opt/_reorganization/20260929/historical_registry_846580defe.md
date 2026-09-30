# recursive_opt — research log

**The single entry point.** Status board first, evidence second. Absorbs the former
`EXECUTION_PLAN.md` and `RESULTS_INDEX.md`. Long-form evidence stays in
`recursive_opt_assessment.md`, referenced here by section (§n) — it is the audit trail,
not the status.

*Last updated 2026-09-02.*

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

**Bottom line.** Recursion has never beaten standard on maximum performance (H1). Its one
demonstrated advantage is *speed* under conditions a real LLM optimiser did not satisfy (H2 vs
H3). The precondition it needs does exist (H5). The two paths by which it could exploit that
precondition are closed in this task pool (H3 refuted, H4 blocked).

---

## 2. Goals

| goal | state | blocker / next |
|---|---|---|
| **G-A** Decide whether recursive_opt is a viable optimisation layer | **answered, conditionally** | Viable mechanism, no demonstrated quality win. Needs a task family satisfying §21.4's five properties. |
| **G-B** Make the instrument trustworthy | **largely done** | 18 defects fixed; menu-collapse detection still opt-in (TODO). |
| **G-C** Find a venue where recursion *can* win | **blocked on task design** | No existing task qualifies. Requires portable-artifact family (fixed calling convention). |
| **G-D** Establish the paired-seed noise floor | **in flight** | EXP-12/13. Everything downstream is gated on this number. |
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
| **MC-b** | ranking-equivalent candidates collapse a menu invisibly | **open** — `TODO(menu-collapse)`, no type check can see it |
| **MC-c** | type audit is **vacuous on prose** — accepts everything unread | **fixed** (`menu_check_kind`) |
| **MC-d** | `effective_menu_size` is opt-in, not recorded per run | **open** — first attempt reverted (read 3 where truth was 1) |
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
3. **Close MC-b / MC-d** — record `effective_menu_size` per run, over the candidates the level evaluated.
4. **Build a portable-artifact family** (§21.4 property 5) — the only route to testing H3 honestly.
5. **Test H6/W1** — the one hypothesis never explored, on a prose surface with n sized to its floor.
