# Execution plan for steps 1–3

*Written 2026-08-30, after four pre-flight analyses. Every number below is measured, not
assumed; the probes are in `artifacts/probe_2026/`.*

## What the pre-flight established

**A. Task headroom (Probe O, 120 evaluations, 216 s).** The task choice for step 1 is
decided by measurement, not reputation:

| task | accuracy | headroom | invalid | detectable @ n=120 | verdict |
|---|---:|---:|---:|---:|---|
| `gsm8k` | 0.975 | 0.025 | 0/40 | 0.025 | **saturated** — no room |
| `bbeh_object_counting` | 0.075 | 0.925 | 0/40 | **0.124** | **USABLE** |
| `bbeh_boolean_expressions` | 0.125 | 0.875 | **5/40** | 0.143 | extraction unreliable |

`bbeh_object_counting` is the replacement: 92.5% headroom, **zero** invalid extractions in
40, and at the design's existing 120 observations per arm it resolves a 12.4-point gain —
which on a 7.5% baseline is a **1.6× relative** improvement. No change to the sample size
is required. `bbeh_boolean_expressions` is rejected: at 12.5% invalid it would trip even a
relaxed constraint.

**B. Step 2 is ~4× cheaper than §9.1 estimated.** `MetaLevel.forward()` already returns
`{'score', 'feedback'}`; `normalize_evaluation_result` already accepts it; `compile_level`
already builds the level; `parameters()` already exposes one plain-string node. A portable
recursive module is a `ModuleRegistryEntry` delegating to existing code — **~30–40 lines,
not 150** — and it makes `legacy_level@1` redundant for the config surface, so net growth
can be zero or negative.

**C. Step 3 is structurally unblocked.** A two-level `o2_policy → o3_prior` spec runs
end-to-end with no errors. But the scored sets differ (`o2` = all families, `o3` = holdout
only) — the UC4 trap. **Both arms must be scored on `o3_prior`.**

**D. Parallelism boundary.**

```
thread-parallel  : SAFE for stateless evaluation (own bundle per worker, <=8)
process-parallel : SAFE for whole arms (separate globals + memory root)
```

The blocking constraints are the module-global task adapter, the global budget, and one
MemoryLite root per run — not the `_BudgetGuard` counter, whose read-modify-write race is
real in principle but did not manifest in 16,000 concurrent charges.

---

## Iteration protocol

Every iteration is the same five steps. **No iteration may end without a commit**, so any
step is revertible with `git revert` alone.

1. **State the hypothesis and its kill condition** — written before running anything.
2. **Pre-flight** — certify the surface at the concurrency the run will use
   (`measurement.certify_task(..., concurrency=N)`), and assert the effect is resolvable.
   Skip nothing here: five of this project's defects were sequential-vs-concurrent or
   effect-vs-noise mismatches.
3. **Run** — processes across arms, threads inside evaluation.
4. **Critique** — the standing checklist below, applied adversarially to *my own* result.
5. **Commit** — result, interpretation, and any retraction, whatever the outcome.

### Standing critique checklist

Applied at every iteration. Each line exists because it caught a real error here.

- [ ] Are both arms scored on the **same task set**? (UC4: `+0.163` was `(qasper−gsm8k)/2`)
- [ ] Did the **artifact change**? An identical artifact means the delta is noise. (§5.2, §7.2)
- [ ] Is the effect above the noise floor **measured at this concurrency**? (§14.2)
- [ ] Is the quality metric **saturated**? (gsm8k 97.5%)
- [ ] Are failures **counted, never imputed** as scores? (D2)
- [ ] Is a constant being read as "quiet, therefore good"? (five separate certifier bugs)
- [ ] Does the reported artifact **match** the reported score? (D16)
- [ ] Is n sufficient, with the interval stated — not just the point estimate? (0/24 → (0, 0.138))

---

## The iterations

### Iteration 1 — portable recursive module (step 2)
*First because 2 and 3 both depend on it, and it needs no provider calls.*

- **Hypothesis:** a recursive level can be expressed as a portable v2 spec with
  `portable=True, promotable=True`, adding ≤40 lines.
- **Kill condition:** requires a behavioral runtime resource (which forces
  `test_mode` → `portable=False`), or exceeds 60 lines.
- **Parallel:** none needed — offline and fast.
- **Checks:** existing suite green; a new test asserting `portable is promotable is True`
  for a config level; `scored_task_ids` reports correctly through the portable path.
- **Budget:** ~1 h, $0.

### Iteration 2 — Experiment-0 design fix (step 1)
- **Hypothesis:** on `bbeh_object_counting` with `invalid_rate ≤ 0.02` pooled (not
  zero-per-unit), the frozen matrix can complete.
- **Kill condition:** measured invalid rate on the new task pushes P(all 40 units pass)
  below 0.8.
- **Parallel:** 8 threads over holdout evaluation; **arms as separate processes**.
- **Checks:** re-certify `bbeh_object_counting` at concurrency 8 *before* freezing;
  recompute P(all units pass) from the measured rate; confirm detectable effect at n=120.
- **Budget:** ~2 h, <$0.10. **Do not re-freeze the lock until this passes.**

### Iteration 3 — recursive vs standard on a zero-noise surface (step 3)
- **Hypothesis:** the `o2→o3` warm prior beats a cold `prior` at equal budget, both scored
  on the same held-out family.
- **Kill condition:** paired delta inside the certified noise floor at n=5.
- **Parallel:** the two arms are independent → **separate processes**; three hypotheses
  (numeric / packing / mixed families) can run **concurrently** as separate processes.
- **Checks:** the whole standing checklist, especially same-task-set and artifact-changed.
- **Budget:** ~2 h, <$0.20.

---

## Code budget

Package is **9,999 lines**. The rule for every iteration: **no net growth.** Additions are
paid for by deletion, not by compression — compression is what produced `spec.py`'s
one-statement-per-function style in the first place (D8).

Identified payment: `TinkerEnvAdapter` (zero references outside its own definition, no test
with a real client), and `legacy_level@1`'s config path once Iteration 1 supersedes it.
18 of 114 public exports are unreferenced outside the package, but most are legitimate
control-plane API and will **not** be removed to hit a number.

---

## Iteration 2 — result: hypothesis KILLED, and the fix is structural

**Hypothesis:** on `bbeh_object_counting` with `invalid_rate <= 0.02` measured pooled, the
frozen 40-unit matrix can complete.
**Kill condition:** P(all 40 units pass) < 0.8.

**Pre-flight (Probe P, 52 samples, concurrency 8, 107 s):**

```
invalid  : 0/52  = 0.0000   95% CI (0.0000, 0.0688)
accuracy : 6/52  = 0.1154   95% CI (0.0540, 0.2297)   headroom 0.885
transport errors: 0
```

**KILL CONDITION TRIGGERED.** And the reason is arithmetic, not marginal:
`invalid_rate <= 0.02` on a 24-sample unit allows `floor(0.02 x 24) = 0` invalid — it is
*identical to the frozen constraint*. Relaxing the threshold changes nothing.

### Why every variant fails

| option | result |
|---|---|
| raise per-unit tolerance to 0.05 / 0.10 | P(all 40) = 0.0000 |
| pool the constraint over all 960 samples | P(pass) = 0.0126 |
| measure 500 clean samples to tighten the bound | P(all 40) = 0.56 — still short |

The structure is the problem. Passing a near-zero invalid constraint across
**40 units × 24 samples = 960 samples** requires essentially every sample to be valid:

| true rate | P(all 40 pass), tolerance 0 | tolerance 1/unit |
|---:|---:|---:|
| 0.0100 | 0.0001 | 0.3807 |
| 0.0050 | 0.0081 | 0.7731 |
| 0.0020 | 0.1463 | **0.9580** |
| 0.0010 | 0.3827 | **0.9892** |

Even the best case needs a true rate below ~0.2%, and **demonstrating** a rate that low
takes ~1,930 clean samples — more than the entire `bbeh_object_counting` v2 pool (52),
and 8× the cost of the main experiment it is meant to protect.

### The actual fix: invalidity is a metric, not a halt condition

A hard constraint that stops a 40-unit matrix on a single stochastic per-sample event is a
design error at *any* threshold. It converts a rare, expected event into total experiment
failure, and it is what stopped the 2026-08-24 run (§16.2) — not the optimizer, and not a
defect.

Recommended amendment:

1. **Report `invalid_rate`; do not gate on it.** It is already in the objective's
   `directions` as a minimise term. That is the right place for it.
2. **Fix the accuracy denominator explicitly** — either exclude invalid samples and report
   the reduced n, or count them as incorrect. Either is defensible; leaving it implicit is
   not.
3. Keep a **run-level** guard for genuine breakage — e.g. halt if pooled `invalid_rate`
   exceeds 0.10, which at the measured rate has a false-stop probability near zero while
   still catching a broken extractor.
4. `bbeh_object_counting` remains the right task: 88.5% headroom, 0/52 invalid, and at the
   design's existing 120 observations per arm it resolves a 12.4-point gain.

**This amendment changes a preregistered stopping rule and therefore requires the
experiment owner's decision, not mine.** It is recorded here rather than applied.

## Iteration 3 — recursive vs standard on the "zero-noise" surfaces — VOID

3 hypotheses × 3 seeds × 2 arms, three concurrent processes, both arms scored on the same level
(`o3_prior`). Result: numeric Δ=0.0, packing Δ=−3.0, mixed Δ=0.0; `artifacts_differ: false` on all three.

**Not a kill of the hypothesis — a void run.** Standing checklist applied:

| check | result |
|---|---|
| Did the artifact change? | **No** on all three → delta is not an effect |
| Is the delta outside the noise floor? | **No** — in-run replicate range 8.00 vs Δ −3.0 |
| Is the surface live and unsaturated? | **No** on 2 of 3 — one distinct value over 27 replicates |
| Did both arms get equal search where scored? | **No** — standard 2 iterations at `o3`, recursive 1 |
| Any candidate invalid by construction? | **Yes** — 12 candidates scored the −1e6 sentinel |

**Kept from this iteration** (the value is entirely methodological):
- **In-run replicate control.** Re-scoring the identical artifact 29× on `online_bin_packing` spans
  8.00 points (sd 2.61). Free, same-run, same-concurrency; independently confirms Probe L's sd 4.41
  and would have caught Probe K's retracted +4.8 before it was claimed. Make it standard in every arm.
- **The budget-equalisation dilemma.** Total compute and scored-level search cannot both be held
  equal. Every comparison must declare which it equalises and report the other.
- **D18 (new, instrument gap).** Certification is not menu-conditional: `admissible_set` is
  `certified` by `probe_liveness` yet flat under the experiment's actual knob. Certified ≠ has
  headroom for *this* experiment.

**Next (Iteration 4) blocked on D18.** Fix menu-conditional headroom first; re-certify candidate
surfaces against their menus; then re-run with scored-level parity and the replicate control.

## Concurrent-strategy round (§20) — instrument fixed, recursion still not beaten

Eight strategies launched across two waves in isolated worktrees; session limits killed several
mid-run. Outcomes, judged against `CRITIC_PANEL.md`:

| strategy | outcome |
|---|---|
| Surface-aware artifact application | **MERGED.** Refuses wrong-kind candidates; `effective_menu_size` catches ranking-equivalence too. |
| W2 amortisation on the routing family | **PARTIAL.** Shared optimum confirmed (ρ 0.51–1.00); K\*=2.25–6.0 vs an uninformed searcher only. |
| LLM control (probe X) | **DECISIVE NEGATIVE.** Cross-task artifact transfer 0/22. |
| Config-knob transfer | **UNTESTABLE HERE.** All 13 bundles are single-example, so batch/ordering knobs are structurally inert. |

**Standing rule added:** every experiment must declare which win condition it tests (§18). W1 needs
noise; W2 works at zero noise. Testing W1 on a deterministic surface is a category error.

**Next, by priority:** (1) enforce `effective_menu_size` in `run_spec` rather than only reporting
it; (2) fix `examples/recursive_opt_example_A_learn_setup.py`, whose O1 search is noise; (3) re-test
knobs on a multi-example prose surface (`gsm8k`/`qasper`) — the only remaining transfer path, and
the only place W1 is testable, so one experiment answers both.
