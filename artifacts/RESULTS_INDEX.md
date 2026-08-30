# recursive_opt — which results are current

*Last updated 2026-08-30, after merging `origin/recursive_opt`.* One page, so no one has to
guess which numbers still hold.

> **Two lines of work were merged here.** This audit (measurement certification, 16 defect
> fixes) and parallel remote work on Experiment 0 (GEPA reflection adapter, transport
> hardening, empty-text handling, a frozen main run). Where both fixed the same bug — the
> optimizer's empty provider response — the implementations were combined rather than one
> discarded; see `opto/optimizers/utils.py::extract_response_content`.

## Read this first

| question | answer | where |
|---|---|---|
| Is recursive optimization better than standard? | **Unproven.** Never validly tested. | §5, §7, §13, §14 |
| Did anything ever beat standard optimization? | **No result survives scrutiny** — including the two produced during this audit. | §5.2, §14.3 |
| Does the optimizer work at all? | It **changes artifacts and moves scores**. Whether it *improves* anything is unproven. | §13, §14.3 |
| Can I trust `notebook_outputs/`? | **No.** Produced below the resolution limit. | §5, §7.1 |
| Which tasks can I measure on? | **3** at realistic concurrency; **2** with a genuine zero floor. | §14.4, `probe_2026/probe_l_results.json` |
| Is Experiment-0's blocking unit fixed? | There was **nothing to fix**. Invalid rate is 0/246, bounded below 1.54%; the 1-in-24 was a rare event. | §16.1 |
| Why did Experiment-0 stop, then? | `invalid_rate <= 0` on 24 samples per unit is **unsatisfiable across 40 units** — P(all pass) < 0.001 at the measured rate. | §16.2 |
| Can Experiment-0 detect an optimizer effect? | **No.** Baseline is 98.4% (1.6 pp headroom); it has 120 observations per arm where ~1,750 are needed for +1 pp. | §16.3 |

## Status of every results file

| file | status | note |
|---|---|---|
| `artifacts/recursive_opt_assessment.md` | ✅ **CURRENT** | The single source of truth, §0–§16. |
| `artifacts/probe_2026/*.json` | ✅ **CURRENT** | Raw data for §7, §11–§14. Probes A–K, 2026-08-29/30. |
| `artifacts/RESULTS_INDEX.md` | ✅ **CURRENT** | This file. |
| `examples/recursive_opt_use_cases_CURRENT_LIMITS.MD` | ⚠️ **SUPERSEDED** | Banner added. UC4 invalid; every delta below the resolution limit. Method still sound. |
| `examples/notebook_outputs/**` (~90 run dirs) | ⚠️ **SUPERSEDED** | ~13 months of runs, all below the instrument's resolution. Keep as evidence, do not cite. |
| `experiments/recursive_opt/multiobjective_reasoning/*` | ◐ **PARTIAL** | The GEPA blocker it reports is fixed. A frozen main run was authorised 2026-08-24 and **started**, stopping on its first unit with `hard constraints not satisfied` (accuracy 22/24, invalid 1/24, zero optimizer calls) — see `reports/prompt18_r3_main_stop_decision.md`. Older `control_plane_blocker.json` is historical. |
| `artifacts/control_plane_v2/*.md` | ◐ **PARTIAL** | Engineering claims hold. Efficacy claims do not — the plane never ran a recursive spec (D14). |
| `artifacts/control_plane_v2/prompt18_readiness.json` | ✅ **CURRENT** | Honestly reports `ready_for_prompt_18: false` with blockers. |

## The numbers that matter

```
UC4's "+0.163"                  : = (qasper - gsm8k)/2, exact to 4 decimals -> not an effect
resolution limit, as found      : 0.239 at n=5  (every historically reported effect was below this)
bounded evaluation sampling     : 14.6x resolution recovered
certified at REAL concurrency   : 3 of 8 probed; 2 with a genuine zero floor
                                  (internal:multi_param, llm4ad:admissible_set)
```

### Both results produced during this audit are retracted

```
Probe F  +0.217 on gsm8k               noise sd 0.32 -> resolvable 0.405   NOT established
Probe K  +4.8   on online_bin_packing  noise sd 4.41 -> resolvable 5.52    NOT established
```

`gsm8k`'s floor was first estimated at 0.033 from **three** repeats; six repeats put it at
**0.32-0.53**. Three samples cannot characterise a rare error term — the same mistake this
audit identified in the work it was auditing.

## Defects fixed (16)

D1–D10 from static analysis; **D11–D16 found only by running the code**.

Severe: `-1e9` sentinel reaching promoted priors (D1); unguarded cross-metric comparison (D3);
unsound `portable` flag (D11); a regression that broke **every** multi-level run (D14); an
evaluator that scored its own crashes (D2); and the config surface silently truncating every
multi-line artifact at its first newline (D16).

**Five further flaws were in the certifier written during this audit**, each a variant of
reading a constant as "quiet, therefore good": a `-1e6` failure sentinel; prose injected into
code and numeric parameters; an absolute effect target on a large-scale objective; a noise
floor measured sequentially while experiments run concurrently; and a Bernoulli term estimated
from three samples. They are documented in §12 and §14 rather than quietly fixed, because they
are the same error class the audit exists to find — and the best evidence that the hard part
here is measurement, not optimization.

## Before running anything

1. Certify the task: `measurement.certify_task(...)`. Refuse `broken` / `saturated` / `too_noisy`.
2. Score every arm on the **same** task set — `promotion_decision` now returns `invalid_comparison` otherwise.
3. Diff the artifacts. If they are identical, the delta is noise by definition.
4. Report the noise floor next to the effect, always.
5. Prefer the zero-noise deterministic surfaces: there, n=1 settles it.
