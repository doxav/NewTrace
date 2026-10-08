# EXP29 — results log

Step-by-step record of P0–P4 (plan: [prior_analysis.md](prior_analysis.md) §8). Each step is committed separately.

## Decisions taken (2026-10-08, by the user)

| | Decision |
|---|---|
| D1, which code may be replaced | **(b) any `opto.` function, method or module constant, by robust monkey patching, with no core-library change**. The runner and scorer (`opto.features.recursive_opt.*`) cannot be patched, so a patch can worsen search but cannot fake a score. |
| D2, where patched code runs | **One process per child run.** Patches are applied inside that process for the duration of the run and restored after. Different versions of the patched surface run in parallel as separate forked workers. A second, different patch set requested concurrently in the same process is refused. |

## P0.1 — engineering: `patches.py` and `child_spec@1`

**New files** (core library untouched):

| File | Lines | Content |
|---|---:|---|
| `opto/features/recursive_opt/patches.py` | ≈ 200 | `validate`, `applied`, `default_source`, `run_isolated` (see below) |
| `opto/features/recursive_opt/child_spec.py` | ≈ 140 | `recursive_opt.module.child_spec@1` and `recursive_opt.evaluator.child_spec@1` |

**`patches.py` functions:**
- `validate`: each patch is a `target` plus `source` or `value`. The target must exist under `opto.`, outside `recursive_opt`. A `source` must be exactly one function with the same parameter names as the original, and must not use denylisted imports or names.
- `applied`: a context manager that applies the patches and restores the originals. A failing patched function falls back to the original, and the fallback is counted.
- `default_source`: the current source of a target, used as the starting value of a code slot.
- `run_isolated`: one forked, non-daemonic process per call. Workers can fork again, so O2 nests.

**`spec.py` change:** the `trace` engine accepts a validated `patches` list (default `[]`). It is applied around the whole level run, and fallbacks are reported in the level metadata.

**`child_spec@1`:**
- The upper level's trainable parameters are slots: dotted paths into a complete child spec.
- Evaluation writes the slots and the episode overrides (restricted to `example_paths`) into the child. It then compiles the child with the normal strict compiler and runs it in its own process.
- It returns the child's final score and feedback, and reports the child's LLM calls as the parent's `forward` usage.
- An invalid child is an invalid candidate, not a crash. Identical children are cached.

**Tests:** `tests/unit_tests/test_recursive_patches_child_spec.py`, 15 tests, offline:
- a patch applies and is restored;
- the default source round-trips;
- value patches work, and a null source means no patch;
- six kinds of invalid patch are rejected (scorer target, non-`opto` target, signature, import, name, missing target);
- a failing patch falls back and is counted;
- a second patch set in the same process is refused, while three versions run in parallel workers;
- engine patches are validated and reach the run;
- `child_spec@1` runs the child with its slot and episode;
- a slot that is not a field is rejected;
- an invalid slot value gives an invalid candidate.

**Full unit suite:** 1,014 passed, 2 skipped. The runtime file inventory and the provenance seal (`prompt18_readiness.json`) were updated because new runtime files were added.

**Known limit:** forking from a multi-threaded parent can deadlock (Python's `DeprecationWarning`). O1 parents therefore run single-threaded, and parallelism comes from separate seed/arm processes and from the child workers.

## P0.2 — numeric task family, live child runs, calibration

**Task** (`EXP29/numeric/task.py`):
- Black-box optimizer programs `propose(history, bounds, seed)` (the EXP15–18 interface), on sphere, Rosenbrock, Rastrigin and Ackley.
- Shifts are in [−4, 4], so starting at the centre is no shortcut. The budget is 64 evaluations.
- Evaluator `exp29.evaluator.bbo@1`. Score = minus the mean log10 normalized best-so-far regret, floored at 1e-6: higher is better, 6 is perfect.

**Headroom probe** (`numeric/probe_headroom.py`, no LLM; 8 strata × 4 instances):

| Program | Score (32 instances) |
|---|---:|
| random | 1.05 |
| seed (EXP15) | 1.38 |
| hand-written ES | 1.39 |
| hand-written quadratic surrogate | 1.64 |

- Under the plain regret AUC at 32 evaluations, the same programs spanned only 0.370 → 0.302: the first random draws dominate it. That metric was rejected.
- The log score keeps order-of-magnitude precision. Example: sphere-5D, seed 0.92 → surrogate 1.65.

**Live child through the control plane** (`numeric/specs.py`, `numeric/run_child.py`):
- Setup: `z-ai/glm-5.3-flash` via OpenRouter, low reasoning, cheapest provider, `OptoPrimeV2` + `PrioritySearch`, one candidate per step.
- 12 iterations on episode mixA-s1:

  | Measure | Value |
  |---|---|
  | Optimizer calls | 11 |
  | Cost (key delta) | **$0.029** (≈ $0.0026 per call) |
  | Wall time | 16 min, dominated by LLM latency |
  | Train score | 1.26 → best 2.20, peak after about 4–10 proposals |
  | Holdout score | seed 1.125 → **1.494** |

- So O0 learning moves the score. P0 children use 8 iterations.

**Fixes needed on the way:**
- **Preflight:** the control-plane LLM preflight sends `max_tokens=8`, which a reasoning model consumes entirely, and it ignored its own documented `RECURSIVE_OPT_SKIP_MODEL_PREFLIGHT` flag. `runmode.preflight_model` now honours the flag.
- **Reasoning effort:** with OpenRouter it must go in `request_params.extra_body.reasoning`.

**`child_spec@1` additions:**
- An example `{"episodes": [...]}` runs several episodes in parallel workers from one call. A single-threaded O1 parent therefore still evaluates episodes in parallel, avoiding a fork from threads.
- `timeout_s` kills a hung child; it becomes an invalid candidate.
- 16 tests pass; full unit suite 1,017 passed.

## P0.3 — certificates: hand-written variants of the O1 targets (live, 24 children, $0.61)

**Setup:**
- 8-iteration O0 children on the three O1 holdout episodes (mixA / mixB / mixC, episode seed 3), 2 repeats each.
- Score = held-out child score. Gain = score minus the seed program's held-out score (mixA 1.216, mixB 1.319, mixC 1.426).
- Script: `numeric/p0_arms.py`; analysis: `numeric/analyze_arms.py` → `results/p0/arms/analysis.json`.

| Arm | Mean holdout | Gain over seed (sd) | Paired vs (a) | Wins |
|---|---:|---:|---:|---:|
| (a) `PrioritySearch` | 1.312 | −0.008 (0.06) | — | — |
| (b) `VariationSearch` default | 1.493 | +0.172 (0.21) | +0.181 | 3/6 |
| (c) `VariationSearch` + hand-written `_next_mode` patch (diverge after 2 stalled steps) | 1.515 | +0.194 (0.23) | +0.202 | 5/6 |
| (d) `PrioritySearch` + hand-written optimizer instruction (strategy hints) | **1.745** | **+0.424 (0.60)** | **+0.432** | 5/6 |

**What this shows:**
- Outcomes are **heavy-tailed**. 9 of 24 children end exactly at the seed's score: the validation gate kept the initial program. A few jump far, up to 2.69 (d, mixB) and 2.38 (d, mixC).
- **T1 certificate passes:** the optimizer instruction (`optimizer_kwargs.objective`) moves O0 most.
- **T2 certificate passes weakly:**
  - the scheduler patch, and `VariationSearch` itself, beat plain `PrioritySearch` in 5/6 and 3/6 pairs;
  - (c) vs (b) is a tie.
- With n = 6 and these tails, none of these differences is statistically resolved. They clear the P0 bar of "a hand-written variant moves O0", so both targets enter P1.
- Cost: $0.0036 per optimizer call; one child ≈ 7 calls ≈ $0.025, 25–28 min wall time with 24 children running concurrently.

**Offline O1 dry run** (no LLM at either level, 1 O1 iteration) passed end to end: slots injected into the child's patches and objective, two meta-train episodes run in parallel workers, meta-validation, meta-holdout. Its holdout equals the seed's (1.2164), as expected.
