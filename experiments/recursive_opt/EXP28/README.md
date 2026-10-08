# EXP28 — EvoX-style exploration for Trace: Trainer level vs recursive_opt level

EXP27 traced Trace's gap to EvoX to exploration: Trace almost never issues productive DIVERGE calls. EXP28 adds that

**Signal as a meta testbed (2026-10-08, [EXP29 §5.1](../EXP29/prior_analysis.md)).** Hand-written causal filters score 0.467–0.539 and the best LLM causal program 0.565, inside the 0.499–0.554 seed spread. Signal is kept for smoke tests and exploit regression, not to resolve meta effects.
missing axis in two places and runs each on Signal (4 arms × 3 seeds × 100 solution calls):

- **Trainer level:** new `opto.trainer.algorithms.VariationSearch` (a PrioritySearch subclass with per-step
  REFINE / DIVERGE / combine instructions). Arms `vs_stagnation` and `vs_combine`.
- **recursive_opt level:** two new coevolution engine options. `meta_brief` gives Trace's policy proposer EvoX's
  label and diversity rules; `diverge_guard` forces DIVERGE with no context after a stall. Arms `brief` and `guard`.

- [PROTOCOL.md](PROTOCOL.md): alternatives table (trainer T1–T7, recursive R1–R5), selection, pre-registered runs and reading
- [RESULTS.md](RESULTS.md): results
- `scripts/run_trainer_signal.py`, `scripts/run_coevo_variation.py`: runners (`--mock` for offline checks)
- `tests/test_exp28.py`: harness checks
