# EXP28 — EvoX-style exploration for Trace: Trainer level vs recursive_opt level

EXP27 traced Trace's gap to EvoX to exploration: Trace almost never issues productive DIVERGE calls. EXP28 adds that
missing axis in two places and runs each on Signal (4 arms × 3 seeds × 100 solution calls):

- **Trainer level:** new `opto.trainer.algorithms.VariationSearch` (a PrioritySearch subclass with per-step
  REFINE / DIVERGE / combine instructions). Arms `vs_stagnation` and `vs_combine`.
- **recursive_opt level:** two new coevolution engine options. `meta_brief` gives Trace's policy proposer EvoX's
  label and diversity rules; `diverge_guard` forces DIVERGE with no context after a stall. Arms `brief` and `guard`.

- [PROTOCOL.md](PROTOCOL.md): alternatives table (trainer T1–T7, recursive R1–R5), selection, pre-registered runs and reading
- [RESULTS.md](RESULTS.md): results
- `scripts/run_trainer_signal.py`, `scripts/run_coevo_variation.py`: runners (`--mock` for offline checks)
- `tests/test_exp28.py`: harness checks
