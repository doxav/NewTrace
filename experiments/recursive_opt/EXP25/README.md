# EXP25 — Signal Processing: best Trace configurations versus stock EvoX

Three arms, three seeds each, 100 solution calls per run: the EXP24 Trace treatment, the EXP23 Trace configuration,
and stock SkyDiscover EvoX. Every candidate is re-scored with the same per-signal evaluator.

| Role | Entry point |
|---|---|
| Results and interpretation | [RESULTS.md](RESULTS.md) |
| Preregistered design | [PROTOCOL.md](PROTOCOL.md) |
| Evaluator (stock-exact, valid score, causality probe) | [signal/whitebox.py](signal/whitebox.py) |
| Raw evidence | [results/README.md](results/README.md) |

From this directory (EXP22 venv, `TRACE_ROOT=~/code/Trace`): `../EXP22/.venv/bin/python -I -m unittest discover -s tests -v` (4 tests),
`scripts/run_signal.py`, `scripts/run_evox_stock.py`, `scripts/analyze.py`, `scripts/causal_best.py`.

[All experiments](../README.md) · Previous: [EXP24](../EXP24/README.md)
