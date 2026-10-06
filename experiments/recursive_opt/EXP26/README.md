# EXP26 — Label fidelity: does stock EvoX's label contract close EXP25's SciPy gap?

EXP25's stock EvoX led native Trace coevolution on the valid score (0.713 vs 0.586) through SciPy look-ahead
smoothers, used in 68–76% of EvoX candidates and almost never by Trace (1 program in six runs). Re-analysis of
EXP25's evidence leaves two mechanisms unseparated: label **content** (stock injects the installed packages and
requires a LIBRARIES/TOOLS block; native labels only permit libraries, and REFINE never names one) and label
**selection** (the native policy chose DIVERGE, the only label naming SciPy, in a median 12% of iterations).
EXP26 holds EXP25's `trace_exp24` configuration fixed and varies only the label source, with full transcripts.

**Status: prepared, not run** — protocol pre-registered, harness and analysis tested offline. [RESULTS.md](RESULTS.md)

| Role | Entry point |
|---|---|
| Why the design looks like this | [docs/DESIGN_ANALYSIS.md](docs/DESIGN_ANALYSIS.md) |
| Preregistered design and decision rules | [PROTOCOL.md](PROTOCOL.md) |
| Results | [RESULTS.md](RESULTS.md) |
| Native arms (imports EXP25's runner) | [scripts/run_signal.py](scripts/run_signal.py) |
| Stock labels and stock arm (import EXP25's stock harness) | [scripts/gen_stock_labels.py](scripts/gen_stock_labels.py), [scripts/run_evox_stock.py](scripts/run_evox_stock.py) |
| Endpoints, mechanism variables, rules R1–R5 | [scripts/analyze.py](scripts/analyze.py) |
| Raw evidence | [results/README.md](results/README.md) |

Library change under test: `generate_labels(..., packages=...)` / `CoevolutionConfig.label_packages`
(commits `b0b0d18913`, `7d2b8059bf`); the default prompt and all EXP24/EXP25 plan fingerprints are unchanged.

From this directory (EXP22 venv, `TRACE_ROOT=~/code/Trace`): `../EXP22/.venv/bin/python -I -m unittest discover -s tests -v`
(6 offline tests). Campaign commands: [results/README.md](results/README.md).

[All experiments](../README.md) · Previous: [EXP25](../EXP25/README.md)
