# EXP24 — Valid guide, white-box feedback and repair projection for native coevolution

EXP23's record PRISM score (30.8766) is a metric exploit: the program crashes on 47 of 50 cases, and PRISM averages
only over solved cases. EXP24 fixes what the search optimizes and sees, generically in `recursive_opt.coevolution`, and
reruns a clean, preregistered comparison of three arms (fixed policy, EvoX-equivalent, Trace proposer) on the valid score.

At the [30 September, 12:12 UTC checkpoint](RESULTS.md#clean-run-checkpoint), four of nine runs have terminal summaries and all nine have reached the computed ceiling. First-hit medians are 12 / 12 / 22 solution calls for fixed / rewrite / Trace. These small-sample observations do not establish a meta-level advantage; final campaign accounting remains incomplete.

| Role | Entry point |
|---|---|
| What was done and the results | [RESULTS.md](RESULTS.md) |
| Preregistered design | [PROTOCOL.md](PROTOCOL.md) |
| Code delta from EXP23 (patch, snapshots, tests) | [docs/CODE_DELTA.md](docs/CODE_DELTA.md) |
| Supporting documents | [docs/README.md](docs/README.md) |
| Raw evidence | [results/README.md](results/README.md) |
| Navigation metadata | [manifest.json](manifest.json) |

Reproduce from this directory (EXP22 venv, `TRACE_ROOT=~/code/Trace`):

```sh
../EXP22/evox/.venv/bin/python -I -m unittest discover -s tests -v       # 5 offline tests
../EXP22/evox/.venv/bin/python -I scripts/prism_exact.py                  # numerical optimum 26.2559717495
../EXP22/evox/.venv/bin/python -I scripts/run_prism.py --arm trace --seed 42 --out results/<campaign>/trace_s42
python3 scripts/analyze.py results/<campaign>
```

[All experiments](../README.md) · Previous: [EXP23](../EXP23/README.md)
