# EXP27 — root causes of Trace's gap to stock EvoX (signal task)

Investigates why the native coevolution engine trails stock EvoX in EXP25/EXP26, by five whys and nine hypotheses.
Part A analyses the 21 recorded runs (no LLM). Part B is a 72-call prompt-ablation probe.

- [PROTOCOL.md](PROTOCOL.md): five whys, hypotheses H1–H9, Part B decision rules, candidate fixes with their checks
- [RESULTS.md](RESULTS.md): findings
- `scripts/mechanisms.py`: Part A → `results/mechanisms.json`
- `scripts/prompt_ablation.py`: Part B → `results/ablation/completions.jsonl`
- `tests/test_exp27.py`: harness checks

Commands, run from this directory:

```
TRACE_ROOT=/home/xav/code/Trace ../EXP22/.venv/bin/python -I scripts/mechanisms.py
TRACE_ROOT=/home/xav/code/Trace ../EXP22/.venv/bin/python -I scripts/prompt_ablation.py --out results/ablation
TRACE_ROOT=/home/xav/code/Trace ../EXP22/.venv/bin/python -m pytest -q tests/test_exp27.py
```
