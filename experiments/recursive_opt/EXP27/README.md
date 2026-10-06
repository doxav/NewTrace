# EXP27 — root causes of Trace's gap to stock EvoX (signal task)

Investigates why the native coevolution engine trails stock EvoX in EXP25/EXP26, by five whys and nine hypotheses.
Part A analyses the 21 recorded runs (no LLM). Part B is a 72-call prompt-ablation probe. Part C is a powered
2×2 factorial (1,000 calls). Part D runs 11 full Trace runs with the causal cue hidden or shown. Part E compares
the two engines' exploration strategies from logs and transcripts.

- [PROTOCOL.md](PROTOCOL.md): five whys, hypotheses H1–H9, Part B decision rules, candidate fixes with their checks
- [RESULTS.md](RESULTS.md): findings
- `scripts/mechanisms.py`: Part A → `results/mechanisms.json`
- `scripts/prompt_ablation.py`: Part B → `results/ablation/completions.jsonl`
- `scripts/factorial.py`: Part C → `results/factorial/`
- `scripts/run_trace_cue.py`, `scripts/analyze_partD.py`: Part D → `results/partD_*/analysis.json`
- `scripts/strategy_timeline.py`, `scripts/instruction_yield.py`: Part E → `results/strategy_timeline.json`, `results/instruction_yield.json`
- `tests/test_exp27.py`: harness checks

Commands, run from this directory:

```
TRACE_ROOT=/home/xav/code/Trace ../EXP22/.venv/bin/python -I scripts/mechanisms.py
TRACE_ROOT=/home/xav/code/Trace ../EXP22/.venv/bin/python -I scripts/prompt_ablation.py --out results/ablation
TRACE_ROOT=/home/xav/code/Trace ../EXP22/.venv/bin/python -m pytest -q tests/test_exp27.py
```
