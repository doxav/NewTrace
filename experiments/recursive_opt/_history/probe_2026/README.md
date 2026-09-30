# Live probes — 2026-08-29

Raw data behind §7 of `artifacts/recursive_opt_assessment.md`.

Model: `openrouter/deepseek/deepseek-v4-flash-0731` (the package's own OpenRouter default;
`.env` supplies `OPENROUTER_API_KEY` but no model). No paid model was chosen beyond that default.

## Probe A — signal vs noise

`probe_a_signal_vs_noise.py` → `probe_a_results.json`, `probe_a.log`

Three prompts × 3 repeats × 2 tasks, `inner_steps=0` (no nested trainer, no optimizer LLM calls),
`max_examples=2`. Separates the between-prompt spread (the signal optimization could exploit) from the
within-prompt stdev (the noise floor of re-running the same prompt).

| task | spread | noise | S/N |
|---|---:|---:|---:|
| `internal:multiobjective_gsm8k` | 0.0335 | 0.0350 | 0.96 |
| `hf:qasper` (excluding the unstable bundle-default arm) | 0.0290 | 0.0391 | 0.74 |

Both below 1: changing the prompt moves the score less than running the same prompt twice.

`probe_a_attempt1_partial.log` is a first attempt that stalled on an untimed provider call; it was
killed and the script given a per-run `SIGALRM` bound. Its 9 gsm8k points agree with the final run.

## Probe B — the corrected UC4

`probe_b_corrected_uc4.py` → `probe_b3.log`, `probe_b_out/`

Both arms scored on the **same** level (`o3_prior`) and therefore the same held-out family, which is
the correction to the historical comparison. 3 seeds, 8 candidates/arm, `starting_artifact` target.

```
comparability : comparable=true, both arms on ["hf:qasper"]
initial   +0.072 (n=3)   standard  +0.184 (n=2)   recursive +0.167 (n=1)
recursive - standard = -0.018        paired delta (seed 1) = -0.006
```

The historical `+0.163` does not survive; −0.018 is inside Probe A's ±0.039 floor. Eight of nine
arm-seeds ended with the unchanged default artifact.

`probe_b.log` / `probe_b2.log` are earlier attempts: the first was refused by
`effects.check_field_effects` for targeting `batch_design`/`batch_size` at `inner_steps=0` (the guard
working correctly), the second hit defect D14. Both are kept because they are evidence.

## Reproducing

```bash
set -a; . ./.env; set +a
export PYTHONPATH=$PWD
export TRACE_LITELLM_MODEL=openrouter/deepseek/deepseek-v4-flash-0731
export RECURSIVE_OPT_TRACEBENCH_MODEL=$TRACE_LITELLM_MODEL
python artifacts/probe_2026/probe_a_signal_vs_noise.py
PROBE_SEEDS=0,1,2 PROBE_TOTAL_CANDIDATES=8 python artifacts/probe_2026/probe_b_corrected_uc4.py
```
