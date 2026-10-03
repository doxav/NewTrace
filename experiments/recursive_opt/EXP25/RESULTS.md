# EXP25 — results (2026-10-01)

All nine runs completed with exactly 100 solution attempts. Scores: stock metric (reproduced exactly), **valid** (all
five signals, documented output length, finite values), and **valid + causal** (no look-ahead on any signal, measured by
rerunning each signal without its last 50 samples). Analysis: [`analysis.json`](results/runs_20260930T235849/analysis.json),
[`causal_analysis.json`](results/runs_20260930T235849/causal_analysis.json).

| Arm | Seed | Best valid (any) | Best valid, causal only | Returned program causal on | Policy switches | Techniques in returned program |
|---|---:|---:|---:|---:|---:|---|
| evox_stock | 42 | 0.7132 | 0.5343 | 0/5 signals | 7 | SciPy `filtfilt`, `butter`, `savgol_filter`, `medfilt` |
| evox_stock | 43 | 0.7225 | 0.5371 | 0/5 | 6 | `filtfilt`, `savgol_filter`, `medfilt` |
| evox_stock | 44 | 0.6785 | 0.5610 | 0/5 | 7 | `filtfilt`, `butter`, `savgol_filter`, `gaussian_filter1d`, `medfilt` |
| trace_exp24 | 42 | 0.5855 | 0.5324 | 0/5 | 7 | NumPy only |
| trace_exp24 | 43 | 0.5926 | 0.5465 | 1/5 | 8 | NumPy only |
| trace_exp24 | 44 | 0.5313 | 0.5313 | 5/5 | 8 | NumPy only |
| trace_exp23 | 42 | 0.5553 | 0.5315 | 0/5 | 7 | NumPy only |
| trace_exp23 | 43 | 0.5615 | 0.5562 | 4/5 | 7 | NumPy only |
| trace_exp23 | 44 | 0.5313 | 0.5210 | 2/5 | 7 | NumPy only |

| Median over 3 seeds | evox_stock | trace_exp24 | trace_exp23 |
|---|---:|---:|---:|
| Best valid (any) | **0.713** | 0.586 | 0.555 |
| Best valid, causal only | 0.537 | 0.532 | 0.532 |

Initial program: 0.499 (causal). No run produced the truncation exploit: every best program is valid (stock = valid).

## Interpretation

1. **Stock EvoX wins on the benchmark's own terms** (every EvoX run above every Trace run, 0.68–0.72 vs 0.53–0.59), with
   scores in the range SkyDiscover publishes (0.72–0.76 with GPT-5).
2. **The lead comes entirely from look-ahead.** EvoX's returned programs use offline zero-phase or centered SciPy smoothers
   (`filtfilt`, `savgol_filter`, `medfilt`) that read future samples on every signal. Restricted to causal (real-time)
   candidates, all three arms are equal (medians 0.532–0.537). The task text asks for real-time filtering, but the evaluator
   does not check causality.
3. **Why EvoX finds SciPy and the native engine does not:** stock EvoX's label generator lists installed libraries and
   insists on using them; its generated labels name `scipy.signal`, `filtfilt`, `savgol_filter` and `butter`. The native
   port's label prompt is shorter and never produced SciPy code. This is a fidelity gap in the native O0 operator (EXP23
   equivalence covered control flow with static labels, not prompt content), not a property of Trace as meta-optimizer.
4. **EXP24 treatment versus EXP23 configuration:** slightly higher on valid-any (medians 0.586 vs 0.555), equal when causal.
   Three runs per arm; no significance claimed.

## Next steps

- Decide the target: benchmark-as-scored (look-ahead allowed) or real-time (causal) filtering.
- Port SkyDiscover's library-aware label generation (and its operator prompt text) into the native engine, re-verify with
  prompt-level equivalence, then rerun Trace versus EvoX on equal O0 prompts.
