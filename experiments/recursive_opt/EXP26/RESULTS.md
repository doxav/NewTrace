# EXP26 — results

**Not run yet.** The protocol was pre-registered on 2026-10-06 ([PROTOCOL.md](PROTOCOL.md)); nothing below may be
filled in before the campaign completes.

## Pre-campaign validation (2026-10-06)

- `scripts/analyze.py` run on EXP25's campaign reproduces all nine published per-run P1 and P2 values exactly and
  every arm median (evox_stock 0.713 / 0.537; trace_exp24 0.586 / 0.532; trace_exp23 0.555 / 0.532). Applied
  retroactively, EXP25 passes the reproduction gate and satisfies R5 (lead is look-ahead only).
- The same run gives the mechanism baseline: SciPy in 68–76% of distinct EvoX candidates per run and 0–1.2% of
  Trace candidates; DIVERGE used in 0–24% of Trace iterations (median 12%).
- Offline: each native arm differs from EXP25's `trace_exp24` configuration only in its label source; `native_pkg`
  receives exactly stock's `get_available_packages()` list; `native_stocklabels` makes no label-generation call and
  its injected text reaches the solution prompt whenever a label is selected; stock's silent fallback to default
  labels is detected.

## Campaign

| Arm | Seed | P1 best valid | P2 best valid, causal | SciPy share | DIVERGE share | Labels name SciPy |
|---|---:|---:|---:|---:|---:|---|
| _pending_ | | | | | | |

Decision rules R1–R5: _pending_.
