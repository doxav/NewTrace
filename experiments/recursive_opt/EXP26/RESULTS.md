# EXP26 — results

**Complete, 12/12** (campaign `results/runs_20261006T112918`, all twelve runs concurrent, 11:32–13:01 UTC
2026-10-06, 100 solution attempts each, no stock label fallback). Protocol: [PROTOCOL.md](PROTOCOL.md).

> **Correction (2026-10-06, from the [EXP27](../EXP27/RESULTS.md) log analysis).** (1) The campaign is *truncated*:
> the OpenRouter key hit its total spend limit (HTTP 403) at one moment for all twelve runs, after iteration 67–70
> for stock and 43–63 for native arms; every later iteration failed. Each `summary.json` still says `success`. At an
> equal 42-iteration budget the ranking is unchanged (raw best: evox_stock 0.686, native 0.652, native_stocklabels
> 0.565, native_pkg 0.552), so R1 and R2 stand. (2) The R3 *interpretation* below is withdrawn: matched by exact
> label text, stock EvoX itself selects DIVERGE in only 8–23% of iterations (median 14%), the same range as the
> native arms, so a low DIVERGE rate does not explain the gap. (3) Stock's lead comes from look-ahead SciPy filters
> (0–8% of stock SciPy candidates are causal); on strictly causal candidates there is no gap. See EXP27.

**Verdict: R3 by the pre-registered rule; its mechanism reading is withdrawn (see the correction above).** Stock EvoX reproduces EXP25 (gate passed), but injecting stock's own labels into the native
engine does not close the gap (median P1 0.576 against 0.709) and does not bring SciPy in (median share 0%).
Label *content* is not the cause; the native label *selection* rarely chooses DIVERGE (13%), so the label
selection policy (M2) is implicated.

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
| evox_stock | 42 | 0.7148 | 0.5514 | 62.3% | n/a | both blocks |
| evox_stock | 43 | 0.7086 | none | 79.7% | n/a | both blocks |
| evox_stock | 44 | 0.5877 | 0.5495 | 49.0% | n/a | both blocks |
| native | 42 | 0.5805 | 0.5250 | 0.0% | 0.0% (no label selected) | one block |
| native | 43 | 0.7025 | 0.5317 | 45.7% | 83.0% | one block |
| native | 44 | 0.6516 | 0.4990 | 46.2% | 43.3% | one block |
| native_pkg | 42 | 0.5543 | 0.5306 | 4.1% | 17.9% | both blocks |
| native_pkg | 43 | 0.5439 | 0.5299 | 1.7% | 12.1% | both blocks |
| native_pkg | 44 | 0.5594 | 0.4990 | 3.0% | 13.0% | both blocks |
| native_stocklabels | 42 | 0.6259 | 0.5651 | 23.1% | 4.4% | both blocks |
| native_stocklabels | 43 | 0.5764 | 0.5536 | 0.0% | 12.6% | both blocks |
| native_stocklabels | 44 | 0.5576 | 0.5576 | 0.0% | 18.5% | both blocks |

Arm medians (P1 / P2 / SciPy share / DIVERGE share): evox_stock 0.709 / 0.550 / 62% / n/a; native
0.652 / 0.525 / 46% / 43%; native_pkg 0.554 / 0.530 / 3% / 13%; native_stocklabels 0.576 / 0.558 / 0% / 13%.
SciPy share = share of distinct candidates importing SciPy; DIVERGE share = share of iterations whose policy
selected DIVERGE (remainder REFINE or no label). evox_stock s43 has no causal-valid candidate (P2 none); its
P2 median is over two runs.

| Rule | Condition | Result |
|---|---|---|
| Gate | evox_stock median P1 ≥ 0.68 | **pass** (0.709) |
| R1 | stocklabels P1 ≥ 0.68 and SciPy ≥ 30% | false (0.576, 0%) |
| R2 | native_pkg within 0.02 of stocklabels | false (0.022 apart; both low) |
| R3 | stocklabels P1 < 0.63 and DIVERGE < 20% | **true** (0.576, 12.6%) |
| R4 | stocklabels P1 < 0.63 and DIVERGE ≥ 20% | false |
| R5 | all P2 medians within 0.03 | false, narrowly (spread 0.033: 0.525–0.558) |

## Interpretation

1. **Labels are not the bottleneck.** All native_pkg and native_stocklabels labels name SciPy in both blocks,
   yet those six runs have SciPy shares of 0–23% and P1 ≤ 0.626. The package-aware port is the worst arm.
2. **Selection is implicated (R3), and the exploratory data agree.** Across the nine native runs, the only two
   with P1 ≥ 0.65 (native s43, s44) are the only two that selected DIVERGE in more than 20% of iterations
   (83%, 43%); both reached about 46% SciPy. Every run with DIVERGE ≤ 18.5% stayed at P1 ≤ 0.626. This
   correlation was not pre-registered and rests on two runs.
3. **Plain native outperformed EXP25's identical `trace_exp24` arm** (median 0.652 against 0.586) only
   because its learned policy happened to favour DIVERGE in two seeds. native s42's policy never selected a
   label at all (4 meta failures). This variance is itself a selection effect.
4. **R5 narrowly fails (spread 0.033).** Under causal filtering, native_stocklabels is best (0.558) and
   evox_stock is 0.550. The causal ranking is not EvoX's raw lead. With three runs per arm, this is consistent
   with EXP25's look-ahead finding, not a refutation of it.

## Next

EXP27 should vary label *selection* with labels held fixed (stock labels injected): the native learned policy,
against a fixed DIVERGE quota (e.g. ≥ 40%), against stock EvoX's selection. P2 should be the primary
endpoint, given EXP25's look-ahead result.
