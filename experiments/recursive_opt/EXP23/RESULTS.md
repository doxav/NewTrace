# EXP23 — corrected results

## Contents

- [Completed native PRISM runs](#completed-native-prism-runs)
- [Why the positive headline is withdrawn](#why-the-positive-headline-is-withdrawn)
- [Other EXP23 evidence and current links](#other-exp23-evidence-and-current-links)

## Completed native PRISM runs

Updated 30 September 2026 from both terminal summaries in `results/prism100_v2/20260929T215903/`. Both arms completed 100 solution attempts. This compares proposers inside the same native coevolution engine; it is not a replicated end-to-end stock EvoX comparison.

| Proposer | Solution attempts | Terminal status | Best stock score | Success rate of stock-selected program | Recorded solution / meta / feedback calls |
|---|---:|---|---:|---:|---|
| Trace | 100 | success | 30.876583 | 3/50 = 0.06 | 100 / 4 / 9 |
| EvoX-style `llm_rewrite` | 100 | success | 26.203272 | 50/50 = 1.00 | 100 / 5 / 11 |

Sources: [Trace report](results/prism100_v2/20260929T215903/trace/report.json), [Trace summary](results/prism100_v2/20260929T215903/trace/summary.json), [rewrite report](results/prism100_v2/20260929T215903/llm_rewrite/report.json), [rewrite summary](results/prism100_v2/20260929T215903/llm_rewrite/summary.json). Summary role counts exclude transport retries; use each run's `calls.jsonl` for HTTP-attempt accounting.

## Why the positive headline is withdrawn

**30.876583 is a metric exploit, not evidence of a better algorithm.** Stock PRISM averages KVPR over solved cases only, then adds success rate. The selected Trace program fails on 47/50 cases, removing difficult cases from that average. Re-evaluating all 50 cases, assigning failed cases the initial placement's KVPR, gives **21.084775**, below the initial program's **21.891622**. This was independently reproduced during the assessment audit. [Selected-source re-scoring](../_analysis/assessment_20260930/prism_rescoring.json), [white-box evaluator](../EXP24/prism/whitebox.py).

The reported best *fully solved* candidate anywhere in Trace's archive scores about **26.233**, versus **26.203** for `llm_rewrite`. Retrospective selection from the archive is different from re-scoring the program actually selected by the flawed metric. One run each and a small difference establish neither superiority nor equivalence; “tie” is too strong as a statistical claim. The full Trace archive re-scoring attempt in this audit timed out after 180 seconds, so its 26.233 maximum remains attributed to the [EXP24 corrected report](../EXP24/RESULTS.md); the [rewrite archive re-scoring](../EXP24/results/analysis/exp23_llm_rewrite_validity.json) is saved separately.

The computed all-case optimum is **26.2559717495** on the 50 fixed PRISM cases, using exhaustive feasibility search with numerical bisection. These two reported archive maxima fall below it. This is not an unseen-task generalization result. [Exact-search implementation](../EXP24/scripts/prism_exact.py), [saved per-case optimum](../EXP24/results/analysis/prism_exact_optimum.json).

The earlier 20:35 and 21:43 UTC snapshots accurately recorded partial raw-score leads but cannot support a scientific advantage after this correction. Their observations are preserved: [intermediate checkpoint](../_analysis/synthesis_20260929/exp23_intermediate.json), [follow-up checkpoint](../_analysis/synthesis_20260929/exp23_followup.json).

## Other EXP23 evidence and current links

[Historical simulator and live analyses](docs/ANALYSIS.md) retain their original scope. The 120 live policy-model calls used simulated task evaluation; they were not a live task-search efficacy comparison. Simulator credit-assignment findings and native engine integration remain useful evidence, but neither rescues the PRISM headline. Statements that native engine support is missing are version-specific and superseded.

- [Reconciled assessment and ranked lessons](../ASSESSMENT.md)
- [EXP24: corrected guide, feedback, projection and fixed-policy control](../EXP24/RESULTS.md)
- [EXP22-EvoX: earlier hybrid comparison](../EXP22/evox/RESULTS.md)
- [Saved native runs](results/prism100_v2/20260929T215903/)
- [Simulator score validation](results/score_validation.json) and [meta comparison](results/meta_comparison.json)
- [Live-policy schedule](results/live_schedule/live/summary.json)
- [Evidence locations](results/README.md) and [supporting documents](docs/README.md)
