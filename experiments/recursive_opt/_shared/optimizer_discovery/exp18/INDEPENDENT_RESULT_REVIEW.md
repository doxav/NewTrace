# EXP-18 independent result review

**PASS within the read-only integrity, selection, aggregation and paired-analysis scope described here.** All independently recomputed values agree with the completed frozen analysis. This is separate from the pipeline’s pending `verify_numerics` result; it does not declare that stage passed.

Independent calculation: 2026-09-12T19:05:35.916104+00:00 to 2026-09-12T19:06:20.760835+00:00. Official-result comparison and final input recheck: 2026-09-12T19:13:14.621095+00:00.

The study is an exploratory six-seed mechanism experiment. The seven registered mechanism contrasts are all inconclusive under the frozen interval rules. Memory and the Pareto parent policy have not demonstrated an advantage here. All selected deployments completed audit without fallback. This result neither disproves useful feedback under other conditions nor supplies the missing independent-generation comparison at N=16.

## Method and authenticated scope

A separate standard-library-only calculation read the preserved EXP-18 files. It did not invoke the production analysis, pipeline, numeric verifier, benchmark objective, candidate runner, model client, credential loader or metadata collector. No EXP-17 performance was read. The production analysis implementation was inspected to identify the frozen equations and output contract, then the arithmetic was implemented independently. Only this review document was created in the repository.

- Original main freeze: `ce88c4b2d9530456e63b497f7bbde7608fbf3d8aca441c13c086d77b6beeea8a`; all 113 frozen files matched their byte hashes.
- Exact grid: outer seeds 18011, 18023, 18037, 18041, 18053, 18067; L/M/P/PM each receive 16 completed responses. Every seed includes A0/B2 and all four selected deployments.
- Model: OpenRouter `deepseek/deepseek-v4-flash-0731`; temperature 0.6, top_p 1, max_tokens 32000, native low reasoning, timeout 300, client retries zero. Recorded requests match the frozen settings; generation is stochastic, not established as deterministic by request seeds.
- 384 distinct completed responses and all their source hashes, requests, TRAIN receipts and attempts were retained; 399 recorded attempts comprise 384 completions and 15 transport failures. All 15 failures retain possible-remote-completion/duplicate-billing uncertainty. No unpaired start or replacement response was found.
- All 2,244 generation-seal members and 24 selection-seal members match their canonical hashes. The generation seal is byte-identical to the earlier independent `operational_status_checks/generation_integrity_001.md` audit, preserving its exact prompt reconstruction, TRAIN-only memory cutoffs and Pareto decision provenance. This result review does not rerank Pareto frontiers or rerun generation.
- Every cached validation row follows the global generation barrier; every audit row follows the global selection barrier. All response/receipt and selection timestamps satisfy the registered order. Each arm’s validation-only representative was frozen before audit.
- All 27,000 physical cache rows passed full frozen-key, row-hash, source/task/local-seed/budget, split, stratum and timestamp checks. Every logical pool/audit row maps to that authenticated cache; there are no missing or unallocated physical rows.
- Every valid curve contains 32 finite nonnegative, nonincreasing regrets. Its AUC, final regret, target attainment and censored hitting time were independently checked. Invalid trajectories retain no numeric metric. This check starts from preserved regret curves; it does not re-execute objectives or independently regenerate normalization reference values. The pipeline’s numeric verifier covers that separate scope.
- Each of the 24 pools contains the unchanged seed and all 16 slots. Eligibility was recomputed from all 48 TRAIN and 24 validation trajectories; selection was recomputed by validation AUC, then earliest index with seed index -1. Every selected source and representative matches.

## Statistical specification and interpretation

Each trajectory AUC is the arithmetic mean of its 32 preserved normalized best-so-far regrets. Instances and two local seeds are averaged within each family/dimension stratum; the six strata receive equal weight. The six outer-seed AUCs are the replication units. The shared task panel is conditioned on; neither tasks nor time points are treated as additional independent outer replications.

For each contrast, its six paired deltas were formed first. A single `random.Random(1515)` stream generated 10,000 samples of six outer-seed indices with replacement, reused jointly across all 13 contrasts. Each sample statistic is the mean of the resampled paired deltas. Percentiles 2.5 and 97.5 use linear interpolation at position `(10000-1)*percentile/100`. This independently reproduces the frozen per-contrast implementation, whose identical RNG reset yields the same joint index draws. No parametric model, p-value or additional replication was introduced.

An interval wholly below zero receives “positive signal”; wholly above zero “negative signal”; all paired deltas exactly zero “no detectable difference”; otherwise “inconclusive”. These are marginal exploratory/descriptive labels, with no simultaneous-coverage or multiplicity guarantee. At n=6 the intervals are fragile. Negative pairwise differences favor the first arm; the interaction sign describes departure from additivity, not superiority of PM over any arm.

The protocol registers four simple effects plus three factorial effects as its seven exploratory mechanism contrasts. The frozen JSON uses `descriptive_secondary` for the four simple effects and `exploratory_factorial` for the three factorial summaries. The crosswalk below preserves both design meaning and actual frozen machine labels. The six additional contrasts are descriptive secondary. No contrast here is confirmatory.

## Deployment results: normalized regret-AUC, lower is better

| Outer seed | A0 | B2 | L | M | P | PM |
| --- | --- | --- | --- | --- | --- | --- |
| 18011 | 0.12622639 | 0.04443282 | 0.02221177 | 0.01565259 | 0.06741975 | 0.10463970 |
| 18023 | 0.14757066 | 0.03889671 | 0.03742249 | 0.02130065 | 0.08541558 | 0.03617325 |
| 18037 | 0.18356567 | 0.05356440 | 0.02027619 | 0.01478713 | 0.02151761 | 0.01986699 |
| 18041 | 0.13540358 | 0.03881975 | 0.04274980 | 0.04182675 | 0.04888749 | 0.01420812 |
| 18053 | 0.14537057 | 0.03460057 | 0.06835238 | 0.05308577 | 0.04008202 | 0.01752647 |
| 18067 | 0.12760659 | 0.03560652 | 0.01921713 | 0.05923854 | 0.01832463 | 0.04420857 |

| Arm | Mean | Median | Audit target attained /144 | Fallback /144 |
| --- | --- | --- | --- | --- |
| A0 | 0.14429058 | 0.14038707 | 86 | 0 |
| B2 | 0.04098679 | 0.03885823 | 93 | 0 |
| L | 0.03503829 | 0.02981713 | 135 | 0 |
| M | 0.03431524 | 0.03156370 | 114 | 0 |
| P | 0.04694118 | 0.04448476 | 121 | 0 |
| PM | 0.03943718 | 0.02802012 | 129 | 0 |

All 864 audit trajectories are valid, each with exactly 32 actual objective evaluations. Candidate-invalid and fallback counts are zero for every seed and every arm. A0 and B2 are fixed controls evaluated on audit; their favorable or unfavorable values were not used to select generated programs.

## All frozen contrasts

| Contrast | Design role / JSON role | Mean delta | Median delta | 95% paired interval | Frozen label |
| --- | --- | --- | --- | --- | --- |
| M-L | registered mechanism / descriptive_secondary | -0.00072305 | -0.00602412 | [-0.01243504, 0.01690947] | inconclusive |
| P-L | registered mechanism / descriptive_secondary | 0.01190289 | 0.00368956 | [-0.00849110, 0.03276106] | inconclusive |
| PM-P | registered mechanism / descriptive_secondary | -0.00750400 | -0.01210308 | [-0.03186697, 0.01714156] | inconclusive |
| PM-M | registered mechanism / descriptive_secondary | 0.00512194 | -0.00497506 | [-0.02271766, 0.04189241] | inconclusive |
| memory | registered mechanism / exploratory_factorial | -0.00411353 | -0.01068552 | [-0.02065131, 0.01490973] | inconclusive |
| pareto | registered mechanism / exploratory_factorial | 0.00851241 | -0.00240030 | [-0.01501853, 0.03568221] | inconclusive |
| interaction | registered mechanism / exploratory_factorial | -0.00678094 | -0.01071321 | [-0.02575737, 0.01568842] | inconclusive |
| PM-L | additional / descriptive_secondary | 0.00439889 | -0.00082922 | [-0.02673227, 0.04055406] | inconclusive |
| L-B2 | additional / descriptive_secondary | -0.00594850 | -0.00893180 | [-0.02178283, 0.01206921] | inconclusive |
| M-B2 | additional / descriptive_secondary | -0.00667155 | -0.00729453 | [-0.02495067, 0.01232324] | inconclusive |
| P-B2 | additional / descriptive_secondary | 0.00595439 | 0.00777460 | [-0.01415898, 0.02652452] | inconclusive |
| PM-B2 | additional / descriptive_secondary | -0.00154961 | -0.00989878 | [-0.02273594, 0.02580366] | inconclusive |
| B2-A0 | additional / descriptive_secondary | -0.10330378 | -0.10262889 | [-0.11648174, -0.09137689] | positive signal |

Memory = ((M−L)+(PM−P))/2; Pareto = ((P−L)+(PM−M))/2; interaction = PM−P−M+L. The sole wholly negative interval above is the descriptive B2−A0 fixed-control comparison. None of the seven mechanism intervals excludes zero; numerically ordered arm means do not establish superiority.

All paired deltas, in the registered seed order:

| Contrast | 18011 | 18023 | 18037 | 18041 | 18053 | 18067 |
| --- | --- | --- | --- | --- | --- | --- |
| M-L | -0.00655918 | -0.01612184 | -0.00548905 | -0.00092305 | -0.01526661 | 0.04002141 |
| P-L | 0.04520798 | 0.04799309 | 0.00124142 | 0.00613769 | -0.02827036 | -0.00089250 |
| PM-P | 0.03721994 | -0.04924234 | -0.00165062 | -0.03467938 | -0.02255555 | 0.02588394 |
| PM-M | 0.08898711 | 0.01487260 | 0.00507986 | -0.02761863 | -0.03555930 | -0.01502997 |
| memory | 0.01533038 | -0.03268209 | -0.00356983 | -0.01780121 | -0.01891108 | 0.03295268 |
| pareto | 0.06709755 | 0.03143285 | 0.00316064 | -0.01074047 | -0.03191483 | -0.00796124 |
| interaction | 0.04377912 | -0.03312049 | 0.00383844 | -0.03375633 | -0.00728894 | -0.01413747 |
| PM-L | 0.08242793 | -0.00124924 | -0.00040920 | -0.02854168 | -0.05082591 | 0.02499144 |
| L-B2 | -0.02222105 | -0.00147422 | -0.03328821 | 0.00393005 | 0.03375181 | -0.01638939 |
| M-B2 | -0.02878023 | -0.01759606 | -0.03877726 | 0.00300700 | 0.01848520 | 0.02363203 |
| P-B2 | 0.02298693 | 0.04651887 | -0.03204679 | 0.01006774 | 0.00548145 | -0.01728189 |
| PM-B2 | 0.06020688 | -0.00272346 | -0.03369740 | -0.02461163 | -0.01707410 | 0.00860206 |
| B2-A0 | -0.08179357 | -0.10867395 | -0.13000127 | -0.09658383 | -0.11077000 | -0.09200007 |

## Invalidity and retained search outcomes

All 384 response slots are counted. Static invalidity and trajectory failures remain separate. An ineligible generated program was not manually repaired or replaced. Every selected program is generated; no arm selected seed index -1.

| Arm | Static invalid /96 | TRAIN+validation eligible /96 | Statically valid but ineligible | Invalid TRAIN rows /4608 | Invalid validation rows /2304 |
| --- | --- | --- | --- | --- | --- |
| L | 11 | 82 | 3 | 648 | 324 |
| M | 5 | 91 | 0 | 240 | 120 |
| P | 5 | 81 | 10 | 633 | 319 |
| PM | 2 | 82 | 12 | 432 | 216 |

Across arms: 23 statically invalid responses (17 missing source, four syntax errors, two protocol violations); 25 further statically valid responses are execution-ineligible; 336/384 generated programs are fully eligible. The source/trajectory distinction is not concealed by successful deployment.

| Seed / arm | Eligible /16 | Static invalid /16 | Invalid TRAIN /768 | Invalid validation /384 |
| --- | --- | --- | --- | --- |
| 18011/L | 11 | 5 | 240 | 120 |
| 18011/M | 14 | 2 | 96 | 48 |
| 18011/P | 12 | 1 | 192 | 96 |
| 18011/PM | 5 | 0 | 288 | 144 |
| 18023/L | 14 | 1 | 72 | 36 |
| 18023/M | 16 | 0 | 0 | 0 |
| 18023/P | 14 | 1 | 96 | 48 |
| 18023/PM | 15 | 1 | 48 | 24 |
| 18037/L | 14 | 1 | 96 | 48 |
| 18037/M | 16 | 0 | 0 | 0 |
| 18037/P | 13 | 1 | 144 | 72 |
| 18037/PM | 16 | 0 | 0 | 0 |
| 18041/L | 13 | 2 | 144 | 72 |
| 18041/M | 15 | 1 | 48 | 24 |
| 18041/P | 14 | 1 | 96 | 48 |
| 18041/PM | 14 | 1 | 96 | 48 |
| 18053/L | 15 | 1 | 48 | 24 |
| 18053/M | 15 | 1 | 48 | 24 |
| 18053/P | 14 | 1 | 56 | 28 |
| 18053/PM | 16 | 0 | 0 | 0 |
| 18067/L | 15 | 1 | 48 | 24 |
| 18067/M | 15 | 1 | 48 | 24 |
| 18067/P | 14 | 0 | 49 | 27 |
| 18067/PM | 16 | 0 | 0 | 0 |

## Resource accounting

| Schedule | TRAIN trajectories | Validation trajectories | Audit trajectories | Actual objective calls | Unused objective allocation | Subprocess executions |
| --- | --- | --- | --- | --- | --- | --- |
| Logical seed-inclusive | 19584 | 9792 | 864 | 880741 | 86939 | 1762830 |
| Unique physical cache | 17424 | 8712 | 864 | 804709 | 59291 | 1610766 |

The registered allocation is 30,240 logical trajectories and 967,680 objective-call slots. Identical-source caching yields 27,000 physical trajectories and 864,000 physical objective allocations, of which 804,709 calls completed and 59,291 remained unused. Logical sums count repeated allocations; they are not additional physical work. Reference normalization preparation, integrity recomputation and interrupted unpersisted overhead are outside these cache call totals. Unknown failed-request billing remains unknown, not zero. This review did not collect new provider metadata.

## Validation-only representatives

| Arm | Outer seed | Slot index (zero-based) | Exact selected source SHA-256 |
| --- | --- | --- | --- |
| L | 18067 | 15 | `66a330a63f1af3e24c83241fa1fc955f4859f28f1318cf3d8f3238dfc9a951e0` |
| M | 18037 | 15 | `d5b3ea6d1529815bfe9524accb95854d55e80b0be6ccfe8d4a9ff381920b7e23` |
| P | 18067 | 12 | `0bc881c4b03dc545f4a9fbf7fa2dbca07c940598dc86e55e73dda9ed50da1c0c` |
| PM | 18041 | 9 | `a787c80c41dbf44a073a0dcf3874d6237e64aabc6db9a2996e760fd4d0c13ced` |

Each exact source is preserved in `run/raw/<outer>/<arm>/selection.json` and its original `slot_<index>/response.json` (ordinary JSON or losslessly compressed `.json.gz`). All 24 selection hashes were checked, not only these four representatives. The registered overall representative is PM/18041/slot09, chosen using validation before audit; it is not the audit-best program selected after seeing outcomes.

## Input hashes and official comparison

The independent calculation tracked 30,207 input files, including all relevant frozen sources, sealed generation records, selected pools, cache records and attempts. Their byte hashes matched before/after the calculation and again during official comparison. Canonical path→byte-hash inventory digest: `8a7bb8ce95f87797b79a1fbcf9c486f9985fce688f11f5de1c6a6a2e1dcc38a6`. The official analysis file was hashed on entry and again after comparison; it was unchanged. Runtime journals and newly written verifier outputs are excluded from this sealed-input stability claim.

| File | Byte SHA-256 | Stability |
| --- | --- | --- |
| run/freeze.json | `9abb465cda1ea55c9cce7e5e784621a9354033c1f6f4c82960f2b3ce917354be` | same at all checks |
| run/generation_frozen.json | `7a8186436ea6933ae36b78025b3aad157db9dee7d7221d3704441e9ffedb9d53` | same at all checks |
| run/selections_frozen.json | `ded3281b54b46055dca32088bf00216c8c8e1a17a4363d82b73ff762701bb038` | same at all checks |
| run/audit_results.json.gz | `dff97a377f5db8cfda1eb22bb761f4be91c79cc88a4d1eed44d94a0797b964eb` | same at all checks |
| PREREG_EXP18.md | `0f84fe008dff9d0795de0e061e157142a16dc39b80d34e2000b8e53ed49dd07c` | same at all checks |
| run/analysis_results.json.gz | `1654ab1ff03bf1d2713bb3ec6bfe870f603cc8ae6e77b32579a074d20f94925a` | same before/after comparison |

The freeze digest at the start of this review is a canonical-JSON digest; the table lists byte hashes, so those values intentionally differ. All six-seed AUC/final-regret aggregates, arm means/medians, 13 paired delta vectors, 13 percentile intervals and interpretation labels agree with `analysis_results.json.gz` within relative tolerance 1e-12 / absolute tolerance 1e-14. Counts, eligibility, selections, source identities, registered roles, resource allocations and fallback counts matched exactly. No discrepancy, dropped seed, hidden invalidity, changed selection, reporting repair or scientific invalidation was required by this review.

## Remaining limits

This is a registered exploratory test conditional on one modest benchmark and six outer seeds. There is no independent N=16 arm, and L already receives immediate TRAIN feedback. Memory therefore estimates the archive increment over that feedback; P estimates the joint frontier-filtering and uniform-parent policy. No algorithmic novelty, benefit from extra recursion depth, amortization, superiority over external engines or operating-system sandbox isolation follows. The current review neither substitutes for the pipeline numeric verifier nor resolves the root cause of earlier trusted-source inspection or transport interruptions; their preserved evidence and operational audits remain applicable.
