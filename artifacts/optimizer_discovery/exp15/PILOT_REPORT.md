# EXP-15 engineering pilot — separate from confirmation

Pilot outer seed 701, two completed responses per arm; separate `pilot` task domain,
6 train / 6 validation / 12 holdout instances, B=32. Scientific implementation at
81cf12c0; original `pilot_01/results.json` is preserved. Subsequent descriptive
reporting adds counts without changing any primary value or selection.

## Observations

| Check | Evidence |
|---|---|
| Real generation | 4 completed responses, exact DeepSeek model, 5 transport attempts |
| A1 | one syntax error, one executable candidate; validation selected seed |
| A2 | one 8,000-token all-reasoning response with no code, one executable candidate; validation selected seed |
| Production path | two actual Trace updates through registered Control Plane engine, test mode false |
| Execution | both generated replacements completed every required train/validation trajectory |
| Selection / deployment | seed selected in both pools; all three pilot deployment AUCs 0.1878731013; no candidate deployment fallback |
| Allocations | 36 search trajectories / 1,152 objective allocations per generative arm; 384 holdout allocations per arm |
| Actual work | 1,536 shared objective calls, 3,072 subprocess launches, 768 unused unique allocations |
| Internal schedule | 222 evaluation requests, 150 deterministic cache hits; no extra LLM proposals |
| Isolation / source identity | `pilot_01/integrity.json`: passed; all global selections preceded holdout |
| Provider receipts | Venice, Relace, DigitalOcean, OpenInference; same unpinned routing policy |

The A2 first request had a transport timeout, followed by the registered retry.
Its error receipt records 436.003 seconds; the client timeout configuration was 300s.
The interrupted request has no returned generation ID, so remote completion/billing
cannot be reconciled. Possible additional billing remains unknown. No completed
response was replaced. The retry's empty completion consumed its scientific slot.

Reported completed-response usage: 4,940 prompt + 21,848 completion = 26,788 total
tokens; reasoning 19,742 (included in completion); cost USD 0.005411162024.
Provider receipt totals agree up to rounding. Transport-failure usage is unknown.
Completed-response durations: 24.72, 112.85, 280.45, 128.99 seconds.

## Independent headroom diagnostics

Fixed policies were chosen before comparative pilot outcomes. On the six pilot
training tasks, normalized regret AUCs were seed 0.1476422397, uniform 0.1753414678,
and midpoint 0.1115573779. The range is 0.0637840900 and the seed has residual
regret; this benchmark distinguishes behavior without saturation at zero.
This does not establish that either generative arm can find better behavior.
No setting was selected because A2 won: both pilot searches retained the seed.

Ten uncached replays of the same seed/task/local-seed trajectory were identical:
AUC range [0.14273211461738702, 0.14273211461738702]. Diagnostics used 896 objective
calls plus 768 normalization-reference evaluations in their separate process.
`pilot_diagnostics.json` retains all trajectories. Pilot search benchmark reference
preparation allocates another 3,072 objective evaluations, separate from arm budgets.

## Confirmatory feasibility and decision

Retain all original scientific choices. Confirmation allocates 80 completed model
responses, at most 640,000 configured completion tokens, 17,280 search objective
allocations per generative arm, and 1,920 holdout allocations per arm: 40,320 total
logical search/deployment objective allocations, plus 3,072 shared normalization
reference evaluations. A0's search seed evaluations are shared with A1/A2 pools.
Caching shared seed evaluations reduces maximum unique scientific objective calls
to 38,400; valid proposal replay requires at most 76,800 subprocess launches for
those trajectories. Invalid attempts can add launcher work but cannot add objective
allocations or scientific opportunities.

Linear completed-response extrapolation: about 535,760 reported tokens and USD
0.10822324, with 3.04 hours of response latency. This is a rough engineering estimate,
not a cost cap: A2 prompt lengths and provider latency vary. The pilot's 235.99s
summed trajectory execution time implies roughly 1.64 hours of local execution at
38,400 calls; the independent diagnostics ran faster. Plan roughly five active
hours plus transport delays, potentially longer. The observed timeout rate would
add substantial time if repeated. No replication reduction is justified by this
small monetary cost. Long wall-clock gaps between pilot attempts also appear in
recorded timestamps; they are not attributed to measured model generation time.

GO for confirmatory execution means the procedure is executable, bounded and
interpretable even if generation remains unreliable. It does not mean H15-A/B is
supported. Keep settings and all five paired outer seeds unchanged.

Before confirmation, added tested environment verification, descriptive missing-
usage accounting, provider metadata collection, and lossless gzip packaging for
JSON records over 450,000 bytes. Original bytes and raw source strings are preserved.
No generation, evaluation, selection, or primary metric semantics changed. Offline
mock crash/resume checks supplement the successful live pilot; no extra pilot LLM
responses were requested to repair invalid outputs.
