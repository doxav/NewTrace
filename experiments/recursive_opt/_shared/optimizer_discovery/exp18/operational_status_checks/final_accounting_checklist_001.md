# Final operational accounting checklist 001

Read-only review, 2026-09-11. Scope: the five named incident/timeout/suspension
records below and the frozen final-analysis path, not a census of later live
events. No comparative efficacy, model/candidate execution, tests, network calls,
credential access or process intervention. Only this new report was written.

**Conclusion:** these observed incidents can be represented honestly by the
existing frozen analysis plus an operational appendix. No omission of a completed
response or change to objective, selection or proposal-slot semantics was found
in this review. There is one ambiguous time-field label and several reporting
coverage limits; none establishes a scientific invalidation.

## Checklist for the final report

| Report item | Exact authority and required interpretation |
| --- | --- |
| Completed responses versus attempts | Use `resources.allocated_proposal_slots`, `completed_responses`, `transport_attempts`, `transport_failures`. They are distinct counts. Final loading requires every registered response, contiguous numeric attempts, one matching successful attempt per slot and no unmatched start. [Analysis:125](../../exp17/analysis.py#L125), [570](../../exp17/analysis.py#L570), [641](../../exp17/analysis.py#L641). |
| Known completed-response tokens/cost | Use `resources.known_usage.usage.<field>.reported_sum`, `reported_responses`, `missing_responses`, for `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `total_tokens`, `cost_usd`. Preserve `response_usage` and `known_usage.field_sources`. Reasoning is separate, not added to completion or total. Missing is unknown, not zero. [Usage fields:58](../../evidence.py#L58), [receipt fallback:255](../../investigation16/production/analysis.py#L255). |
| Receipt coverage | `resources.known_usage.provider_receipts`; missing count is `completed_responses - provider_receipts`. The matching receipt ID must equal the response ID. Only absent response fields are filled from receipt `tokens_prompt`, `tokens_completion`, `native_tokens_reasoning`, `total_cost`; no automatic reconstruction of a missing `total_tokens`. Native tokenizer counts and upstream inference cost must not be silently added/substituted. [Receipt fallback:257](../../investigation16/production/analysis.py#L257). |
| Failed-request billing uncertainty | Report `resources.possible_remote_completion_or_duplicate_billing_attempts` and `transport_failure_usage`, alongside exact failed-attempt paths. Current reviewed failures have no usage or usable pending provider IDs. Known completed cost is not total verified spend; do not report failed requests as free. A later successful response does not establish the earlier attempt was unbilled. [Analysis:264](../../exp17/analysis.py#L264), [incident 001:3317](incident_001.json#L3317). |
| Auxiliary metadata requests | Count saved `raw/.../slot_*/metadata_attempts/*.json` failures separately from generative attempts. Preserve `error_type`, `http_status`, `id`, and receipt presence. These failures and receipt `generation_time`/`latency` are **not summarized** by frozen `_resources`; an appendix must derive them from raw metadata, without changing scientific values. The collector requires a completed response ID and cannot reconcile an unknown failed request by local slot name. [Collector:70](../../investigation16/production_driver.py#L70). |
| Monotonic client duration | `resources.attempt_wall_s_reported` sums attempt `wall_s`, which is actually calculated with `time.monotonic()`. Report it as **summed monotonic client-attempt seconds including shared-lock queueing**, not civil elapsed or provider compute time. It excludes between-attempt retry sleeps. A response's own `wall_s` covers its successful attempt only. [Timer:124](../../investigation16/generation.py#L124), [failure:141](../../investigation16/generation.py#L141), [success:185](../../investigation16/generation.py#L185), [queue:47](../../exp17/driver.py#L47). |
| Whole-slot and phase duration | Use `timing.generation_slots`, and `timing.generation/selection/audit`: `start`, `end`, `elapsed_s.wall_ns`, `elapsed_s.monotonic_ns`, `elapsed_s.boottime_ns`, `suspend_s_estimate`, `clock_reset_detected`. Despite their `_ns` names, entries **inside `elapsed_s` are seconds**. Preserved start clocks include explicit-resume gaps; monotonic time is not CPU or active-process time. Phase generation also includes local work. Do not add these overlapping durations to attempt totals. [Clock arithmetic:145](../../investigation16/search_experiment.py#L145), [persistent start:171](../../investigation16/search_experiment.py#L171), [loaded/output timing:611](../../exp17/analysis.py#L611). |
| Provider duration and timeout | Report raw receipt coverage and documented units separately. The prior client audit established a forwarded 300-second timeout per network-operation category, **not a total request deadline**. Queue wait, repeated reads, DNS/address work and civil suspension cannot be inferred from the total alone. Do not claim hidden retries, dropped timeout, provider progress or token consumption from a long pending slot. [Client audit:121](client_timeout_audit_01.md#L121). |

## Reconcile the observed incidents once

Deduplicate by exact study/outer/arm/slot/attempt path, not by incident snapshot.
Incident 001 overlaps the later snapshots. The union of incidents 002/003 contains
these **five distinct recorded failed attempts**, all retaining the possible
remote-completion/duplicate-billing flag:

| Study and slot | Attempt | Recorded monotonic seconds | `transient` |
| --- | ---: | ---: | --- |
| EXP17 `17001/C/slot_03` | 1 | 1419.203217 | false |
| EXP17 `17001/C/slot_04` | 1 | 1515.547328 | false |
| EXP18 `18011/L/slot_02` | 1 | 1562.294465 | true |
| EXP18 `18011/L/slot_02` | 2 | 0.011085 | false |
| EXP18 `18011/L/slot_03` | 1 | 831.352810 | false |

Sources: [incident 002:5](incident_002.json#L5),
[incident 003:6](incident_003.json#L6). These are minimum already-observed failed
attempt counts, not final counts for the still-running studies. Their flags mean
uncertainty, not five proven remotely completed or charged generations.
`transient=false` records the frozen classifier's decision, not a permanent
network diagnosis. Explicit resumes retain earlier attempts; the four-attempt
limit is **per invocation**, not a lifetime per-slot cap.
[Slot-wrapper semantics:113](../../investigation16/generation.py#L113).

[Suspension 001:2](suspension_001.json#L2) records approximately **6317.53 seconds**
of additional boottime-minus-monotonic time since each pending slot's start. This
supports reporting host suspension separately. Both studies observed the same
host pause; do not add the two observations as distinct pauses or infer a precise
provider-generation state during it. Neither the incident nor the timeout audit
requires retroactively replacing an accepted response.

## Defect classification and final integration

- **Reporting ambiguity, not an arithmetic defect:** `attempt_wall_s_reported` says “wall” but contains
  monotonic client-duration sums. Raw timing and the analysis values remain
  available and correct under the interpretation above. Clarify the label in the
  final prose/export; preserve the frozen field and its values.
- **Coverage limits:** metadata-GET failure counts, per-request provider duration,
  incident narratives and remote-failure reconciliation are not all propagated
  into `analysis_results.json`. Add a separate operational appendix referencing
  the raw records; do not present the aggregate alone as a complete billing or
  elapsed-time account.
- **No established scientific accounting defect:** the reviewed analysis retains
  recorded transport failures and uncertainty, reuses completed responses, and
  rejects unmatched starts at final loading. It can accommodate the documented
  resumes without editing frozen generation/evaluation/selection code. This is a
  static finding about these incidents, not certification of unfinished runs.

The final narrative should therefore say: “Known completed-response usage/cost
is X with stated coverage; Y failed attempts retain unknown remote expenditure;
client monotonic, civil/boottime and reported provider durations are separate;
the documented host pause is included only in the applicable clocks.”

## Exact source anchors

SHA-256 of inspected bytes. Paths below are relative to `artifacts/optimizer_discovery/`.
The `exp17/analysis.py` hash matches both EXP17 and EXP18 main freezes; no analysis
or efficacy routine was executed to establish that identity.

| Artifact | SHA-256 |
| --- | --- |
| `exp17/analysis.py` | `204e43f8f62de913538804d506ebcc1a040102f84e85ddee8b8ae30a3e08bb04` |
| `investigation16/production/analysis.py` | `a5b2c759b7ce2c491c76820acc7ed1c12ef0138a538cc658dca6bead3097656e` |
| `investigation16/generation.py` | `6a5a66b9b12fd4f4217668a5803bd92ad660a5e35d78a42cd45ec7882dd6d24c` |
| `exp17/driver.py` | `be19f5bc162e1f4891c81a2e68825bb3873171363c2ebfe0b11b8201b38b0236` |
| `investigation16/production_driver.py` | `dd74df50ae466a119ef080e62bce5116ea62399d91e0f4240baec3fcdf722d16` |
| `investigation16/search_experiment.py` | `9050d74c9a41aea1822cc353fba6b829c8cda0b5de9ffbd54646e4cd7aa12e3e` |
| `evidence.py` | `23320a4a44598a363d58623f95a88762fb6c6a4fef4ed09afd510461719fe36a` |
| `exp18/operational_status_checks/incident_001.json` | `b0b91f24afaed0f39904471dc5e625200ed76ee762b482646bc51114adf4b3ea` |
| `exp18/operational_status_checks/incident_002.json` | `db6e38471a3c099a527ffe8a54fa5bca0c3023311a0d1edcb3b972d66ec5fa37` |
| `exp18/operational_status_checks/incident_003.json` | `393041c983cafbf18cae78e3e9579b28db2f8611cd59d725ff4287bddbe53772` |
| `exp18/operational_status_checks/client_timeout_audit_01.md` | `8b19d15db9d87ec2eb61bdb9483ad0d02308b4f987b5755f62d866c06119792c` |
| `exp18/operational_status_checks/suspension_001.json` | `3b8a8716038402f9693091d1a6323734a32059ae38df9194fc07e5ab0021269a` |
