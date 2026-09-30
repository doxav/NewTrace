# EXP-16 — diagnostic investigation, not a replacement confirmation

Opened 2026-09-09 from EXP-15 completion commit
`13ebda2242e1c18022591737b113030ca2ce2da2`, on
`codex/investigation-feedback-exp16`. Initial worktree: only the user's untracked
`artifacts/probe_2026/probe_aa_results.json`. Preserve that file and all EXP-15
evidence. No push, merge, PR, contact, or new production dependency. This inquiry
does not change H15-B's inconclusive result.

## Question and evidence standard

Which mechanisms limit useful training feedback, and which interventions improve
generation reliability, search efficiency, or generalization? A mechanism observed
in code is distinguished from a causal intervention result. A successful
engineering intervention is not evidence of an A2 performance advantage. New
diagnostic outcomes are exploratory; any final superiority claim needs a fresh
registered confirmation, including new task instances and outer seeds.

Investigate separately:

1. Completion/reasoning cap and upstream routing; invalid-response opportunity loss.
2. Objective information: early versus final performance; raw scales; loss of trace.
3. Six task trajectories versus one trainer row; training instance count and local
   optimizer replication; train/validation ranking and selection instability.
4. Benchmark headroom, central optimum prior, initialization-dominated AUC, surface
   complexity, and specialization versus generalization across families/dimensions.
5. Proposal count, effective diversity, incumbent ancestry, mutation without score
   feedback, greedy versus population/archive search, and meta-policy transfer.
6. Sampling uncertainty at five outer seeds and realistic effect-size scenarios.
7. Historical routing/code/prose gains, their original controls and retractions,
   followed by deterministic offline reproduction where the environment permits.

Independent read-only EXP-15 audits and historical reproductions may proceed in
parallel. No new model is selected because it makes A2 win. Use existing production
Trace for subsequent iterative search tests; fixed-context generation probes below
are mechanism probes, not a substitute recursive arm.

## Stage G1 — frozen token-cap intervention

Before any live G1 call, preserve exact requests, source hashes, environment, code
hashes and this protocol's hash in `generation/freeze.json`. This stage contains
24 completed responses: six blocks × two fixed prompt contexts × two caps.

- Model: `deepseek/deepseek-v4-flash-0731`, OpenRouter, existing `make_live_llm`.
- Temperature 0.6, top_p 1.0; native extra_body reasoning effort low; timeout 300 s;
  concurrency 1; cache false; empty retries 0; wrapper attempts 1; client retries 0.
- Intervention: max_tokens **8000 versus 32000**, with every other request field
  identical within a block/context pair. Request seeds 16001 through 16006 are
  accepted hints, not assumed deterministic common random numbers.
- Context I: exact EXP-15 independent prompt and handwritten seed.
- Context L: exact EXP-15 iterative prompt format, with the previously
  validation-selected representative (outer 41, source 1684f91a...) as current
  source, its newly evaluated sparse training feedback, and the unchanged seed as
  previous attempt. This fixed, complex parent exercises longer iterative requests.
  No EXP-15 evaluation result is supplied.
- Context/cap order: alternate contexts across blocks, alternate cap-first order
  within each context/block. Retain every completed response including invalidity.
- Fresh training tasks: six balanced family/dimension strata generated using
  SHA256 namespace `EXP-16/G1`, split `train`, instance index 0. Same transformations
  and ranges as EXP-15; semantics checked disjoint from every earlier split.
- Evaluate exact generated sources on all six tasks at B=32 and common local
  seeds, using the unchanged subprocess evaluator. Invalidity remains typed; no
  numerical imputation. Preserve every partial trajectory and allocation.
- Main G1 outcomes: response truncation, source validity, full-panel eligibility,
  completion/reasoning tokens, latency/cost. Conditional candidate performance is
  descriptive and does not choose the token cap. Report all 12 paired contrasts.
- No scientific retries. Initial call plus at most three transient transport
  retries at 2/4/8 s; record every attempt and possible duplicate billing. Completed
  responses cannot be replaced. Uncertain in-flight calls block that slot pending
  reconciliation. Provider unavailability does not permit model substitution.
- Cap recommendation: prefer 32000 only if it reduces truncation or invalidity;
  retain uncertainty from n=12/context heterogeneity. Even zero observed failures
  does not establish near-perfect reliability. If no improvement, report that.

Public model metadata on 2026-09-09 advertises support for `max_tokens` and low
reasoning effort. Actual requests/receipts, not advertised limits, verify use.
Reasoning consumes the output-token budget according to OpenRouter's documentation:
https://openrouter.ai/docs/guides/best-practices/reasoning-tokens

## Later stages — register exact interventions before running them

- Offline fixed-policy headroom and selection experiments on new tasks, including
  different instance/local-seed panel sizes and central-versus-wide shifts.
- Matched generation ablations distinguishing objective instruction, trajectory
  summaries, feedback quantity, and code-only mutation. Change one factor per
  contrast; equal response budgets and generation settings; retain negative results.
- Actual production search pilot comparing promising feedback with independent
  generation, followed by a search-strategy ablation if justified. Separate
  discovery diagnostics from independent validation of chosen modifications.
- Final recommendations state supported mechanisms, rejected hypotheses, remaining
  uncertainty, exact implementation changes and conditional projected gains. Do
  not promise a recursive advantage which the evidence has not established.

Each later stage gets an amendment and immutable request/config freeze before its
execution. G1 settings cannot be silently edited to absorb a disappointing outcome.
