**Priorité actuelle — EXP21, 24 septembre 2026 :** réapprovisionner le compte OpenRouter (solde −0,364179 USD) et relever le plafond de clé actuellement 14 USD. Puis reprendre cinq chaînes HTTP 402 et quatre inédites : 33 propositions restent, réponses reçues conservées. 33/42 mesures terminées, 961 tests passent. Combinaisons, O1/O2 et confirmation attendent le développement complet. [État exact et commandes](../../_shared/o1_learning/EXP21.md).

# Execution review and one-hour continuation plan — 16 September 2026

**Campagne précédente du 23 septembre — EXP20 terminée :** [EXP20 avec Qwen](../../_shared/o1_learning/EXP20.md)
a exécuté les six graines, 72 réponses DeepSeek, toutes les évaluations TRAIN /
VALIDATION et les 48 TEST. Exactitude finale : initial 28,47 %, standard 47,92 %,
curriculum 43,75 %. Le contraste principal curriculum−standard vaut −3,37 points,
intervalle apparié [−8,53 ; +2,38] : pas d’avantage établi du curriculum.
Aucune reprise nécessaire pour EXP20. Ne pas relancer ses slots achevés. Les propositions
BBEH ci-dessous sont un plan historique ; la prochaine comparaison ciblée et les
limites d’attribution sont décrites dans EXP20 ; EXP21 a ensuite été autorisée séparément, voir son état ci-dessus.

**Précision du 23 septembre :** la reconstruction S4 révèle à la fois un progrès
TRAIN temporairement défavorable aux compositions VALIDATION et des scores
internes mêlant des populations/lots différents. Avant le transfert proposé
ci-dessous, isoler le choix TRAIN/VALIDATION avec une règle de comparaison propre
(scores séparés, mêmes lots par comparaison, même actualisation). La recette S4
n’est pas encore une recommandation générale de suppression de VALIDATION.
Voir la [précision causale dans EXP19](../../EXP19/RESULTS.md).

**Decision:** EXP-19 is complete and does not need recovery or a wholesale rerun.
EXP-17 is a different, suspended experiment and cannot be completed within one
hour under its frozen protocol. The next useful work is a narrowly scoped test of
whether EXP-19's parent-selection result transfers to a real task. **No model calls
or suspended processes were restarted by this review.**

Evidence: [current EXP-19 report](../../EXP19/RESULTS.md),
[machine-readable review](../../_shared/o1_learning/restart_review_20260916.json),
[research history](../navigation/RESEARCH_LOG.md), [file index](../navigation/RESULTS_INDEX.md).

## 1. Exact saved state and results already reportable

| Work | Checked state | What we can say now | Recovery/rerun needed? |
|---|---|---|---|
| Trace/curriculum engineering | Complete; 47 tests rerun successfully today | The dict changes captured and consumed traces by level. Failed→solved TRAIN observations invoke the buffer and change subsequent batches. | No live rerun. Keep the documented backend/Guide-score limitations. |
| EXP-19-S3 | All 72 responses and three paired seeds present; aggregates reproduce | TEST means: initial 39.58%, standard 47.92%, O1-selected 50.00%; none reaches 90% in 8 responses. No demonstrated acceleration. | No. It is a valid bounded negative/inconclusive result. |
| EXP-19-S4 | All 24 responses, all 12 paired prompt batches, frozen selections and TEST present | 100% TEST on 3/3 seeds in 4 responses versus 72.22% standard mean. Mean delta +27.78 points; descriptive bootstrap [+22.92, +31.25]. | No recovery. Replication on another task is new work. |
| EXP-19 recursive diagnostic | Completed evidence, including empty O2 output | O2 exhausted 16000 tokens; retained configuration equals standard. This does not test a successful distinct recursive treatment. | Fresh bounded diagnostic only if depth becomes the research question; not a replacement response. |
| Earlier S2 interrupted child | Request saved, no response receipt | Remote completion/billing is unknown. The interrupted nesting is not a completed result. | Preserve it; do not blindly resend. Corrected S3 already supplies a separate nested diagnostic. |
| EXP-17 | 545/736 responses: I 272, C 273; 68 complete arm/seed runs; one pending request 17035/C/slot 01 | Partial generation evidence only. No frozen final selections or held-out efficacy result. | 191 responses plus remaining TRAIN, all VALIDATION, selection, audit and analysis. Not this one-hour session. |

The EXP-19 audit was executed again today and is **byte-identical** to the saved
integrity report. The 47 new tests pass again. No new model responses were needed.
Its total remains 255 completed responses, 2,332,712 tokens and USD 0.21856749005
reported cost; unknown interrupted billing is additional. The S4 task remains six
fixed Boolean tables with new compositions, not BBEH or new concepts. The result
was diagnosed manually at O1; it was not automatically discovered by O2.

## 2. What still needs work — without repeating completed science

1. **Transfer:** establish whether selecting parents on TRAIN during fitting,
   followed by separate external VALIDATION selection, helps a less trivial task.
2. **Curriculum efficacy:** activation is proven; a performance benefit is not.
   Test after a useful baseline exists, with actual transition/replay evidence and
   all extra evaluations counted.
3. **Trace efficacy:** S1/S2 had a projection defect; S3 repaired and retested a
   subset. There is no reason to rerun the whole early grid on the same short
   traces. A longer semantic workflow is needed to test the remaining question.
4. **Other axes:** dynamic surfaces, long-horizon credit, broad optimizer/trainer
   choices and useful extra recursion depth remain unvalidated. They cannot all
   be evaluated convincingly in a single hour. Earlier narrow nulls do not refute
   these general ideas.
5. **Execution engineering:** the current study runner refuses partial run
   directories; it does not yet implement a general safe mid-chain resume.
   Its wall-time budget is checked between resource charges, not an external
   interrupt of a blocked request. A hard deadline and resume tests are required
   before claiming the next run is safe within an hour.
6. **Code delivery:** production edits remain uncommitted, as does the user's
   staged IO checkout. The project's pinned hook tools differ from both installed
   Ruff versions. Reconcile the actual delivery toolchain separately; do not
   rewrite evaluated source or suppress warnings to claim universal lint green.

## 3. Why simply adding workers is insufficient

- The machine currently exposes 20 logical CPUs and about 29 GiB available RAM.
  Twelve model-waiting worker processes plus a **shared** pool of 8 local evaluator
  workers are plausible. Twelve processes each spawning eight more workers are
  not the plan. Mutable Trace registries and client state must be process-isolated.
- S3's 72 responses accumulated 56.22 minutes of active request time, with median 29.09 s,
  p90 134.24 s and maximum 298.82 s. S4's 24 responses accumulated 35.38 minutes,
  median 42.15 s and maximum 458.72 s including retries. These are observations, not
  endpoint throughput guarantees.
- An ideal rescheduling of the same six S4 learning chains gives 35.47 min with
  one worker, 12.09 min with three, 11.54 min with six. Twelve cannot improve that
  six-chain critical path. More workers help independent replications; they do
  not parallelize dependent updates within one learner.
- A 300 s timeout with four attempts and 2/4/8 s backoff permits about 1214 s per
  response slot. Four dependent slots can exceed 80 minutes even with many workers.
  A deadline must include **all attempts and waits**, not reset for every retry.
- `BudgetState.elapsed_s` uses `monotonic()`, and enforcement occurs on charge.
  On this Linux host that clock excludes sleep. Use an external supervisor and
  CLOCK_BOOTTIME for the session limit; if the machine sleeps past the deadline,
  checkpoint/stop immediately on wake rather than restart its time allowance.
  Local control cannot guarantee cancellation of a remotely submitted request.

EXP-17 is especially unsuitable for this deadline: concurrency 1 is frozen.
The 545 saved timings imply **16.51 hours for the 191 remaining responses alone** at
the observed mean, excluding subsequent evaluation. Seven frozen source files
now differ. Any eventual continuation needs the archived source/environment in
an isolated checkout, pending-request reconciliation and the unchanged protocol.
Changing its concurrency or dropping seeds would not be a faithful resume.
Do not send SIGCONT to the processes in the changed workspace.

## 4. Recommended next scientific question

**Does the parent-selection intervention transfer to learning an executable
solver for real BBEH Boolean expressions?** Use an unchanged seed, standard
Trace learning, and the same learning path with parent selection on TRAIN;
retain common external VALIDATION selection. A negative result remains useful.

There is a material preparation gap. The existing
[reasoning module](../../multiobjective_reasoning/components.py)
makes **two LLM calls per evaluated example**. The PAL notebook makes one code
request per example. Neither is a cheap local evaluator. For illustration,
12 chains ×5 candidate states ×(24 TRAIN+12 VALIDATION) ×2 forward calls already
requires 4320 forward responses, before final TEST or trainer reevaluations.
Counting only 48 optimizer responses would be a false runtime budget.

For the proposed hour, optimize a portable **solver source**, with local
`solve(question) -> answer` execution and deterministic scoring. Reuse the
existing dataset, control-plane and subprocess mechanisms; add only the narrow
adapter needed for this task. This **changes the artifact from the notebook's
PAL prompt to reusable solver code** and must be a new experiment, not described
as a faithful PAL rerun. A prompt-only PAL experiment needs its own measured,
much smaller all-call budget and is not the recommended first hour.

Before timed execution, require:

- a reviewed seed solver, source validity checks, sanitized bounded subprocess
  execution, exact-answer evaluator, typed failures, and meaningful unit tests;
- cached, hashed official dataset records; inventory earlier-used IDs before
  claiming a fresh holdout;24 TRAIN/12 VALIDATION/24 TEST only if that inventory
  supports it; no reading held-out outputs for engineering;
- TRAIN/development-only feasibility/headroom checks against fixed diagnostic
  policies, without choosing a design because the modified arm wins;
- exact production evaluation accounting in an offline dry run; no hidden LLM
  feedback or forward calls; same TRAIN schedule across the two learning arms;
- a separate deadline supervisor and durable response/slot receipts; kill/resume
  tests during a request, after response persistence and after evaluation.

**This adapter and deadline supervisor are not implemented by this review.**
The one-hour comparison is therefore a conditional executable design, not a
claim that a ready BBEH runner already exists. If the whole next work session,
including implementation, must fit 60 minutes, use that session for this preparation
and its pilot; do not also promise a completed transfer result.

## 5. Fixed one-hour execution once those gates pass

| Item | Fixed proposed allocation |
|---|---|
| Arms | A0 unchanged seed; A1 standard Trace; A2 TRAIN parent selection, external validation common |
| Replication | 6 paired outer seeds; dataset/task IDs frozen separately from generation seeds |
| Optimizer budget | 4 completed responses per generated arm/seed: **48 main responses** |
| Pilot | At most 12 separate responses, not reused in the main result; exercises the intended concurrency |
| Model | `deepseek/deepseek-v4-flash-0731` through OpenRouter; 0.6/top_p=1/low reasoning; 16,000 tokens; no client cache or empty-output replacement |
| Parallelism | 12 independent learning chains/processes; 1 request at a time per chain; global API semaphore 12; shared local pool 8 |
| Sequential dependency | Each chain's next proposal waits for its preceding TRAIN evaluation; never parallelize those dependent proposals |
| Request bound | 300 s **total per slot including transport retries**; existing bounded attempt count/delays additionally constrained by remaining time; no automatic timeout replacement |
| Evaluation bound | Preflight must prove ≤3000 local example evaluations before TEST; per-example subprocess wall cap 2 s; allocations/actual calls/unused allocations kept distinct |
| Held-out budget | 6 seeds ×3 arms ×24 examples = 432 local evaluations after every selection is frozen |
| Primary result | Paired final held-out accuracy and calls-to-90%; nonattainment censored; failed generation and execution reported separately |
| Cost | Report preparation, optimizer responses, local execution and tokens separately; no amortization claim |

The six seeds, exact dataset IDs, prompt/trace settings, fallback/invalidity rules,
selection ties and uncertainty procedure must be written and frozen **before**
main calls. Six seeds improve on three but do not establish universal superiority.
Choose a fresh ID after checking the ledger; EXP-20 currently appears available,
but this planning review does not reserve it or collect its data.

| Session time | Work and boundary |
|---|---|
| 0–5 min | Verify source/dataset hashes, run targeted tests and the no-live accounting check in parallel. Fail closed on drift. |
| 5–10 min | Separate engineering pilot, up to 12 concurrent requests, all within the total-slot deadline. Verify quotas, usable output, local latency and projected schedule. |
| 10–12 min | Freeze the main design and all slot IDs. No outcome-based change in sample size or treatment. |
| 12–47 min | Run all 12 chains. Four sequential slots ×300 s gives 20 min of generation per chain; 3000×2 s/8 represents 12.5 min of local workload at full utilization. This is a planning estimate, not an upper bound on dependency/queue latency. Pilot measurements must support the 35 min window. |
| 47–55 min | Freeze every final/prefix selection, then run TEST and paired analysis. 432×2 s/8 gives about 1.8 min of local TEST work at full utilization, with remaining time for checks and report. |
| 55–60 min | Recompute, check hashes/slots, write the short result or explicit incomplete status, and stop all owned workers. |

Submit the paired arms in balanced order across seeds and use a fair shared API queue; no arm receives a larger successful-response allowance or a privileged routing policy.

The supervisor is armed at session start, independently of worker progress.
No new generation is launched after its stage deadline; an outstanding slot at
that boundary is checkpointed as incomplete/ambiguous, not relabeled invalid
model output. The minute 60 deadline is an upper bound on **local execution while
the host is awake**, not a guarantee of a scientifically complete result during
an external outage. A pilot timeout or insufficient throughput prevents the main
launch; it does not silently reduce 6 seeds to the fastest successful subset.
If a mandatory chain is incomplete, TEST stays closed and the session reports an incomplete experiment. Completed invalid responses still consume their slots and remain scientific outcomes; transport-incomplete slots are distinct. No analysis is restricted to the fastest completed pairs.
All deadlines include retries. A restart never overwrites a response, duplicates
an ambiguous pending request, or grants the same run a fresh hour automatically.

## 6. What follows this hour

If transfer is measurable and the baseline has usable headroom, the next separate
bounded study can use a 2×2 design for curriculum off/on and compact semantic trace
off/on, with the selected trainer fixed. That provides individual effects,
combination and ablation from one coherent experiment. Record actual replay
transitions and trace payloads; distinguish equal proposal budgets from differing
local evaluation costs. Do not add optimizer type, surface, goal and depth to the
same study. Each changes the question and increases the required replication.

No projected accuracy gain is guaranteed. The guarantees to build and test are
budget enforcement, preserved evidence, honest incomplete status and a clear
answer to one question per run.
