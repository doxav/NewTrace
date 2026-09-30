# EXP-18 — prospective memory and instance-Pareto diagnostic

Status: PILOT PROTOCOL AND DRAFT EXPLORATORY DESIGN, before new model calls.
This is separate from EXP-17 and EXP-16. No outcomes are pooled or used to alter
EXP-17's confirmatory C−I comparison.

Six paired outer seeds18011,18023,18037,18041,18053,18067; namespace
`EXP18-MECHANISMS-v1`. Each of four arms receives16 completed responses, total384.
Common original seed, model/settings (32,000 tokens/low), benchmark family,
24 TRAIN/12 validation/12 audit instances ×2 local seeds, B32, source screening,
eight local workers, selection and permanent deployment fallback follow EXP-17.
New instances and seeds are disjoint. A0 and fixed B2 are audit controls only.

## Two intervention factors

All arms get the invariant prompt, exact current-parent source and the same compact
current-parent TRAIN summary: aggregate normalized AUC only when the entire panel
is valid, allocated/completed/valid trajectory counts and typed status counts.
No raw per-task values paired with normalization constants, no validation, no hidden
parameters and no LLM-generated critique are included.

L: scalar TRAIN-best parent, no prior-attempt archive.
M: scalar parent plus deterministic prior-attempt archive.
P: uniformly sampled nondominated TRAIN-instance parent, no archive text.
PM: nondominated parent plus the same archive rule.

These controls isolate the increment of the historical archive from the introduction
of immediate score feedback. They do not estimate the effect of individual archive
fields separately. P changes the complete parent-selection policy, including
frontier filtering and uniform sampling; it is not a claim about filtering alone.

Memory uses at most seven most recent distinct earlier nonempty sources excluding
the already displayed current parent, plus all previous-slot typed summaries.
Same arm/outer and prior completion only. Source hashes, response/receipt times,
parent identity and TRAIN receipt hashes must align. Source is shown whole or
explicitly omitted, never silently truncated. Exact serialized memory budget:
65,536 characters. All omitted/duplicate/no-code/invalid counts remain recorded.
No cache-miss evaluation is allowed to fill a historical context. All arms record
the same allocated TRAIN panel immediately after each completed response.

The Pareto vector contains24 means, each averaging the two local trajectories for
one TRAIN instance. Only complete TRAIN-valid programs enter parent selection.
Include the seed. Strict componentwise nondominance, no floating tolerance;
source and equal-vector duplicates retain the earliest index. Uniform choice from
the frontier uses a deterministic SHA256-derived local Random seed, independently
of model RNG and holdout. Record the frontier, chosen/scalar parent and whether
the choice will feed a subsequent proposal. Use the real production PrioritySearch
path with a narrow parent-selection override; do not substitute a second loop.

## Order, analysis and interpretation

Whole-arm orders, paired with their reverses:
18011 L/M/P/PM;18023 PM/P/M/L;
18037 M/PM/L/P;18041 P/L/PM/M;
18053 P/L/M/PM;18067 PM/M/L/P.
Thus every pairwise precedence is balanced3/3. Single generation concurrency.
Sixteen sequential single-parent updates in every arm; no width/depth tradeoff.

Freeze every selection and each arm's validation-selected representative before
any audit. Per-seed deployment AUC is the same metric/weights as EXP-17.
Registered exploratory contrasts: M−L, P−L, PM−P, PM−M;
memory main effect=((M−L)+(PM−P))/2;
Pareto main effect=((P−L)+(PM−M))/2;
interaction=PM−P−M+L. Bootstrap the six outer seeds jointly,10,000 draws/seed1515.
All contrasts are exploratory, with no multiplicity guarantee or superiority claim
from ordered means. Report all seeds/invalidities/fallback and realized exposure
to distinct prior sources/nontrivial frontiers. No independentN16 arm is included,
so this study cannot establish a memory advantage over equal-budget independentN16.

## Engineering checks before scientific execution

Separate live engineering namespaces: EXP18-E1-M-v1, outer18901, M10 responses;
EXP18-E1-SHORT-v1, outer18903, L/P/PM two responses each. Combined with EXP17-E1,
the planned engineering allocation is20 completed model responses. The long M
pilot tests naturally accumulated history after seven prior attempts; invalid and
duplicate outputs may reduce distinct usable examples and remain visible.
The short pilot exercises actual P/PM production paths; scripted integration tests
must additionally demonstrate a nontrivial frontier and an actual non-scalar parent.

Test first: provenance, invalid/empty output, honest omissions, no hidden data,
all allocated TRAIN receipts before reuse, determinism, no extra calls/evaluations,
real stored parent identity, global barriers and crash/resume with identical memory
snapshots. Freeze exact source/feedback/selection/analysis code after engineering.
Poor outputs are not replaced; no outcome-driven prompt/metric/seed change.
The mechanism study completes for positive, null, negative or ordinary generation
failure outcomes. Engineering failures must be fixed and documented before freeze.
