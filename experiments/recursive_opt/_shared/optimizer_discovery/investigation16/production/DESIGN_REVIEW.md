# P1 design review — resources and interpretation before efficacy results

Read-only review of the P1/P1-E1 protocols, frozen analysis specification, S0 audit
and corrected T1 timing report. No F1/P1 efficacy results were opened. No candidate
or model was executed. The only new calculations below are arithmetic. This note
is supplementary and does not amend any frozen protocol, code, test or decision rule.

## Allocation arithmetic is consistent

Six outer seeds × four generative arms (I/C/R/W) × eight completed responses gives
**192 responses**. A0 has no model calls. Each generative pool includes the seed,
so there are 6×4×9=216 logical pool entries. Invalid responses retain their slots;
unused evaluations cannot be converted into extra search proposals.

| Phase | Logical trajectories | Objective-call allocation |
|---|---:|---:|
| Training: 216 entries × 24 instances × 2 local seeds | 10,368 | 331,776 |
| Validation: 216 entries × 12 instances × 2 local seeds | 5,184 | 165,888 |
| Audit: 6 outer seeds × 5 arms × 12 instances × 2 locals | 720 | 23,040 |
| Total, B=32 | **16,272** | **520,704** |

Two executions per valid proposal yield 1,041,408 subprocess allocations without
deployment failure. The four generated deployment arms have 576 audit trajectories.
Each may incur one failed proposal before permanent fallback; a nondeterminism
failure can execute the candidate twice. Thus at most 576×2=1,152 extra subprocesses,
or **1,042,560 total**, assuming the trusted seed/evaluator succeeds. Fallback adds
no objective evaluation and never resets history. Ordinary invalid training runs
terminate early. Cache reuse can reduce physical execution substantially; logical
allocations and physical persisted-cache counts must remain distinct.

The unique 48-task normalization design has 48×128=6,144 reference objective calls,
separate from the table. Repeated physical reconstructions after process restarts
and interrupted work lost before persistence are not fully instrumented. Neither
6,144 nor the persisted-cache total should be presented as an unconditional exact
count of every CPU operation performed by the machine.

## What the contrasts identify

**R−I is central**: does the complete training-informed production search procedure
improve deployment AUC over fresh independent generation at the same response
allocation? All arms share the corrected invariant task/metric instructions. This
comparison is conditional on that revised instrument, task family, B32 and N8.
It is not a causal decomposition of every change from historical EXP-15.

**R−C is the mechanism comparison**: what does explicit evaluated Trace text and its
aggregate training AUC add beyond editing a training-selected current parent?
C is not devoid of all feedback: training results still determine its parent.
Because parent trajectories can diverge after an update, R−C estimates the effect
of the two complete adaptive procedures, not a same-parent single-prompt treatment.
It does not isolate the aggregate AUC scalar from richer trace content.

**W−R is a search-schedule comparison** at the same eight responses: width two/four
rounds versus width one/eight rounds. Breadth and sequential update opportunities
change together. This can support a concrete schedule choice; it cannot establish
that breadth alone, arbitrary population search, or deeper recursion is superior.
The first width-two parents may be copies of the same seed. **R−A0 is contextual**
evidence that generation/search improves the starting policy; it cannot replace a
missing R−I advantage.

Primary results include the common deployment fallback. Candidate-only validity,
seed selection and fallback frequency must accompany any favorable AUC. A2/EXP-15
and R/P1 are different experimental procedures; their means are not randomized
before/after evidence for a particular root cause. P1 neither measures amortization
nor establishes algorithmic novelty or an advantage from nested recursion depth.

## Six outer seeds give a bounded exploratory comparison, not high precision

S0 observed an A2−A1 paired SD of 0.038660 from only five outer seeds, with mean
−0.005068 and a registered interval spanning both signs. If that SD persisted,
six outer seeds would give SE≈0.01578 and an illustrative normal 95% half-width
≈0.03093. Moving from five to six reduces that SE by only 8.7%. Under the same rough
normal planning approximation, an effect near 0.044 AUC would be needed for 80%
power at a two-sided 5% threshold. These calculations are **not P1's registered
bootstrap decision rule**, not a forecast of P1 variance, and not a power guarantee.
The old SD is unstable; the improved measurement panels could change it.

More train instances/local seeds can stabilize incumbent and final selection.
They do not turn 192 responses, 720 audit trajectories, or individual trajectory
points into independent outer replications. The six paired outer seeds share the
same frozen audit instances; their bootstrap uncertainty is conditional on that
panel, rather than encompassing arbitrary future benchmark-panel variation.
Report all four contrasts and their fragile intervals. An interval spanning zero
means inconclusive, not proof of absence; numerical sample-mean ordering alone
cannot justify a superiority claim. S1's fixed-bank selection improvement is not
a projected magnitude for R−I or R−C.

## The E1 gate and realistic scheduling expectations

P1-E1 runs exactly two actual production R responses on separate 24×2 training
trajectories. It checks long prompts, callback/budget/source integrity, typed
failures and completed-slot resume. Its gate requires both responses completed,
feedback within the fixed text limit, all integrity checks, and at least one
training-eligible generated program. No validation/audit outcome or pilot program
enters P1. Passing establishes a necessary engineering check on this sample;
it does not establish reliable eight-step prompts, W-specific live behavior,
efficacy or a low future failure rate. Both-invalid output is a generation
feasibility issue to inspect, not automatic evidence of an implementation bug.

Scheduling reference metadata supplied by the coordinator, without fitness:
G1's 24 completed responses averaged 95.836 active seconds; the first ten completed
F1 responses averaged 302.649 seconds, median 206.224, maximum 1,006.257. Simple
192-response scaling gives **5.11 or 16.14 hours of generation**, respectively.
T1's healthy eight-worker measurements give another **about 1.86 hours of offline
evaluation** under linear scaling. Adding those components gives rough scenarios
near 7–18 hours, not lower/upper guarantees: P1's 24×2 feedback is longer, model
latency has a long tail, candidate complexity/cache use vary, and transport retries,
orchestration and host suspension can add time. HTTPX timeout300 is not an absolute
generation deadline when response bytes continue to arrive.

The same metadata scales to approximately 1.28M or 3.30M actual reported tokens and
$0.166 or $1.029 of reported cost, using G1/F1 totals respectively. These are only
rough scenarios from different prompt/cap mixtures and limited metadata, not a
price quote or guaranteed spend ceiling. The configured completion allowance is
192×32,000=6,144,000 tokens; prompt tokens and uncertain transport billing are
additional. Preserve metadata coverage and never equate missing usage with zero.
Use E1's measured end-to-end resource evidence before launch. Do not reduce P1
replication or alter its design because an efficacy result is disappointing.

## Wording for the final recommendation and any EXP-17

A defensible positive formulation is: “Under this frozen benchmark and response
allocation, the revised production procedure showed an exploratory advantage over
independent generation; the paired interval and all failure rates are reported.”
Use R−C to qualify whether explicit evaluated feedback has evidence beyond
training-selected code inheritance, and W−R to qualify the particular schedule.
Do not promise that the same gain will recur or attribute it to all bundled
instrument corrections individually.

An inconclusive/negative P1 still completes this prospective diagnostic when its
protocol and evidence are valid. Recommendations may retain independently validated
engineering and measurement corrections without claiming an efficacy gain. Any
EXP-17 requires a new preregistration, fresh splits/outer seeds, a practical minimum
effect worth detecting and independently justified replication/resources. P1 can
inform those choices; its programs/data then become development evidence and cannot
be reused as untouched confirmatory holdout. Do not automatically trigger another
efficacy study until R wins, or select a favorable stratum, metric or contrast after
seeing outcomes. This review proposes no protocol amendment or additional run.
