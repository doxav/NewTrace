# Experiment 0 v1 eligibility failure analysis

Experiment 0 v1 stopped at task eligibility. It stopped before the live
one-unit micro-smoke, every optimizer comparison, the three-seed pilot, and
any holdout-informed model or task selection. The preserved v1 JSON files and
all 264 persisted run files are copied byte-for-byte under `manifests/v1/` and
`reports/v1/`; their hashes are recorded in
`manifests/v1/evidence_hashes.json`.

## Protocol-design false negative

The scientific objective is explicitly multi-objective: maximize accuracy and
minimize `forward_token_ratio`. Its final efficiency-success rule permits an
accuracy delta whose lower 95% confidence bound is at least `-0.02` when the
upper 95% confidence bound for the token-ratio delta is below zero. The v1 task
gate nevertheless required `probe_accuracy_spread > 0` even when a feasible
probe preserved accuracy and substantially reduced tokens.

GSM8K demonstrates the contradiction. P0 through P3 all had pooled
train-plus-validation accuracy `0.833333`, while tokens per example ranged
from about `303.3` to `1264.3`; P1 used roughly half of P0's `606.7` tokens at
the same accuracy, and every probe had invalid rate zero. Rejecting this task
solely because accuracy was flat is a measurement-design false negative, not
evidence that its optimization surface is uninformative.

The original validation pool contained only two examples. Every GSM8K probe
scored `1.0` on the four train examples and `0.5` on the two validation
examples, so the pooled `0.833333` obscured the relevant validation baseline
and two examples were insufficient to establish prompt insensitivity.

V1 also excluded the source task named `boolean_expressions` by taking the
maximum invalid rate over every manual probe. P2 was invalid on part of the
pool, but P0, P1, and P3 were valid. A deliberately poor probe is useful
surface evidence and should be rejected by the experiment's hard invalidity
constraint; it should not invalidate an otherwise suitable task.

Finally, all provider monetary costs were recorded as `0.0`. Because the runs
used nonzero tokens and the provider did not supply a reliable positive cost,
the monetary cost was unavailable, not zero. V1 therefore could not causally
determine the "cheapest task" in dollars. V2 uses a transparent forward-token
proxy when monetary costs are unavailable and preserves the monetary forecast
requirement after the live micro-smoke.

The pinned source is `hubert233/BigBenchExtraHard`. Its experiment-facing tasks
are BBEH object counting and BBEH boolean expressions, not ordinary BBH. V1
identifiers remain unchanged as immutable evidence; v2 uses corrected BBEH
names.

## Why a versioned repair is permitted

V1 revealed a pre-optimizer measurement-design defect. No engine result,
holdout result, or optimizer comparison was available when the defect was
identified. V2 therefore changes only eligibility measurement semantics and
sample resolution. It does not change the scientific purpose, P0, the four
manual probes, model profiles, objective, weights, invalidity constraint,
arms, validation-gate ablation, paired seeds, or final success criteria.
