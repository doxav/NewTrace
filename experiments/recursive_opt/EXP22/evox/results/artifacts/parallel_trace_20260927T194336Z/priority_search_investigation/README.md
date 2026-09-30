# PrioritySearch over OptoPrimeV2: investigation, not an activated amendment

Scope: retain the current stagnation-based trigger and the 100 solution-HTTP-attempt budget. The user clarified that a fixed ten-attempt update schedule was not requested. No live runtime, frozen source, existing run, or protocol was changed. No API requests were made.

## Current execution

`raw_spec.json` declares `llm_roles.optimizer = "main"`. `main` is the `llm_profiles.main` model/transport profile, not an optimizer or loop. `_resolve_role_clients` materializes that profile into the guarded optimizer client. The registered `exp22.engine.coevolution@1` engine passes that client to `run_kernel`. The inherited EvoX `run_discovery` loop controls solution generation and stagnation. `TraceMetaCoEvolutionController._trace_proposal` constructs OptoPrimeV2 and performs one backward/step per trigger. No PrioritySearch trainer is currently instantiated.

The trigger is ten controller iterations without an increase greater than 0.01 in tracked best score. One controller iteration can consume several solution HTTP attempts through retries. The trigger is not ten HTTP attempts and is not periodic. The patience is derived from 10% of the configured horizon; for horizon 100 it is ten.

## Verified PrioritySearch behavior

The experiment's frozen Trace checkout contains a standard `trace` control-plane engine with explicit `optimizer`, `trainer`, `iterations`, `num_candidates`, `optimizer_kwargs`, and `trainer_kwargs`. It defaults to OptoPrimeV2 and PrioritySearch. The EXP22 custom engine does not use that engine configuration or implementation.

The offline probe exercises the actual frozen PrioritySearch loop, real OptoPrimeV2 backward propagation, and a mocked proposal step. It uses one candidate, one batch, one proposal, one thread, no test evaluation, and no separate validation dataset:

| Trainer steps | Optimizer proposal attempts | Module evaluations |
|---:|---:|---:|
| 1 | 0 | 1 |
| 4 | 3 | 7 |
| 10 | 9 | 19 |

These counts describe this controlled probe, not all PrioritySearch configurations. Step zero initializes and samples. Subsequent steps propose, validate new candidates, and sample selected candidates. Disabling `validate_exploration_candidates` does not disable validation of new proposals. With one candidate and one batch, a proposal sequence of values 2, 0, 3 selected parents 1, 2, 2: the worse child did not replace the best parent for the next proposal.

Therefore calling `train(num_steps=1)` on a freshly constructed trainer at every trigger would produce no optimizer proposals and discard archive continuity. Calling `train(num_steps=10)` does not mean ten optimizer calls or 100 benchmark solution attempts. Merely changing EXP22's engine to `trace` is also insufficient: its PolicyModule currently returns a measured observation, not a live benchmark rollout, and the existing evaluator expects a completed kernel result. The standard engine also performs a final evaluation that must not accidentally start an extra search.

## Recommended integration

Retain the persistent solution controller and put one persistent PrioritySearch-derived policy-search adapter alongside it. Reuse PrioritySearch's candidate representation, priority queue, parent selection, and OptoPrimeV2 proposal machinery. Extend the experiment adapter rather than modifying the frozen frameworks. A normal unmodified `PrioritySearch.train()` call is not a compatible replacement for the delayed online policy-scoring lifecycle.

1. Initialize one trainer and optimizer per run, plus the stock policy candidate. Do not recreate the trainer at every trigger. Initialize its sampling/proposal state once without making a paid proposal.
2. At a stagnation event, finalize the active policy's measured window using the existing policy score. Attach the score and losslessly encoded feedback to that exact policy candidate, then update PrioritySearch's archive.
3. Let PrioritySearch select one archived parent and request one OptoPrimeV2 proposal. Build its Trace rollout from that parent's own recorded observation; do not attach the currently active policy's measurements to a different archived parent. Keep current population context explicitly separate. Reuse the existing source-hash checks with the selected parent hash.
4. Validate the child using the existing stock policy validator. A valid child is deployed for its measured search window while preserving the solution population. Its downstream performance remains pending until the next trigger or run end. Do not give it an invented score or reuse its parent's score as measured child performance.
5. Feed the child's measured result back into the trainer archive when its window completes. At run end finalize this score without generating another policy.

This is deliberately a PrioritySearch adapter for deferred online evaluation. Standard PrioritySearch validates a child before ranking/deploying the best candidate; this experiment must run the child first to measure its downstream score. The adapter must document that distinction and must not claim to be an unmodified PrioritySearch training loop. It can reuse `propose`, `exploit`/`explore`, and `update_memory`, with rollout construction and delayed validation supplied by the experiment.

A different option is an unmodified PrioritySearch outer training loop whose module evaluation runs an entire benchmark window. That would introduce additional parent/child evaluations, require explicit population snapshots or resets and a revised allocation of the 100 solution attempts, and change the experiment more substantially.

## Explicit control-plane configuration

Expose trainer and optimizer choice in a new version of the custom engine. The following is a proposed configuration fragment, not an existing runnable configuration:

```json
{
  "engine": {
    "name": "exp22.engine.coevolution@2",
    "config": {
      "trainer": "PrioritySearch",
      "optimizer": "OptoPrimeV2",
      "num_candidates": 1,
      "num_proposals": 1,
      "num_batches": 1,
      "num_threads": 1,
      "policy_evaluation": "deferred_search_window",
      "trigger": "inherited_stagnation"
    }
  },
  "llm_roles": {"optimizer": "main"}
}
```

The existing module horizon remains the single source of truth for the 100 solution-attempt limit. All added fields must be validated and actually consumed, and raw/normalized/resolved plans must expose them. The custom execution function must construct the named trainer; adding decorative JSON alone changes no behavior.

## Attempt accounting and limits of the guarantee

Keep distinct counters for actual solution HTTP attempts, meta HTTP attempts, optimizer proposal invocations, accepted policy deployments, and controller iterations. Availability probes and guide calls are separate roles. Budget checks must occur before dispatch, including every retry; each dispatched call must reach one recorded outcome, including failures. Preserve the existing successful-response identity checks and the separately pending provider-recovery decision.

For each trigger, configure exactly one parent/batch/proposal and one thread. Record empty or invalid proposals as attempts without pretending that a new policy was deployed. Preserve proposal errors rather than allowing trainer filtering to hide them. Bound any transport retry through the same role counter. Reserve the remaining solution budget before work and prohibit request 101; a completed run must assert 100 dispatched solution attempts and 100 recorded outcomes. An interrupted run must report its actual count, never success.

A stagnation schedule cannot guarantee a fixed number of policy proposals, and a model cannot guarantee a fixed number of valid or distinct policies. Provider failure can also prevent completion. What can be guaranteed locally is the cap, complete accounting, one proposal invocation per reached trigger, and an exact-count completion criterion. Guaranteed completion requires provider availability or a separately authorized recovery policy.

Archive scores come from different evolving solution populations and search windows. PrioritySearch's best archived score is useful selection evidence, not a controlled causal comparison of policies on an identical population. Retain this limitation when reporting results.

## Implementation and verification work required

- `src/control_plane.py`: version/register the new engine, validate its configuration, construct the persistent trainer and pass the same resolved `main` client.
- `src/kernel.py`: replace per-trigger optimizer construction with the trainer adapter; retain the existing solution loop, stagnation condition, population migration, and window scorer.
- Policy event logging: distinguish `active_policy_before`, `selected_parent_policy`, `candidate_policy`, and the candidate solution's separate `parent_solution_id`. The existing switch event records the previously active policy, which is not necessarily the selected parent once archive selection is introduced.
- Adapt transport and final checks only where needed for pre-dispatch limits and explicit meta-attempt accounting; do not silently activate the pending 429 amendment.
- Offline tests: regression returns to best archived parent; parent feedback/source hash agreement; persistent archive across triggers; retries count individually; 99 consumed attempts allow exactly one further solution request; refusal/empty/invalid proposals remain counted; no hidden trainer validation/test/final-evaluation requests; exact 100/outcomes agreement; no post-budget proposal; preserved solution population; interrupted runs remain partial.

Verification performed for this investigation:

```sh
experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/artifacts/parallel_trace_20260927T194336Z/priority_search_investigation/offline_probe.py
ruff check experiments/recursive_opt/EXP22/artifacts/parallel_trace_20260927T194336Z/priority_search_investigation/offline_probe.py
```

The probe assertions pass. Implementation of the live adapter and its integration tests remains future work; this investigation does not claim they are complete.
