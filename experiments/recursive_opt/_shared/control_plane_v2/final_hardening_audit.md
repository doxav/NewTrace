# Prompt 17.7 final hardening audit

- Starting remote/local HEAD: `14b832c82341bbc55e9c662ebaebcba4e3e8e95b`
- Branch: `recursive_opt`
- Provider or paid calls: none
- Baseline runtime footprint: 8,750 physical lines; `spec.py`: 2,633 lines

## Falsification results

Each suspected issue was probed before editing. “Confirmed” means the baseline behavior contradicted the requested experiment-validity contract; “partly refuted” records behavior that was already correct and must be preserved.

| issue | result | minimal causal evidence | permitted correction |
|---:|---|---|---|
| 1 | confirmed for Trace; fixed/GEPA partly refuted | One direct `_EvaluatedModule.forward` call made two stochastic workflow calls, scored nonce 2, and attached through wrapper argument nodes rather than the exact output. Fixed made 1 forward for 1 evaluation; injected GEPA made 2 forwards for its 2 evaluations. | Add an explicit output/legacy evaluator mode, score the already-produced output, and attach the exact output node. Preserve fixed/GEPA single-forward behavior and explicit legacy compatibility. |
| 2 | confirmed | With all listed environment variables set, `_optimizer_kwargs` injected `env_marker`, an environment-selected LLM, and `_trainer_kwargs` injected another `env_marker`. Canonical Trace called `optimize` without an environment-isolation option. | Add `allow_env_overrides` with backward-compatible default `True`; canonical v2 passes `False`. |
| 3 | confirmed | A profile containing JSON-safe `request_params` failed normalization as an unknown key. | Normalize, validate, fingerprint, manifest, and apply per-attempt request parameters with explicit precedence. |
| 4 | confirmed | `ExecutionPlan` had no code provenance; changing the callable behind the same evaluator ref left `_resume_identity` unchanged. | Record source/package/registry/runtime digests and bind the authoritative digests into resume identities. |
| 5 | confirmed | Readiness contained `final_sha`; its verifier skipped unless `RECURSIVE_OPT_FINAL_SHA` was supplied. | Replace the SHA gate with clean-checkout-verifiable runtime and registry digests; keep Git informational only. |
| 6 | confirmed | The required job installed `-e .`; `.[gepa]` appeared only in a `workflow_dispatch`-only job. | Install the GEPA extra in the required offline job and run the final-hardening tests there. |
| 7 | confirmed | Reserving eight candidates reported only `accounted.candidates=8`, indistinguishable from observed proposals/evaluations. | Retain the legacy limit counter and add reserved/proposed/evaluated counters from observable engine events. |
| 8 | constraint satisfied at baseline | Runtime starts 53 lines below the 8,803 semantic-closure baseline; no framework or dependency addition is needed. | Keep the final correction within +100 runtime lines net from 8,750 and itemize the result. |

## Final evidence

All confirmed defects were closed without provider calls or a new framework.

- Portable evaluators now use an explicit output contract. Fixed, Trace, and injected GEPA tests prove one workflow forward for every evaluator invocation. The stochastic-output test proves the evaluator receives the exact traced node attached to the trainer result, and standard optimizer backpropagation reaches the originating parameter through that graph. Explicit `legacy_module` evaluators remain executable only as nonportable compatibility.
- Canonical Trace calls `optimize(..., allow_env_overrides=False)`. An adversarial test changes all nine hidden trainer/optimizer/model environment controls and obtains the same fingerprint, resolved optimizer/trainer configuration, artifact, and result. The legacy API default remains `True`.
- `request_params` is JSON-safe normalized profile data. It affects fingerprints, manifests, each fallback request independently, and provider kwargs. Profile request controls override module-supplied kwargs; canonical `max_tokens` and `temperature` override colliding request controls. Identity, endpoint, credential, and secret-bearing keys are rejected recursively.
- Resolved modules, evaluators, dataset resolvers, codecs, and engines carry callable/source/package records. Resume identity includes the runtime-tree and selected-registry digests, and a same-ref evaluator implementation change forces reevaluation.
- Readiness uses authoritative source digests rather than a future commit SHA: runtime tree `6315c6fc23d7f4e51effeb936f0b8c5938a36d821dd7b85346f7e2d8407ef07c`; golden UC4 registry `f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`.
- The required offline workflow installs `.[gepa]` and runs the hardening suite. This environment cannot run or observe the unpushed workflow, so `required_gepa_ci` and `ready_for_prompt_18` remain false. Required manual action: push the patch and observe `recursive-opt v2 offline (required)` pass.
- Candidate reporting retains the existing `candidates` limit counter and separately reports reserved, proposed, and evaluated observations.
- Runtime footprint is 8,850 physical lines, +100 from the 8,750 starting tree and at the +100 limit. `spec.py` contributes +99 and `optimize.py` +1.

Validation with provider keys removed and external sockets blocked:

- focused hardening: `21 passed in 2.12s`;
- mandated recursive/Trace/GEPA regression: `305 passed, 2 skipped, 1 warning in 16.12s`;
- complete unit suite: `485 passed, 3 skipped, 1 warning in 37.27s`;
- clean-kernel notebook: `1 passed in 4.11s`;
- isolated worktree focused readiness rerun: `26 passed in 5.32s`;
- Ruff on all changed Python files: passed;
- `git diff --check`: passed;
- workflow YAML parsing and required-job structural assertion: passed.

The skips are existing optional graph/telemetry/Graphviz dependency cases. No test was suppressed, no provider experiment was run, and Prompt 18 was not started.
