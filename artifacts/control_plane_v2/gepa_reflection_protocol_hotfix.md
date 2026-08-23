# GEPA 0.1.4 reflection protocol hotfix

## Pre-patch reproduction

- Inspected control-plane HEAD: `8077703218d087c8b458d029ca581ce7dcd745b6`.
- Installed dependency: `gepa==0.1.4`.
- `_run_gepa_engine` passed `prepared["clients"].get("optimizer")`
  directly to GEPA as `ReflectionConfig.reflection_lm`.
- GEPA's `TrackingLM` called that object as `reflection_lm(prompt: str)`.
- The recursive-opt `_GuardedRoleClient` retained its canonical provider
  protocol and delegated the positional string to the chat provider client.

A provider-free strict-chat reproducer therefore produced the live failure
shape before any patch:

```text
GEPA -> TrackingLM -> _GuardedRoleClient("GEPA reflection prompt")
AttributeError: 'str' object has no attribute 'get'
```

The failure is a protocol mismatch, not an evaluator-contract failure. GEPA
owns a text-to-text reflection interface, while recursive-opt owns a guarded
chat-provider interface:

```text
GEPA:          reflection_lm(prompt: str) -> str
recursive-opt: client(messages=[{"role": "user", "content": prompt}])
               -> choices[0].message.content
```

## Earlier false-green mechanisms

The real GEPA public-contract smoke set `max_candidate_proposals=0`. It
correctly covered the public evaluator pair but stopped after seed evaluation;
its reflection callable received zero calls and could not cover this seam.

Experiment 0's `offline_contract._FakeClient` also contained an explicit
positional-string branch for `gepa_reflection`. It accepted GEPA's foreign
protocol directly and returned a string, bypassing the canonical recursive-opt
chat interface. That fake therefore concealed the missing production adapter.

These are independent GEPA 0.1.4 integration seams:

1. The earlier public evaluator hotfix changed recursive-opt's public callback
   from an internal triple to GEPA's `score | (score, side_info)` contract.
2. This reflection hotfix must adapt GEPA's text callable to the existing
   guarded chat client and extract the provider's textual response.

The diagnosis is confirmed. The repair belongs only in the GEPA engine adapter;
the guarded role-client contract, accounting, budgets, request parameters,
Experiment-0 objective, task, pools, and holdout policy remain unchanged.

## Readiness checkpoint

The adapter calls the unchanged guarded optimizer client exactly once with one
user chat message, accepts a legitimate direct string or extracts only textual
`choices[0].message.content`, and rejects malformed responses. Its call site
preserves `None` when no optimizer client exists. Runtime physical size remains
8,850 lines (`spec.py`: 2,732): the 19 adapter lines are offset only by 19 blank
separator lines in the same runtime file, for a zero-line production delta.

Provider-free/network-blocked pre-CI results:

- direct reflection adapter and actual GEPA reflection seam: `5 passed`;
- control-plane v2: `50 passed`;
- final hardening: `21 passed`;
- recursive spec: `47 passed`;
- objectives/vector/multi-objective: `89 passed`;
- mandated recursive regression: `312 passed, 2 skipped, 1 warning`;
- Experiment-0 eligibility tests: `13 passed`;
- complete unit suite: `492 passed, 3 skipped, 1 warning`;
- clean-kernel notebook: `1 passed, 49 deselected`.

The two recursive skips require optional graph/telemetry backends. The complete
suite's third skip requires the Graphviz `dot` executable. No GEPA or
Experiment-0 test skipped. The warning is the existing LangGraph serializer
default deprecation.

Production-computed pre-CI digests:

- `runtime_tree_sha256=37072c1364a02c277a677bf43ad8132a32a9f233488c80cd2b6bf1a7e344f33e`
- `registry_sha256=f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`

The registry digest is unchanged because the adapter does not modify canonical
registry entries. The old runtime digest and required CI describe the
pre-repair implementation and remain historical only. Until the new required
workflow is observed green, `required_gepa_ci=false` and
`ready_for_prompt_18=false`.
