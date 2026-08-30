# GEPA reflection-client blocker

## Classification

This is a control-plane interoperability defect discovered by Experiment 0,
not an experiment-task, evaluator, objective, budget, or provider defect.
Experiment 0 must not patch it because Prompt 18-R3 explicitly freezes the
control plane.

The real GEPA 0.1.4 path calls its configured reflection callable with a plain
prompt string. Recursive-opt supplies `_GuardedRoleClient`, whose underlying
LiteLLM callable expects an OpenAI-style `messages` list and returns a
LiteLLM `ModelResponse`. The seam therefore needs both input and output
adaptation:

```text
GEPA prompt string
  -> messages [{"role": "user", "content": prompt}]
  -> guarded optimizer-role provider call
  -> final response text
  -> GEPA
```

Budget and role-usage accounting must remain around the provider call.

## Provider-free reproducer

Run from the repository root:

```python
from typing import Any
from gepa.lm import TrackingLM
from opto.features.recursive_opt import spec as S

class FakeGuard:
    limits = {"total_tokens": 60000}

    def consume(self, _name: str, _amount: int = 1) -> None:
        return None

    def require(self, _name: str, _amount: int = 1) -> None:
        return None

class MessagesOnlyClient:
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", args[0] if args else None)
        for message in messages:
            message.get("role")
        raise AssertionError("unreachable")

wrapped = S._GuardedRoleClient(
    MessagesOnlyClient(),
    "optimizer",
    {},
    FakeGuard(),
    8192,
    0.0,
    {"reasoning": {"effort": "low"}},
    "openrouter/deepseek/deepseek-v4-flash-0731",
)
TrackingLM(wrapped)("GEPA reflection prompt")
```

Observed without network or provider credentials:

```text
AttributeError: 'str' object has no attribute 'get'
```

## Live evidence

In `reports/live_micro_smoke.json`, A and B pass every micro gate. GEPA arm C
reaches reflection, emits the same exception, retries per task, and reports
that reflective mutation did not propose a candidate. GEPA then returns P0,
so the canonical run itself has status `success`, but Experiment 0 correctly
fails these gates:

- `proposal_nontrivial`;
- `usage_populated` for the optimizer role;
- `output_persistence_and_resume` (not attempted after earlier failures).

C accounted two attempted optimizer calls but zero optimizer provider tokens.
Its output is therefore not evidence about GEPA optimization quality.

## Required separate hardening proof

A separate control-plane hotfix must minimally adapt the configured optimizer
role client at the GEPA reflection boundary and prove, with GEPA 0.1.4 and no
provider calls, that:

1. GEPA can call the reflection function with a string;
2. the provider-facing callable receives a messages list;
3. GEPA receives final text rather than a `ModelResponse`;
4. optimizer calls and tokens are attributed once;
5. budgets still stop reflection;
6. the public evaluator remains `score | (score, side_info)`;
7. the public `optimize_anything()` integration produces a nontrivial proposal.

After the hotfix, required CI and control-plane readiness/digest gates must be
re-established before rerunning the C micro arm from a fresh output namespace.
