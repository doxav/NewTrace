# Prompt 18-R3F temporary-outage resume decision

## Trigger

The seed-1 / budget-12 / Trace-B unit stopped after the frozen three transport
attempts failed with `[Errno -3] Temporary failure in name resolution`. The
user subsequently confirmed that a temporary local Internet outage caused the
failure and explicitly instructed the experiment to resume.

## Preconditions checked before resumption

- `https://openrouter.ai/api/v1/models` returned HTTP 200.
- The locally referenced OpenRouter credential authenticated successfully via
  `https://openrouter.ai/api/v1/key` without being printed or persisted.
- OpenRouter advertised the exact frozen model
  `deepseek/deepseek-v4-flash-0731`.
- Experiment source digest remained
  `851c47563a1365733988babf49cdbf5be98d9a7be534b8aaf82cf7f70550c6b0`,
  exactly matching the existing control-plane lock.
- Runtime digest remained
  `420b5351063a56b0ad274a6c39b6aaa4dc95b9094434600e89ea79f3eccc8872`.

## Resume semantics

The 12 canonical units completed before the outage remain the resumable matrix
prefix. The interrupted thirteenth unit produced no interpretable scientific
result and is retained only as infrastructure-failure evidence. Its output tree
is moved intact under the main namespace's
`pre_temporary_internet_outage/seed-1/budget-12/B` archive.

Seed 1 / budget 12 / Trace B is rerun from zero after connectivity restoration.
No partial candidates, evaluations, optimizer state, tokens, or metrics from
the failed attempt enter the replacement unit. All subsequent units retain the
same frozen source lock.

No task, data, holdout, P0, model, request parameters, reasoning, objective,
hard constraint, seed, candidate budget, arm order, concurrency, timeout,
transport attempt ceiling, watchdog, or statistical rule changes.

The previous failure and stop-reporting `KeyError: 'invalid_rate'` remain valid
diagnostic evidence. If another infrastructure failure occurs, execution must
stop again; it must not be converted into scientific evidence.
