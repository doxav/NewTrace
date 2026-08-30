# Live transport resilience hotfix

## Pre-patch diagnosis

The Prompt 18-R3F diagnosis is confirmed at branch HEAD
`8cdde772ac45f5ba0646cef12e959fe20754ab75`.

### A. Transient retry classification

`opto/utils/auto_retry.py` recognized broad fragments such as `connection
error`, `network`, and `timeout`, plus selected LiteLLM exception class names.
It did not recognize either observed OpenRouter failure:

- `Connection reset by peer`
- `Server disconnected without sending a response.`

A provider-free pre-patch reproducer invoked
`retry_with_exponential_backoff(..., max_retries=3, base_delay=0)` with each
exact message. Both calls were logged as `Non-retryable error`, each provider
stub was called exactly once, and the original `RuntimeError` escaped. The
helper also inspected only the immediate exception type and message, not
`__cause__` or `__context__`.

### B. No explicit canonical request timeout

The canonical call path was:

`spec._make_guarded_role_client`
→ `runmode.make_live_llm`
→ `opto.utils.llm.LiteLLM`
→ `litellm.completion`.

`_make_guarded_role_client` explicitly passed `request_timeout_s=None`.
Experiment 0 did not declare a timeout in either LLM profile, and none of
`RECURSIVE_OPT_LLM_TIMEOUT_S`, `RECURSIVE_OPT_LLM_MAX_RETRIES`, or
`RECURSIVE_OPT_LLM_BASE_DELAY_S` was set during the reproduction. Therefore a
canonical provider request had no frozen, first-class timeout policy and could
remain blocked at the transport layer.

### C. Cooperative wall budget

`_BudgetGuard.check_wall_time()` compares elapsed monotonic time only when
Python regains control and calls `consume`, `require`, or another explicit
check. It cannot interrupt a worker blocked inside network I/O. Consequently
the main `wall_time_s=7200` limit was a cooperative budget, not a hard
two-hour unit deadline. The preserved R3E evidence records the process blocked
inside concurrent evaluation for more than twelve hours before manual
interruption.

### D. Implicit Trace concurrency

Experiment 0 built Trace arms with `trainer_kwargs={}`. PrioritySearch passed
the resulting `num_threads=None` into `trainer.evaluators`, whose `batch_run`
path constructed `ThreadPoolExecutor(max_workers=None)`. Python therefore
selected a machine-dependent worker count. GEPA, independently, was already
frozen with `parallel=false`.

## Evidence classification

The completed pre-fix A run remains valid historical evidence but is excluded
from the restarted paired matrix. The incomplete B run is valid infrastructure
failure evidence only: no canonical result was produced, so it supports no
Trace efficacy conclusion.

The supported causal conclusion is that the observed transient transport
messages were not retried, provider calls had no explicit canonical timeout,
the wall budget could not preempt blocked I/O, and Trace concurrency was
implicit. The record does not establish which upstream component reset or
disconnected the socket, nor does it prove that either transport message alone
caused every worker to remain blocked.

## Surgical correction

Canonical LLM profiles now normalize, validate, fingerprint, and persist
`request_timeout_s`, `transport_max_attempts`, and
`transport_base_delay_s`. Canonical v2 constructs each primary/fallback
provider with those values and `allow_env_overrides=false`; legacy/demo calls
retain their prior environment behavior by default. Experiment 0 freezes the
policy at 180 seconds, three total attempts, and a 1.0-second base delay for
both forward and optimizer roles.

Retry classification now follows exception type, message, `__cause__`, and
`__context__` for the requested transport families. It does not retry parser,
validation, authentication, invalid-request, or programming failures. A
recovered transport request remains one logical guarded role call and records
only the eventual provider response usage. Safe counters retain reset,
disconnect, retry, recovery, and exhaustion totals without retaining error or
reasoning text.

Experiment 0 freezes Trace `num_threads=4` for B and D; GEPA remains
`parallel=false`. Each main unit runs in a spawned child process and the parent
terminates its process group after the frozen 7,200-second wall budget, with a
fixed five-second shutdown grace. A timeout produces an infrastructure record
and stops the matrix; it cannot create a candidate or scientific result.

## Provider-free and required-CI verification

- transport/profile/watchdog and main-runner seams: 28 passed;
- focused control-plane/objective matrix: 225 passed;
- mandated recursive regression: 330 passed, 2 accepted optional
  graph/telemetry skips;
- complete unit suite: 510 passed, 3 accepted optional graph/telemetry/Graphviz
  skips;
- Experiment-0 tests: 40 passed, with its A/B/C/D offline contract passing;
- clean-kernel notebook: 1 passed;
- changed-file Ruff and `git diff --check`: passed.

No Trace, GEPA, or Experiment-0 critical test skipped. The authoritative
pre-CI runtime digest is
`420b5351063a56b0ad274a6c39b6aaa4dc95b9094434600e89ea79f3eccc8872`;
the registry digest remains
`f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`.
The first expanded workflow run, `32856225573` / job `97828761794`, failed at
collection because the newly included Experiment-0 tests lacked the existing
`datasets==3.6.0` test extra. No runtime test executed in that failed job. The
workflow retained its explicit `.[gepa]` installation, added `.[test]`, and
cached only the exact pinned benchmark revisions before socket isolation.

Required run `32856513753`, job `97829714487`, then completed successfully on
`c6a109ca2aa2d5cf62074fd16b34570582092aa3` in 1m44s. The required test step
ran sockets disabled and the standalone offline contract ran with
`HF_HUB_OFFLINE=1`. Digest-based readiness is true for that implementation.
