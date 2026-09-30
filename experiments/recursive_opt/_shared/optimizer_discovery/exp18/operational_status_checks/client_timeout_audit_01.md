# Installed client retry and timeout audit 01

Read-only code inspection on 2026-09-10, provenance snapshot
`1789052124807244652` ns. Interpreter: `/tmp/phase0-venv/bin/python`, Python 3.13.13.
It resolves the packages below from `/home/xav/miniconda3/lib/python3.13/site-packages`.

| Package | Installed version |
| --- | --- |
| LiteLLM | 1.75.0 |
| OpenAI Python SDK | 2.38.0 |
| HTTPX | 0.28.1 |
| HTTPCore | 1.0.9 |

No live/client/model call, HTTP request, request-body/header/environment tracing,
credential read, candidate execution or running-process interruption was performed.
The only executable parameter check constructed the pure value `httpx.Timeout(300)`;
no HTTP client was constructed. No source, installed package or frozen file changed.

**Confirmed conclusion:** the reviewed OpenRouter path forwards the 300-second
timeout, but it is not a total wall-clock deadline. The configured path adds no
OpenAI SDK retries because it does not invoke that SDK's request loop. Project and
HTTPX defaults support one HTTP attempt per slot-wrapper attempt, subject to the
explicitly described LiteLLM global-state caveat below. Elapsed slot time alone
cannot demonstrate hidden requests or an ignored timeout.

## Actual forwarding path

1. [Successor shared client](../../exp17/driver.py) at lines 41–56 creates the
   existing client and then blocks on an exclusive `fcntl.flock` before calling it.
   This lock wait has no 300-second timeout.
2. [Existing live-client factory](../../investigation16/production_driver.py) at
   lines 119–135 supplies `max_retries=1`, `request_timeout_s=300`,
   `allow_env_overrides=False`, `empty_response_retries=0` and no budget wrapper.
3. [Project make_live_llm](/home/xav/code/Trace/opto/features/recursive_opt/runmode.py:152)
   resolves these explicit settings. Its compatibility wrapper uses
   `kwargs.setdefault("timeout", 300)`, preserving the already explicit request
   timeout. The token-name conversion applies only to GPT-5 names, so this DeepSeek
   model retains `max_tokens=32000`. Empty-response retries are not installed.
4. [Frozen generation settings](../../investigation16/search_experiment.py) at
   lines 358–372 supply `temperature=0.6`, `top_p=1`, `max_tokens=32000`, native
   `extra_body.reasoning.effort=low`, `timeout=300`, a stable request seed and
   `num_retries=0` on every request.
5. [Project LiteLLM factory](/home/xav/code/Trace/opto/utils/llm.py:304) forwards
   those keyword arguments to `litellm.completion` inside the project's
   [bounded retry wrapper](/home/xav/code/Trace/opto/utils/auto_retry.py:93).
   That wrapper iterates `range(max_retries)`: **1 means one attempt**, not an
   initial call plus one retry. A retryable failure is wrapped immediately after
   that sole attempt.
6. Installed [LiteLLM completion](/home/xav/miniconda3/lib/python3.13/site-packages/litellm/main.py:1006)
   reads `num_retries`; its non-None check at line 1080 assigns the explicit zero
   to `max_retries`. The timeout logic at line 1124 retains 300 and converts it
   to `300.0`.
7. Crucially, the [OpenRouter dispatch branch](/home/xav/miniconda3/lib/python3.13/site-packages/litellm/main.py:2523)
   invokes `base_llm_http_handler.completion`, passing that timeout. The installed
   OpenRouter transformation module's introductory comment says calls occur through
   OpenAI/openai.py, but the executable dispatch uses the generic HTTP handler.
   Provider-compatible response types and exception classes do not imply an OpenAI
   SDK network request.
8. The non-stream [generic handler](/home/xav/miniconda3/lib/python3.13/site-packages/litellm/llms/custom_httpx/llm_http_handler.py:463)
   obtains the synchronous HTTPX handler and forwards `timeout` to
   `_make_common_sync_call`, then to
   [HTTPHandler.post](/home/xav/miniconda3/lib/python3.13/site-packages/litellm/llms/custom_httpx/http_handler.py:731).
   That method supplies the explicit timeout to `httpx.Client.build_request` and
   calls `send`. Its constructor's shorter default timeout is therefore not the
   value used by these requests.

The OpenRouter transformer inherits the supported `max_tokens` mapping and
[flattens extra_body into the outgoing data](/home/xav/miniconda3/lib/python3.13/site-packages/litellm/llms/openrouter/chat/transformation.py:65).
Thus the reviewed transformation preserves the completion cap and native reasoning
field. This is a static forwarding finding, not an inspection of a live request
body or proof of an upstream provider's internal enforcement. A token cap does not
set a duration limit.

## Retry layers and their limits

| Layer | Established behavior on the reviewed path |
| --- | --- |
| Scientific slot wrapper | Initial attempt plus at most three transport retries per explicit invocation, with 2/4/8-second delays; records each attempt. Explicit later resumes retain earlier attempts. |
| Project retry wrapper | `max_retries=1` gives one call; no extra internal attempt. |
| Empty-response wrapper | Disabled by `empty_response_retries=0`; completed poor/empty responses are not replaced. |
| LiteLLM completion error wrapper | Explicit zero with default global retry state disables its retry branch; caveat below. |
| OpenRouter HTTP translation | Inherited retry count is zero and retry predicate false; the common loop makes one attempt. |
| HTTPX transport | Default `HTTPTransport(retries=0)`; the project supplies no custom retrying transport on this path. |
| OpenAI SDK request loop | Not called by this OpenRouter HTTP dispatch. |

The generic translator's loop is `range(max(configured_retry_count, 1))`.
[BaseConfig](/home/xav/miniconda3/lib/python3.13/site-packages/litellm/llms/base_llm/chat/transformation.py:181)
returns false for its HTTP-error retry predicate and zero for its retry count;
neither the OpenRouter nor inherited GPT transformation overrides these methods.
This is one attempt, not a hidden repair request.

[HTTPX HTTPTransport](/home/xav/miniconda3/lib/python3.13/site-packages/httpx/_transports/default.py:135)
defaults to zero connection retries. LiteLLM's synchronous handler constructs
`httpx.Client` with its default transport, or an IPv4 transport that also does not
set a nonzero retry count. HTTP connection/address attempts are not themselves
additional completed model proposals.

The installed OpenAI SDK does have
[DEFAULT_MAX_RETRIES=2](/home/xav/miniconda3/lib/python3.13/site-packages/openai/_constants.py:8)
and its [request loop](/home/xav/miniconda3/lib/python3.13/site-packages/openai/_base_client.py:1014)
uses `range(max_retries + 1)`. Those facts cannot explain extra requests in this
OpenRouter branch: the SDK HTTP request loop is not on the reviewed call path.

### Substantiated conditional LiteLLM caveat

The synchronous [LiteLLM error wrapper](/home/xav/miniconda3/lib/python3.13/site-packages/litellm/utils.py:1280)
uses `kwargs.get("num_retries") or litellm.num_retries or None`. Consequently,
explicit zero would not override a separately configured nonzero global retry
value; an explicit retry policy can also select a count. The installed
[global default](/home/xav/miniconda3/lib/python3.13/site-packages/litellm/__init__.py:380)
is `None`, and no assignment to that global or a retry policy was found in the
reviewed project/live-driver path. The frozen requests do not supply such a policy.

This is a real conditional weakness in the installed wrapper, **not evidence that
it caused retries in these runs**. This audit did not inspect live-process globals
or instrument requests. The warranted conclusion is that zero retries holds for
the explicit request plus the unmodified default state used by the reviewed path;
an unconditional claim covering arbitrary global mutations would be too strong.
No package or frozen-code change is proposed for the ongoing runs on this basis.

## What 300 seconds actually bounds

The pure installed-library check returned:

```json
{"connect": 300, "read": 300, "write": 300, "pool": 300}
```

[HTTPX Timeout](/home/xav/miniconda3/lib/python3.13/site-packages/httpx/_config.py:86)
assigns a scalar timeout to these four categories. There is no total-request field.
Several mechanisms can therefore make total elapsed time exceed 300 seconds
without changing this setting or issuing another model request:

- **Global queue:** the project's `flock` wait occurs before the provider call.
  The frozen slot wrapper starts its `wall_s` timer before this wait.
- **Separate network operations:** connection, TLS handshake, writes and reads
  each receive their operation timeout; they do not share one shrinking deadline.
- **Response progress:** [HTTPCore's HTTP/1.1 body loop](/home/xav/miniconda3/lib/python3.13/site-packages/httpcore/_sync/http11.py:196)
  repeatedly reads with the same timeout, and
  [SyncStream.read](/home/xav/miniconda3/lib/python3.13/site-packages/httpcore/_backends/sync.py:124)
  sets the socket timeout before each receive. Bytes arriving between timeout
  intervals can keep the response active for much longer. Non-stream API mode
  still reads the complete response body before returning; see
  [HTTPX Client.send](/home/xav/miniconda3/lib/python3.13/site-packages/httpx/_client.py:879).
  This audit did not observe the pending response's bytes or establish that this
  particular mechanism is occurring.
- **DNS and address iteration:** [SyncBackend.connect_tcp](/home/xav/miniconda3/lib/python3.13/site-packages/httpcore/_backends/sync.py:188)
  calls Python's `socket.create_connection`. In the installed
  [socket implementation](/home/xav/miniconda3/lib/python3.13/socket.py:822),
  `getaddrinfo` runs before per-socket `settimeout`; each returned address then gets
  its own connect attempt. The Python timeout is not a total deadline over resolver
  work plus every address. Resolver/OS behavior was not measured here.

This confirms a limitation of the configured timeout's semantics, rather than
showing it was dropped. The separately enforced two-second optimizer-program
subprocess timeout is a different mechanism and is not assessed or changed here.

## Operational interpretation

A long pending slot alone cannot establish hidden retries, provider deadlock,
generation progress, token consumption or a timeout violation. It also cannot
establish that a previous failed request was unbilled. The previous incident audit's
unknown remote-completion/billing flags remain appropriate.

No source-level evidence found here requires invalidating completed responses or
altering the frozen experiment. Any future decision to add a hard wall-clock
deadline or change retry handling would be an explicit change to execution
semantics, not a transparent correction of `timeout=300`. The root operator retains
control of monitoring and the registered interruption/resume procedure; this audit
did not interrupt, restart or attach tracing to a process.

## Source identity anchors

Paths in this table are relative to the installed `site-packages` directory above.
These hashes describe the inspected files, not a claim that package source bytes
were part of the earlier scientific freeze.

| File | SHA-256 |
| --- | --- |
| `litellm/main.py` | `666dc11c010ee9ebf1bf075b78853d0fb439a955bcda41bc8efbf1775d88407c` |
| `litellm/utils.py` | `231df86ca68b9ffa8383f16e9d02d54149ba78cf5c2011591590239770ad3dbf` |
| `litellm/__init__.py` | `0c341e86305e9c778fbfbebc3d56fe41aad85be497c6226dc82ed9c8276badb3` |
| `litellm/llms/custom_httpx/llm_http_handler.py` | `673ffa36f55bb883e1b90defc4932db18c4c583028421d84d5926bf4df218b41` |
| `litellm/llms/custom_httpx/http_handler.py` | `3d1bb85e41c0c266a470c74a85528023f49fb7bf807a5bc0e4f359eaa17262c2` |
| `litellm/llms/openrouter/chat/transformation.py` | `8a0779d034a828c8e922fbc1647bae60b56b5a7b3d69a69cd7d7d1aa48cacb5e` |
| `litellm/llms/base_llm/chat/transformation.py` | `28e1d9c3f59a13337ba5499fba1e32f59a415e9236727e6d300ecc1ae57e7e74` |
| `httpx/_config.py` | `a4fa7653ec2271f70ab05f8a6111352d876dddee84446788a1767c1a3a372d67` |
| `httpx/_client.py` | `c43f941baefe58c91e96d00039e1868fe719d91453026d7db1647194563bff8d` |
| `httpx/_transports/default.py` | `03379a454c95c0271c4f2c8d25ec437f49f57457f3cf2769748420bf0ecf2e79` |
| `httpcore/_backends/sync.py` | `6e113877d88af549b176c76c826d847ca9d76819acd9682019853eee3994ba35` |
| `httpcore/_sync/http11.py` | `205a1b0f531de491652462969e1d7f4377a98a452bc88f2aa34f6ff0c8892040` |
| `openai/_base_client.py` | `de8a3777d353122a242dd7376bf814668d7557675b81fe36b994d3f65a598c83` |
| `openai/_constants.py` | `5a60b0813e2d1a616c4ab96d6f6e9fddb33c7ed532158933877d8dcb9ca5f92a` |
