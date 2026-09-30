# Runtime timeout diagnostic — local HTTP only

This engineering probe tests the installed client stack; it makes no model call
and does not modify a running experiment or frozen source. Every network
connection in the probe process is restricted to IPv4 loopback. An explicit
nonsecret placeholder credential is supplied; no local secret is loaded.

Predicted mechanism from installed source inspection: `timeout=300` reaches
HTTPX and applies separately to network operations, including each response-body
read. It is not an absolute request deadline. A nonstreaming response may therefore
take longer than the configured read timeout if bytes continue to arrive.

Use the actual repository `make_live_llm` wrapper, LiteLLM 1.75 OpenRouter adapter,
HTTPHandler, HTTPX and HTTPcore, substituting only the HTTP endpoint with a local
server and the model response with declared synthetic JSON. Keep client retries
disabled (`max_retries=1`, `num_retries=0`, empty retries zero).

Cases:

1. Fast response, wrapper timeout 300, no request override: record actual HTTPX
   timeout extensions and forwarded model/temperature/top-p/token/reasoning fields.
2. Response sends 12 chunks of JSON-legal whitespace at 0.1-second intervals
   before the final JSON body. Wrapper default 300, request override 0.5 seconds.
   Expected: success after at least 1.2 seconds, exactly one HTTP request, while
   actual HTTPX read timeout is 0.5 seconds.
3. Headers arrive, but no body bytes for one second, request timeout 0.5 seconds.
   Expected: typed read-timeout failure after roughly 0.5 seconds and exactly one
   HTTP request. Preserve error type only; do not log request headers.

These tests establish transport semantics, not the exact bytes or server-side
cause of any past OpenRouter response. An observed remote duration above 300
seconds is compatible with this mechanism. Active TLS traffic alone does not
identify whether the remote payload was heartbeat whitespace or response content.
