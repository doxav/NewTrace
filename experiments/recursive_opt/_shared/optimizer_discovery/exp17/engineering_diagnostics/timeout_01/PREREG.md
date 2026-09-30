# EXP-17 timeout diagnostic 01 — preregistration

Registered before execution at wall timestamp 1789041583352086415. Engineering only.

Candidate `3110652b6dc6a902e504216ffdc9103b583dacbf15ca41d712a595309dc46947` is unchanged. Compare it with trusted seed `5b74e5a3fe2fc90fcb42a603b058fa7befbf65f38775caa77acd30cdf619b640` on the exact 12-observation last history of cache `02f60de7a8007d2dc735e6a37fc1e3d99edfd1144af6ebc20cfb7987098a7b91` and on empty history, using the same legal bounds and local seed. The selected record is the lexicographically first completed failing TRAIN cache row of this candidate. All 48 completed rows were available: {'nondeterministic': 23, 'timeout': 25}. No objective value is recomputed.

Round 0: candidate/failed, seed/failed, candidate/empty, seed/empty. Round 1 is the exact reverse. Eight calls to existing `propose_point`, at most sixteen existing child executions, original two-second timeout. No retries or parameter changes.

Exact sources, input payloads, order and hashes are in protocol.json, inputs.json and sources.json. The observer wraps the unchanged child launcher only to record counts, first/replay results, clocks, and CPU; it asserts the existing sanitized environment. Generated code executes only through the existing subprocess boundary.

Host load is uncontrolled. Child CPU and relative costs can describe computational work but cannot prove that a specific system condition caused the original failures. The existing nondeterministic label also covers a failed second replay. All outcomes remain preserved, including timeouts, exceptions and mismatches. No candidate/source/settings edits or further diagnostic reruns are allowed.
