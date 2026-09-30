# T1 — bounded engineering check of offline evaluator concurrency

Register before evaluation. This engineering probe compares worker counts while
holding source, tasks, local randomness, evaluator, timeout and objective budget
fixed. It tests throughput and identical scientific execution, not algorithmic
performance or model generation. No LLM calls or production dependencies.

Sources: unchanged EXP-15 seed and exact validation representative A2/41,
SHA256 `1684f91acdc36c0ca6aac70afeb9cc2c4eed7ab847926d5880590e059266abb7`.
Tasks: `G.fresh_tasks("T1","train",1)`, six balanced new instances, disjoint from
EXP-15 and S1. Local blocks16501/16502 use `G.local_seed("T1",block,task)`.
Every group evaluates the same24 source/task/local combinations at B32, exact
unchanged `B.evaluate`, hard timeout2s and two subprocess executions per valid point.

Two preregistered rounds, worker counts in these exact orders:

1. [4,1,16,8]
2. [16,8,4,1]

Thus192 trajectories,6,144 objective calls,12,288 expected valid subprocess
executions. Shared normalization preparation costs6×128=768 objective calls,
outside measured candidate evaluation. Freeze source/config/environment/protocol
hashes, exact tasks and job IDs before execution. Persist each trajectory unchanged
and never retry a completed invalid execution. Interrupted unfinished local
trajectories may replay under the same ID; a partially resumed condition is excluded
from throughput timing comparisons and explicitly retained as interrupted.

Each condition measures wall time, research-process CPU time, accumulated child
CPU time and actual valid/invalid/objective/subprocess counts. Comparisons project
out execution duration only and require exact equality of every remaining evaluator
result field for each common source/task/local combination across all8 conditions.
Record exceptions separately rather than suppressing them or fabricating metrics.
Any mismatch/timeout/exception prevents a claim that this worker count preserved
behavior. Do not selectively rerun an unfavorable condition.

Report both timing observations per worker count, their arithmetic mean and speedup
relative to worker1. Two repeats under one host load are an engineering estimate,
not a controlled performance confidence interval or an OS scheduling guarantee.
Recommendation: use the fastest tested configuration only if all its executions
are valid and exactly match the common scientific trajectory reference; report
uncertainty and resource context. This does not increase LLM concurrency, objective
allocations or proposal slots. It only changes local evaluation scheduling.
