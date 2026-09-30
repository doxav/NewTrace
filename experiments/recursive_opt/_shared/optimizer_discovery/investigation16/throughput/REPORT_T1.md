# T1 — offline concurrency preserves behavior and reduces elapsed evaluation time

The unchanged seed and exact old A2/41 validation representative were evaluated
on six fresh tasks with two local seeds. Every condition used the same24
combinations,32 objective calls per trajectory,2s timeout and exact existing
subprocess/replay boundary. The worker-count schedule was frozen before execution:
[4,1,16,8], then[16,8,4,1]. No model calls or production dependencies were added.

All192 original trajectories are valid and match exactly across all eight
conditions, comparing every evaluator-result field except execution duration.
There are no exceptions, timeouts, source changes or scientific mismatches.
The original run contains6,144 objective calls and12,288 subprocess executions,
plus768 shared reference calls. Original freeze:
`f14364db36b9082a9e035cfd03ebb6b3565910cfe80e27cf1f5035c180457ffa`.

## Suspension audit and explicit timing correction

The user reported machine suspension. Per-trajectory timestamps identify exactly
one affected condition: round0/workers1 spans06:38:24–08:59:59 UTC on2026-09-09.
Its realtime span is8,494.890s, while recorded monotonic execution time is87.100s,
a difference of8,407.790s. All seven other conditions have realtime-minus-monotonic
differences below2ms. Thus monotonic timing omitted the suspend interval, but the
mixed pre-/post-resume serial measurement is unsuitable as a clean reference.

The original evidence and original result file remain unchanged. The original
`timing_eligible` flag tests checkpoint resume, not suspension; the separate
[clock audit](timing_audit.json) adds that missing check. The original headline
speedups7.08×/9.40× versus serial are explicitly superseded by the corrected values
below. No objective/selection result is invalidated.

[T1-R1](TIMING_REPAIR_R1.md) was registered before repeating only that one serial
condition, once, in a separate directory. It took55.690s; its realtime/monotonic gap
is−1.003ms. All24 repeated scientific trajectories exactly match the originals.
The repetition adds768 objective calls,1,536 subprocess executions and768 reference
calls. It is a timing-only engineering replacement, not a rerun selected for a
better scientific outcome. No unaffected condition was repeated.

## Corrected timing observations

Each value covers the same24 trajectories. Two observations under this host's
changing load are an engineering estimate, not a confidence interval or scheduling
guarantee. Host:20 logical CPUs, affinity includes all20. Research-process and
child CPU time are preserved alongside wall time in raw timing files.

| Offline workers | Two healthy elapsed observations (s) | Mean (s) | Mean s/trajectory | Speedup vs healthy serial |
|---:|---:|---:|---:|---:|
| 1 | 55.690,52.628 | 54.159 | 2.25663 | 1.00× |
| 4 | 19.710,15.540 | 17.625 | .73437 | 3.07× |
| 8 | 10.178,9.550 | 9.864 | .41099 | 5.49× |
| 16 | 7.490,7.367 | 7.429 | .30953 | 7.29× |

Sixteen workers is the fastest tested configuration, as specified by the T1
criterion. Eight workers is a reasonable conservative operational choice for P1:
it preserves behavior in the measured sample and leaves more CPU capacity for
new programs with different computational cost and other active processes. This
is an explicit resource-margin choice, not a claim that8 is fastest or universally
safe. All generated programs still face the same hard timeout; monitor typed
failures without silently changing the confirmed run's worker setting.

For the proposed P1 upper allocation of15,552 train/validation plus720 audit
trajectories (16,272 total), simple scaling at .410991s/trajectory gives about
6,688s or **1.86h of offline evaluation** with8 workers. This is not a runtime
upper bound: candidate complexity, scheduler load, invalid early termination,
batch filling and cache reuse can change realized time. It excludes LLM generation
latency and any non-overlapped orchestration. A48-trajectory training panel costs
about19.7s under the same rough scaling; a24-trajectory validation panel about9.9s.

This measured scheduling improvement supports affording S1's more informative
evaluation panels. It does not improve optimizer objective values, create extra
proposal slots or increase live LLM concurrency; that remains1.

The previously completed S1 timings were independently checked after the pause
report: its training and audit realtime/monotonic discrepancies are below0.9ms.
They are not affected by this suspension.

## Verification and artifact locations

- [Original results](results.json), [original raw conditions](raw),
  [corrected separate results](timing_repair_r1/results.json).
- Four timing/concurrency tests pass; source and protocol hashes remain frozen.
- Every completed trajectory is retained:192 original plus24 timing-repair
  trajectories,6,912 objective calls,13,824 subprocess executions,1,536 reference
  calls. All216 outcomes are valid; the new24 agree with the original reference.
- Original analysis recomputes without altering its immutable result; the
  corrected summary is separately identified and never substituted silently.

Commands:

```bash
/tmp/phase0-venv/bin/python -m artifacts.optimizer_discovery.investigation16.throughput.run_throughput analyze
/tmp/phase0-venv/bin/python -m pytest -q artifacts/optimizer_discovery/investigation16/throughput/test_throughput.py artifacts/optimizer_discovery/investigation16/throughput/test_timing_repair.py
```

Do not rerun the timing-repair execution command merely to regenerate a report;
its raw output and corrected JSON already preserve the measured replacement.
