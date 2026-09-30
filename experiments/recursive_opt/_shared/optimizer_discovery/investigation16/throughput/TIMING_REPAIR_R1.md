# T1-R1 — one timing-only replacement after documented machine suspension

The user reported machine suspension. Recorded per-trajectory realtime stamps
confirm that T1 round0/workers1 spanned 06:38:24–08:59:59 UTC on2026-09-09:
8,494.890 calendar seconds versus87.100 monotonic seconds, a gap of8,407.790s.
All other seven condition spans differ from their measured monotonic wall time by
under2ms (ordinary condition setup/teardown overhead). All192 scientific trajectories
remain valid and identical. Monotonic time excluded the suspension, but CPU clock
and post-resume behavior make that mixed serial timing an unsuitable reference.

Preserve the entire original T1 run, summaries and timing_eligible fields. The
original eligibility flag addresses checkpoint resume, not machine suspension;
this reporting amendment adds the missing clock audit without rewriting evidence.
No objective values, source hashes, validity decision or algorithmic outcome changes.

Before another timing run, freeze this amendment and the repair script. Repeat
**only round0/workers1**, once, under `timing_repair_r1/raw/`. Use the exact original
24 source/task/local combinations,32 objective evaluations,2s timeout and original
runner. The adapter changes only its output-root binding in a separate process;
the frozen source file and all scientific inputs remain unchanged. This is24
additional planned trajectories,768 objective calls,1,536 subprocess executions,
plus768 normalization-reference calls in the new process, separately accounted.

Compare every new scientific result against the original matching combination,
excluding execution duration only. Do not replace an outcome if it changes.
Audit the replacement's realtime span against monotonic elapsed time. If it also
shows a clock discrepancy above1s, retain and flag it; do not loop until a favorable
time occurs. No repeat is authorized merely because it is slower than the old time.

Write a separate corrected timing summary using this predetermined replacement
and the seven unaffected original observations. Show both original and corrected
ratios. Preserve all source/task/seed/count/exception information. This correction
changes engineering throughput interpretation only, not scientific optimizer results.
No model call and no new production dependency.
