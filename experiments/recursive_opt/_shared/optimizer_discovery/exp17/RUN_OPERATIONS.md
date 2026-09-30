# Successor study operations

These instructions cover the unfrozen orchestration helper. The registered drivers
remain responsible for scientific configuration, request identities, proposal slots,
selection barriers and immutable evidence. The helper neither loads credentials nor
retries a failed stage. Do not edit frozen sources to recover an interrupted run.

From the repository root, after the corresponding engineering gate and main freeze
exist, an explicit launch is:

```bash
/tmp/phase0-venv/bin/python -m artifacts.optimizer_discovery.exp17.run_pipeline exp17
/tmp/phase0-venv/bin/python -m artifacts.optimizer_discovery.exp17.run_pipeline exp18
```

Each invocation delegates `generate`, `receipts`, `select`, `audit`, `analyze`, then
`verify_numerics`, in that order. It stops on the first unsuccessful child. A new
invocation creates a new `run/runtime/launch_NNN/` journal and delegates all stages
again; completed proposal reuse and uncertain-request reconciliation belong to the
frozen driver. A completed poor or invalid response is never replaced.

## Inspect an interruption before resuming

`launch_started.json` records the parent PID, commands and freeze identity. Each
`step_NN_started.json` records intent; `step_NN_process.json` records the child PID,
command, parent PID and recording time immediately after process creation, before
waiting. `step_NN_finished.json` records the observed exit code, or an exception
type without its message. These records are immutable and stdout/stderr are inherited.

A killed parent can leave its child running. A missing finish record does not prove
that the child or its remote request stopped. Inspect the recorded PID, process
command and start time (for example `ps -p <PID> -o pid,ppid,lstart,args`) before
resuming. PIDs can be reused; the PID alone is not an identity guarantee. A kill
between process creation and journal persistence can also leave a child with no
process record, so inspect processes associated with the recorded command and run.
Do not print process environments or credential sources while investigating.

The pipeline's `.pipeline.lock` and the driver's separate `.process.lock` reject
concurrent writers at their respective boundaries. Keep the lock files: deleting a
lock file can bypass an active file lock. Wait for an active driver to finish before
resuming. If a request may have completed remotely, follow the frozen reconciliation
policy; do not infer that an absent local response authorizes a replacement.

## Resume numeric verification after analysis exists

If generation, selection, audit and analysis completed and only numeric verification
was interrupted, run the existing verifier directly after confirming no writer is
active:

```bash
/tmp/phase0-venv/bin/python -m artifacts.optimizer_discovery.exp17.verify_numerics artifacts/optimizer_discovery/exp17/run
/tmp/phase0-venv/bin/python -m artifacts.optimizer_discovery.exp17.verify_numerics artifacts/optimizer_discovery/exp18/run
```

Choose the command for the affected study. This adds a numbered
`numeric_verification/attempt_NNN.json`; it preserves earlier attempts. It performs
local objective/metric integrity recomputation from preserved observations, with
no model calls or candidate subprocesses. Its additional integrity evaluations are
reported separately from scientific budgets. It does not repair earlier values or
replace an existing analysis report.

The verifier also inventories operational journal files. If the parent is delayed
after spawning the verifier, its child-PID record can appear during that inventory
and produce a conservative changed-input failure. Confirm that only this recorded
operational addition changed, preserve the failed attempt, and use the targeted
command after all writers stop. Do not dismiss changes to scientific inputs.

## Provider metadata and reporting revisions

A full pipeline replay includes receipt collection, including within `generate`.
Previously unavailable provider metadata may then become available. Even when all
scientific values and selections remain identical, newly known usage/cost can change
the recomputed aggregate. The immutable `analyze` command will reject a different
`analysis_results.json` instead of overwriting the original report. This is a
reporting provenance issue, not permission to regenerate proposals or change an
experiment's scientific semantics.

Preserve the original analysis and receipt evidence. If updated usage is required,
make a separately named, explicitly documented reporting revision after verifying
that scientific per-seed results, selections and contrasts are unchanged. There is
no automatic revision mechanism in this helper. A targeted numeric-verifier resume
does not refresh or certify equivalence of an earlier saved analysis report; any
metadata discrepancy must still be disclosed separately.
