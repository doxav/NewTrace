# Portable optimizer benchmark and executor boundary

Engine-independent host API:

```python
from pathlib import Path
from artifacts.optimizer_discovery import benchmark as benchmark

source = Path("optimizer.py").read_text()
# Host owns task parameters and normalization; never place them in candidate workspace.
row = benchmark.evaluate(source, host_task, local_seed, budget=32, deployment=True)
```

`benchmark.make_tasks(phase, split)` reconstructs the frozen benchmark. The host
computes normalization and metrics. External search engines can submit exact source
strings and consume typed training evaluations; they need not adopt recursive_opt.
All future comparative arms must use the frozen source screen, complete task panels,
local seeds, slot accounting, validation selection and deployment fallback.

The current launcher is `optimizer_program._execute_once`: each proposal gets a
fresh directory containing optimizer.py, worker.py, request.json, stdout.txt and
stderr.txt. request.json contains only history, bounds and seed. A host-side task
manifest may contain hidden benchmark parameters; a candidate-side task manifest
must never contain them, instance IDs or split labels. Captured stdout/stderr and
typed failures are retained in evaluation records. Source and observations are
file-based JSON strings; large host records are losslessly gzip-compressed.

A future Project-1 executor can replace `_execute_once(source, public_payload,
timeout_s)` and return the same ProposalResult, preserving deterministic double
replay and the evaluator's budget/fallback rules. Optimizer source, mathematical
objectives, metrics and recursive_opt search objective need no changes. No snapshot,
branch, restart or sandbox infrastructure was added here.

The existing subprocess boundary sanitizes the credential environment and separates
the candidate API from objective data. It is **not an operating-system security
sandbox** and does not prove filesystem confinement against adversarial code.
The declared conservative AST screen is a protocol check, not that stronger guarantee.

Next integration: in a new preregistered experiment, implement a thin FunSearch or
OpenEvolve proposal adapter producing this exact source artifact. Allocate eight
completed proposals per outer seed and the identical train/validation evaluation
schedule. The engine may consume training feedback only. Collect every proposal,
then apply the common validation selection and global holdout barrier. Additional
engine-internal model/repair calls must consume declared slots; no hidden budget.

Completed EXP-15 evidence uses lossless compression for large traces, the event
stream and exact source exports. To recompute its unchanged frozen analysis:

```bash
python -m artifacts.optimizer_discovery.reporting artifacts/optimizer_discovery/exp15/raw
```

This materializes the exact archived event bytes temporarily, checks equality with
the preserved aggregate and removes its temporary log. It makes no model or candidate
calls. To deploy the validation-selected representative as a normal source file:

```bash
gzip -dc artifacts/optimizer_discovery/exp15/selected/A2_seed_41.py.gz > optimizer.py
```

The export index records the hash of those exact decompressed bytes. Archived
scientific evidence and source strings have not been reformatted.
