# EXP17 timeout diagnostic 01 — engineering evidence only

The original failure **did not reproduce** in this registered sequential check.
All eight `propose_point` calls and all sixteen first/replay subprocess executions
were valid under the unchanged two-second limit. This does not replace any
failed pilot row or establish that the candidate is valid on a complete trajectory.

The inspected pilot candidate had 48 completed TRAIN rows: 25 `timeout` and
23 `nondeterministic`. The fixed diagnostic input was the complete actual
12-observation history immediately before the final failed proposal in the
lexicographically first failing cache row, with its original four-dimensional
bounds and local seed 1975121986. An empty history with the same bounds and seed
provided the second registered input.

Two rounds used reversed order, comparing the exact generated source with the
unchanged trusted seed on identical inputs. Each row below includes both fresh
subprocesses required by `propose_point`; CPU is reaped-child user plus system
CPU. No objective values were computed in this diagnostic.

| Call | Policy | History | Outer status | First/replay | Elapsed seconds | Child CPU seconds |
| --- | --- | --- | --- | --- | ---: | ---: |
| 0 | Candidate | 12 observations | valid | valid / valid | 0.3309 | 0.2371 |
| 1 | Seed | 12 observations | valid | valid / valid | 0.0980 | 0.0652 |
| 2 | Candidate | empty | valid | valid / valid | 0.0657 | 0.0621 |
| 3 | Seed | empty | valid | valid / valid | 0.0655 | 0.0528 |
| 4 | Seed | empty | valid | valid / valid | 0.0657 | 0.0512 |
| 5 | Candidate | empty | valid | valid / valid | 0.0977 | 0.0601 |
| 6 | Seed | 12 observations | valid | valid / valid | 0.0977 | 0.0576 |
| 7 | Candidate | 12 observations | valid | valid / valid | 0.2303 | 0.2005 |

At the fixed nonempty history, the candidate used mean child CPU 0.2188 seconds
versus 0.0614 for the seed, about 3.56 times as much. Its elapsed time was
0.230–0.331 seconds including replay. This is evidence of greater computational
cost on this one input, while remaining below the registered limit in both calls.
It is not evidence that this input intrinsically requires more than two seconds.

Source inspection finds that the candidate scores 400 acquisition proposals and
reconstructs the same history covariance matrix, Cholesky factorization and
training solve inside each proposal's GP-posterior computation. Its Latin
hypercube routine also performs a preliminary permutation loop whose result is
discarded. These are plausible sources of unnecessary work. No intervention
isolated their individual costs, and the source was not repaired or edited.

Concurrent host load was uncontrolled. Recorded 1/5/15-minute load averages were
approximately 2.02/4.82/5.79 on a host reporting 20 CPUs during these calls. Load
averages do not determine CPU scheduling or explain the earlier failures.
The earlier failure and subsequent success are compatible with runtime-state
dependence, but this diagnostic does **not** identify the system cause. It also
does not measure behavior at longer histories or under the pilot worker schedule.

The existing `nondeterministic` status covers either unequal valid replay points
or an invalid second execution after a valid first execution. Original pilot
records do not retain the separate first/replay status, so their label alone
does not establish stochastic candidate behavior. The diagnostic's sixteen
underlying executions were all valid and each pair returned equal points.

## Provenance and verification

- Candidate SHA-256:
  `3110652b6dc6a902e504216ffdc9103b583dacbf15ca41d712a595309dc46947`.
- Trusted seed SHA-256:
  `5b74e5a3fe2fc90fcb42a603b058fa7befbf65f38775caa77acd30cdf619b640`.
- Preregistered protocol SHA-256:
  `15fb9173b0ba2e603c2b991944a9e7e5039dbf88a46e18d4fc286cc2514e40d3`.
- Results SHA-256:
  `361c2ab51b909ec2f13b579fb8c837b4fa5feebef19985c630890226022bc85b`.
- Verification SHA-256:
  `b71dd8362c1a49460fbc31a14dc6f96afc8fe1e3accf800bbce9909078fd0993`.

`verify.py` independently checked preregistration chronology, exact call order,
source and input hashes, original cache-history provenance, all 48 original
failed cache rows, subprocess totals, replay semantics, per-call/aggregate
agreement and unchanged hashes of all 52 registered original files. It performed
no candidate, model or objective execution. `verification.json` records a pass.

The first shell launch by direct script path failed at Python import before
entering `main`; `launch_01.json` preserves that zero-call event. Launching the
unchanged source using its module name supplied the repository import path:

```bash
/tmp/phase0-venv/bin/python -m artifacts.optimizer_discovery.exp17.engineering_diagnostics.timeout_01.run
/tmp/phase0-venv/bin/python -m artifacts.optimizer_discovery.exp17.engineering_diagnostics.timeout_01.verify
```

Only the second launch executed the eight registered calls. There were no
additional proposal reruns, timeout changes, model requests or objective calls.
All results are separate engineering evidence and are excluded from efficacy
analyses and the pilot/main allocation grid. No existing frozen source, prompt,
grid, selection rule or original evidence was changed.
