# EXP-15 verification record

Baseline at 90651f47: 559 passed, 2 skipped (baseline_tests.txt).
Pre-freeze affected regression: 581 passed, 2 skipped (pre_freeze_tests.txt).
Broader offline run after freeze: 766 passed, 3 skipped, one pre-existing warning
(full_offline_tests.txt). No frozen source changed between the final freeze and this run.

Affected command:

```bash
mapfile -t phase1_tests < <(rg --files tests/unit_tests | rg '/test_recursive.*\.py$' | rg -v '/test_recursive_opt_review_regression.py$' | sort)
/tmp/phase0-venv/bin/python -m pytest -q -rs --disable-socket --allow-hosts=127.0.0.1,localhost "${phase1_tests[@]}" tests/unit_tests/test_objectives.py tests/unit_tests/test_evaluators_vector.py tests/unit_tests/test_trainers_multiobjective.py
```

Broader command:

```bash
/tmp/phase0-venv/bin/python -m pytest -q -rs --disable-socket --allow-hosts=127.0.0.1,localhost tests/unit_tests --ignore=tests/unit_tests/test_recursive_opt_review_regression.py
```

The excluded module requires external Trace-Bench/backend behavior: one test is a
live provider integration and the other exercises missing-backend failure through
that external adapter. The same exclusion was used in the baseline; no test was
weakened or marked xfail. Outbound sockets were disabled for offline tests.
Three existing skips: absent Graphviz executable and two optional graph/telemetry
backends. Existing SyntaxWarning: invalid escape sequence in test_optimizer_xml_parsing.py:336.
The provider-dependent optimizer suite was not substituted for the real registered
pilot/confirmatory calls.

New/interface-targeted command: 47 passed in 21.92s (new_tests.txt).
The production spec.py lint inventory matches the baseline exactly by rule; see
core_lint_comparison.json. No unrelated cleanup was performed.

New implementation, tests and formatting checks:

```bash
/tmp/phase0-venv/bin/python -m pytest -q tests/unit_tests/test_recursive_exp15.py tests/unit_tests/test_recursive_optimizer_benchmark.py tests/unit_tests/test_recursive_optimizer_program.py
/tmp/phase0-venv/bin/python -m ruff check artifacts/optimizer_discovery/{benchmark,exp15,evidence}.py tests/unit_tests/test_recursive_exp15.py tests/unit_tests/test_recursive_optimizer_benchmark.py
/tmp/phase0-venv/bin/python -m black --target-version py313 --check artifacts/optimizer_discovery/{benchmark,exp15,evidence}.py tests/unit_tests/test_recursive_exp15.py tests/unit_tests/test_recursive_optimizer_benchmark.py
/tmp/phase0-venv/bin/python -m artifacts.optimizer_discovery.exp15 preflight
git diff --check
```

Raw candidate code is retained as strings in JSON, with gzip for large records.
It is not reformatted as implementation source. Packaging round-trips exact bytes.
No production dependencies were added. Full-repository automatic formatting was
not run because it would modify unrelated historical code/evidence. Targeted
formatting/lint and the broad offline regression cover this change.

Final evidence integrity and aggregate recomputation are recorded in
final_integrity.json and resource_audit.json. Staged credential/size checks accompany
the final data commit. Every registered seed and all 80 completed response slots
are represented; no holdout trajectory required fallback.

Additional audit during confirmation, without changing frozen scientific source:
49 new/interface-targeted tests passed in 13.39s (resume_audit_tests.txt). This adds
an actual production-A2 interruption/resume regression: the first completed response
is preserved byte-for-byte; only the unfinished second slot makes another model
request. A missing outer seed blocks aggregation and a missing selection blocks the
global holdout freeze. A modified frozen source byte fails preflight. All these
checks use mocks only in unit tests; confirmatory generation remains real.

Reporting repair R15-LATENCY-01: 50 targeted tests passed in 14.02s
(reporting_repair_tests.txt). A new descriptive helper matches the completed
response to its own attempt ID before calculating latency. It is not imported
by generation, evaluation, selection or primary analysis. The frozen preflight
still passes. The earlier manual latency summaries are preserved and explicitly
superseded; all original scientific records and decisions remain unchanged.

Selected-program export: 51 targeted tests passed in 14.38s (export_tests.txt).
The export test additionally exercises lineage diffs, preserves trailing whitespace
in the raw source and rejects a tampered response or frozen selection. It passes
independently after those stronger assertions. Exports use deterministic gzip and
record both the exact evaluated-source hash and compressed-file hash. No generated
program is reformatted or executed by the reporting helper.

Trace archive packaging: the first full A2 gzip trace was 899,427 bytes, above the
repository's 500 KiB added-file gate. The first raw-data commit included that file;
a following packaging commit replaces its working-tree representation with a
115,372-byte XZ archive. Exact uncompressed JSON bytes and the original gzip hash
are preserved and checked. No source string or trace field changes. The original
gzip remains in Git history. The final size gate is rerun rather than relaxed.

52 targeted tests passed in 13.97s (archive_tests.txt), including refusal to archive
an unfinished generation, byte-preserving roundtrip, hash integrity and idempotence.
The frozen completed-block resume was also exercised on the actual seed-11 block
without a live client after archiving; it returned without new generation.
`reporting.read_trace` reads and verifies these post-run archives. The frozen
runner itself still writes its original JSON/gzip representation. This archive
operation is reporting/storage only and does not amend scientific semantics.

Final broader offline regression on the current implementation and reporting tools:
771 passed, 3 existing optional skips in 59.00s (final_offline_tests.txt), using the
same all-unit-tests command and external-backend module exclusion above. No new
skip or xfail was added. The original frozen generation/evaluation source remains
unchanged; the additional tests cover reporting and lossless archive integrity.

Reporting repair R15-EXPORT-PATH-01: invoking the exporter first with a relative
path and then its absolute equivalent changed path labels in the index, so the
immutable writer correctly refused the second write. The failing regression is
preserved in export_path_red.txt.gz (one intended failure). Canonical path labels in
the reporting helper resolve it without changing any frozen scientific module.
52 targeted tests pass in 14.25s (export_path_green.txt). Both path forms now return
the identical index, whose bytes match the previously committed export at 6ffdd524;
all sources and lineage bytes remain unchanged. Targeted Ruff/Black and the frozen
preflight pass. No scientific result was invalidated or replaced.

After this final reporting change, the full offline command passes again:
**771 passed, 3 existing optional skips in 54.89s** (completion_offline_tests.txt).
The same external-backend module is excluded; no exclusions, skips or xfails changed.

Final raw audit verifies all source hashes, all request settings, 80 unique response
IDs, five actual production Trace blocks with eight updates each, the rotated
sequential request order, equal logical evaluation allocations, validation isolation
and the global selection barrier. Every one of the 1,020 cache file names matches
its complete key hash, all cached numeric metrics recompute from retained observations,
and the complete analysis equals raw/results.json exactly. All cache records are
ordinary JSON, so the frozen cache's unsupported compressed-record edge was not
exercised. There were no extra candidate or model calls in this audit. Reference
normalization recomputation is accounted separately in resource_audit.json.

Completed event-stream packaging: events.jsonl was 1,463,776 bytes, over the 500 KiB
file gate. The exact JSONL stream is retained as a 45,361-byte deterministic gzip,
with both hashes in raw/events_archive.json. The reporting adapter materializes
the original bytes temporarily for the unchanged frozen analyzer, checks exact
result equality, then removes only its temporary copy. No event is dropped or
rewritten. The archived source stream is the final evidence, not an untracked log.
The original frozen runner is unchanged and can still create new unarchived runs.

An intended missing-function failure is retained in events_archive_red.txt.
53 targeted tests pass in 14.26s (events_archive_green.txt), including completeness,
roundtrip, tamper refusal and temporary-file cleanup. Actual complete EXP-15 analysis
matches its original results after archival (events_archive_integrity.json).

Recompute the primary aggregate from the delivered archive without model or
candidate calls; the adapter verifies equality with the preserved result:

```bash
/tmp/phase0-venv/bin/python -m artifacts.optimizer_discovery.reporting artifacts/optimizer_discovery/exp15/raw
```

The full `evidence.verify(Experiment(raw_root, "confirmation"))` scientific audit was
run before archival, while the complete event stream was materialized. It requires
that stream when rerun. `reporting.read_trace` verifies losslessly archived
production traces, and `reporting.export_programs` verifies all selected A2 artifacts.

Final release regression after all reporting/archive changes: **772 passed, 3 existing
optional skips in 54.99s** (release_offline_tests.txt), with the same socket restrictions
and external-backend module exclusion. All 53 targeted tests, targeted Ruff/Black,
frozen preflight and exact archived-result recomputation pass. No frozen source changed.

The failing path-export test log is gzip-compressed without changing its trailing
whitespace; test_log_packaging.json records both hashes. The staged whitespace gate
passes without an exemption.
