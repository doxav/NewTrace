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

Final evidence-integrity, staged credential scan, source-hash verification and
aggregate recomputation are recorded separately after confirmation completes.

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
