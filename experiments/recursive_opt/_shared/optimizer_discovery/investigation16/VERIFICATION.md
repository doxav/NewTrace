# EXP-16 verification record

Starting SHA: `13ebda2242e1c18022591737b113030ca2ce2da2`.
Branch: `codex/investigation-feedback-exp16`. No production dependency, commit,
push, merge, PR or external message has been created for this investigation.
The earlier Phase-0/EXP-15 scientific sources and evidence remain unchanged.

## Passing verification before P1-E1 execution

The complete unit suite passes **831 tests, 3 existing optional skips** in129.47s:

```bash
/tmp/phase0-venv/bin/python -m pytest -q -rs \
  --disable-socket --allow-hosts=127.0.0.1,localhost \
  tests/unit_tests \
  --ignore=tests/unit_tests/test_recursive_opt_review_regression.py
```

Output: `frozen_unit_tests.txt`. The excluded module requires the external
Trace-Bench/backend integration; this is the same exclusion used for the original
baseline. The three skips are absent Graphviz `dot` and two optional graph/telemetry
backends. No new xfail or skip was introduced. Outbound sockets are disabled;
loopback is allowed for offline HTTP timeout tests.

All13 additional research test modules pass separately, **117 tests**. Together
the two non-overlapping file groups execute **948 passing tests**. Exact commands,
exit codes and output paths are in `verification_isolated/results.json`. Each uses:

```bash
/tmp/phase0-venv/bin/python -m pytest -q -rs \
  --disable-socket --allow-hosts=127.0.0.1,localhost PATH_TO_RESEARCH_TEST
```

The individual module runs avoid collisions between repeated test basenames and
the existing module/evidence-directory naming pattern, such as `generation.py`
beside `generation/`. The first combined collection failed on duplicate module
names; `--import-mode=importlib` also collided with that module/directory pattern.
Both failed collection logs are retained as `frozen_engineering_offline_tests.txt`
and `frozen_engineering_offline_importlib_tests.txt`. No test was removed to obtain
the passing result, and no frozen source was renamed or modified.

Earlier passing records include812 unit tests with the then-under-review new
driver temporarily excluded,19 final driver tests, and66 combined analysis,
owner, driver and Trace tests. The larger final unit and isolated-module runs
supersede those intermediate coverage totals without discarding their logs.

Targeted Black and Ruff pass on the new implementation/test files. Do not run an
automatic formatter over raw generated source or earlier frozen evidence.
`git diff --check` passes. All scientific source hashes are checked by the stage
preflights; exact-byte packaging and credential checks are recorded separately.

## Test-first and independent-review findings

The stage reports and `WORK_LOG.md` preserve intended failing tests and subsequent
passing checks. Independent review caught, before P1-E1 generation: native Trace's
two feedback-envelope forms, ordinary invalid final exploration artifacts, gzip
record handling, incomplete engineering gates, cache-key integrity, response/receipt
identity and an invalidity denominator that incorrectly included the seed.

The source/analysis/protocol freeze for P1-E1 is
`4be570f58e0522d76d17e9f67b9bb1338eed060dc800319c5b083f23fa1abe75`.
`production_engineering/frozen_sources.zip` preserves93 exact source/dependency
files, including all production `opto` Python files. Its SHA256 is
`ea11026cc21bf4fe5c589522ba97f57899f363e34dd5803dbbdc7430b2bbf04a`.
All bytes round-trip and the actual active credential is absent from the archive
inputs. Source and archive hashes are in `production_engineering/source_snapshot.json`.

## Live evidence status

G1 completed24 real responses; its read-only integrity audit passes. F1 remains
in progress under its exact freeze, with completed responses preserved across a
temporary provider/DNS interruption. No F1/P1 efficacy conclusion is included in
this verification record until their complete raw evidence is available. The
final evidence, chronology, usage and aggregate checks will be appended after
execution; test success alone is not a scientific outcome.

## Additional post-audit numeric verifier

The independent helper `production/verify_numerics.py` adds10 passing tests; its
combined run with the26 frozen analysis tests reports36 passed in3.24s. Thus the
non-overlapping total is **958 passing tests**, with the same3 optional skips.

```bash
/tmp/phase0-venv/bin/python -m pytest -q \
  artifacts/optimizer_discovery/investigation16/production/test_verify_numerics.py \
  artifacts/optimizer_discovery/investigation16/production/test_analysis.py
```

The helper SHA256 is
`fa07e8c59190d0caa810c051168520f89edfc62c02f3f124e59455b69500edf7`.
It first requires completed generation, selection and audit plus the frozen
structural analysis checks. Only then does it independently rebuild normalization
and objective values. It executes no candidate or model, changes no saved value,
and accounts for its additional reference/observation calls separately. No real
P1 or E1 evaluation data was read during implementation; the10 tests use synthetic
completed bundles and exercise premature-access guards and corrupt evidence.
This invocation's output is retained in the agent tool session35617; no separate
log file was written. Unlike the earlier offline-suite invocation, this command
did not set socket-blocking flags; its tests use synthetic data and make no network
calls. Black, Ruff and `git diff --check` pass for these two new files. They do not change
the frozen scientific implementation or the already sealed E1 archive.

## Format and client-transmission checks during F1

The unfrozen F1 read-only analysis/format test module now passes17 tests (five
additional cases beyond the12 previously counted). Its exact output and checks
are in `feedback_experiment/generation_failure_verification.json`. A null final
content exposed a reporting-helper error; it was fixed before the full F1 audit,
without changing execution, source extraction, metric definitions or evidence.

Two new message-transmission tests pass alongside the three existing timeout
tests: **5 passed, 7 documented HTTPX/Pydantic warnings in5.42s**.

```bash
/tmp/phase0-venv/bin/python -m pytest -q \
  artifacts/optimizer_discovery/investigation16/runtime/test_message_probe.py \
  artifacts/optimizer_discovery/investigation16/runtime/test_timeout_probe.py
```

These use a loopback server and no actual credential or external model connection.
The exact recorded F1 request was also sent once through that local client path;
both messages and generation settings were preserved. Evidence SHA256:
`911d688fe6b7fcec95a9caf74eb8405d86b58693855e879b07943c15a3dd9321`.
The current non-overlapping total is therefore **965 passing tests**, with the
same3 existing optional skips. Black, Ruff and `git diff --check` pass for the
new and amended helper/test files. The warning count above is not a test skip.

## B2 initialization diagnostic

Six new B2 tests and the three existing B1 unit tests pass: **9 passed in7.70s**,
without skips. The non-overlapping current total is **971 passing tests** plus
the same3 optional baseline skips.

```bash
/tmp/phase0-venv/bin/python -m pytest -q \
  artifacts/optimizer_discovery/investigation16/benchmark/test_first_point.py \
  tests/unit_tests/test_investigation16_benchmark.py
```

The B2 report records prepare/run/analyze commands, the initial missing-module
test failure and subsequent passing implementation. All96 new trajectories and96
preserved B1 controls pass an independent audit of source changes, input pairing,
objective values, normalization, metrics and aggregates. Its canonical digest is
`46fec532666d17529a4b23181f32e83de8b61761cd194c71b34b5b9b62d0cdb0`.
The971 total counts tests, not those192 scientific trajectories. The independent
audit's9216 objective calls are additional integrity work, not search evaluations.
Black, Ruff, source/credential integrity and `git diff --check` pass. No frozen
source, prior program or existing control was changed.

## Prospective supplementary baseline control

Seventeen new tests pass with the 26 unchanged primary-analysis tests: **43 passed**.
The current non-overlapping total is **988 passing tests**, plus the same three
optional baseline skips. An earlier broader control/owner/driver/Trace run also
passed 80 tests; overlapping tests are not added to the total.

```bash
/tmp/phase0-venv/bin/python -m pytest -q \
  artifacts/optimizer_discovery/investigation16/production/test_baseline_control.py \
  artifacts/optimizer_discovery/investigation16/production/test_analysis.py
```

Independent review confirmed pre-generation registration, the post-primary-audit
barrier, identical 144 evaluation inputs, unchanged fallback, separate storage,
and immutable completed evidence. Black, Ruff and whitespace checks pass. The
tests include early audit refusal, corrupt checkpoints, interrupted resume,
final-result preservation and linkage to started-attempt journals. These checks
use synthetic evidence; no real supplementary evaluation or main P1 result was
opened. Helper SHA256:
`2401c76c5d9b456037b6db13de119edc8f9b0cc885524a48411857126e5dd896`.

## B2 scientific figure

`benchmark/b2/plot_paired_outcomes.py` renders four panels from the preserved 96
pairs, with explicit logarithmic axes and no clipping or jitter. Both coordinator
and independent author inspected the PNG; the PDF contains one page. The script
verified all 192 raw-row hashes before/after, all pair counts and aggregate means
against the independent review. Rendering ran with candidate evaluation, objective
and normalization entry points blocked. No new scientific evaluation or model
call was made, and the 988-test total is unchanged. Black, Ruff, compilation and
whitespace checks pass. PNG SHA256:
`6336ba36f5763cabbfb142e52313a55e5fc50efa079c6ff1cbfd3b5624d532e9`.

## Provider-ended generation classification

Six additional red/green tests cover F1's returned `finish_reason=error` response,
null final content and provider receipt. The unfrozen F1 analysis now passes
**23 tests in 9.71s**, bringing the non-overlapping total to **994 passing tests**
with the same three optional baseline skips. These additions annotate termination
origins separately from extraction/source validity; they do not alter slots,
metrics, fallback or paired comparisons. All nine frozen F1 source hashes remain
unchanged. Black, Ruff and whitespace checks pass. No actual validation/fitness
was opened. Descriptive evidence digest:
`4334e13d41410599732072587a09017d0ef2ebea326a0fbabad15d5ec491da14`.

## Completed F1 and prospective P1 start

F1 final verification passes: all24 responses,24 receipts,26 transport attempts
and60 sealed evaluation artifacts; exact recomputation of all four contrasts,
196 scientific inputs unchanged, nine frozen execution sources unchanged.
The final focused regression run has69 passing tests in10.97s (existing tests,
not an increase to the994 non-overlapping total). Exact commands are retained in
`feedback_experiment/final_verification.json`. Black, Ruff, whitespace and the
actual-credential scan pass. An independent statistical review reproduces every
paired result and confirms the exploratory fixed-parent interpretation.

The independent P1-E1 gate replay forbids generation, objective/candidate execution,
key loading and network access; it passes with all1277 files byte-identical.
The main/control preflight then confirms both source/configuration freezes and
the97-file source ZIP, including all79 opto Python files. Its chronology proves
main freeze < control freeze < source archive < generation start < first request.
See `production_run/preflight_independent_review.json` for exact hashes, clocks,
budgets and the scope of this read-only check.

The first P1-generation snapshot credential scan covers4182 files and206 compressed
members, with no actual credential matches or .env files; the user's untracked
probe artifact retains its original hash. This scan is a checkpoint, not the final
scan of the still-running P1. Record: `runtime/credential_scan_p1_started.json`.

Completed-study workbook export: Node `--check` and the builder command in
`report_data/README.md` pass (`build_03.log`). It checks332 formula results and738
numeric comparisons from204 hashed source records; formula-error search reports
zero matches. These are artifact checks, not additional unit tests. All17 sheet
previews and nine supplemental ranges were visually reviewed. Boolean COUNTIFS
representation and presentation corrections are documented; no scientific source
changed and no P1 data/model/objective execution was used.

An independent standard-library ZIP/XML review then recalculates all557 exported
formulas (365 numeric and192 string results), checks all cached values, archive
relationships and204 unchanged source hashes. No Excel errors, external links or
macros are present. Source projections retain all80 EXP-15,24 G1,96 B2 and24 F1
rows, including21 EXP-15 ineligible candidates,26 B2 AUC losses and F1's four
invalid generations/24 deployment fallbacks. See
`report_data/independent_workbook_review.json`; this is an export-integrity check,
not an additional scientific experiment or completion of P1.

## Final P1 completion and regression — 2026-09-10

P1 and its seven-step post-generation pipeline are complete. The independent final
review verifies 192 unique completed responses, 194 transport attempts, 24 frozen
selections and the global generation/selection/audit chronology. All 97 source ZIP
members match. It recomposes the full aggregate exactly; a separate arithmetic
implementation reproduces the six registered contrasts to at most 3.47e−17.
The negative R−I, R−C and R−B2 results remain in the reports and workbooks.

The numeric verifier checks all 14,592 cache rows, including 865 invalid partial
rows, 440,961 objective observations and 6,144 normalization references, without
rerunning candidates or models. The independent B2 review checks another 4,608
objective values and 1,536 references. These **447,105 + 6,144 mathematical
integrity calls** are separate from scientific search budgets. The final review
reads 82,371 evidence files without changes. Its preserved procedure and limitations
are in `production/INDEPENDENT_FINAL_REVIEW.md` and `.json`.

The final focused production regression passes **93 tests in 73.32s**, no skips:

```bash
/tmp/phase0-venv/bin/python -m pytest -q -rs \
  --disable-socket --allow-hosts=127.0.0.1,localhost \
  tests/unit_tests/test_investigation16_production_driver.py \
  tests/unit_tests/test_investigation16_search_experiment.py \
  tests/unit_tests/test_investigation16_trace_schedule.py \
  artifacts/optimizer_discovery/investigation16/production/test_analysis.py \
  artifacts/optimizer_discovery/investigation16/production/test_verify_numerics.py \
  artifacts/optimizer_discovery/investigation16/production/test_baseline_control.py
```

Output: `production/final_regression_01.log`. These overlap the earlier tests and
are not added again. Four additional program-inspection tests pass in 0.02s;
the non-overlapping accumulated total is **998 passing tests**, with the same
three optional baseline skips. This is an accumulated count, not one simultaneous
998-test command. The earlier broad offline 831-test run remains applicable:
no frozen generation/evaluation implementation changed afterward.

```bash
/tmp/phase0-venv/bin/python -m pytest -q artifacts/optimizer_discovery/investigation16/production/test_program_inspection.py
/tmp/phase0-venv/bin/python artifacts/optimizer_discovery/investigation16/production/program_inspection.py
```

Inspection verifies 192 sources, 24 selections, 144 native feedback envelopes,
6,912 projected trajectories and 582 unchanged source records. All six selected
R exports exactly match evaluated bytes. Four representative diffs are preserved
as JSON strings so context whitespace is retained without changing raw source.
Black py313, Ruff, local links and whitespace checks pass for these new helpers.

## Final reporting exports and credential integrity

`production/present_results.py` reads completed records only. It exports all 36
seed/arm rows, six registered contrasts, 24 search pools and four usage rows,
checking four source hashes before/after. Its standalone PNG/PDF figures were
visually inspected. No model, candidate or objective call is made by reporting.

The separate P1 workbook passes its five-sheet visual review and independent
CSV/JSON/ZIP/XML checks: 658 CSV projections, 670 Excel projections, 138 formulas
and nine source hashes. All 18 ineligible candidates, four R−I losses, six missing
B2 selection indices and the single R seed selection remain explicit. Commands,
hash and ten previews are indexed in `production/presentation/P1_WORKBOOK_QA.md`.

The earlier diagnostics workbook was reexported by its unchanged builder during
agent resumption. This changed the ZIP hash, not scientific data. Its current hash
is `ef871459db0413a0adda0d4a44f6f27f93b633f8fed2e09c870943f2bc58b2ce`.
The historical review with the old hash is preserved. A new
`report_data/independent_workbook_review_r2.json` passes all 557 formulas, 204 sources
and 4,259 projected cells; all 26 preview hashes remain identical. This packaging
event requires no scientific invalidation. See `report_data/README.md`.

The private final scan passes **87,067 files and 345 compressed members**, including
raw/gzip/ZIP/XLSX contents. It checks the active credential, full local `.env` bytes
and its secret-source value without exposing them. No candidate, model, objective
or post-key subprocess is invoked. The key and private values are not persisted.
Scanner happy-path, absence and cross-chunk detection checks pass separately with
synthetic bytes. Its exact command and sanitized result are:

```bash
PYTHONPATH=. /tmp/phase0-venv/bin/python artifacts/optimizer_discovery/investigation16/runtime/final_integrity_scan.py
git diff --check
```

Record: `runtime/final_integrity_scan.json`. It verifies starting/final HEAD
`13ebda2242e1c18022591737b113030ca2ce2da2`, the investigation branch, an empty staged
diff and changes to tracked files limited to the two research ledgers. The existing
user probe retains SHA256
`ca8a08e5c2eca14c1004e2af3241ae8d9ebf2909d386eee9e1e2611f5b5e84e8`.
Prior tracked scientific evidence and production files are unchanged. Final
documentary closure records only these public findings after the scan.

Black/Ruff checks on the reporting and scanner helpers pass:

```bash
/tmp/phase0-venv/bin/python -m black --check --target-version py313 artifacts/optimizer_discovery/investigation16/runtime/final_integrity_scan.py artifacts/optimizer_discovery/investigation16/production/present_results.py
/tmp/phase0-venv/bin/python -m ruff check artifacts/optimizer_discovery/investigation16/runtime/final_integrity_scan.py artifacts/optimizer_discovery/investigation16/production/present_results.py
```

No new dependency, commit, push, merge, PR or external message is created. Invalid
candidate output is preserved verbatim. There is no unresolved execution blocker;
unproven feedback efficacy and generalization are scientific limitations.
