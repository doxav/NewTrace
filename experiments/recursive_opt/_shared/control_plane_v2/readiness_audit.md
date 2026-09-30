# recursive-opt v2alpha semantic-readiness audit

Baseline: `21a0ad3d2f4f835ce2ffb1eef18c36a622265418` (`recursive_opt`, clean tracked worktree).

Method: inspect the exact baseline source, run the existing offline control-plane/spec suite (`79 passed`), and run an untouched-baseline counterfactual probe covering schema shape, legacy compilation, Trace configuration, behavioral resources, initial artifacts, targets, preflight, seeds, budgets, outputs, holdout context, objective controls, and knowledge-store resolution. “Confirmed” means the suspected blocker survived an attempted falsification. Line references in this table are anchored to the baseline SHA and intentionally remain stable after corrective edits.

| id | confirmed/refuted | code evidence | causal test | consequence |
|---:|---|---|---|---|
| 1 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:65-90,1374-1455` defines canonical flat blocks and stores recursion below `surface.levels`. | Baseline probe: normalized flat input reported `levels in spec=False`; the existing normalization test asserts flat blocks. | Replace the canonical shape with top-level fully specified `levels`; retain flat input only as migration shorthand. |
| 2 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:404-421,1180-1212,1654-1678` validates `surface.levels` but expands only arms/seeds/matrix. | A legacy spec with one declared level compiled to one experiment unit whose module only returned level metadata. | Compile immutable level plans and execute them inside every experiment unit. |
| 3 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:300-309,1065-1070,1316-1324` implements `legacy_levels@1` as a metadata module. | Calling the migrated module returned level id `a`; it did not execute the level. | Migrate legacy level declarations into real canonical level specs; remove the metadata-module execution claim. |
| 4 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:742-746,927-980` routes Trace to `_run_module_engine` and mutates only through `resources["fit"]`. | Trace with no fit override produced the same artifact as a configuration variant; the existing fixed/Trace test supplies a direct restoring fit callback. | Route Trace through the existing `recursive_opt.optimize.optimize` path. |
| 5 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:927-1000` never reads `engine.config`. | Changing only `iterations`/`num_candidates` left artifact and evaluation identical. | Validate and pass the declared Trace configuration to the existing optimizer/trainer stack. |
| 6 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:316-319,334-373,650-657,1270-1273` exposes module/engine/codec registration but mutates evaluators internally. | Public-symbol inventory has no `register_evaluator`; existing tests can only use built-ins or hidden resources. | Add public exact-version evaluator registration/resolution. |
| 7 | confirmed | No dataset registry exists in `21a0ad3:opto/features/recursive_opt/spec.py`; `DatasetAccess` receives inline values directly at lines 253-282. | Public-symbol inventory has no `register_dataset`; exact dataset refs cannot compile. | Add a public versioned dataset registry and inline/ref resolver. |
| 8 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:422-480,757-760,898-903,938-960` accepts behavioral fit/evaluator/outputs/GEPA functions independently of the spec. | Two runs with the same fingerprint and different fit callbacks produced different artifacts. | Strict portable runs must reject behavior overrides; explicit test mode must mark results non-portable/non-promotable and manifest override identity. |
| 9 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:375-382` builds from config but never restores `module.artifact`. | Declaring artifact state `artifact-state` still built `config-state`. | Validate and restore the declared artifact before any engine operation. |
| 10 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:285-298` makes every component trainable; no builder/runner reads `surface.targets`. | Unknown target `missing-target` normalized and built; the real `planner` parameter remained trainable. | Resolve targets before execution and mark only those parameters trainable. |
| 11 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:571-587,946-951` resolves role metadata and passes mappings in context, not clients. | Evaluator context exposed role dictionaries; no engine constructed or invoked a role client. | Construct exact role clients, inject them into the corresponding runtime paths, and meter each call once. |
| 12 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:589-607,2007-2040` exposes a manual preflight helper but `run_spec` does not call it. | Replacing `preflight_llm_profiles` with a counter recorded zero calls during `run_spec`. | Automatically preflight exact resolved live models before execution. |
| 13 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:1180-1212` copies a seed into runtime but no runner applies it. | With external RNG reset, runtime seeds 1 and 2 produced the same random-fit artifact. | Add scoped Python/NumPy seeding and pass seeds to optimizer/trainer/GEPA. |
| 14 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:705-732` computes budget usage after all work. | With candidate/eval limits zero, the evaluator still ran once and only the returned report said `candidates` exceeded. | Add a shared pre-operation budget guard for candidates, evaluations, LLM calls, tokens, and wall time. |
| 15 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:705-732,1424-1430` carries `on_exceed` but never branches on it. | `on_exceed=fail` with a zero candidate limit returned `success`. | Validate and implement `fail`, `raise`, and `return_best_valid`. |
| 16 | confirmed | `outputs` is normalized at `21a0ad3:opto/features/recursive_opt/spec.py:1423`; the execution path contains no output writer. | A run with a temporary output directory created zero files. | Persist the complete declared manifest/results atomically and honor format/artifact controls. |
| 17 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:429-480` resumes only from a supplied mutable map keyed by fingerprint/unit. | Existing `test_resume_reuses...` constructs and reuses the same in-process dictionary. | Replace it with persisted, identity-complete level resume records and stale/partial rejection. |
| 18 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:946-957` passes the complete normalized spec into fit callbacks. | A faulting fit read `context["spec"]["datasets"]["holdout"] == [3]` successfully. | Build structural phase-specific views with no holdout path. |
| 19 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:766-793` reads holdout and passes it as GEPA `test_set`. | Existing GEPA test explicitly asserts the holdout-valued `test_set`. | Keep holdout outside `optimize_anything`; evaluate it only after best-candidate extraction. |
| 20 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:520-569` retains aggregation/feedback descriptors, while runners evaluate once and do not apply them. | Changing only aggregation weights and feedback channels left metrics and feedback equal. | Make aggregation and feedback-channel behavior causal or reject unsupported values. |
| 21 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:609-633,1619-1641` validates the store string but consumes an externally supplied `MemoryLite`. | `acme.unknown_store@1` compiled successfully. | Resolve the declared knowledge store through a registry. |
| 22 | confirmed | Defaults at `21a0ad3:opto/features/recursive_opt/spec.py:1412-1421` claim promotion/rollback policies; retrieval at lines 609-633 does not use them. | Existing knowledge test invokes `MemoryLite.set_artifact_status` directly rather than a runner policy. | Reject non-empty policies or move them to an explicitly experimental extension. |
| 23 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:939-945` binds only `knowledge[0]`. | Static branch inspection falsified the “all selected cards” requirement; current test only proves retrieval ordering/status. | Bind every selected promoted card and record every id. |
| 24 | confirmed | `21a0ad3:opto/features/recursive_opt/spec.py:485-527,938-945` resolves bindings from `resources["outputs"]`; no level runner populates upstream outputs. | Existing counterfactual calls `apply_bindings` directly with handcrafted level output maps. | Feed actual earlier-level results into dependent level bindings. |
| 25 | confirmed | `21a0ad3:artifacts/control_plane_v2/migration_report.json` uses `migrated_replayable`; `test_historical_migration_report...` proves normalization/hash only. | No classified migrated spec is executed semantically by that test. | Reclassify by semantic executability and replay representative config/family-policy/prior specs or name exact missing dependencies. |
| 26 | confirmed | `21a0ad3:tests/unit_tests/test_recursive_control_plane_v2.py:482-526` restores a better artifact in custom `fit`. | Baseline test passes without invoking `recursive_opt.optimize.optimize`. | Replace with an anti-no-op test over a real trainable `ParameterNode` and existing optimize path. |
| 27 | confirmed | `21a0ad3:tests/unit_tests/test_recursive_control_plane_v2.py:549-643` injects `fake_optimize_anything`. | The package/version test inspects `pyproject.toml` only; no installed-package contract is exercised. | Retain fake conversion tests and add an exact installed GEPA no-paid-call API contract. |
| 28 | confirmed | `21a0ad3:artifacts/control_plane_v2/proof.md:1-145` names the baseline only; `code_footprint_after.json` says `worktree` and repeats the baseline SHA. | `git rev-parse HEAD` is `21a0ad3...`, but neither artifact identifies a distinct corrective implementation commit. | Commit the corrective implementation and rerun proof at that exact SHA. |

## Untouched-baseline probe output

```text
D1 False surface.levels 0
D2_D3 1 a
D4_D5 True True
D8 True True
D9 config-state
D10 ['planner:5'] [True]
D12 0
D13 True
D14_D15 1 success ['candidates']
D16 []
D18 [3]
D20 True True
D21 True
```

All 28 diagnoses are therefore confirmed at the required starting commit. No diagnosis is being patched on the basis of speculation alone.

## Corrective disposition

The baseline findings above are immutable audit evidence. The following rows record the corrective state in this worktree; test numbers refer to `tests/unit_tests/test_recursive_control_plane_v2.py`.

| id | disposition | corrective code evidence | causal proof |
|---:|---|---|---|
| 1 | closed | `normalize_spec`, `_normalize_level` | 01–03: canonical one/two-level form and flat shorthand migration |
| 2 | closed | `_ExecutionUnit.levels`, `execute_plan` | 05: two levels execute in order |
| 3 | closed | `_migrate_legacy_level`, `_run_legacy_level_engine` | 04 plus all 47 legacy spec regressions |
| 4 | closed | `_run_trace_engine` calls `recursive_opt.optimize.optimize`; `_EvaluatedModule.parameters` exposes only trainable targets | 07 drives a scripted real optimizer, mutates the declared target, and proves a protected component remains unchanged |
| 5 | closed | validated Trace engine config reaches optimizer/trainer | 08 changes only iterations and changes candidate accounting |
| 6 | closed | public `register_evaluator`/`_evaluator_entry` | 11 executes a registered exact evaluator ref |
| 7 | closed | public `register_dataset`/`_resolve_datasets` inside `_seed_scope` | 12 resolves all three splits through an exact ref; 25 proves seeded resolver sampling |
| 8 | closed | `_validate_runtime_resources`, `_prepare_output_root` | 13 rejects hidden behavioral resources unless explicit nonportable test mode permits and persists the named adapter identity |
| 9 | closed | `_build_level_module` validates/restores artifact | 09 changes only artifact and changes accuracy |
| 10 | closed | `_apply_trainable_targets`, `_EvaluatedModule.parameters` | 07 and 10 prove selected/nonselected targets, real optimizer isolation, exact snapshot/restore, and pre-execution unknown-target failure |
| 11 | closed | `_build_role_clients`, `_GuardedRoleClient` | 14 and 16 prove construction, fallback order, and exactly-once role usage |
| 12 | closed | `execute_plan` automatic live preflight | 15 records exact primary and fallback models |
| 13 | closed | `_seed_scope` covers unit compilation/execution and GEPA seed mapping | 25 proves same-seed equality and different-seed divergence for evaluators, registered datasets, optimizer construction, Python RNG, and NumPy RNG |
| 14 | closed | shared `_BudgetGuard` before consuming operations | 23 proves zero-budget means zero evaluator calls |
| 15 | closed | `_should_raise` and budget-exhausted result path | 24 proves fail, raise, and return-best-valid |
| 16 | closed | `_prepare_output_root`, atomic `_write_json`, persistence helpers | 13 and 26 verify the resolved override identity and every required record, including artifact suppression only when requested |
| 17 | closed | exact resume identities plus canonical result checksums in `_load_resume`/`_load_final_result` | 27 proves zero evaluator calls in a second process and rejects/repairs partial or tampered results |
| 18 | closed | `_phase_context` exposes a holdout-free spec/datasets/input view | 21 fault-injects all three forbidden access paths |
| 19 | closed | `_run_gepa_engine` omits `test_set`; final holdout evaluation follows extraction | 22 asserts holdout never enters OptimizeAnything |
| 20 | closed | `_aggregate_evaluations`, feedback-channel filtering, descriptor defaulting, validity-aware Trace/GEPA projection | 17–20b prove selection/constraints/rollback and causal aggregation, feedback, intent, and invalid-candidate rejection |
| 21 | closed | `_resolve_knowledge_store`, `_knowledge_store` | 28 rejects an unregistered store ref |
| 22 | closed | semantic validation rejects nonempty promotion/rollback rules | existing knowledge-policy rejection test plus 13b anti-no-op validation matrix |
| 23 | closed | `_prepare_level_inputs` iterates every retrieved card | 29 proves both ids in lineage and downstream binding |
| 24 | closed | `execute_plan` populates `_upstream` from actual earlier results | 05–06 prove actual propagation and counterfactual behavior |
| 25 | closed | six semantic migration classifications and precise representative dependencies | 30 plus `migration_report.json.representatives` |
| 26 | closed | the real Trace path uses `optimize` on `_EvaluatedModule` | 07 performs a real scripted candidate update through `optimize`; no direct artifact restore or custom fit callback implements the improvement |
| 27 | closed | exact installed GEPA 0.1.4 types and keyword-only evaluator contract | 22b runs without a provider or paid call |
| 28 | closed | proof, footprint, and readiness name completion-audit commit `05dabf68e77ef2b9c59a8fc20c68bf4f8d2c1eaf` | 35 runs with that exact SHA after the commit |

Additional anti-no-op closure in test 13b rejects `runtime.strict_refs=false`, binding-level `ordering_only`, nonempty `objective.aggregation.weights`, descriptor-form `directions`, unsupported role `base_url`, and per-level output directory/format overrides. These fields now fail explicitly instead of altering only a fingerprint.
