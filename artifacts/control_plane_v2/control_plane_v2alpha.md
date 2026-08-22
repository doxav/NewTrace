# ADR: recursive-opt control plane v2alpha

- Status: accepted; semantic closure at `c92f0af4af3e72a12b0228dbed215f86f8c9475b`, completion-audit correction pending final commit
- Date: 2026-08-22
- Baseline SHA: `21a0ad3d2f4f835ce2ffb1eef18c36a622265418`
- Public schema: `recursive-opt/v2alpha`
- Public kind: `recursive_optimization`

## Context

The current package already has the execution primitives—`trace.Module`, the recursive levels, `MemoryLite`, objective selection, budgets, and `run_spec`—but orchestration is split between legacy dictionaries, direct calls, examples, and notebook helpers. The control plane must consolidate those paths without creating a parallel framework.

The explicitly mandated targeted baseline is green. The complete repository suite has unrelated failures that reproduce at the pristine SHA; those are recorded in `baseline.md` and are not suppressed or repaired here.

## Decision

`run_spec` remains the public façade. A single public dictionary shape represents both one run and experiments. Canonical processing is ordered and fail-fast:

```text
raw mapping
→ migrate legacy
→ normalize/materialize defaults
→ structural validation
→ semantic validation
→ resolve versioned refs
→ expand arms/seeds/matrix
→ immutable ExecutionPlan
→ ordered immutable level plans inside each execution unit
→ one canonical runner and selected engine adapters
→ canonical RunResult
```

There will be no public `RunSpec`/`ExperimentSpec` split and no Pydantic, Hydra, or jsonschema dependency. Immutable internal dictionaries/tuples remain directly JSON-serializable. SHA-256 fingerprints are computed from canonical JSON with sorted keys and no secret values.

## Invariants

1. Every portable spec declares the exact schema version and kind.
2. Globals (`runtime`, `llm_profiles`, `knowledge`, `outputs`, `budget`, `experiment`) are materialized once. Every canonical item in top-level `levels` independently materializes `surface`, `module`, `engine`, `objective`, `llm_roles`, `datasets`, `bindings`, and `outputs`.
3. Unknown structural keys fail. Vendor data is allowed only below a namespaced `extensions` entry.
4. Normalized specs contain only JSON values, no callable, and no credential value. Credentials are referenced as `env:NAME`.
5. Reproducible mode rejects moving `latest` model aliases and records every explicit fallback.
6. Strict refs are versioned registry identifiers. Arbitrary `module:symbol` imports are forbidden.
7. The common compilation target is `trace.Module`; `trace.Model` remains one supported factory.
8. A declared causal dependency requires a typed binding. Ordering-only dependencies must say so explicitly.
9. Train/validation/holdout access is capability-based; holdout access is denied during fitting, proposal, induction, and candidate selection.
10. Invalidity is explicit and independent from numeric score. Negative scores may be valid.
11. Dict scores are never averaged implicitly. Weighted and Pareto selection reuse `opto.trainer.objectives`.
12. Runtime LLM accounting is the source of truth and is retained by role.
13. Knowledge retrieval occurs in the runner and is injected through a typed binding. Promoted knowledge is the default reusable set.
14. Engine capabilities are checked before execution. GEPA does not claim native Pareto support.
15. The notebook is a strict client: declare/load, normalize, explain, run, display.

## Compatibility

JSON-compatible legacy specs and flat-v2 shorthand are migrated into the same canonical top-level `levels` shape. Legacy callables cannot become portable refs implicitly; explicit test adapters mark results `portable=false` and `promotable=false`. No evaluator, dataset, module, or provider response is invented when migration evidence is absent.

## Footprint decision

New control-plane behavior is added first to existing `spec.py`, objective behavior to `opto/trainer/objectives.py`, and knowledge behavior to `memory.py`. New public classes require two concrete consumers. Compatibility wrappers remain routing-only.

The current graph/Trace IO import is user-provided source material, not the accepted final footprint: it adds 5,023 staged runtime lines and includes OTEL/telemetry capabilities outside this mission. Before final proof it must be reduced to the contracts actually consumed (`GraphAdapter`, `GraphExecutor`, `GraphModule`, serialization/codecs/capabilities), with optional LangGraph kept behind a narrow integration boundary.

Phase 10 resolved this decision: 384 graph lines remain and the 4,004-line Trace IO import was removed. `recursive_opt.module.graph@1` resolves a `GraphExecutor` from explicit runtime resources, while `LangGraphAdapter` is optional and does not leak LangGraph into the core contract.

Any positive final runtime-plus-notebook delta must be itemized against a required invariant in the final footprint report. A temporary positive delta is accepted only while a directly replaced notebook/legacy path is still scheduled for deletion; it is not sufficient for completion.

### Corrective footprint result

Against the semantic-closure baseline, recursive-opt runtime is **8,750** physical lines versus **8,803** (`-53`), and `spec.py` is **2,633** versus **2,688** (`-55`). Notebook code remains 16 lines, so runtime plus notebook is also `-53`. The public package export count remains 114: `register_evaluator` and `register_dataset` replace two obsolete exports. No footprint exception is required; `code_footprint_after.json` contains the per-file measurement.

### Completion-audit correction

The final causal audit found that a scripted real Trace optimizer could still mutate a non-target component through candidate restoration. The Trace wrapper now exposes only parameters marked trainable. The same audit made invalid evaluations unrankable for Trace and GEPA, applies the execution-unit seed while resolving registered datasets, records test-override identity in the resolved manifest, delays holdout materialization until fitting and selection finish, and protects persisted level and final results with canonical integrity digests. Tests exercise each behavior through the runtime path, including a cross-process zero-call resume.

## Public-field classification

The classifications below are normative. “Active” includes fields consumed in canonical compilation/execution and legacy-only controls consumed by the migrated legacy module. Unsupported values fail validation rather than remaining fingerprint-only metadata.

| block | active | validation-only / migrated shorthand | unsupported and rejected | extension-only |
|---|---|---|---|---|
| identity | `schema_version`, `kind` | caller-supplied `fingerprint` is verified | unknown top-level keys | `extensions.<namespace>` |
| runtime | `reproducible`, `offline`, `resume`, `memory_root`, `reuse_priors`, `tracebench`, `scoring`, `prior_promotion`, `trainer_kwargs`, `run_id`, `seed`, `test_mode` | `strict_refs=true` is the invariant | `strict_refs=false`; hidden behavioral resources in portable mode | none |
| level graph | `id`, `depends_on`, level `ordering_only` | none | binding `ordering_only=true`; decorative dependency edges | none |
| surface/module | `surface.kind`, `surface.targets`, `module.ref/config/artifact/inputs` | registry config/artifact validators are mandatory | unknown targets/config/artifacts | none |
| engine | `name`; validated Trace/GEPA configs and their supported knobs | fixed config must be empty | engine capabilities/objective modes and unknown knobs | none |
| objective | evaluator ref, intent, metric descriptors, selection, hard constraints, aggregation default, feedback channels | metrics-list plus `directions` migrates to descriptors | descriptor-form `directions`; aggregation weights (selection weights are authoritative); unknown sources/modes | none |
| LLM | profiles, role bindings/overrides, exact model, env key ref, ordered fallbacks, temperature, max tokens | resolved model is materialized | `latest` in reproducible mode, role/global `base_url`, malformed secret refs | none |
| datasets/bindings | inline splits or exact ref/split/config; typed from/to/codec | none | arbitrary import refs, untyped/decorative bindings | none |
| knowledge | store, retrieval, statuses, scope fields, top-k, injection codec | empty promotion/rollback maps reserve compatibility shape | nonempty promotion/rollback policies | future policy experiments must use namespaced extensions |
| outputs | global directory, JSON format, artifact flag; per-level artifact flag | JSON is the sole format; per-level directory/format must inherit globals | other formats or per-level path/format overrides | none |
| budget/experiment | every limit, `on_exceed`, seeds, arms, matrix | legacy `return_best` migrates to `return_best_valid` | unknown policies, negative limits, empty matrix axes | none |

## Consequences

- Strict normalization turns silent typos and hidden behavior into compile-time errors.
- Frozen normalized values use tuples for sequences; JSON round-trip restores lists without changing the fingerprint.
- Old callable-heavy examples remain executable only through explicit local compatibility and cannot be promoted as portable artifacts.
- Live provider checks are automatic for non-offline runs; live paid tests remain manual/secrets-protected while mandatory CI is deterministic and offline.
