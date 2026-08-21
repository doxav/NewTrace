# ADR: recursive-opt control plane v2alpha

- Status: accepted for implementation
- Date: 2026-08-21
- Baseline SHA: `6fc278a398709fe79a0fc9be22bae99bffd8cba6`
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
→ engine runner
→ canonical RunResult
```

There will be no public `RunSpec`/`ExperimentSpec` split and no Pydantic, Hydra, or jsonschema dependency. Immutable internal dictionaries/tuples remain directly JSON-serializable. SHA-256 fingerprints are computed from canonical JSON with sorted keys and no secret values.

## Invariants

1. Every portable spec declares the exact schema version and kind.
2. All canonical blocks are materialized: `surface`, `module`, `engine`, `runtime`, `objective`, `llm_profiles`, `llm_roles`, `datasets`, `knowledge`, `bindings`, `outputs`, `budget`, and `experiment`.
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

JSON-compatible legacy specs are migrated into the canonical shape. Legacy callables cannot become portable refs implicitly; a later `LocalObjectRegistry` compatibility path will mark such specs `portable=false` and `promotable=false`. No evaluator or module is invented when migration evidence is absent.

## Footprint decision

New control-plane behavior is added first to existing `spec.py`, objective behavior to `opto/trainer/objectives.py`, and knowledge behavior to `memory.py`. New public classes require two concrete consumers. Compatibility wrappers remain routing-only.

The current graph/Trace IO import is user-provided source material, not the accepted final footprint: it adds 5,023 staged runtime lines and includes OTEL/telemetry capabilities outside this mission. Before final proof it must be reduced to the contracts actually consumed (`GraphAdapter`, `GraphExecutor`, `GraphModule`, serialization/codecs/capabilities), with optional LangGraph kept behind a narrow integration boundary.

Phase 10 resolved this decision: 384 graph lines remain and the 4,004-line Trace IO import was removed. `recursive_opt.module.graph@1` resolves a `GraphExecutor` from explicit runtime resources, while `LangGraphAdapter` is optional and does not leak LangGraph into the core contract.

Any positive final runtime-plus-notebook delta must be itemized against a required invariant in the final footprint report. A temporary positive delta is accepted only while a directly replaced notebook/legacy path is still scheduled for deletion; it is not sufficient for completion.

### Final footprint exception

The measured budget scope is 8,819 lines versus 8,716 at baseline: **+103**. The neutral default was therefore missed by 103 lines. This is accepted as an explicit exception rather than hidden by counting tests, docs, graph, or shared objective code. Every changed line in the footprint scope is assigned below using physical before/after counts and the single added `spec.py` hunk:

| scope | physical delta | required invariant |
|---|---:|---|
| `recursive_opt/__init__.py` | +52 | Export the one public façade, normalized plan/result, registries, bindings, roles, knowledge, and usage contracts. |
| `recursive_opt/memory.py` | +138 | Knowledge-card schema, status/scope/supersession persistence, promoted-only retrieval, and rollback. |
| `recursive_opt/runmode.py` | +68 | Runtime-owned per-role provider usage and call accounting. |
| `recursive_opt/spec.py` lines 73–484 | +412 | Immutable registry entries, `ExecutionPlan`, `RunResult`, dataset capabilities, compilation, resume, and execution. |
| `recursive_opt/spec.py` lines 485–704 | +220 | Typed causal bindings, objective compilation, role resolution/preflight, and runner-side knowledge retrieval. |
| `recursive_opt/spec.py` lines 705–1049 | +345 | Budget accounting, fixed/Trace/GEPA runners, canonical projection/results, and registered evaluators. |
| `recursive_opt/spec.py` lines 1050–1327 | +278 | Generic component/graph/legacy module snapshot adapters, deterministic expansion, and built-in registrations. |
| `recursive_opt/spec.py` lines 1328–1778 | +451 | Strict migration, default materialization, fingerprinting, structural/semantic validation, and secret/callable rejection. |
| `recursive_opt/spec.py` legacy integration glue | +42 | Route v2 through the existing `run_spec`, promoted retrieval, imports, and validation compatibility. |
| `recursive_opt/traces.py` | -27 | Remove the redundant graph compiler path and narrow optional imports. |
| notebook code cells | -1,876 | Delete 75 helpers, direct optimization, private APIs, and manual orchestration; retain only load/normalize/explain/run/display. |
| **combined** | **+103** | Exact residual required control-plane implementation. |

The `spec.py` bands total its +1,748 physical-line delta; the recursive runtime totals +1,979 and the notebook removes 1,876 code lines. `code_footprint_after.json` contains the reproducible per-file measurements. Further compression would mainly trade maintainability for physical-line counting or remove baseline behavior, neither of which is justified.

## Consequences

- Strict normalization turns silent typos and hidden behavior into compile-time errors.
- Frozen normalized values use tuples for sequences; JSON round-trip restores lists without changing the fingerprint.
- Old callable-heavy examples remain executable only through explicit local compatibility and cannot be promoted as portable artifacts.
- Live provider checks are manual/secrets-protected; mandatory CI remains deterministic and offline.
