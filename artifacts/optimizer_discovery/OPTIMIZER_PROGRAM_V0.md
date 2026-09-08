# Patrick interface — OptimizerProgramV0

One portable UTF-8 file, `optimizer.py`:

```python
def propose(history: list[dict], bounds: list[list[float]], seed: int) -> list[float]:
    """Propose one point for a minimization problem."""
    import random
    rng = random.Random(seed + len(history))
    return [rng.uniform(low, high) for low, high in bounds]
```

`history` contains only past `{ "x": [coordinate, ...], "value": objective_value }`
observations. Smaller values are better. `bounds` contains one finite increasing
`[low, high]` pair per dimension. Endpoints are legal. `seed` is an integer held
constant across one trajectory; combine it with history length for new random draws.
The returned list/tuple must have exactly one finite real number per dimension.
Booleans are rejected. The callable must have the three named positional parameters
without defaults/variadics; annotations are optional in generated artifacts.

No objective callable/source, split label, validation/holdout data or LLM client is
passed to this API. State must derive from history and seed: each call is a fresh
process. Imports use the Python standard library only (`-I -S`). Generation method
is irrelevant: recursive_opt, FunSearch and OpenEvolve can all write this same file.
No search framework or new memory architecture is needed. A future Project-1
workspace can store the file unchanged.

## Host API

`parse_program(response)` accepts raw Python or exactly one Python code fence.
It does not repair malformed programs. `propose_point(source, history, bounds, seed)`
returns `ProposalResult(status, point, stdout, stderr)`. Invalid statuses include
syntax_error, import_error, missing_propose, signature_error, exception, timeout,
shape_error, nonfinite, out_of_bounds, nondeterministic and process_error.
Invalid points are `None`; host configuration errors raise descriptive exceptions.
Two fresh invocations must agree exactly. This detects observed nondeterminism;
it is not a proof that an arbitrary program will always be deterministic.

`evaluate_program(source, seed=0, budget=8)` is only the public calibration fixture:
2-D shifted sphere on [-5,5]^2, shift [1.25,-0.75]. One accepted proposal consumes
one objective evaluation. Repeatability checks consume no objective budget.
Failure stops the trajectory and preserves completed observations and logs;
`best_value` is absent for incomplete/invalid trajectories. All proposal coordinates
are retained as the evaluation behavior signature. No holdout exists in Phase 0.

## Canonical integration

`optimizer_spec(source, seed=0, budget=8, engine="fixed")` registers the versioned
`recursive_opt.evaluator.optimizer_program@1` evaluator and uses the existing
`recursive_opt.module.reasoning_workflow@1` component module. Its `optimizer`
component is trainable source text; a normal Trace update can replace it, and
canonical snapshots persist the complete source. The evaluator returns the existing
`EvaluationResult` with a minimized `value` metric, typed validity and execution
artifacts. `engine="trace"` and the existing GEPA adapter use this same evaluator.
GEPA/FunSearch/OpenEvolve search implementations are not added here.

CodeArtifactLevel's current convention is an in-process `self` method invoked by a
traced bundle. Executing arbitrary propose code through that path would violate the
subprocess boundary. Reusing the canonical trainable component module is the smaller
adapter: it preserves Trace dependency edges while the evaluator executes the text
outside the parent. The real-trainer integration test proves source replacement,
selection, minimization and behavior evidence without a parallel optimizer loop.

## Execution limits

Fresh temporary working directory; allowlisted PATH/LANG; no inherited credentials;
worker contains only proposal validation, not the objective implementation. A hard
2-second timeout kills the process group, including descendants. Stdout/stderr files
are capped at 64 KiB by OS file-size limits; up to 8 KiB of each are retained per
invocation with recognizable key strings redacted. No personal filesystem paths
are required by the program or runner.

**A subprocess is not a security sandbox.** Candidate code retains OS-user filesystem
and network capabilities and can deliberately evade the protocol or inspect accessible
files. These checks handle ordinary generated code, not adversarial submissions.
Do not claim credential security against hostile filesystem reads. Project-1 sandbox
infrastructure remains separate work; no security isolation is claimed here.

## Menu semantics

Canonical result metadata contains `menu_observations` and `menu_evidence` automatically.
Candidate counts deduplicate exact source artifacts; evaluation_observation_count
retains duplicate evaluations. declared_menu_size is a declared Cartesian menu count,
or null for adaptive/unknown menus. Only actual search-phase inputs are included;
final/holdout evaluations are excluded. Candidate comparisons use the intersection
of their evaluated input panels; common_example_count describes that scope.

Evaluator `artifacts.behavior_signature` represents actual choices/outputs, such as
proposal trajectories. Byte-different programs can therefore collapse to one, while
equal objective values can correspond to distinct behavior. Without that field,
equivalence is explicitly `metric_vector` (legacy score_spread: `scalar_score`),
with behavior_equivalence_known=false. This weaker evidence does not establish
behavioral headroom. Repeated inconsistent observations or disjoint panels return
null effective size and null collapse, with a reason. All-invalid is size zero.
These are observations on the measured panel, never proofs over an entire domain.

Legacy trainer rollouts are observed directly; normalized rejection flags and typed
invalid payloads are excluded. Legacy numeric ranking floors remain for compatibility,
and legacy evaluators without behavior signatures cannot certify behavioral diversity.
This is an explicit narrowing of MC-b, not retrospective certification of old runs.
