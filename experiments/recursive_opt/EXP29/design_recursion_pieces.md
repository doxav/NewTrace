# EXP29 — design of the two missing control-plane pieces

Date: 2026-10-08. Design only; nothing implemented. Companion to [prior_analysis.md](prior_analysis.md).

The two pieces are:
1. A generic, declarable way for an upper level to run a lower-level spec and get its score.
2. A way to pass validated code into a trainer, optimizer or memory hook.

Every claim about current behaviour cites `opto/features/recursive_opt/spec.py` (abbreviated `spec.py:<line>`).

---

## 0. Requirements, from the failures of EXP15, EXP21 and EXP23

| # | Requirement | Why |
|---|---|---|
| R1 | The whole recursive experiment is **one JSON spec**: no callables, fingerprinted, strictly validated | 0 of 85 migrated specs could be replayed: each study's nested runner lived in Python, outside the spec |
| R2 | The child (lower level) is **itself a normal v2 spec**, run by the same `compile_plan` / `execute_plan` | One runner; the child keeps the holdout gating, budget, resume and provenance it already has |
| R3 | Whatever O1 changes in the child goes through the **same strict validation** as a hand-written spec | An O1 proposal must not reach hidden controls (`_set_dotted_path` already rejects unknown fields, `spec.py:1906`) |
| R4 | Child LLM usage is charged to the parent budget | Equal-budget comparisons must include the cost of O1 |
| R5 | Identical (artifact, seed, instance) child runs are cached | EXP21 relied on cache hits; resume identity already exists (`spec.py:731`) |
| R6 | The O1 parameter is shown to the optimizer as a named, documented parameter whose initial value is the **current default** | O1 then starts from today's implementation ("optimize `VariationSearch`"), and the default arm is the zero-change point |
| R7 | O2 is the same mechanism applied twice | No new code per recursion depth |
| R8 | Hook code fails safe: on error, fall back to the default and count the failure | A bad proposal must cost one candidate, not crash a child run (cf. coevolution `PolicySlot`) |

---

## 1. Piece 1: an upper level runs a child spec

### Alternatives considered (eliminate, then refine)

| Option | Idea | Verdict |
|---|---|---|
| A. Bindings between levels of one plan (`depends_on` + typed bindings, `spec.py:395`) | O1 level outputs → O0 level inputs | **Eliminated.** It runs O1 once, then O0 once ("prepare then run"). O1 is never *scored by* O0, so it cannot be optimized. |
| B. A new "meta" engine | An engine that both proposes O1 artifacts and runs children | **Eliminated.** It duplicates the `trace` / `gepa` engines, and O1 could no longer choose its own optimizer. |
| C. Extend `recursive_opt.module.recursive_level@1` | Add a surface to the existing portable recursive level | **Eliminated.** It is built on legacy `LevelConfig` decoding (`spec.py:1706`, surfaces `config`, `family_policy`, `prior` only), so its artifact is config text, not a child spec. |
| D. A new evaluator `child_spec@1`, with the O1 artifact in `reasoning_workflow@1` components | Evaluator injects the components into a child spec and runs it | **Viable but awkward.** The injection map must live in the dataset examples, because objectives have no evaluator config. The O1 parameters are anonymous component strings. |
| **E. A new module `recursive_opt.module.child_spec@1`** | A `trace.Module` whose trainable parameters are **slots** of a child-spec template, and whose `forward(example)` runs the child | **Recommended.** It is exactly the package's founding idea (README §2: "a recursion level is itself a `trace.Module` whose `forward()` runs the optimization of the level below"). Slots are named, documented ParameterNodes; the evaluator stays generic. |

### Recommended design (E)

```jsonc
// O1 level (abridged). The child is an ordinary v2 spec, inline.
{
  "id": "O1",
  "module": {
    "ref": "recursive_opt.module.child_spec@1",
    "config": {
      "child": { /* a complete recursive-opt/v2alpha spec: O0 on one task-family instance */ },
      "slots": {
        "next_mode": {
          "path": "levels.O0.engine.config.hooks.trainer.next_mode",
          "kind": "code",
          "description": "When to refine, diverge or combine."
        },
        "diverge_text": {
          "path": "levels.O0.engine.config.trainer_kwargs.variation_instructions.diverge",
          "kind": "text"
        }
      },
      "example_paths": ["runtime.seed", "levels.O0.datasets"],
      "child_metric": "evaluation.metrics.score"
    }
  },
  "surface": {"kind": "module", "targets": ["next_mode", "diverge_text"]},
  "engine": {"name": "trace", "config": {"trainer": "PrioritySearch", "optimizer": "OptoPrimeV2", "iterations": 20}},
  "objective": {"evaluator_ref": "recursive_opt.evaluator.module_output@1", "metrics": {"score": {"direction": "maximize", "source": "evaluation.metrics.score"}}},
  "datasets": {
    "train":      [ {"runtime.seed": 1, "levels.O0.datasets": { /* episode A */ }}, "..." ],
    "validation": [ "... episodes C ..." ],
    "holdout":    [ "... episodes E ..." ]
  }
}
```

**Building the module.**
- One `ParameterNode` per slot, named after the slot.
- Its initial value is the value already at `path` in the child template.
- When that value is `null` for a code hook, the initial value is the source of the default method (R6).

**`forward(example)`:**
1. Deep-copy the child template.
2. Write each slot value at its `path` with the existing `_set_dotted_path` (R3: the path must already exist).
3. Write each key of the episode `example` at its path. The key must be listed in `example_paths`.
4. `compile_plan`. This is strict normalization, so an invalid O1 proposal becomes an invalid candidate, not a crash.
5. `execute_plan` with:
   - an output root under the parent's, keyed by the child fingerprint (R5);
   - the parent's runtime resources (LLM factory).
6. Return `{score, usage, child fingerprint, status}`. The usage goes through the existing `_charge_reported_usage` (R4).

**Snapshot and restore** work on a JSON dict `{slot: value}`, as `reasoning_workflow@1` does today.

**O2** is a `child_spec@1` level whose `child` is itself an O1 spec (R7).

**Split semantics** (taken from EXP22-QA's META-TRAIN / META-VALIDATION / META-TEST):
- An O1 example is a whole episode, with its own TRAIN / VALIDATION / MEASURE data.
- The child reports the episode's measure score.
- O1 selection sees only the train and validation episodes. The O1 holdout episodes stay closed until the end of the run (`DatasetAccess`, `spec.py:135`).

**Size estimate.**
- About 150 lines in `spec.py`: build, forward, snapshot, restore, config validator.
- About 10 tests:
  - slot injection;
  - an unknown path is rejected;
  - an invalid proposal becomes an invalid candidate;
  - usage is charged;
  - the cache hits;
  - O2 runs through the same module;
  - holdout episodes never reach the fit phase;
  - a default-slot child equals a direct run of the template.
- Ports that prove generality: EXP21 `o1_qa.meta` (O1 over configuration) and the EXP15/18 numeric study.

---

## 2. Piece 2: validated code in a trainer, optimizer or memory hook

### Alternatives considered

| Option | Verdict |
|---|---|
| Callables in `trainer_kwargs` | **Eliminated:** specs forbid callables (invariant 4, ADR). |
| A new keyword argument per hook in each class (`mode_policy=`, `parent_score=`, …) | **Eliminated as the general route:** every new target needs a core edit and a new spec field. Kept only for **text** surfaces that are plain data (e.g. `variation_instructions: dict`). |
| Monkey-patch any method named in the spec | **Eliminated:** it could replace `train` or `step` and silently change budgets. |
| **Declared hook points + a generic materializer** | **Recommended.** |

### Recommended design

**1. Declaring hook points.** Each hookable class declares, as a class attribute, the methods recursive_opt may replace. This is a few lines per class and no import of `opto.features`:

```python
class PrioritySearch(...):
    HOOKS = ('compute_exploration_priority', 'compute_exploitation_priority', 'filter_candidates')

class VariationSearch(PrioritySearch):
    HOOKS = PrioritySearch.HOOKS + ('next_mode', 'instruction')   # current _next_mode / _instruction, made public

class OptoPrimeV2(...):
    HOOKS = ('problem_instance',)   # what evidence is shown to the LLM and how it is rendered
```

**2. In the spec**, as JSON text under the `trace` engine config. `hooks` is a new validated knob; `null` means the default:

```json
"engine": {"name": "trace", "config": {
  "trainer": "VariationSearch",
  "hooks": {"trainer":   {"next_mode": "def next_mode(self):\n    ...", "compute_exploration_priority": null},
            "optimizer": {"problem_instance": null}}
}}
```

**3. Materializer** (in `recursive_opt`, about 120 lines). Before `optimize()`, for each non-null hook:
- Reject the name unless it is listed in the class's `HOOKS`.
- Parse the source as **one function** whose parameter names equal those of the original method (`inspect.signature`).
- Statically reject a short denylist: imports of `os`, `subprocess`, `socket`, `sys`, `importlib` and `shutil`; `open`, `exec`, `eval` and `__import__`; dunder attribute access other than `__name__`.
- Build `type(f'{cls.__name__}Hooked', (cls,), {name: guarded})`, passed to `optimize()` as a class (`load_trainer_class` accepts classes). Optimizers get the same treatment.
- `guarded(self, *a, **k)` calls the hook. On exception, a per-call time limit, or an output that fails the method's return check, it calls the original method and increments `hook_failures[name]` (R8).
- `hook_failures` and the call counts are added to the level metrics. The O1 feedback therefore says when a proposal fell back.

**4. Text surfaces need no code hook.** Expose the instruction texts as data (`VariationSearch(variation_instructions=...)`); the `OptoPrimeV2` objective already is (`optimizer_kwargs.objective`).

**5. Memory hooks, two kinds:**
- *Optimizer memory* (how past attempts enter the prompt) is reached through `OptoPrimeV2.HOOKS`.
- *Cross-run knowledge* (which artifacts are retrieved and promoted): `knowledge.retrieval` and the reserved `promotion_rule` / `rollback_rule`. These become hookable through `extensions.recursive_opt.knowledge_hooks`, with the same materializer. They are only needed for P4.

**Size estimate.**
- About 120 lines for the materializer plus the validation of the `hooks` knob.
- `HOOKS` declarations on 3 classes.
- `VariationSearch`: make `next_mode` / `instruction` public (aliases keep the private names); add `variation_instructions`.
- Tests:
  - a null hook is identical to no hook;
  - a valid hook changes behaviour;
  - a name outside the allowlist, a signature mismatch or a denylisted construct is rejected at compile time;
  - a runtime exception falls back to the default and is counted;
  - deepcopy and pickling of the hooked class work.

---

## 3. Decisions to take before implementing

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Hook allowlist location | (a) `HOOKS` declared on core classes (small edits in `opto/trainer`, `opto/optimizers`); (b) any method, accepted by signature check only (no core edit) | **(a)**: explicit, reviewable, cannot reach `train` / `step`. Touches core files, so it should go through a PR to `experimental`. |
| D2 | Isolation of hook code | (a) in-process, with static denylist + fallback (fast; hooks run many times per step); (b) a subprocess per call (safe, far too slow for per-candidate hooks) | **(a)**, stated as **not a security sandbox**: acceptable for our own optimizer's output in research runs, not for untrusted code. |
| D3 | What the child reports | (a) the episode's measure score (EXP22-QA design); (b) the child's training-best score | **(a)**: it measures the O0 artifact's generalization. O1 claims use O1-holdout episodes only. |
| D4 | Child template location | (a) inline in the O1 spec; (b) a versioned dataset/registry ref | **(a)** for EXP29 (simplest, fingerprinted). Add (b) only if specs become too large. |
| D5 | Parallel child runs inside one O1 evaluation | sequential first; a `workers` option later | **Sequential** for numeric and Signal (seconds per child). QA needs workers (EXP22-QA estimated up to 828 reader calls per O0 chain). |
| D6 | Fate of `recursive_level@1` | keep / deprecate | **Keep** as a legacy route; document `child_spec@1` as the portable recursion route. |

D1 and D2 are the only decisions that change the core library or its safety posture. D3–D6 have defaults that can be revised later.
