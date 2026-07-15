"""
opto.features.recursive_opt.levels
==================================

The spine of recursive (meta) optimization on Trace.

WHAT "RECURSIVE / META OPTIMIZATION" MEANS HERE
-----------------------------------------------
Normal Trace optimizes a *task artifact* (a prompt or a code block) so an agent, functions,
or multiple-agents solve a task better.
Meta-optimization optimizes *the thing that does the optimizing*: the trainer, the batch design, the trace representation, the guide, the memory policy.

Recursive optimization stack is:

    O0  optimize a task artifact                         (e.g. a solver prompt/code)
    O1  optimize HOW O0 is optimized                     (the A-list below)
    O2  optimize the O1 policy per problem family
    O3  induce transferable priors across families

The generic deign trick :
**a recursion level is itself a ``trace.Module``.
** Its ``forward()`` runs the optimization of the level below and returns that level's held-out score;
its trainable parameters are whatever defines the level below.
Because every level is "just a Module", the *same* ``opto.trainer.train`` + ``opto.optimizers`` optimizer + ``Guide`` optimize all of them.
No new core machinery is needed — that is the whole point.

TWO COMPLEMENTARY TRAINABLE SURFACES  (answers "can we optimize the actual Trainer/batch-design/Trace classes, and future ones?")
-------------------------------------------------------------------------------
There are two *different kinds* of trainable parameter, and you choose per goal:

1. SELECTION / CONFIG surface  ->  ``LevelConfig`` + ``MetaLevel``
   A small, low-dimensional set of choices over EXISTING components
   ("use BeamsearchAlgorithm, batch_size=8, failure_balanced, typed memory").
   The optimizer SELECTS and CONFIGURES. Cheap, stable, good first step.
   This is what examples_A demonstrates.

2. CODE / IMPLEMENTATION surface  ->  ``CodeArtifactLevel`` (uses ``@trace.bundle(trainable=True)``)
   The trainable parameter is the *source code* of a component (a batch-design function, a trace summarizer, a Trainer's ``update`` rule, ...).
   The optimizer REWRITES THE CODE, so it can invent improved or entirely NEW components, not only pick from an enum.
   This is how you optimize the actual classes involved
   in training/optimization AND future classes that don't exist yet — you give
   it a baseline implementation and a feedback signal, and it writes a better one.
   This is what examples_B demonstrates.

You can mix surfaces:
an O1 ``MetaLevel`` can SELECT among existing trainers while a parallel ``CodeArtifactLevel`` REWRITES the chosen trainer's hot path.

Trace API used by this feature:
    opto.trace.model / node / bundle(trainable=True) / Module / ParameterNode
    opto.trainer.train(model=, train_dataset=, algorithm=, optimizer=, guide=)
    opto.optimizers.OptoPrime / OptoPrimeMulti
"""

from __future__ import annotations

import copy
import textwrap
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from opto import trace
from opto.trace import node, Module
from opto.trainer.guide import Guide


# =========================================================================== #
# SURFACE 1 — SELECTION / CONFIG
# =========================================================================== #
@dataclass
class LevelConfig:
    """The *selection/config* surface for one optimization level.

    Each field is a knob over an EXISTING component.
    The optimizer selects and configures; it does not change any component's code.
    Keep the set small:
    a low-dimensional config is what makes O1 search stable (the classic credit-assignment objection to meta-optimization).

    NOTE ON EXTENSIBILITY: these fields are plain strings/ints, NOT enums, on purpose.

    To register a brand-new Trainer or batch design,
    just (a) add the class to ``opto.trainer.algorithms`` (or your own module)
    and (b) allow its name as a value here — no change to the recursion machinery.

    To go further and *invent* a new component, use ``CodeArtifactLevel`` instead (surface 2).
    """

    # --- A.1 starting artifact + knowledge ---
    starting_artifact: str = ""  # initial prompt / code / SKILL.md / HARNESS.md
    initial_knowledge: str = ""  # constrained, task-FAMILY priors (no task leakage)

    # --- A.2 batch size & design ---
    batch_size: int = 4
    batch_design: str = "random"  # random | failure_balanced | diversity | curriculum | <your_new_one>

    # --- A.3 trace type & horizon ---
    trace_type: str = "internal"  # internal | otel | sysmon | hybrid
    credit_horizon: str = "episode"  # step | episode | truncated | full

    # --- A.4 memory ---
    memory_policy: str = "typed"  # none | fifo | typed | retrieval

    # --- A.5 LLM optimizer agentic design + tools ---
    optimizer: str = "OptoPrime" # OptoPrime | OptoPrimeMulti | OPRO | TextGrad | <your_new_one>
    optimizer_tools: Tuple[str, ...] = ()  # ("trace_search","pytest","note",...)

    # --- A.6 guide ---
    guide: str = "LLMJudge"  # LLMJudge | ExactMatch | staged | deterministic

    # --- A.7 trainer ---
    trainer: str = "MinibatchAlgorithm"  # Minibatch | Beamsearch | UCBSearch | <your_new_one>
    num_epochs: int = 1
    num_threads: int = 4  # >1 => async forward (the async-trainer behaviour)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


CONFIG_ALLOWED_VALUES: Dict[str, Tuple[str, ...]] = {
    "batch_design": ("random", "failure_balanced", "diversity", "curriculum"),
    "trace_type": ("internal", "otel", "sysmon", "hybrid"),
    "credit_horizon": ("step", "episode", "truncated", "full"),
    "memory_policy": ("none", "fifo", "typed", "retrieval"),
    "optimizer": ("OptoPrime", "OptoPrimeMulti", "OptoPrimeV2", "OptoPrimeMultiV2", "OPRO", "TextGrad"),
    "guide": ("LLMJudge", "ExactMatch", "ExactMatchGuide", "staged", "deterministic"),
    "trainer": (
        "MinibatchAlgorithm",
        "BeamsearchAlgorithm",
        "UCBSearchAlgorithm",
        "PrioritySearch",
        "PrioritySearchMulti",
        "ParetobasedPS",
        "SequentialUpdate",
        "SequentialSearch",
        "BeamSearch",
    ),
}

INVALID_CONFIG_SCORE = -1_000_000_000.0

# Bounded fallback penalty for an invalid candidate when no scoring.clip is
# configured. Must sort BELOW any real score (real scores are typically in
# [-1, 1] or [0, 1]) while NOT destroying reported means the way -1e9 would.
DEFAULT_INVALID_FLOOR = -1.0


def invalid_result(reason: str, floor: Optional[float] = None, **extra) -> dict:
    """Standard payload for an undecodable candidate (DRY across all surfaces).

    The penalty is the worst LEGAL score (the scoring clip floor) when known —
    an astronomical internal sentinel must never leak into stats, normalization
    baselines, or reported results (root cause of the -666,666,666 confirm runs).
    """
    return {"score": float(floor) if floor is not None else INVALID_CONFIG_SCORE,
            "feedback": reason, **extra}


def register_config_values(field: str, values: Iterable[str], *, replace: bool = False) -> None:
    """Register allowed enum-like values for a LevelConfig field.

    This keeps generated configs validated while preserving the extension path for
    new trainer, optimizer, batch-design, or trace labels.
    """
    if not field:
        raise ValueError("field must be a non-empty string")
    normalized = tuple(str(value).strip() for value in values if str(value).strip())
    if not normalized:
        raise ValueError("values must contain at least one non-empty entry")
    if replace:
        CONFIG_ALLOWED_VALUES[field] = normalized
        return
    existing = CONFIG_ALLOWED_VALUES.get(field, ())
    CONFIG_ALLOWED_VALUES[field] = tuple(dict.fromkeys((*existing, *normalized)))


def validate_config_field(field: str, value: Any) -> None:
    """Validate one enum-like config field when a value registry exists."""
    allowed = CONFIG_ALLOWED_VALUES.get(field)
    if allowed is None:
        return
    if str(value).strip() == "":
        return  # empty = unset/default (e.g. bundle-default starting_artifact): always a legal control arm
    if str(value) not in allowed:
        raise ValueError(
            f"Invalid value for {field}: expected one of {list(allowed)}, got {value!r}. "
            "Register new values with register_config_values(...) before using them."
        )


def validate_level_config(cfg: "LevelConfig", fields: Tuple[str, ...]) -> None:
    """Validate all enum-like fields in a LevelConfig for the selected surface."""
    for field in fields:
        if hasattr(cfg, field):
            validate_config_field(field, getattr(cfg, field))


def describe_config_fields(fields: Tuple[str, ...]) -> str:
    """Describe the editable LevelConfig surface for LLM optimizers."""
    lines = ["YAML-like LevelConfig. Edit only these fields:"]
    for field in fields:
        allowed = CONFIG_ALLOWED_VALUES.get(field)
        if allowed:
            lines.append(f"- {field}: one of {', '.join(allowed)}")
        elif field == "batch_size":
            lines.append("- batch_size: positive integer")
        else:
            lines.append(f"- {field}: value compatible with LevelConfig")
    lines.append("Do not add extra fields; unknown fields are removed before scoring.")
    return "\n".join(lines)


# =========================================================================== #
# O0 — optimize one task artifact directly (selection surface, single node)
# =========================================================================== #
@trace.model
class ArtifactLevel(Module):
    """O0: a single trainable artifact (prompt/code/harness) used by a task.

    ``parameters()`` exposes one trainable node (the artifact); any opto optimizer rewrites it.
    ``forward`` is the agent USING the artifact,
    so the artifact is on the traced path (a disconnected node cannot be optimized).
    """

    def __init__(self, cfg: LevelConfig, agent_fn: Callable[[Any, Any], Any]):
        super().__init__()
        self.artifact = node(
            cfg.starting_artifact or "TODO: solve the task.",
            trainable=True,
            name="artifact",
        )
        self._agent_fn = agent_fn  # (artifact_node, task_input) -> output

    def forward(self, x: Any):
        return self._agent_fn(self.artifact, x)


# =========================================================================== #
# O1+ — META level: forward() runs the level below; params = below's config
# =========================================================================== #
def encode_cfg(cfg: "LevelConfig", fields: Tuple[str, ...]) -> str:
    """Serialize selected config fields as ``key: value`` lines (the trainable text)."""
    d = cfg.to_dict()
    return "\n".join(f"{k}: {d[k]}" for k in fields)


def decode_cfg(text: str, base_cfg: "LevelConfig", fields: Tuple[str, ...]) -> "LevelConfig":
    """Parse ``key: value`` lines back into a LevelConfig (typed, validated).

    Shared by MetaLevel (O1), FamilyPolicyLevel (O2) and PriorInductionLevel (O3)
    so there is exactly one config (de)serialisation contract.
    """
    cfg = copy.deepcopy(base_cfg)
    for line in str(text).splitlines():
        if ":" not in line:
            continue
        k, v = (s.strip() for s in line.split(":", 1))
        if k in fields and hasattr(cfg, k):
            cur = getattr(cfg, k)
            # LLM-generated configs commonly wrap an enum/string value in quotes
            # (e.g. starting_artifact: "Plan step by step"). The raw enum has no
            # quotes, so strip a single matching surrounding pair before parsing —
            # otherwise a perfectly valid choice is rejected as an unknown value.
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                v = v[1:-1]
            try:
                if isinstance(cur, tuple):
                    parsed = tuple(v.split(",")) if v else ()
                elif isinstance(cur, int) and not isinstance(cur, bool):
                    parsed = int(v)
                    if parsed <= 0:
                        raise ValueError("must be positive")
                else:
                    parsed = type(cur)(v)
            except (TypeError, ValueError) as exc:
                expected = "positive int" if isinstance(cur, int) else type(cur).__name__
                raise ValueError(
                    f"Invalid value for {k}: expected {expected}, got {v!r}"
                ) from exc
            validate_config_field(k, parsed)
            setattr(cfg, k, parsed)
    validate_level_config(cfg, fields)
    return cfg


def canonicalize_cfg_text(text: str, base_cfg: "LevelConfig", fields: Tuple[str, ...]) -> str:
    """Return validated config text containing only the supported trainable fields."""
    return encode_cfg(decode_cfg(text, base_cfg, fields), fields)


@trace.model
class MetaLevel(Module):
    """O1/O2 over the SELECTION surface.

    Given forward(family) -> {"score","feedback"}
    Optimize the CURRENT (trainable) config on tasks from ``family``.

    Both the numeric score and the textual feedback flow back through the single
    trainable config node, so an LLM optimizer can read *why* a config
    under-performed and propose a better config (the OPTO ``(f, g)`` pair).
    """

    def __init__(
        self,
        cfg: LevelConfig,
        inner_runner: Callable[[LevelConfig, Any], Tuple[float, str]],
        trainable_fields: Tuple[str, ...] = (
            "batch_size",
            "batch_design",
            "trace_type",
            "memory_policy",
            "optimizer",
            "guide",
            "trainer",
        ),
        memory: Optional[object] = None,
        invalid_floor: Optional[float] = None,
    ):
        super().__init__()
        self._fields = trainable_fields
        self._base_cfg = cfg
        self._invalid_floor = invalid_floor if invalid_floor is not None else DEFAULT_INVALID_FLOOR
        # ONE node holds the whole (sub-)config as text -> low-dimensional search.
        self._cfg_node = node(
            self._encode(cfg),
            trainable=True,
            name="level_config",
            description=describe_config_fields(self._fields),
        )
        self._inner_runner = inner_runner
        self._memory = memory  # MemoryLite for active knowledge building

    def _encode(self, cfg: LevelConfig) -> str:
        return encode_cfg(cfg, self._fields)

    def _decode(self, text: str) -> LevelConfig:
        return decode_cfg(text, self._base_cfg, self._fields)

    def canonicalize(self) -> None:
        """Normalize generated config text to the validated field subset."""
        self._cfg_node._data = canonicalize_cfg_text(
            self._cfg_node.data,
            self._base_cfg,
            self._fields,
        )

    @trace.bundle(allow_external_dependencies=True)
    def _run_inner(self, cfg_text: str, family: Any):
        """Decode the config, run the inner optimization, and record the result."""
        try:
            cfg = self._decode(cfg_text)
        except ValueError as exc:
            return invalid_result(f"invalid generated config: {exc}", self._invalid_floor)
        score, feedback = self._inner_runner(cfg, family)
        if self._memory is not None:
            self._memory.record(
                level="O1",
                cfg=cfg.to_dict(),
                family=str(family),
                score=score,
                feedback=feedback,
            )
            if hasattr(self._memory, "record_artifact"):
                self._memory.record_artifact(
                    level="O1",
                    family=str(family),
                    kind="config_candidate",
                    content=self._encode(cfg),
                    score=float(score),
                    metrics={"feedback": str(feedback), "cfg": cfg.to_dict()},
                )
        return {"score": float(score), "feedback": str(feedback)}

    def forward(self, family: Any):
        try:
            self.canonicalize()
        except ValueError:
            pass
        return self._run_inner(self._cfg_node, family)

    # convenience for the offline driver in the examples
    def propose(self, **field_values):
        cfg = copy.deepcopy(self._base_cfg)
        for k, v in field_values.items():
            setattr(cfg, k, v)
        self._cfg_node._data = self._encode(cfg)

    def warm_start_from_memory(self, family: Any) -> None:
        """Apply a promoted memory prior to the trainable config node."""
        if self._memory is None:
            return
        cfg = self._memory.apply_priors(copy.deepcopy(self._base_cfg), str(family))
        self._cfg_node._data = self._encode(cfg)


# =========================================================================== #
# O2 / O3 — TRAINABLE recursive levels (not manual loops / majority vote)
# =========================================================================== #
def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


@trace.model
class FamilyPolicyLevel(Module):
    """O2: learn a *per-family* config policy as ONE trainable parameter.

    The trainable node is a policy text with one line per family::

        combinatorial => batch_design=failure_balanced, trainer=BeamsearchAlgorithm
        qa_reasoning  => batch_design=curriculum, trainer=UCBSearchAlgorithm

    ``forward()``:
    1. decodes the policy
    2. runs the inner optimization for every task of every family
    3. returns the mean score plus a per-family breakdown that names the worst family.
    4. An LLM optimizer reads that breakdown and rewrites the policy — i.e. O2 is a genuine trainable level, not a Python ``max`` loop.
    """

    def __init__(
        self,
        families: Dict[str, List[str]],
        run_task: Callable[[LevelConfig, str], Tuple[float, str]],
        base_cfg: Optional[LevelConfig] = None,
        policy_fields: Tuple[str, ...] = ("starting_artifact", "trace_type", "batch_design"),
        memory: Optional[object] = None,
        invalid_floor: Optional[float] = None,
    ):
        super().__init__()
        self._families = families
        self._run_task = run_task
        self._invalid_floor = invalid_floor if invalid_floor is not None else DEFAULT_INVALID_FLOOR
        self._base = base_cfg or LevelConfig()
        self._fields = policy_fields
        self._memory = memory
        seed = "\n".join(
            f"{fam} => " + ", ".join(f"{f}={getattr(self._base, f)}" for f in policy_fields)
            for fam in families
        )
        self._policy_node = node(seed, trainable=True, name="family_policy")

    def _decode_policy(self, text: str) -> Dict[str, LevelConfig]:
        out: Dict[str, LevelConfig] = {}
        for line in str(text).splitlines():
            if "=>" not in line:
                continue
            fam, rhs = (s.strip() for s in line.split("=>", 1))
            kv_lines = "\n".join(p.strip().replace("=", ": ", 1) for p in rhs.split(",") if p.strip())
            out[fam] = decode_cfg(kv_lines, self._base, self._fields)
        return out

    def canonicalize(self) -> None:
        """Normalize generated family policy text to known families and fields."""
        policy = self._decode_policy(self._policy_node.data)
        lines = []
        for fam in self._families:
            cfg = policy.get(fam, self._base)
            body = ", ".join(
                f"{field}={getattr(cfg, field)}" for field in self._fields
            )
            lines.append(f"{fam} => {body}")
        self._policy_node._data = "\n".join(lines)

    @trace.bundle(allow_external_dependencies=True)
    def _run_policy(self, policy_text: str):
        try:
            policy = self._decode_policy(policy_text)
        except ValueError as exc:
            return invalid_result(f"invalid generated family policy: {exc}",
                                  self._invalid_floor, per_family={})
        per_family = {}
        for fam, tasks in self._families.items():
            cfg = policy.get(fam, self._base)
            per_family[fam] = _mean(self._run_task(cfg, t)[0] for t in tasks)
        score = _mean(per_family.values())
        worst = min(per_family, key=per_family.get) if per_family else None
        fb = "; ".join(f"{f}={s:.3f}" for f, s in per_family.items())
        if worst is not None:
            fb += f". Weakest family: {worst} ({per_family[worst]:.3f}) — try a different per-family config for it."
        if self._memory is not None:
            self._memory.record(level="O2", cfg={"policy": policy_text}, family="<multi>",
                                 score=score, feedback=fb)
            self._memory.record_artifact(level="O2", family="<multi>", kind="policy",
                                          content=policy_text, score=score)
        return {"score": float(score), "feedback": fb, "per_family": per_family}

    def forward(self, _: Any = None):
        try:
            self.canonicalize()
        except ValueError:
            pass
        return self._run_policy(self._policy_node)

    def propose(self, policy_text: str):
        self._policy_node._data = policy_text


@trace.model
class PriorInductionLevel(Module):
    """O3: learn ONE transferable config, scored on HELD-OUT families.

    The trainable node is a single shared config; ``forward()`` applies it to
    families NOT used to induce it and returns the transfer score + which
    held-out families it fails on. Optimizing this node maximises genuine
    cross-family transfer — replacing the majority-vote heuristic with a
    trainable objective.
    """

    def __init__(
        self,
        train_families: Dict[str, List[str]],
        holdout_families: Dict[str, List[str]],
        run_task: Callable[[LevelConfig, str], Tuple[float, str]],
        base_cfg: Optional[LevelConfig] = None,
        fields: Tuple[str, ...] = ("starting_artifact", "trace_type", "batch_design"),
        memory: Optional[object] = None,
        invalid_floor: Optional[float] = None,
    ):
        super().__init__()
        self._invalid_floor = invalid_floor if invalid_floor is not None else DEFAULT_INVALID_FLOOR
        self._train = train_families
        self._holdout = holdout_families
        self._run_task = run_task
        self._base = base_cfg or LevelConfig()
        self._fields = fields
        self._memory = memory
        self._prior_node = node(encode_cfg(self._base, fields), trainable=True, name="transfer_prior")

    @trace.bundle(allow_external_dependencies=True)
    def _run_prior(self, prior_text: str):
        try:
            cfg = decode_cfg(prior_text, self._base, self._fields)
        except ValueError as exc:
            return invalid_result(f"invalid generated transfer prior: {exc}",
                                  self._invalid_floor, per_family={})
        per_family = {
            fam: _mean(self._run_task(cfg, t)[0] for t in tasks)
            for fam, tasks in self._holdout.items()
        }
        transfer = _mean(per_family.values())
        worst = min(per_family, key=per_family.get) if per_family else None
        fb = "held-out transfer: " + "; ".join(f"{f}={s:.3f}" for f, s in per_family.items())
        if worst is not None:
            fb += f". Worst held-out family: {worst} — adjust the prior to generalise to it."
        if self._memory is not None:
            self._memory.record(level="O3", cfg={"prior": prior_text}, family="<holdout>",
                                 score=transfer, feedback=fb)
            self._memory.record_artifact(level="O3", family="<holdout>", kind="prior",
                                          content=prior_text, score=transfer)
        return {"score": float(transfer), "feedback": fb, "per_family": per_family}

    def forward(self, _: Any = None):
        try:
            self.canonicalize()
        except ValueError:
            pass
        return self._run_prior(self._prior_node)

    def propose(self, **field_values):
        cfg = copy.deepcopy(self._base)
        for k, v in field_values.items():
            setattr(cfg, k, v)
        self._prior_node._data = encode_cfg(cfg, self._fields)

    def canonicalize(self) -> None:
        """Normalize generated transfer-prior text to the validated field subset."""
        self._prior_node._data = canonicalize_cfg_text(
            self._prior_node.data,
            self._base,
            self._fields,
        )


# =========================================================================== #
# SURFACE 2 — CODE / IMPLEMENTATION
# =========================================================================== #
class ComponentSpec:
    """Describes a library component whose CODE we want to improve/invent.

    name       : human label ("batch_design", "trace_summarizer", "trainer.update")

    baseline   : a Python function (the starting implementation; its SOURCE is the trainable parameter).
                 MUST take ``self`` as first arg (Trace's trainable-bundle convention) so it works as a method.

    evaluate   : (component_callable, family) -> (score, feedback);
                 runs the candidate code on real (or stub) Trace-Bench inner problems and returns a rich signal.
                 The optimizer reads ``feedback`` to know
                 HOW to rewrite the code.

    objective  : optional natural-language objective passed to the optimizer (e.g. "maximize held-out pass rate while keeping the function O(n log n)").
    """

    def __init__(
        self,
        name: str,
        baseline: Callable,
        evaluate: Callable[[Callable, Any], Tuple[float, str]],
        objective: str = "",
    ):
        self.name = name
        self.baseline = baseline
        self.evaluate = evaluate
        self.objective = objective


def _normalize_eval_result(result):
    """Normalize evaluator outputs to ``(score: float, feedback: str, metrics: dict)``.

    Supported evaluator contracts (so one level handles single- AND multi-objective evaluators:
    ``make_code_evaluator`` returns ``(score, feedback)``
     while ``make_multiobjective_evaluator`` returns ``(metrics_dict, feedback, scalar)``):
      1) ``(score, feedback)``
      2) ``(metrics_dict, feedback, scalar_score)``
      3) ``scalar_score`` (bare float)
    """
    metrics: Dict[str, float] = {}
    feedback = ""
    score = None
    if isinstance(result, tuple):
        if len(result) == 2:
            score, feedback = result
        elif len(result) == 3:
            first, feedback, scalar = result
            if isinstance(first, dict):
                metrics = {k: float(v) for k, v in first.items()}
            score = scalar if scalar is not None else first
        else:
            raise ValueError(
                "Evaluator must return (score, feedback) or (metrics, feedback, scalar)."
            )
    else:
        score = result
    if isinstance(score, dict):
        metrics = {k: float(v) for k, v in score.items()}
        if "score" not in metrics:
            raise ValueError(
                "Dict score outputs must contain a 'score' entry or provide a scalar third return."
            )
        score = metrics["score"]
    return float(score), str(feedback), metrics


def _canonicalize_def_name(source: str, expected_name: str) -> str:
    """Rename the first top-level ``def <x>`` in ``source`` to ``expected_name``.

    The trainable bundle resolves its recompiled callable by a fixed
    ``_fun_name`` (the baseline's name). When an optimizer emits improved code
    under a different function name (commonly ``spec.name``), the recompiled
    namespace would not contain ``_fun_name`` and the candidate would be lost.
    Aligning the def-line keeps the optimizer's code while making it resolvable.
    Returns ``source`` unchanged when it already defines ``expected_name`` or has
    no recognizable top-level def.
    """
    import re

    if re.search(rf"^\s*def\s+{re.escape(expected_name)}\b", source, flags=re.MULTILINE):
        return source
    m = re.search(r"^\s*def\s+(\w+)", source, flags=re.MULTILINE)
    if not m:
        return source
    return re.sub(r"^(\s*def\s+)\w+", r"\1" + expected_name, source, count=1,
                  flags=re.MULTILINE)


@trace.model
class CodeArtifactLevel(Module):
    """Optimize the SOURCE CODE of a component (improve / invent a new one).

    The trainable parameter is the body of ``self._impl``,
    a ``@trace.bundle(trainable=True)`` method seeded from ``spec.baseline``.
    The LLM optimizer rewrites this code;
    ``forward(family)`` runs the current code on Trace-Bench inner problems
    via ``spec.evaluate`` and returns ``{"score","feedback"}``.

    This is how recursive_opt PROPOSES IMPROVED VERSIONS of existing lib
    components (a Trainer's hot path, a batch sampler, a trace representation),
    rather than only selecting from a fixed menu.

    NOTE: we install the baseline as the bundle body via ``__set_impl`` so the
    deployed code is exactly the user-provided baseline (Trace reads it through
    ``inspect.getsource``; the baseline must be importable from a file, not a
    REPL lambda).
    """

    def __init__(self, spec: ComponentSpec, memory: Optional[object] = None):
        super().__init__()
        self._spec = spec
        self._memory = memory
        # The trainable code node. We wrap spec.baseline so its source is the param.
        self._impl = trace.bundle(trainable=True)(spec.baseline)
        # The bundle resolves its recompiled callable by the baseline's __name__
        # (_fun_name). An optimizer naturally emits improved code named after
        # spec.name, so when baseline.__name__ != spec.name the recompiled
        # namespace lacks _fun_name and forward() silently scores 0.0 even though
        # current_code() holds the best candidate. We record the expected name so
        # forward() can canonicalize the def-line of generated code to match.
        self._fun_name = getattr(spec.baseline, "__name__", spec.name)

    @trace.bundle()
    def _attach_eval(self, anchor, payload: Dict[str, Any]):
        # ``anchor`` is the last traced output of the trainable code. Taking it as
        # an input keeps the returned node CONNECTED to the trainable code param,
        # so a live optimizer's ``backward`` on this output reaches the code.
        return payload

    def current_code(self) -> str:
        """Return the current (possibly optimized) source code of the component."""
        params = self.parameters()
        return str(params[0].data) if params else ""

    def forward(self, family: Any):
        # Build a plain callable from the (trainable) code and evaluate it.
        # We unwrap the bundle's Node output to the raw value so evaluators can
        # use it directly, while remembering the LAST output node so a live
        # optimizer can call ``backward`` on it (which reaches the code param).
        #
        # Canonicalize the generated code's def-line to the bundle's _fun_name so
        # optimizer-emitted code named after spec.name still resolves (otherwise
        # forward() silently scores 0.0 — see _canonicalize_def_name). Read the
        # name from the LIVE bundle (it may have been swapped) rather than a cache.
        params = self.parameters()
        fun_name = None
        try:
            fun_name = self._impl.info.get("_fun_name")
        except Exception:
            fun_name = getattr(self, "_fun_name", None)
        if params and fun_name:
            current = str(params[0].data)
            canonical = _canonicalize_def_name(current, fun_name)
            if canonical != current:
                params[0]._data = canonical
        self._calls = []

        def component(*args, **kwargs):
            try:
                res = self._impl(self, *args, **kwargs)  # traced call -> Node
            except trace.ExecutionError as exc:
                self._calls.append(exc.exception_node)
                raise
            self._calls.append(res)
            return res.data if hasattr(res, "data") else res

        score, feedback, metrics = _normalize_eval_result(
            self._spec.evaluate(component, family)
        )
        self._last_node = self._calls[-1] if self._calls else None
        if self._last_node is None:
            raise RuntimeError(
                "Component evaluator did not invoke the candidate callable; no "
                "traced path exists from the final output back to the trainable "
                "code, so the optimizer cannot improve it."
            )
        if self._memory is not None:
            self._memory.record(
                level="O1-code",
                cfg={"component": self._spec.name},
                family=str(family),
                score=score,
                feedback=feedback,
                metrics=metrics,
            )
            if hasattr(self._memory, "record_artifact"):
                artifact_metrics = dict(metrics or {})
                artifact_metrics.setdefault("feedback", str(feedback))
                self._memory.record_artifact(
                    level="O1-code",
                    family=str(family),
                    kind="code",
                    content=self.current_code(),
                    score=score,
                    metrics=artifact_metrics,
                )
        payload: Dict[str, Any] = {"score": float(score), "feedback": str(feedback)}
        if metrics:
            payload["metrics"] = metrics
        # Re-anchor to the trainable code path so live ``backward`` reaches it.
        return self._attach_eval(self._last_node, payload)


# =========================================================================== #
# Guide that bridges any level's output into the trainer (score, feedback) API
# =========================================================================== #
class RecursiveGuide(Guide):
    """Maps a level's output into the ``Guide`` contract.

    Signature matches ``opto.trainer.guide.Guide.__call__``:
        (task, response, info, **kw) -> (score: float, feedback: str)
    """

    def __init__(self, inner_guide: Optional[Callable] = None):
        self._inner = inner_guide  # used for O0 task scoring (e.g. LLMJudge)

    def get_feedback(
        self, query: str, response: Any, reference: Any = None, **kwargs
    ) -> Tuple[float, str]:
        """Return scalar score and feedback for recursive level outputs."""
        data = response.data if hasattr(response, "data") else response
        if isinstance(data, dict) and "score" in data:  # meta-level output
            return float(data["score"]), str(data.get("feedback", ""))
        if self._inner is not None:  # O0 task output
            return self._inner(query, response, reference, **kwargs)
        ok = str(data).strip() == str(reference).strip()  # deterministic fallback
        return (1.0 if ok else 0.0), ("correct" if ok else f"expected: {reference}")

    def get_score_dict(
        self, query: str, response: Any, reference: Any = None, **kwargs
    ) -> dict:
        """Per-objective scores for the Trainer's multi-objective (Pareto) path.

        Returns the level's ``objectives`` dict when present (e.g. capability
        artifacts expose {"accuracy", "cost"}); otherwise a single ``score`` key.
        Only consulted by trainers when an ``ObjectiveConfig`` is supplied, so the
        scalar path is unaffected.
        """
        data = response.data if hasattr(response, "data") else response
        if isinstance(data, dict) and isinstance(data.get("objectives"), dict):
            return {k: float(v) for k, v in data["objectives"].items()}
        score, _ = self.get_feedback(query, response, reference, **kwargs)
        return {"score": float(score)}


def best_config_from(level: MetaLevel) -> str:
    """Current optimized config text of a MetaLevel.

    Reads the *live* trainable parameter (like ``CodeArtifactLevel.current_code``)
    rather than a cached attribute, so it reflects what the Trainer wrote back.
    Always returns a non-empty, canonicalized config (falls back to the encoded
    base config if the trained text is blank or unparizable).
    """
    params = level.parameters()
    text = str(params[0].data) if params else ""
    if not text.strip():
        return encode_cfg(level._base_cfg, level._fields)
    try:
        return canonicalize_cfg_text(text, level._base_cfg, level._fields)
    except Exception:
        return text


class TimedGuide(Guide):
    """Wrap any Guide and expose evaluation wall-time as a Pareto objective.

    Enables time-aware multi-objective training/tie-breaking, e.g.
    ``ObjectiveConfig(mode="pareto", minimize={"wall_time"})`` — or declaratively
    via the spec level keys ``"timed_guide": true`` +
    ``"objective_config": {"mode": "pareto", "minimize": ["wall_time"]}``.
    Thread-safe: elapsed time is tracked per thread (parallel rollouts).
    """

    def __init__(self, inner: Guide):
        self._inner = inner
        import threading
        self._local = threading.local()

    def get_feedback(self, query, response, info=None, **kwargs):
        import time as _time
        t0 = _time.perf_counter()
        out = self._inner.get_feedback(query, response, info, **kwargs)
        self._local.elapsed = _time.perf_counter() - t0
        return out

    def get_score_dict(self, query, response, info=None, **kwargs):
        base = {}
        if hasattr(self._inner, "get_score_dict"):
            base = dict(self._inner.get_score_dict(query, response, info, **kwargs) or {})
        base.setdefault("wall_time", float(getattr(self._local, "elapsed", 0.0)))
        return base

    def __getattr__(self, name):  # transparent for anything else trainers read
        return getattr(self._inner, name)


@trace.model
class CapabilityArtifact(Module):
    """Trainable capability implementation (a SKILL-style text/policy).

    Promoted from example C: the artifact text is the trainable parameter;
    ``forward`` evaluates it through a (multi-objective) evaluator returning
    ``(score_dict, feedback, scalar)`` and yields a score DICT, so it works with
    plain trainers (scalar) and Pareto ``ObjectiveConfig`` trainers (objectives).
    Declarative form: spec surface ``"capability"`` with keys ``seed`` and
    ``evaluator``.
    """

    def __init__(self, seed_impl: str, evaluator, memory=None):
        super().__init__()
        self.impl = node(seed_impl, trainable=True, name="capability")
        self._eval = evaluator
        self._memory = memory

    @trace.bundle(allow_external_dependencies=True)
    def _evaluate_impl(self, impl_text, family):
        # Taking the trainable ``self.impl`` node as input keeps this output
        # CONNECTED to the parameter so optimizers can backpropagate to it.
        impl_text = impl_text.data if hasattr(impl_text, "data") else str(impl_text)

        def capability(task):
            return {"answer": impl_text, "task": task}

        score_dict, feedback, scalar = self._eval(capability, family)
        if self._memory is not None:
            self._memory.record(
                level="capability",
                cfg={"len": len(impl_text)},
                family=str(family),
                score=scalar,
                feedback=feedback,
                metrics=score_dict,
            )
        return {"score": scalar, "feedback": feedback, "objectives": score_dict}

    def forward(self, family):
        return self._evaluate_impl(self.impl, family)

    def current_text(self) -> str:
        return str(self.parameters()[0].data)
