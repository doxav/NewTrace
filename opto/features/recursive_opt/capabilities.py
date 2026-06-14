"""
recursive_opt.capabilities  —  new capabilities  (C.2 / C.3 / C.4)
==================================================================

C.2  AgenticOptimizer    : wrap a standard opto optimizer so the LLM optimizer
                           can call *tools* during optimization (search traces,
                           run pytest, run a subset eval, note/memorize).
C.3  Tinker integration  : expose a Tinker RL/SFT task family as an *outer-loop
                           environment* whose reward feeds the same recursive
                           stack (the optimizer proposes artifacts; Tinker scores).
C.4  HITL gate           : a guide / update-gate that pauses for human approval
                           when an update is risky or evidence is thin.

These wrap the real optimizer/guide contracts so they drop into
``opto.trainer.train`` unchanged.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from opto.optimizers import OptoPrime
from opto.optimizers.optimizer import Optimizer


# --------------------------------------------------------------------------- #
# C.2  Agentic LLM optimizer: optimizer that uses tools during optimization.
# --------------------------------------------------------------------------- #
class AgenticOptimizer(Optimizer):
    """Wrap an opto optimizer; let it call tools before proposing an update.

    Tools are plain callables registered by name. Before each ``step`` the
    wrapper runs a bounded tool loop (budgeted) that can *gather evidence*
    (search past failures, run the artifact's tests, run a subset eval) and
    inject the results into the feedback the inner optimizer reads. This is the
    "AgenticTrace" idea: optimizer sees computational paths, not just a scalar.
    """

    def __init__(
        self,
        parameters: list,
        *,
        tools: Optional[Dict[str, Callable]] = None,
        tool_budget: int = 3,
        base_optimizer_cls: Callable = OptoPrime,
        **kw: Any,
    ) -> None:
        super().__init__(parameters)
        self.opt = base_optimizer_cls(parameters, **kw)
        self.tools = tools or {}
        self.tool_budget = tool_budget
        self._notes: List[str] = []

    # opto optimizer surface -------------------------------------------------
    def zero_feedback(self):
        return self.opt.zero_feedback()

    def backward(self, target, feedback, *a, **kw):
        evidence = self._run_tools(feedback)
        enriched = (
            feedback if not evidence else f"{feedback}\n[tool-evidence]\n{evidence}"
        )
        return self.opt.backward(target, enriched, *a, **kw)

    def step(self, *a, **kw):
        return self.opt.step(*a, **kw)

    def update(self, *a, **kw):
        return self.opt.update(*a, **kw)

    def __getattr__(self, name):
        # Transparent wrapper: delegate anything else (llm, log, parameters, ...)
        # to the inner optimizer so any Trainer can drive this like the original.
        return getattr(self.opt, name)

    # tool loop --------------------------------------------------------------
    def _run_tools(self, feedback: str) -> str:
        out = []
        for name, fn in list(self.tools.items())[: self.tool_budget]:
            try:
                res = fn(feedback)
                if name in ("note", "memorize"):
                    self._notes.append(str(res))
                out.append(f"{name}: {res}")
            except Exception as e:  # tools never break the loop
                out.append(f"{name}: <error {e}>")
        return "\n".join(out)


def default_optimizer_tools(
    memory=None,
    run_subset: Callable = None,
    pytest_fn: Callable = None,
    *,
    family: Optional[str] = None,
) -> Dict[str, Callable]:
    """Standard optimizer toolset used in the examples.

    ``family`` scopes the ``trace_search`` tool: ``None`` (default) searches all
    families globally; pass a concrete id to restrict retrieval to that family.
    """
    tools: Dict[str, Callable] = {}
    if memory is not None:
        tools["trace_search"] = lambda fb, fam=family: [
            e.feedback[:120] for e in memory.similar_failures(family=fam, k=2)
        ]
    if run_subset is not None:
        tools["run_subset"] = lambda fb: run_subset()
    if pytest_fn is not None:
        tools["pytest"] = lambda fb: pytest_fn()
    tools["note"] = lambda fb: f"observed: {fb[:80]}"
    return tools


def parse_optimizer_tool_policy(
    policy: Any,
    available_tools: Tuple[str, ...] | List[str] | Dict[str, Callable],
    *,
    default_tools: Optional[Tuple[str, ...] | List[str]] = None,
    max_tools: Optional[int] = None,
    strict: bool = False,
) -> List[str]:
    """Parse a generated optimizer-tool policy into known tool names.

    The policy may be a list/tuple, a JSON object with a ``tools`` key, a JSON
    list, or compact text such as ``"tools: trace_search, run_subset"``. Unknown
    names are ignored by default so generated policies degrade safely; set
    ``strict=True`` for validation tests.
    """
    known = tuple(available_tools.keys()) if isinstance(available_tools, dict) else tuple(available_tools)
    if max_tools is not None and int(max_tools) < 0:
        raise ValueError("max_tools must be non-negative")

    def flatten(value: Any) -> List[str]:
        """Return candidate names from nested JSON/list/text policy values."""
        if value is None:
            return []
        if isinstance(value, dict):
            for key in ("tools", "tool_policy", "order", "use"):
                if key in value:
                    return flatten(value[key])
            return []
        if isinstance(value, (list, tuple)):
            out: List[str] = []
            for item in value:
                out.extend(flatten(item))
            return out
        text = str(value).strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
        except Exception:
            loaded = None
        if loaded is not None and loaded is not value:
            parsed = flatten(loaded)
            if parsed:
                return parsed
        tokens: List[str] = []
        for line in text.replace(",", "\n").splitlines():
            line = line.strip().strip("-* ")
            if not line:
                continue
            if ":" in line:
                key, rest = line.split(":", 1)
                if key.strip().lower() in {"tools", "tool_policy", "order", "use"}:
                    line = rest.strip()
            tokens.extend(part.strip().strip("'\"`") for part in line.split() if part.strip())
        return tokens

    candidates = flatten(policy)
    if not candidates and default_tools:
        candidates = list(default_tools)

    selected: List[str] = []
    unknown: List[str] = []
    for name in candidates:
        if name in known and name not in selected:
            selected.append(name)
            if max_tools is not None and len(selected) >= int(max_tools):
                break
        elif name not in known:
            unknown.append(name)
    if strict and unknown:
        raise ValueError(f"unknown optimizer tool(s): {unknown}; available={list(known)}")
    return selected


def select_optimizer_tools(
    available_tools: Dict[str, Callable],
    policy: Any,
    *,
    default_tools: Optional[Tuple[str, ...] | List[str]] = None,
    max_tools: Optional[int] = None,
    strict: bool = False,
) -> Dict[str, Callable]:
    """Return the ordered subset of optimizer tools selected by a policy."""
    names = parse_optimizer_tool_policy(
        policy,
        available_tools,
        default_tools=default_tools,
        max_tools=max_tools,
        strict=strict,
    )
    return {name: available_tools[name] for name in names}


# --------------------------------------------------------------------------- #
# C.3  Tinker integration: a stable RL/SFT family as an outer-loop testbed.
# --------------------------------------------------------------------------- #
class TinkerEnvAdapter:
    """Adapt a Tinker task to the recursive (inner_runner) contract.

    The recursive stack does not care whether the reward came from a unit test,
    an LLM judge, or a Tinker rollout. This adapter exposes
        run(cfg, family) -> (score, feedback)
    by (a) materialising the artifact from cfg, (b) handing it to a Tinker
    rollout/eval, (c) returning mean reward + a textual reward breakdown.

    `tinker_client` is duck-typed: any object with
        .rollout(policy_text, task) -> dict(reward=float, transcript=str)
    works (real `tinker` SDK, or a local stub for smoke tests).
    """

    def __init__(self, tinker_client, task_sampler: Callable[[Any], List[Any]]):
        self.client = tinker_client
        self.task_sampler = task_sampler

    def run(self, cfg, family: Any) -> Tuple[float, str]:
        tasks = self.task_sampler(family)
        rewards, transcripts = [], []
        for t in tasks:
            r = self.client.rollout(policy_text=cfg.starting_artifact, task=t)
            rewards.append(float(r.get("reward", 0.0)))
            transcripts.append(str(r.get("transcript", ""))[:200])
        mean = sum(rewards) / max(len(rewards), 1)
        fb = (
            f"tinker mean_reward={mean:.3f} over {len(rewards)} rollouts; "
            f"worst={min(rewards) if rewards else 0:.3f}. "
            f"sample_transcript={transcripts[0] if transcripts else ''}"
        )
        return mean, fb


# --------------------------------------------------------------------------- #
# C.4  Human-in-the-loop optimization gate.
# --------------------------------------------------------------------------- #
class HITLGate:
    """Validation gate that escalates risky/low-evidence updates to a human.

    Wraps the trainer's accept/reject decision. An update is auto-accepted only
    if (validation improvement >= threshold) AND (support >= min_support).
    Otherwise it calls `approver(diff, evidence) -> bool`. `approver` can be a
    CLI prompt, a web UI, a Slack approval, or an auto-allow stub for tests.
    Every decision is logged for the audit trail.
    """

    def __init__(
        self,
        approver: Callable[[str, str], bool],
        threshold: float = 0.0,
        min_support: int = 3,
        risky_fields: Tuple[str, ...] = ("optimizer", "trainer"),
    ):
        self.approver = approver
        self.threshold = threshold
        self.min_support = min_support
        self.risky_fields = risky_fields
        self.audit: List[Dict[str, Any]] = []

    def decide(
        self,
        *,
        old_score: float,
        new_score: float,
        support: int,
        diff: str,
        evidence: str,
        changed_fields: Tuple[str, ...] = (),
    ) -> bool:
        improved = (new_score - old_score) >= self.threshold
        risky = any(f in self.risky_fields for f in changed_fields)
        if improved and support >= self.min_support and not risky:
            decision, via = True, "auto"
        else:
            decision, via = bool(self.approver(diff, evidence)), "human"
        self.audit.append(
            dict(
                old=old_score,
                new=new_score,
                support=support,
                risky=risky,
                decision=decision,
                via=via,
            )
        )
        return decision


def auto_allow(diff: str, evidence: str) -> bool:  # stub approver for tests
    return True
