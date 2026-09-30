"""Narrow configurable proposal schedule over the existing production Trace engine."""

from __future__ import annotations

import contextlib
import io
import time
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _safe
from opto.features.recursive_opt import spec as control
from opto.features.recursive_opt.optimizer_program import optimizer_spec
from opto.optimizers.optimizer import Optimizer


def _completion_status(result: dict[str, Any]) -> str | None:
    """Recognize completed searches while preserving a typed invalid final artifact."""
    if result.get("valid") is True and result.get("status") != "error":
        return "search_completed"
    evaluation = result.get("evaluation") or {}
    if (
        result.get("valid") is False
        and result.get("status") == "invalid"
        and result.get("error") == "invalid_program"
        and evaluation.get("valid") is False
        and evaluation.get("status") == "invalid"
        and evaluation.get("error") == "invalid_program"
    ):
        return "search_completed_final_artifact_invalid"
    return None


def generate_recursive(
    owner: E.Experiment,
    outer: int,
    *,
    parents_per_round: int = 1,
    proposals_per_parent: int = 1,
) -> None:
    """Run a declared width/depth schedule through Control Plane and PrioritySearch.

    The owner supplies registered prompts, immutable response recording and the
    common evaluator, as in EXP-15. Only production trainer scheduling is made
    configurable here. There is no second candidate-search loop.
    """
    if (
        type(parents_per_round) is not int
        or type(proposals_per_parent) is not int
        or min(parents_per_round, proposals_per_parent) < 1
        or owner.slots % (parents_per_round * proposals_per_parent)
    ):
        raise ValueError("response slots must be divisible by positive schedule width")
    if outer not in owner.seeds:
        raise ValueError("unregistered outer seed")
    schedule = {
        "slots": owner.slots,
        "parents_per_round": parents_per_round,
        "proposals_per_parent": proposals_per_parent,
        "iterations": owner.slots // (parents_per_round * proposals_per_parent) + 1,
    }
    arm = getattr(owner, "arm", "A2")
    persist = getattr(owner, "persist", E.persist)
    directory = owner.root / str(outer) / arm
    persist(directory / "schedule.json", schedule)
    completion = directory / "generation_complete.json"
    trace_path = directory / "trace.json"
    if E.exists(completion) or E.exists(trace_path):
        if (
            sum(
                E.exists(directory / f"slot_{i:02d}/response.json")
                for i in range(owner.slots)
            )
            != owner.slots
        ):
            raise RuntimeError("completed production search has missing response slots")
        if not E.exists(trace_path):
            raise RuntimeError("completed production search has missing Trace evidence")
        saved = E.read(trace_path)
        config = saved["plan"]["engine"]["config"]
        if (
            saved["completed_update_callbacks"] != owner.slots
            or _completion_status(saved["result"]) is None
            or config["iterations"] != schedule["iterations"]
            or config["num_candidates"] != parents_per_round
            or config["trainer_kwargs"]["num_proposals"] != proposals_per_parent
        ):
            raise RuntimeError(
                "saved Trace does not prove the registered search completed"
            )
        if not E.exists(completion):
            persist(
                completion,
                {
                    "slots": owner.slots,
                    "completed_ns": time.time_ns(),
                    "recovered_from_trace": True,
                    "status": _completion_status(saved["result"]),
                },
            )
        return
    owner.panel(B.SEED_SOURCE, outer, "train")
    counter = [0]

    class SlotOptimizer(Optimizer):
        """Translate actual propagated Trace updates into immutable response slots."""

        def _step(self, *args: Any, **kwargs: Any) -> dict[Any, str]:
            """Generate once per production update and reject unallocated requests."""
            if len(self.parameters) != 1 or not self.parameters[0].feedback:
                raise RuntimeError(
                    "production update requires one source and Trace feedback"
                )
            if counter[0] >= owner.slots:
                raise RuntimeError(
                    "production schedule exceeded the response allocation"
                )
            parent = self.parameters[0].data
            replayed = E.exists(directory / f"slot_{counter[0]:02d}/response.json")
            hook = getattr(owner, "proposal_from_trace", None)
            if hook is None:
                result = owner.proposal(outer, arm, counter[0], parent)
                feedback_hash = None
            else:
                propagated = self.trace_graph.user_feedback
                if not isinstance(propagated, str) or not propagated.strip():
                    raise RuntimeError(
                        "production update has no propagated user feedback"
                    )
                result = hook(outer, counter[0], parent, propagated)
                feedback_hash = B.source_hash(propagated)
            owner.event(
                {
                    "event": "trace_update_replay" if replayed else "trace_update",
                    "outer": outer,
                    "slot": counter[0],
                    "parent_sha256": B.source_hash(parent),
                    "source_sha256": result["source_sha256"],
                    "propagated_feedback_present": True,
                    "propagated_feedback_sha256": feedback_hash,
                    "replayed_completed_response": replayed,
                }
            )
            counter[0] += 1
            return {self.parameters[0]: result["source"]}

    context_id = B.digest(["EXP-16", str(owner.root), owner.phase, outer, schedule])
    E._TRAIN_CONTEXTS[context_id] = {
        "owner": owner,
        "outer": outer,
        "optimizer": SlotOptimizer,
    }
    reference = "recursive_opt.evaluator.exp16_training@1"
    control.register_evaluator(reference, E._training_evaluator)
    control.register_engine(
        "exp16_trace_v1",
        control.EngineRegistryEntry(
            run=E._trace_engine,
            capabilities=frozenset(
                {"scalar", "weighted", "pareto", "rich_trace", "trace_module"}
            ),
        ),
    )
    raw = optimizer_spec(B.SEED_SOURCE, seed=outer, budget=owner.budget, engine="trace")
    raw["objective"] = {
        "evaluator_ref": reference,
        "intent": "Minimize training normalized anytime regret",
        "metrics": {
            "auc": {"direction": "minimize", "source": "evaluation.metrics.auc"}
        },
        "selection": {"mode": "scalar", "score_key": "auc"},
    }
    raw["datasets"] = {
        "train": [{"panel": "train", "context_id": context_id}],
        "validation": [],
        "holdout": [],
    }
    raw["engine"] = {
        "name": "exp16_trace_v1",
        "config": {
            "optimizer": "EXP16SlotOptimizer",
            "trainer": "PrioritySearch",
            "optimizer_kwargs": {},
            "iterations": schedule["iterations"],
            "num_candidates": parents_per_round,
            "validation_gate": False,
            "trainer_kwargs": {
                "num_threads": 1,
                "num_proposals": proposals_per_parent,
                "test_frequency": None,
                "log_frequency": 1000,
                "validate_exploration_candidates": False,
                "use_best_candidate_to_explore": True,
                "long_term_memory_size": None,
                "score_function": "mean",
            },
        },
    }
    log = io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = control.execute_plan(control.compile_plan(raw))[0]
    evidence = {
        "plan": raw,
        "result": result.to_dict(),
        "logs": _safe(log.getvalue()),
        "completed_update_callbacks": counter[0],
    }
    status = _completion_status(evidence["result"])
    if counter[0] != owner.slots or status is None:
        attempt = 1
        while E.exists(directory / f"trace_attempt_{attempt:03d}.json"):
            attempt += 1
        persist(directory / f"trace_attempt_{attempt:03d}.json", evidence)
    if counter[0] != owner.slots:
        raise RuntimeError(
            "production schedule did not consume the response allocation"
        )
    if status is None:
        raise RuntimeError("production search failed: " + str(result.error))
    persist(trace_path, evidence)
    persist(
        completion,
        {"slots": owner.slots, "completed_ns": time.time_ns(), "status": status},
    )
