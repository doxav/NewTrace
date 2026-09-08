"""Registered EXP-15 orchestration around the production Trace search path."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

from artifacts.optimizer_discovery import benchmark as B
from artifacts.optimizer_discovery.phase0 import _load_key, _safe
from opto.features.recursive_opt import spec as control
from opto.features.recursive_opt.measurement import (
    is_transient_provider_error,
)
from opto.features.recursive_opt.optimizer_program import optimizer_spec, parse_program
from opto.features.recursive_opt.runmode import make_live_llm
from opto.optimizers.optimizer import Optimizer
from opto.trainer.objectives import EvaluationResult

INVARIANT = """Generate one complete Python optimizer.py exporting exactly propose(history, bounds, seed),
with those three positional parameters and no defaults. Minimize an unknown black-box
numerical function over finite bounds. History is a list of {x: coordinate list, value: finite number}.
Return one finite in-bounds coordinate list of length len(bounds). Lower values are better.
Tasks are deterministic smooth numerical functions in dimensions 2 and 4, including convex
and curved-valley landscapes. The budget is 32 objective evaluations. The function receives
no objective implementation, hidden parameters, known optimum, task identity or split label.
Each invocation is a fresh process. Use only history and seed for state, e.g. random.Random(seed+len(history)).
Identical arguments must produce identical points. Standard library only; allowed imports:
math, random, statistics, itertools, functools, collections, heapq, operator, typing, bisect.
No external I/O, files, network, environment or process inspection, dynamic execution,
introspection, private/dunder attributes, LLM calls or third-party dependencies.
Hard per-proposal timeout: two seconds. Keep the implementation simple and portable.
Return exactly one Python code block with the complete file; no explanations.
The common starting optimizer is:
"""

_TRAIN_CONTEXTS: dict[str, dict[str, Any]] = {}


def _training_evaluator(output: Any, example: Any, context: Any) -> EvaluationResult:
    """Evaluate the registered training panel without exposing its host parameters."""
    if set(example) != {"panel", "context_id"} or example["panel"] != "train":
        raise RuntimeError("generation evaluator accepts training only")
    binding = _TRAIN_CONTEXTS[example["context_id"]]
    owner, outer = binding["owner"], binding["outer"]
    source = (output.data if hasattr(output, "data") else output)["components"][
        "optimizer"
    ]
    rows = owner.panel(source, outer, "train")
    valid = all(row["valid"] for row in rows)
    return EvaluationResult(
        valid=valid,
        status="ok" if valid else "invalid",
        metrics={"auc": B.aggregate(rows, "auc")} if valid else {},
        feedback=json.dumps(owner.feedback(source, outer), sort_keys=True),
        artifacts={"behavior_signature": [r["observations"] for r in rows]},
        error=None if valid else "invalid_program",
    )


def _trace_engine(unit: Any, level: Any, resources: Any) -> Any:
    """Bind the declared slot adapter, then delegate to the existing production engine."""
    context_id = level.datasets["train"][0]["context_id"]
    binding = _TRAIN_CONTEXTS[context_id]
    return control._run_module_engine(
        unit, level, {**resources, "optimizer": binding["optimizer"]}, fit=True
    )


def persist(path: Path, value: Any) -> None:
    """Atomically preserve immutable sanitized JSON; refuse conflicting replacement."""
    text = _safe(json.dumps(value, indent=2, allow_nan=False)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"refusing to overwrite completed evidence: {path.name}")
        return
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(text)
    os.replace(temporary, path)


def read(path: Path) -> Any:
    """Read a retained JSON artifact without interpreting generated code."""
    return json.loads(path.read_text())


def preflight(path: Path = B.ROOT / "exp15/freeze.json") -> dict[str, Any]:
    """Refuse confirmation unless actual code and manifest match the committed freeze."""
    if not path.exists():
        raise RuntimeError("confirmatory freeze is missing")
    frozen = read(path)
    for name, expected in frozen["files"].items():
        if B.source_hash(Path(name).read_text()) != expected:
            raise RuntimeError(f"confirmatory freeze mismatch: {name}")
    if B.MANIFEST["status"] != "FROZEN_CONFIRMATORY":
        raise RuntimeError("manifest is not a confirmatory freeze")
    return frozen


def paired(deltas: list[float]) -> dict[str, Any]:
    """Bootstrap paired outer-seed differences with a frozen RNG and percentile rule."""
    if not deltas or not all(math.isfinite(x) for x in deltas):
        raise ValueError("paired analysis needs all finite outer-seed differences")
    config = B.MANIFEST["bootstrap"]
    rng = random.Random(config["seed"])
    samples = sorted(
        statistics.mean(rng.choices(deltas, k=len(deltas)))
        for _ in range(config["replicates"])
    )
    interval = []
    for percentile in config["percentiles"]:
        position = (len(samples) - 1) * percentile / 100
        lower = math.floor(position)
        upper = math.ceil(position)
        interval.append(
            samples[lower] + (samples[upper] - samples[lower]) * (position - lower)
        )
    interpretation = (
        "positive signal"
        if interval[1] < 0
        else (
            "negative signal"
            if interval[0] > 0
            else (
                "no detectable difference"
                if all(x == 0 for x in deltas)
                else "inconclusive"
            )
        )
    )
    return {
        "deltas": deltas,
        "mean": statistics.mean(deltas),
        "median": statistics.median(deltas),
        "paired_bootstrap_95": interval,
        "interpretation": interpretation,
        "replication_unit": "outer_seed",
    }


class Experiment:
    """Persist proposal slots and use the common evaluator for both search arms."""

    def __init__(self, root: Path, phase: str, *, client: Any = None) -> None:
        """Bind a run to its manifest before any requests or objective evaluations."""
        if phase not in ("pilot", "confirmation"):
            raise ValueError("phase must be pilot or confirmation")
        if phase == "confirmation":
            preflight()
        self.root, self.phase, self.client = root, phase, client
        self.seeds = B.MANIFEST[
            "pilot_outer_seeds" if phase == "pilot" else "outer_seeds"
        ]
        self.slots = B.MANIFEST[
            "pilot_proposal_slots" if phase == "pilot" else "proposal_slots"
        ]
        self.budget = B.MANIFEST["inner_budget"]
        self.evaluator_version = B.digest(
            [
                B.VERSION,
                B.source_hash(Path(B.__file__).read_text()),
                B.source_hash(
                    Path("opto/features/recursive_opt/optimizer_program.py").read_text()
                ),
            ]
        )
        persist(
            root / "run.json",
            {
                "experiment": "EXP-15",
                "phase": phase,
                "manifest_sha256": B.digest(B.MANIFEST),
                "evaluator_version": self.evaluator_version,
            },
        )

    def event(self, value: dict[str, Any]) -> None:
        """Append accounting events without embedding credential-bearing objects."""
        with (self.root / "events.jsonl").open("a") as handle:
            handle.write(
                _safe(json.dumps({"time_ns": time.time_ns(), **value}, allow_nan=False))
                + "\n"
            )

    def panel(
        self, source: str, outer: int, split: str, *, deployment: bool = False
    ) -> list[dict[str, Any]]:
        """Use frozen deterministic cache keys and block premature holdout access."""
        if outer not in self.seeds:
            raise ValueError("unregistered outer seed")
        if split == "holdout" and not (self.root / "selections_frozen.json").exists():
            raise RuntimeError("holdout requires every selection to be frozen")
        rows = []
        for task in B.make_tasks(self.phase, split):
            seed = B.local_seed(self.phase, outer, task)
            key = {
                "source_sha256": B.source_hash(source),
                "task_identity": B.task_identity(task),
                "phase": self.phase,
                "split": split,
                "local_seed": seed,
                "budget": self.budget,
                "deployment": deployment,
                "evaluator_version": self.evaluator_version,
            }
            path = self.root / "cache" / f"{B.digest(key)}.json"
            hit = path.exists()
            if hit:
                entry = read(path)
                if entry["key"] != key:
                    raise RuntimeError("evaluation cache identity mismatch")
                row = entry["result"]
            else:
                row = B.evaluate(
                    source,
                    task,
                    seed,
                    budget=self.budget,
                    deployment=deployment,
                    timeout_s=B.MANIFEST["proposal_timeout_s"],
                )
                persist(path, {"key": key, "result": row})
            self.event(
                {
                    "event": "evaluation",
                    "key": key,
                    "cache_hit": hit,
                    "objective_calls": 0 if hit else row["objective_calls"],
                }
            )
            rows.append(row)
        return rows

    def feedback(self, source: str, outer: int) -> dict[str, Any]:
        """Expose bounded raw training observations without hidden benchmark constants."""
        rows = self.panel(source, outer, "train")
        return {
            "valid": all(row["valid"] for row in rows),
            "tasks": [
                {
                    "valid": row["valid"],
                    "status": row["status"],
                    "observations": row["observations"][:2] + row["observations"][-2:],
                    "best_observed_value": min(
                        (r["value"] for r in row["observations"]), default=None
                    ),
                }
                for row in rows
            ],
        }

    def proposal(self, outer: int, arm: str, slot: int, parent: str) -> dict[str, Any]:
        """Issue or replay exactly one response slot, preserving all transport attempts."""
        if (
            outer not in self.seeds
            or arm not in ("A1", "A2")
            or type(slot) is not int
            or not 0 <= slot < self.slots
        ):
            raise ValueError("unregistered proposal slot")
        directory = self.root / str(outer) / arm / f"slot_{slot:02d}"
        for path in directory.glob("attempt_*.json"):
            if read(path).get("status") == "in_flight":
                raise RuntimeError(
                    "uncertain remote completion requires provider reconciliation"
                )
        messages = [{"role": "user", "content": INVARIANT + B.SEED_SOURCE}]
        if arm == "A2":
            feedback = {"current": self.feedback(parent, outer)}
            if slot:
                previous = read(
                    directory.parent / f"slot_{slot-1:02d}" / "response.json"
                )
                feedback["previous_attempt"] = self.feedback(previous["source"], outer)
            bounded = json.dumps(feedback, sort_keys=True, allow_nan=False)[
                : B.MANIFEST["feedback"]["max_training_feedback_chars"]
            ]
            messages.append(
                {
                    "role": "user",
                    "content": "Improve the current optimizer using TRAINING FEEDBACK only.\nCURRENT SOURCE:\n"
                    + parent
                    + "\nTRAINING FEEDBACK:\n"
                    + bounded,
                }
            )
        settings = {
            key: B.MANIFEST["model"][key]
            for key in ("temperature", "top_p", "max_tokens", "extra_body", "timeout")
        }
        settings.update(
            {
                "seed": B.stable_seed("request", self.phase, outer, slot),
                "num_retries": 0,
            }
        )
        request = {
            "slot": slot,
            "outer_seed": outer,
            "arm": arm,
            "model": B.MANIFEST["model"]["model"],
            "messages": messages,
            "settings": settings,
            "parent_sha256": (
                B.source_hash(parent) if arm == "A2" else B.MANIFEST["seed_sha256"]
            ),
        }
        persist(directory / "request.json", request)
        if (directory / "response.json").exists():
            completed = read(directory / "response.json")
            receipt = directory / f'attempt_{completed["attempt"]}.json'
            if not receipt.exists():
                persist(
                    receipt,
                    {
                        "status": "completed",
                        "id": completed["id"],
                        "wall_s": completed["wall_s"],
                        "recovered_from_persisted_response": True,
                    },
                )
            return completed
        if list(directory.glob("started_*.json")):
            started = list(directory.glob("started_*.json"))
            if any(
                not (directory / p.name.replace("started_", "attempt_")).exists()
                for p in started
            ):
                raise RuntimeError(
                    "uncertain remote completion requires provider reconciliation"
                )
        if self.client is None:
            raise RuntimeError("live client is required for an unfinished proposal")
        offset = len(list(directory.glob("attempt_*.json")))
        for index in range(4):
            attempt_id = offset + index + 1
            persist(
                directory / f"started_{attempt_id}.json",
                {"time_ns": time.time_ns(), "status": "in_flight"},
            )
            captured = io.StringIO()
            started_at = time.monotonic()
            try:
                with (
                    contextlib.redirect_stdout(captured),
                    contextlib.redirect_stderr(captured),
                ):
                    response = self.client(messages=messages, **settings)
            except Exception as error:  # noqa: BLE001 - retain every provider failure
                transient = is_transient_provider_error(error)
                persist(
                    directory / f"attempt_{attempt_id}.json",
                    {
                        "status": "transport_failure",
                        "error_type": type(error).__name__,
                        "error": _safe(str(error)),
                        "transient": transient,
                        "wall_s": time.monotonic() - started_at,
                        "logs": _safe(captured.getvalue())[:16000],
                    },
                )
                if not transient or index == 3:
                    raise RuntimeError(
                        "transport attempts exhausted; slot remains uncompleted"
                    ) from None
                time.sleep(B.MANIFEST["model"]["transport_retry_delays_s"][index])
                continue
            content = control._optimizer_response_text(response)
            try:
                source = parse_program(content)
                parse_status = "parsed"
            except ValueError:
                source, parse_status = "", "unparsable"
            usage = getattr(response, "usage", {}) or {}
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            safe_usage = {
                k: usage[k]
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                if usage.get(k) is not None
            }
            details = usage.get("completion_tokens_details") or {}
            if details.get("reasoning_tokens") is not None:
                safe_usage["reasoning_tokens"] = details["reasoning_tokens"]
            cost = usage.get("cost_usd", usage.get("cost"))
            if cost is not None:
                safe_usage["cost_usd"] = cost
            result = {
                "completed": True,
                "completed_ns": time.time_ns(),
                "id": getattr(response, "id", None),
                "model": getattr(response, "model", None),
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
                "content": content,
                "source": source,
                "source_sha256": B.source_hash(source),
                "parse_status": parse_status,
                "source_status": B.source_status(source),
                "usage": safe_usage,
                "wall_s": time.monotonic() - started_at,
                "attempt": attempt_id,
            }
            persist(directory / "response.json", result)
            persist(
                directory / f"attempt_{attempt_id}.json",
                {
                    "status": "completed",
                    "id": result["id"],
                    "wall_s": result["wall_s"],
                    "logs": _safe(captured.getvalue())[:16000],
                },
            )
            print(
                json.dumps(
                    {
                        "phase": self.phase,
                        "outer": outer,
                        "arm": arm,
                        "slot": slot,
                        "status": result["source_status"],
                        "tokens": safe_usage.get("total_tokens"),
                    }
                ),
                flush=True,
            )
            return result
        raise RuntimeError("unreachable proposal state")

    def generate(self, outer: int, arm: str) -> None:
        """Generate A1 independently or let production PrioritySearch drive all A2 updates."""
        directory = self.root / str(outer) / arm
        if (directory / "generation_complete.json").exists():
            if len(list(directory.glob("slot_*/response.json"))) != self.slots:
                raise RuntimeError("completed generation has missing slots")
            return
        self.panel(B.SEED_SOURCE, outer, "train")
        if arm == "A1":
            for slot in range(self.slots):
                result = self.proposal(outer, arm, slot, B.SEED_SOURCE)
                self.panel(result["source"], outer, "train")
        elif arm == "A2":
            owner = self
            counter = [0]

            class SlotOptimizer(Optimizer):
                """Translate one production Trace update into one retained proposal slot."""

                def _step(self, *args: Any, **kwargs: Any) -> dict[Any, str]:
                    """Consume actual propagated feedback and update the trainable source."""
                    if len(self.parameters) != 1 or not self.parameters[0].feedback:
                        raise RuntimeError(
                            "A2 requires one source parameter with Trace feedback"
                        )
                    parent = self.parameters[0].data
                    result = owner.proposal(outer, "A2", counter[0], parent)
                    owner.event(
                        {
                            "event": "trace_update",
                            "outer": outer,
                            "slot": counter[0],
                            "parent_sha256": B.source_hash(parent),
                            "source_sha256": result["source_sha256"],
                            "propagated_feedback_present": True,
                        }
                    )
                    counter[0] += 1
                    return {self.parameters[0]: result["source"]}

            context_id = B.digest([str(self.root), self.phase, outer])
            _TRAIN_CONTEXTS[context_id] = {
                "owner": self,
                "outer": outer,
                "optimizer": SlotOptimizer,
            }
            reference = "recursive_opt.evaluator.exp15_training@1"
            control.register_evaluator(reference, _training_evaluator)
            control.register_engine(
                "exp15_trace_v1",
                control.EngineRegistryEntry(
                    run=_trace_engine,
                    capabilities=frozenset(
                        {"scalar", "weighted", "pareto", "rich_trace", "trace_module"}
                    ),
                ),
            )
            raw = optimizer_spec(
                B.SEED_SOURCE, seed=outer, budget=self.budget, engine="trace"
            )
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
            raw["engine"]["config"] = {
                "optimizer": "EXP15SlotOptimizer",
                "trainer": "PrioritySearch",
                "optimizer_kwargs": {},
                "iterations": self.slots + 1,
                "num_candidates": 1,
                "validation_gate": False,
                "trainer_kwargs": {
                    "num_threads": 1,
                    "num_proposals": 1,
                    "test_frequency": None,
                    "log_frequency": 1000,
                    "validate_exploration_candidates": False,
                    "use_best_candidate_to_explore": True,
                    "long_term_memory_size": None,
                    "score_function": "mean",
                },
            }
            raw["engine"]["name"] = "exp15_trace_v1"
            log = io.StringIO()
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = control.execute_plan(control.compile_plan(raw))[0]
            if counter[0] != self.slots:
                raise RuntimeError(
                    "production Trace proposal count differs from the allocation"
                )
            if result.status == "error" or not result.valid:
                raise RuntimeError(
                    "production Trace search failed: " + str(result.error)
                )
            persist(
                directory / "trace.json",
                {
                    "plan": raw,
                    "result": result.to_dict(),
                    "logs": _safe(log.getvalue()),
                },
            )
        else:
            raise ValueError("unknown generative arm")
        persist(
            directory / "generation_complete.json",
            {"slots": self.slots, "completed_ns": time.time_ns()},
        )

    def pool(self, outer: int, arm: str) -> list[dict[str, Any]]:
        """Include the unchanged seed and every response slot in final selection."""
        directory = self.root / str(outer) / arm
        if not (directory / "generation_complete.json").exists():
            raise RuntimeError("selection requires completed generation")
        candidates = [
            {
                "index": -1,
                "source": B.SEED_SOURCE,
                "source_sha256": B.MANIFEST["seed_sha256"],
            }
        ]
        for slot in range(self.slots):
            result = read(directory / f"slot_{slot:02d}" / "response.json")
            candidates.append(
                {
                    "index": slot,
                    "source": result["source"],
                    "source_sha256": result["source_sha256"],
                }
            )
        return candidates

    def select(self, outer: int) -> None:
        """Evaluate the same validation allocation after both arms finish generation."""
        target = self.root / str(outer) / "selection.json"
        if target.exists():
            return
        pools = {arm: self.pool(outer, arm) for arm in ("A1", "A2")}
        selected: dict[str, Any] = {"selected_ns": time.time_ns()}
        for arm, candidates in pools.items():
            for candidate in candidates:
                source = candidate["source"]
                training = self.panel(source, outer, "train")
                validation = self.panel(source, outer, "validation")
                eligible = all(r["valid"] for r in [*training, *validation])
                candidate.update(
                    {
                        "eligible": eligible,
                        "train": training,
                        "validation": validation,
                        "validation_auc": (
                            B.aggregate(validation, "auc") if eligible else None
                        ),
                    }
                )
            eligible = [c for c in candidates if c["eligible"]]
            if not eligible or not candidates[0]["eligible"]:
                raise RuntimeError("trusted seed failed selection eligibility")
            best = min(eligible, key=lambda c: (c["validation_auc"], c["index"]))
            persist(self.root / str(outer) / arm / "pool.json", candidates)
            selected[arm] = {
                k: best[k]
                for k in ("index", "source", "source_sha256", "validation_auc")
            }
        persist(target, selected)

    def freeze_selections(self) -> None:
        """Freeze every seed's selections and the representative before any holdout."""
        target = self.root / "selections_frozen.json"
        if target.exists():
            frozen = read(target)
            for outer, expected in frozen["selection_hashes"].items():
                if B.digest(read(self.root / outer / "selection.json")) != expected:
                    raise RuntimeError("frozen selection was modified")
            return
        if not all(
            (self.root / str(s) / "selection.json").exists() for s in self.seeds
        ):
            raise RuntimeError("every selection must be complete before holdout")
        selections = {
            str(s): read(self.root / str(s) / "selection.json") for s in self.seeds
        }
        representative = min(
            self.seeds,
            key=lambda s: (
                selections[str(s)]["A2"]["validation_auc"],
                self.seeds.index(s),
            ),
        )
        persist(
            target,
            {
                "frozen_ns": time.time_ns(),
                "selection_hashes": {s: B.digest(v) for s, v in selections.items()},
                "representative_outer_seed": representative,
                "representative": selections[str(representative)]["A2"],
            },
        )

    def holdout(self) -> None:
        """Evaluate all deployment policies only after the global selection barrier."""
        if not (self.root / "selections_frozen.json").exists():
            raise RuntimeError("holdout requires frozen selections")
        self.freeze_selections()
        for outer in self.seeds:
            path = self.root / str(outer) / "holdout.json"
            if path.exists():
                continue
            selection = read(self.root / str(outer) / "selection.json")
            rows = {"opened_ns": time.time_ns()}
            for arm in ("A0", "A1", "A2"):
                source = B.SEED_SOURCE if arm == "A0" else selection[arm]["source"]
                rows[arm] = self.panel(source, outer, "holdout", deployment=True)
            persist(path, rows)

    def analyze(self) -> dict[str, Any]:
        """Recompute all paired scientific values exclusively from complete retained rows."""
        rows = []
        for outer in self.seeds:
            path = self.root / str(outer) / "holdout.json"
            if not path.exists():
                raise RuntimeError("analysis cannot omit missing outer seeds")
            raw = read(path)
            if (
                raw["opened_ns"]
                <= read(self.root / "selections_frozen.json")["frozen_ns"]
            ):
                raise RuntimeError("holdout predates the selection freeze")
            row: dict[str, Any] = {"outer_seed": outer}
            for arm in ("A0", "A1", "A2"):
                if len(raw[arm]) != 12 or any(
                    not t["valid"] or t["objective_calls"] != self.budget
                    for t in raw[arm]
                ):
                    raise RuntimeError("incomplete deployment trajectory")
                row[arm] = {
                    "auc": B.aggregate(raw[arm], "auc"),
                    "final_regret": B.aggregate(raw[arm], "final_regret"),
                    "target_attainment": B.aggregate(raw[arm], "attained"),
                    "capped_target_evaluations": B.aggregate(
                        raw[arm], "capped_target_evaluations"
                    ),
                    "fallback_trajectories": sum(r["fallback_used"] for r in raw[arm]),
                    "candidate_valid_trajectories": sum(
                        r["candidate_valid"] for r in raw[arm]
                    ),
                }
            rows.append(row)
        arms = {
            arm: {
                "mean_auc": statistics.mean(r[arm]["auc"] for r in rows),
                "median_auc": statistics.median(r[arm]["auc"] for r in rows),
            }
            for arm in ("A0", "A1", "A2")
        }
        contrasts = {
            f"A2-{arm}": paired([r["A2"]["auc"] - r[arm]["auc"] for r in rows])
            for arm in ("A0", "A1")
        }
        return {
            "experiment": "EXP-15",
            "phase": self.phase,
            "per_seed": rows,
            "arms": arms,
            "contrasts": contrasts,
            "accounting": self.audit(),
            "selections": read(self.root / "selections_frozen.json"),
        }

    def audit(self) -> dict[str, Any]:
        """Count all retained slots, failures and cache execution costs without dropping rows."""
        proposals = [
            read(p) for p in sorted(self.root.glob("*/A*/slot_*/response.json"))
        ]
        expected = len(self.seeds) * 2 * self.slots
        if len(proposals) != expected:
            raise RuntimeError("proposal-slot accounting is incomplete")
        events = (
            [
                json.loads(line)
                for line in (self.root / "events.jsonl").read_text().splitlines()
            ]
            if (self.root / "events.jsonl").exists()
            else []
        )
        cache = [read(p)["result"] for p in (self.root / "cache").glob("*.json")]
        pools = [
            read(self.root / str(s) / arm / "pool.json")
            for s in self.seeds
            for arm in ("A1", "A2")
        ]
        usage = {
            key: sum(p["usage"].get(key, 0) for p in proposals)
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "total_tokens",
                "cost_usd",
            )
        }
        return {
            "proposal_slots": expected,
            "completed_responses": len(proposals),
            "transport_attempts": len(
                list(self.root.glob("*/A*/slot_*/attempt_*.json"))
            ),
            "usage": usage,
            "source_screen_invalid": sum(
                p["source_status"] != "valid" for p in proposals
            ),
            "ineligible_generated_candidates": sum(
                not c["eligible"] for pool in pools for c in pool if c["index"] >= 0
            ),
            "allocated_search_trajectories_per_arm": len(self.seeds)
            * (self.slots + 1)
            * 12,
            "allocated_search_objective_calls_per_arm": len(self.seeds)
            * (self.slots + 1)
            * 12
            * self.budget,
            "allocated_holdout_objective_calls_per_arm": len(self.seeds)
            * 12
            * self.budget,
            "actual_shared_objective_calls": sum(r["objective_calls"] for r in cache),
            "unused_unique_objective_allocations": sum(
                r["unused_objective_allocation"] for r in cache
            ),
            "subprocess_executions": sum(r["subprocess_executions"] for r in cache),
            "evaluation_requests": sum(e["event"] == "evaluation" for e in events),
            "cache_hits": sum(e.get("cache_hit", False) for e in events),
            "trace_updates": sum(e["event"] == "trace_update" for e in events),
            "normalization_reference_evaluations": 24
            * B.MANIFEST["normalization_reference_size"],
        }


def main() -> None:
    """Run pilot/confirmation stages or recompute analysis without new model calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "analyze", "preflight"])
    parser.add_argument("--phase", choices=["pilot", "confirmation"], default="pilot")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
        print("Confirmatory preflight passed")
        return
    root = args.root or B.ROOT / "exp15" / (
        "pilot_01" if args.phase == "pilot" else "raw"
    )
    client = None
    if args.command == "run":
        _load_key()
        client = make_live_llm(
            model="openrouter/" + B.MANIFEST["model"]["model"],
            cache=False,
            max_retries=1,
            empty_response_retries=0,
            request_timeout_s=300,
            allow_env_overrides=False,
            budget_resource=None,
        )
    exp = Experiment(root, args.phase, client=client)
    if args.command == "run":
        for index, outer in enumerate(exp.seeds):
            for arm in (("A1", "A2") if index % 2 == 0 else ("A2", "A1")):
                print(f"Starting {args.phase} outer={outer} arm={arm}", flush=True)
                exp.generate(outer, arm)
            exp.select(outer)
        exp.freeze_selections()
        exp.holdout()
    result = exp.analyze()
    persist(root / "results.json", result)
    print(
        json.dumps(
            {
                "arms": result["arms"],
                "contrasts": result["contrasts"],
                "accounting": result["accounting"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
