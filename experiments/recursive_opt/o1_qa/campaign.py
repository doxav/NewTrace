"""EXP20 recording adapters; all learning uses the production Control Plane trainer."""

from __future__ import annotations

import fcntl
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from litellm import ModelResponse

from opto.features.recursive_opt.budget import BudgetExceeded
from opto.features.recursive_opt.runmode import make_live_llm
from opto.trace import ExecutionError, bundle
from opto.trainer.algorithms.priority_search import PrioritySearch

from . import pilot, task

MODULE_REF = "o1_qa.module.recorded_hotpot@1"
EVALUATOR_REF = "o1_qa.evaluator.recorded_hotpot@1"
ACTIVE: Journal | None = None


def canonical_request(request: dict[str, Any]) -> dict[str, Any]:
    """Stabilize unordered Trace XML node listings and ephemeral object addresses."""

    def clean(text: str) -> str:
        """Preserve every named node and value while fixing listing order."""
        text = re.sub(r"(<[^>\n]+ object at )0x[0-9a-f]+(>)", r"\1<address>\2", text)
        identities = dict(re.findall(r'"id": "(\d+)", "label": "([^"]+)"', text))
        for identity, label in identities.items():
            text = text.replace(f'"{identity}"', json.dumps(label))
        for tag in ("node", "variable"):
            block = rf'<{tag} name="[^\"]+"[^>]*>.*?</{tag}>'
            group = rf"(?:{block}\s*)+"
            text = re.sub(
                group,
                lambda m: "\n\n".join(sorted(re.findall(block, m.group(0), re.S)))
                + "\n\n",
                text,
                flags=re.S,
            )
        return text

    return {
        **request,
        "messages": [
            {**m, "content": clean(m["content"])} for m in request["messages"]
        ],
    }


def retain(path: Path, value: Any) -> None:
    """Write once, or verify an identical deterministic replay."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"replay mismatch: {path.name}")
    else:
        pilot.write_once(path, value)


def live_response(
    folder: Path, profile: Mapping[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    """Use bounded production transport; only explicit HTTP rejection permits a resumed attempt."""
    attempt = 0
    while (folder / str(attempt) / "request.json").exists():
        current = folder / str(attempt)
        if (current / "response.json").exists():
            return pilot.request_once(current, request, lambda **_: None)
        failure = current / "failure.json"
        if (
            not failure.exists()
            or json.loads(failure.read_text())["remote_completion"]
            != "explicit_rejection"
        ):
            raise RuntimeError("ambiguous remote request requires reconciliation")
        attempt += 1
    # Four waves maximum, each with the profile's original bounded transport policy.
    if attempt >= 4:
        raise RuntimeError("provider rejection retry waves exhausted")
    current = folder / str(attempt)
    events: list[dict[str, Any]] = []

    def record(event: str, kind: str | None) -> None:
        """Persist transport labels without exception payloads or credentials."""
        item = {"event": event, "failure_kind": kind}
        pilot.write_once(current / f"transport_{len(events)}.json", item)
        events.append(item)

    client = make_live_llm(
        profile["resolved_model"],
        cache=False,
        max_retries=profile["transport_max_attempts"],
        base_delay=profile["transport_base_delay_s"],
        request_timeout_s=profile["request_timeout_s"],
        allow_env_overrides=False,
        empty_response_retries=0,
        retry_event_callback=record,
        budget_resource=None,
    )
    try:
        return pilot.request_once(current, request, client)
    except Exception:
        failure = current / "failure.json"
        if (
            failure.exists()
            and json.loads(failure.read_text())["remote_completion"]
            == "explicit_rejection"
            and attempt < 3
        ):
            time.sleep(4 * (2**attempt))
            return live_response(folder, profile, request)
        raise


class Journal:
    """Per-chain immutable events and a process-safe, per-seed shared response cache."""

    def __init__(self, directory: Path, cache: Path, seed: int, limit: int) -> None:
        """Bind identity without loading secrets or making requests."""
        self.directory, self.cache, self.seed, self.limit = (
            directory,
            cache,
            seed,
            limit,
        )
        self.calls = 0
        self.reader_calls = 0
        self.evaluations = 0
        self.batches: list[dict[str, Any]] = []
        self.failure: str | None = None
        self.phase = "train"

    def client(
        self,
        profile: Mapping[str, Any],
        role: str,
        provider: Callable[..., Any] | None = None,
    ) -> Any:
        """Create a role wrapper; recorded provider metadata never enters cache identity."""
        journal = self
        config = task.S._thaw(profile)
        config.pop("profile", None)  # Role alias is not a provider/request setting.

        class Client:
            deterministic_trace = True

            def __call__(self, **request: Any) -> Any:
                """Replay immutable responses, with optimizer slots separate from reader cache hits."""
                if role == "optimizer":
                    request = canonical_request(request)
                    if journal.calls >= journal.limit:
                        raise RuntimeError("completed proposal budget exhausted")
                    index = journal.calls
                    folder = journal.directory / "optimizer" / f"{index + 1:02d}"
                    retain(
                        folder / "identity.json",
                        {"profile": config, "request_hash": task.digest(request)},
                    )
                    evidence = (
                        pilot.request_once(folder, request, provider)
                        if provider
                        else live_response(folder, config, request)
                    )
                    journal.calls += 1
                    return ModelResponse(**evidence["response"])
                key = task.digest(
                    {"seed": journal.seed, "profile": config, "request": request}
                )
                folder = journal.cache / key
                folder.mkdir(parents=True, exist_ok=True)
                with (folder / "lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    existed = any(folder.glob("**/response.json"))
                    try:
                        evidence = (
                            pilot.request_once(folder, request, provider)
                            if provider
                            else live_response(folder, config, request)
                        )
                    except Exception:
                        journal.failure = "reader_transport_or_cache_error"
                        raise
                path = journal.directory / "reader" / f"{journal.reader_calls:05d}.json"
                # A resumed event keeps its original paid/cache attribution.
                event = {
                    "request_hash": task.digest(request),
                    "cache_key": key,
                    "phase": journal.phase,
                    "optimizer_responses_before": journal.calls,
                }
                if path.exists():
                    old = json.loads(path.read_text())
                    if any(old[k] != v for k, v in event.items()):
                        raise ValueError("reader replay mismatch")
                else:
                    pilot.write_once(path, {**event, "cache_hit": existed})
                journal.reader_calls += 1
                return ModelResponse(**evidence["response"])

            def __deepcopy__(self, memo: dict[int, Any]) -> Any:
                """Share accounting across production module and optimizer copies."""
                memo[id(self)] = self
                return self

        return Client()


@bundle()
def invalid_output(parameters: list[Any], kind: str) -> dict[str, Any]:
    """Keep invalid-policy feedback connected to trainable nodes without a numeric metric."""
    return {"invalid_policy": kind}


class RecordedPolicy(task.QAPolicy):
    """The same task policy, with typed invalidity and per-attempt artifact evidence."""

    def forward(self, example: Any) -> Any:
        """Record exact parameters even for rejected candidates and execution failures."""
        assert ACTIVE is not None
        row = getattr(example, "data", example)
        artifact = task.snapshot(self)
        key = task.digest(artifact)
        retain(ACTIVE.directory / "artifacts" / f"{key}.json", artifact)
        try:
            output = super().forward(example)
        except (ValueError, TypeError) as error:
            output = invalid_output(self.parameters(), type(error).__name__)
        except ExecutionError as error:
            output = invalid_output(self.parameters(), type(error).__name__)
        data = output.data
        valid = "invalid_policy" not in data
        record = {
            "artifact_hash": key,
            "row_id": row["id"],
            "slot": ACTIVE.calls,
            "valid": valid,
            "error": None if valid else data["invalid_policy"],
            "answer": data.get("answer"),
            "phase": ACTIVE.phase,
            "metrics": (
                task.evaluate(output, row, {"phase": "fit"}).metrics if valid else {}
            ),
        }
        retain(
            ACTIVE.directory / "evaluations" / f"{ACTIVE.evaluations:05d}.json", record
        )
        ACTIVE.evaluations += 1
        return output


def evaluate(output: Any, row: Any, context: Mapping[str, Any]) -> Any:
    """Expose typed invalidity to the canonical evaluator instead of inventing accuracy."""
    data = getattr(output, "data", output)
    if "invalid_policy" in data:
        return task.EvaluationResult(
            False,
            "invalid_policy",
            {},
            f"Invalid optimizer artifact: {data['invalid_policy']}",
        )
    return task.evaluate(output, row, context)


def validate_record(artifact: Mapping[str, Any]) -> None:
    """Validate snapshot encoding; execution still applies the full task validity contract."""
    if not isinstance(artifact, Mapping) or set(artifact) != set(task.INITIAL):
        raise ValueError("recorded artifact requires the four declared parameters")
    if any(type(value) not in (str, int, bool) for value in artifact.values()):
        raise ValueError("recorded parameters must be JSON primitive values")


class CampaignTrainer(PrioritySearch):
    """Production search with observation hooks and an equal replay allocation for control."""

    def sample(self, agents: Any, verbose: bool = False, **kwargs: Any) -> Any:
        """Reserve the same replay evaluations in both arms; control discards replay feedback."""
        if (
            self.train_sampler.loader.curriculum is None
            and self.train_sampler._prev_batch is not None
            and agents
        ):
            self.train_sampler.sample(
                agents, use_prev_batch=True, observe_curriculum=False
            )
        return super().sample(agents, verbose=verbose, **kwargs)

    def propose(self, samples: Any, verbose: bool = False, **kwargs: Any) -> Any:
        """Retain each feedback batch and cap completed responses including semantic retries."""
        assert ACTIVE is not None
        if ACTIVE.calls >= ACTIVE.limit:
            return []
        event = {
            "slot_before": ACTIVE.calls,
            "row_ids": [
                getattr(r.x, "data", r.x)["id"] for batch in samples for r in batch
            ],
            "curriculum_history": (
                list(self.train_sampler.loader.curriculum.history)
                if self.train_sampler.loader.curriculum
                else []
            ),
        }
        before = ACTIVE.calls
        try:
            result = super().propose(samples, verbose=verbose, **kwargs)
        except BudgetExceeded:
            if ACTIVE.calls != ACTIVE.limit:
                raise
            result = []
        except RuntimeError as error:
            if "no final textual content after 2 metered attempts" not in str(error):
                raise
            result = []
        event["slot_after"] = ACTIVE.calls
        event["candidate_hashes"] = []
        for candidate in result:
            artifact = task.snapshot(candidate.get_module().module)
            key = task.digest(artifact)
            retain(ACTIVE.directory / "artifacts" / f"{key}.json", artifact)
            event["candidate_hashes"].append(key)
        retain(ACTIVE.directory / "batches" / f"{len(ACTIVE.batches):02d}.json", event)
        ACTIVE.batches.append(event)
        if ACTIVE.calls == before:
            raise RuntimeError("production proposal did not consume a response")
        return result


def register() -> None:
    """Register file-recording adapters while retaining canonical task and trainer semantics."""
    task.register()
    import opto.trainer.algorithms as algorithms

    algorithms.EXP20CampaignTrainer = CampaignTrainer
    if MODULE_REF not in task.S._MODULE_REGISTRY:
        task.S.register_module(
            MODULE_REF,
            task.S.ModuleRegistryEntry(
                build=lambda level, resources: RecordedPolicy(
                    level["module"]["config"], resources["llm_clients"].get("forward")
                ),
                snapshot=task.snapshot,
                restore=task.restore,
                validate_artifact=validate_record,
                validate_config=task.validate_artifact,
                capabilities=frozenset(
                    {"multi_component", "json_snapshot", "trace_module"}
                ),
            ),
        )
        task.S.register_evaluator(EVALUATOR_REF, evaluate)


def fit(
    root: Path,
    train: list[dict[str, Any]],
    *,
    seed: int,
    arm: str,
    calls: int,
    factory: Callable[..., Any] | None = None,
    trace_mode: str = "internal",
) -> dict[str, Any]:
    """Run one entire production chain; resume replays completed responses without replacement."""
    global ACTIVE
    if arm not in {"standard", "curriculum"}:
        raise ValueError("unknown learning arm")
    folder = root / "chains" / str(seed) / arm
    if (folder / "result.json").exists():
        return json.loads((folder / "result.json").read_text())
    register()
    from opto.trace.nodes import GRAPH

    GRAPH.clear()
    ACTIVE = Journal(folder, root / "reader_cache" / str(seed), seed, calls)
    spec = task.specification(
        train, seed=seed, curriculum=arm == "curriculum", calls=calls
    )
    level = spec["levels"][0]
    level["module"]["ref"], level["objective"]["evaluator_ref"] = (
        MODULE_REF,
        EVALUATOR_REF,
    )
    level["objective"]["trace_config"]["mode"] = trace_mode
    retain(folder / "spec.json", spec)
    retain(folder / "artifacts" / f"{task.digest(task.INITIAL)}.json", task.INITIAL)

    def clients(profile: Any, role: str) -> Any:
        """Use the same frozen role settings through the existing guarded clients."""
        return ACTIVE.client(profile, role, factory(profile, role) if factory else None)

    production = task.S.run_spec(
        spec, resources={"llm_factory": clients, "trainer": "EXP20CampaignTrainer"}
    ).to_dict()
    if ACTIVE.failure or ACTIVE.calls != calls:
        raise RuntimeError(
            "incomplete production chain; preserve evidence and resume uncompleted work"
        )
    if production.get("error"):
        raise RuntimeError("production integration error; inspect before continuing")
    records = [
        json.loads(p.read_text())
        for p in sorted((folder / "evaluations").glob("*.json"))
    ]
    result = {
        "seed": seed,
        "arm": arm,
        "optimizer_responses": ACTIVE.calls,
        "actual_train_evaluations": len(records),
        "reader_logical_calls": ACTIVE.reader_calls,
        "candidate_count": len(list((folder / "artifacts").glob("*.json"))),
        "batch_events": ACTIVE.batches,
        "curriculum_events": production["metadata"]["curriculum_events"],
        "unique_questions_in_feedback": len(
            {i for e in ACTIVE.batches for i in e["row_ids"]}
        ),
        "production": production,
    }
    retain(folder / "result.json", result)
    return result
