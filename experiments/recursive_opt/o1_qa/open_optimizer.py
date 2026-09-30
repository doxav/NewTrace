"""EXP22: bounded open optimizer artifacts on the existing Trace execution path."""

from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import hashlib
import json
import multiprocessing
import os
import random
import statistics
import subprocess
import sys
import textwrap
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from opto.optimizers.optoprime_v2 import OptoPrimeV2
from opto.trace.nodes import GRAPH
from opto.trainer.objectives import EvaluationResult

from . import (
    axes as X,
    campaign as C,
    campaign_analysis as A,
    meta as M,
    prepare,
    task,
)

ROOT = prepare.ROOT / "experiments/recursive_opt/_shared/o1_learning/exp22"
ARMS = ["B", "P", "C", "PC", "I-PC"]
EPISODES = [
    "pilot0",
    "pilot1",
    "discovery0",
    "discovery1",
    "selection",
    "test0",
    "test1",
    "test2",
    "test3",
    "test4",
    "test5",
]
COUNTS = {
    "diagnostic": 24,
    **{
        f"{episode}_{split}": count
        for episode in EPISODES
        for split, count in [("train", 36), ("validation", 24), ("measure", 48)]
    },
}
DIAGNOSTICS = ["initial", "code", "prompt", "both", "oracle"]
BASE_META = {
    "update_instruction": (
        "Improve exact-match on NEW HotpotQA questions by updating the retrieval code "
        "and reader instruction. Only four of ten documents reach the reader. "
        "Use TRAIN evidence to distinguish missing supporting documents, reader "
        "reasoning errors, answer formatting, and regressions after previous edits. "
        "Fix the responsible component and preserve successful behavior. Do not "
        "memorize questions or answers. Return complete executable rank(question, "
        "passages) code when changing it, with no imports or I/O. It must return a "
        "permutation of indices. Instructions must be general, not copied examples. "
        "The reader model, four-document limit and evaluation metric cannot change."
    ),
    "selector_source": (
        "def select_evidence(events, limit):\n"
        "    chosen = []\n"
        '    for kind in ["regression", "retrieval", "reader", "repaired", "correct"]:\n'
        "        for event in reversed(events):\n"
        '            if event["kind"] == kind and event["id"] not in chosen:\n'
        '                chosen.append(event["id"])\n'
        "                break\n"
        "    for event in reversed(events):\n"
        '        if event["id"] not in chosen:\n'
        '            chosen.append(event["id"])\n'
        "    return chosen[:limit]\n"
    ),
}
STRONG_CODE = """def rank(question, passages):
    def tokens(text):
        return set(''.join(c.lower() if c.isalnum() else ' ' for c in text).split())
    stop = {'the', 'a', 'an', 'of', 'in', 'and', 'is', 'was', 'what', 'which', 'who', 'to', 'are'}
    q = tokens(question) - stop
    docs = [tokens(p['title'] + ' ' + ' '.join(p['sentences'])) - stop for p in passages]
    titles = [tokens(p['title']) - stop for p in passages]
    def score(i):
        rarity = sum(1.0 / sum(w in d for d in docs) for w in q & docs[i])
        return rarity + 2 * len(q & titles[i])
    order = sorted(range(len(passages)), key=lambda i: (-score(i), i))
    lead = order[0]
    links = [i for i in order[1:] if titles[i] and titles[i] <= docs[lead]]
    return [lead] + links + [i for i in order[1:] if i not in links]
"""
STRONG_PROMPT = (
    "Answer using the supplied documents. Identify the entities asked about, connect "
    "the supporting facts, and perform the requested comparison when needed. Give "
    "only the shortest complete answer: a name, place, date, number, or yes/no. "
    "Use the spelling in the documents. Do not include explanations or uncertainty "
    "phrases in the final answer. Output exactly one line: FINAL: <answer>."
)
MODULE = "exp22.module.hotpot@1"
EVALUATOR = "exp22.evaluator.hotpot@1"
EVENTS: list[dict[str, Any]] = []
CURRENT_BATCH: list[str] = []
SELECTOR_FAILURES: list[dict[str, Any]] = []
EVIDENCE_CALLS = 0
META_CONTEXT: dict[str, Any] = {}
META_EVENTS: list[dict[str, Any]] = []
META_EVALUATOR = "exp22.evaluator.meta@1"
META_INSTRUCTION = (
    "Improve the OPTIMIZER of a retrieval-and-reader program, not the answers to questions. "
    "Each evaluation runs actual six-update lower learning on two separate development episodes. "
    "Maximize average exact-match over validation-selected prefixes 0..6, not just final accuracy. "
    "The lower program can change Python rank(question, passages) and the small-reader instruction; "
    "it always supplies four of ten documents to Qwen. The lower optimizer is DeepSeek. "
    "update_instruction is the free instruction actually sent to that lower optimizer. "
    "selector_source must define exactly select_evidence(events, limit) returning at most limit "
    "distinct event ID strings in presentation order. limit=5. Events are prior TRAIN observations "
    "with id, row_key, kind (retrieval, reader, regression, repaired, correct, invalid), question, "
    "accuracy, previous_accuracy, answer, expected, support_recall, support_titles, selected_documents, "
    "format_valid and slot. IDs are local; do not memorize questions or answers. "
    "Select useful, nonredundant evidence to improve credit assignment and preserve successes. "
    "No imports, I/O, private attributes or extra LLM calls. Supported Python builtins: "
    "abs all any bool dict enumerate float int len list max min range reversed round set sorted str sum tuple zip. "
    "Never change the evaluator, budgets, models or fixed lower settings. "
    "Only enabled variables may change. Return full source or full instruction, not patches. "
    "Instruction <=4000 characters; selector source <=8000. Invalid responses consume slots. "
    "No final TEST information is available. Costs of all descendant runs are accounted separately."
)


def previous_identities() -> set[str]:
    """Exclude all prior EXP20/21 panels, not just their training subsets."""
    return set().union(
        *(
            X.identity_keys(row)
            for panels_ in (prepare.panels(), X.panels())
            for rows in panels_.values()
            for row in rows
        )
    )


def panels() -> dict[str, list[dict[str, Any]]]:
    """Deterministically allocate independent balanced episodes before observing outcomes."""
    excluded = previous_identities()
    rows = [row for row in prepare.load_rows() if not X.identity_keys(row) & excluded]
    return task.split_rows(rows, COUNTS, seed=22001)


def selector_wrapper(source: str) -> str:
    """Embed exact source in the existing restricted runner; sentinel preserves short output."""
    if not isinstance(source, str) or not 1 <= len(source) <= 8000:
        raise task.InvalidPolicy("selector source must contain 1..8000 characters")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise task.InvalidPolicy("selector syntax error") from error
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise task.InvalidPolicy("define exactly select_evidence(events, limit)")
    fn = tree.body[0]
    if (
        fn.name != "select_evidence"
        or [a.arg for a in fn.args.args] != ["events", "limit"]
        or fn.args.posonlyargs
        or fn.args.kwonlyargs
        or fn.args.vararg
        or fn.args.kwarg
        or fn.args.defaults
        or fn.decorator_list
    ):
        raise task.InvalidPolicy("required signature: select_evidence(events, limit)")
    wrapper = (
        "def rank(question, passages):\n" + textwrap.indent(source, "    ") + "\n"
        "    events = passages[:-1]\n"
        '    ids = [event["id"] for event in events]\n'
        "    selected = select_evidence(events, int(question))\n"
        "    if len(selected) > int(question):\n"
        "        return []\n"
        "    indices = [ids.index(value) for value in selected]\n"
        "    return indices + [len(ids)] + [i for i in range(len(ids)) if i not in indices]\n"
    )
    task.validate_source(wrapper)
    return wrapper


def validate_meta(value: Mapping[str, Any]) -> None:
    """Validate only the two declared open surfaces; never coerce categorical substitutes."""
    if not isinstance(value, Mapping) or set(value) != set(BASE_META):
        raise ValueError(
            "meta artifact requires update_instruction and selector_source"
        )
    if (
        not isinstance(value["update_instruction"], str)
        or not value["update_instruction"].strip()
        or len(value["update_instruction"]) > 4000
    ):
        raise ValueError("update_instruction must contain 1..4000 nonempty characters")
    selector_wrapper(value["selector_source"])


def select_evidence(
    source: str, events: list[dict[str, Any]], limit: int, *, timeout_s: float = 2.0
) -> list[str]:
    """Execute exact selector source without credentials and validate its ID subset."""
    if type(limit) is not int or not 1 <= limit <= 5:
        raise ValueError("evidence limit must be an integer in 1..5")
    ids = [e["id"] for e in events]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("event IDs must be unique strings")
    result = task.run_ranker(
        selector_wrapper(source),
        {"question": str(limit), "passages": events + [{"sentinel": True}]},
        timeout_s=timeout_s,
    )
    return [ids[i] for i in result[: result.index(len(ids))]]


def diagnostic_input(
    name: str, row: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create frozen admissible interventions and a clearly privileged four-document oracle."""
    if name not in DIAGNOSTICS:
        raise ValueError("unknown diagnostic")
    artifact, data = dict(task.INITIAL), copy.deepcopy(row)
    if name in ("code", "both"):
        artifact["ranker_source"] = STRONG_CODE
    if name in ("prompt", "both", "oracle"):
        artifact["answer_instruction"] = STRONG_PROMPT
    if name == "oracle":
        required = {title for title, _ in row["supporting_facts"]}
        data["context"] = sorted(data["context"], key=lambda p: p[0] not in required)
        artifact["ranker_source"] = (
            "def rank(question, passages):\n    return list(range(len(passages)))\n"
        )
    return artifact, data


def paired(a: list[float], b: list[float]) -> dict[str, Any]:
    """Report every paired episode and a fixed descriptive bootstrap, including losses."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("at least two complete paired episodes required")
    deltas = [x - y for x, y in zip(a, b)]
    rng = random.Random(22099)
    draws = sorted(
        statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000)
    )
    return {
        "deltas": deltas,
        "mean_delta": statistics.mean(deltas),
        "median_delta": statistics.median(deltas),
        "bootstrap_95": [draws[249], draws[9749]],
        "interpretation": (
            "positive signal"
            if draws[249] > 0
            else "negative signal" if draws[9749] < 0 else "inconclusive"
        ),
    }


def source_hashes() -> dict[str, str]:
    """Seal dirty-tree source contents, including all reused research and production paths."""
    paths = sorted(
        set(
            prepare.SOURCE_PATHS
            + [
                str(p.relative_to(prepare.ROOT))
                for directory in (
                    "experiments/recursive_opt/o1_qa",
                    "opto/features/recursive_opt",
                    "opto/trainer",
                    "opto/optimizers",
                    "opto/trace",
                )
                for p in (prepare.ROOT / directory).rglob("*.py")
            ]
        )
    )
    return {
        p: hashlib.sha256((prepare.ROOT / p).read_bytes()).hexdigest() for p in paths
    }


def register_pilot() -> dict[str, Any]:
    """Register hypotheses, independent panels and fixed diagnostic interventions before calls."""
    path = ROOT / "pilot_protocol.json"
    if path.exists():
        return X.read(path)
    data = panels()
    value = {
        "experiment": "EXP22-PILOT",
        "utc": datetime.now(timezone.utc).isoformat(),
        "starting_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "initial_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ),
        "staged_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--cached", "--binary"])
        ).hexdigest(),
        "source_sha256": source_hashes(),
        "pilot_implementation_source": Path(__file__).read_text(),
        "profiles": A.profiles(),
        "splits": {
            k: {"ids": [r["id"] for r in v], "hash": task.digest(v)}
            for k, v in data.items()
        },
        "baseline_meta": BASE_META,
        "diagnostic_code": STRONG_CODE,
        "diagnostic_prompt": STRONG_PROMPT,
        "variants": DIAGNOSTICS,
        "primary_hypothesis": "PC-B learning-curve gain on fresh episodes",
        "mechanism_prior": "PC greatest flexibility; P lower search complexity; no claim PC wins",
        "diagnostic_gates": {
            "seed_below": 0.85,
            "oracle_at_least": 0.4,
            "admissible_improvement_correct_answers": 2,
        },
        "workers": 8,
        "completed_diagnostic_reader_allocations": 120,
        "scope": "PILOT ONLY, no confirmation access or learned-arm winner selection",
    }
    C.retain(path, value)
    return value


def diagnostic_job(args: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    """Evaluate one diagnostic question in an isolated process with immutable request receipts."""
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

    _load_key()
    name, row = args
    artifact, data = diagnostic_input(name, row)
    root = ROOT / "pilot_diagnostic"
    result = A.evaluate_panel(root, 22003, artifact, [data], name)
    return {"variant": name, "id": row["id"], **result}


def diagnostic() -> dict[str, Any]:
    """Run all paired diagnostic slots without choosing an optimization winner."""
    protocol = register_pilot()
    data = panels()["diagnostic"]
    jobs = [
        (name, row)
        for i, row in enumerate(data)
        for name in DIAGNOSTICS[i % 5 :] + DIAGNOSTICS[: i % 5]
    ]
    rows = []
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for offset in range(0, len(jobs), 8):
            futures = [
                pool.submit(diagnostic_job, job) for job in jobs[offset : offset + 8]
            ]
            rows.extend(f.result() for f in futures)
            print(
                json.dumps({"diagnostic_complete": len(rows), "total": len(jobs)}),
                flush=True,
            )
    scores = {
        v: statistics.mean(r["accuracy"] for r in rows if r["variant"] == v)
        for v in DIAGNOSTICS
    }
    gates = protocol["diagnostic_gates"]
    checks = {
        "not_saturated": scores["initial"] < gates["seed_below"],
        "reader_usable": scores["oracle"] >= gates["oracle_at_least"],
        "accessible_headroom": (
            max(scores[v] for v in ("code", "prompt", "both")) - scores["initial"]
        )
        * len(data)
        >= gates["admissible_improvement_correct_answers"] - 1e-9,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "REVIEW_BEFORE_META",
        "EM": scores,
        "checks": checks,
        "rows": rows,
        "scope": "PILOT, not O1 gain evidence",
    }
    C.retain(ROOT / "pilot_diagnostic" / "summary.json", result)
    return result


class EvidencePolicy(C.RecordedPolicy):
    """Existing task pipeline with two open surfaces and immutable retrieval limits."""

    def __init__(self, artifact: Mapping[str, Any], reader: Any) -> None:
        """Deactivate fixed native nodes before optimizer parameter discovery."""
        super().__init__(artifact, reader)
        self.top_k.trainable = False
        self.bridge_expansion.trainable = False

    def forward(self, example: Any) -> Any:
        """Reject any attempt to change the resource constraint, including malformed updates."""
        if self.top_k.data != 4 or self.bridge_expansion.data is not False:
            raise ValueError("EXP22 requires exactly four documents and no expansion")
        return super().forward(example)


def evaluate(output: Any, row: Any, context: Mapping[str, Any]) -> Any:
    """Archive actual TRAIN observations and transitions without changing the task metric."""
    result = C.evaluate(output, row, context)
    data = getattr(output, "data", output)
    row = getattr(row, "data", row)
    key = task.digest(row["question"])
    accuracy = result.metrics.get("accuracy")
    previous = next((e for e in reversed(EVENTS) if e["row_key"] == key), None)
    observation = X.read(
        C.ACTIVE.directory / "evaluations" / f"{C.ACTIVE.evaluations - 1:05d}.json"
    )
    selected = data.get("retrieval", {}).get("documents", [])
    required = {title for title, _ in row["supporting_facts"]}
    recall = len(required & {p["title"] for p in selected}) / len(required)
    kind = (
        "invalid"
        if not result.valid
        else (
            "regression"
            if previous and previous["accuracy"] == 1 and accuracy == 0
            else (
                "repaired"
                if previous and previous["accuracy"] == 0 and accuracy == 1
                else (
                    "correct"
                    if accuracy == 1
                    else "retrieval" if recall < 1 else "reader"
                )
            )
        )
    )
    event = {
        "id": f"e{len(EVENTS):04d}",
        "row_key": key,
        "kind": kind,
        "question": row["question"][:800],
        "accuracy": accuracy,
        "previous_accuracy": previous["accuracy"] if previous else None,
        "artifact_hash": observation["artifact_hash"],
        "previous_artifact_hash": previous["artifact_hash"] if previous else None,
        "answer": data.get("answer"),
        "expected": row["answer"],
        "support_recall": recall,
        "support_titles": sorted(required),
        "selected_documents": [
            {"title": p["title"], "text": "".join(p["sentences"])[:800]}
            for p in selected
        ],
        "format_valid": data.get("format_valid"),
        "slot": C.ACTIVE.calls,
    }
    EVENTS.append(event)
    C.retain(C.ACTIVE.directory / "train_events" / f"{len(EVENTS):04d}.json", event)
    return result


class EvidenceOptimizer(OptoPrimeV2):
    """Narrow native prompt adapter: execute learned code, keep the production update path."""

    def __init__(
        self, parameters: list[Any], *, selector_source: str, **kwargs: Any
    ) -> None:
        """Bind one frozen O1 artifact to this O0 learning chain."""
        self.selector_source = selector_source
        super().__init__(parameters, **kwargs)

    def construct_prompt(
        self, summary: Any, mask: Any = None, *args: Any, **kwargs: Any
    ) -> tuple[str, str]:
        """Prevent native graph fields or optimizer memory from bypassing the evidence budget."""
        global EVIDENCE_CALLS
        original = X.stable_request(
            {
                "messages": [
                    {"role": "user", "content": str(self.problem_instance(summary))}
                ]
            }
        )
        failure = None
        try:
            selected = select_evidence(self.selector_source, EVENTS, 5)
        except (ValueError, TypeError, KeyError) as error:
            failure = {"slot": C.ACTIVE.calls, "kind": type(error).__name__}
            SELECTOR_FAILURES.append(failure)
            selected = select_evidence(BASE_META["selector_source"], EVENTS, 5)
        by_id = {e["id"]: e for e in EVENTS}
        batch = [
            next((e for e in reversed(EVENTS) if e["row_key"] == key), None)
            for key in CURRENT_BATCH
        ]
        scores = [
            e["accuracy"] for e in batch if e is not None and e["accuracy"] is not None
        ]
        feedback = {
            "batch_count": len(batch),
            "valid_count": len(scores),
            "batch_accuracy": statistics.mean(scores) if scores else None,
            "evidence": [by_id[i] for i in selected],
        }
        projected = copy.copy(summary)
        projected.graph = [
            (1, "rank(question, passages) -> permutation -> first four documents"),
            (
                2,
                "make_prompt(answer_instruction, question, documents) -> Qwen -> answer",
            ),
            (3, "exact_match(answer, expected) -> TRAIN feedback"),
        ]
        projected.documentation = {"contract": task.DESCRIPTIONS["ranker_source"]}
        projected.inputs, projected.others, projected.output = {}, {}, {}
        projected.user_feedback = json.dumps(feedback, ensure_ascii=False)
        system, user = super().construct_prompt(projected, mask, *args, **kwargs)
        C.retain(
            C.ACTIVE.directory / "evidence" / f"{EVIDENCE_CALLS:03d}.json",
            {
                "raw_native_trace": original,
                "selected_ids": selected,
                "feedback": feedback,
                "selector_hash": hashlib.sha256(
                    self.selector_source.encode()
                ).hexdigest(),
                "fallback": failure,
                "actual_prompt": X.stable_request(
                    {"messages": [{"role": "user", "content": user}]}
                ),
            },
        )
        EVIDENCE_CALLS += 1
        return system, user


class EvidenceTrainer(C.CampaignTrainer):
    """Production PrioritySearch retains its scoring and proposal accounting."""

    def propose(self, samples: Any, verbose: bool = False, **kwargs: Any) -> Any:
        """Expose only current TRAIN batch identities to the fixed aggregate renderer."""
        global CURRENT_BATCH
        CURRENT_BATCH = [
            task.digest(getattr(r.x, "data", r.x)["question"])
            for batch in samples
            for r in batch
        ]
        return super().propose(samples, verbose=verbose, **kwargs)


def register() -> None:
    """Register experiment adapters without changing frozen EXP20/21 implementations."""
    X.register()
    if (ROOT / "resume_rejections.json").exists():
        C.live_response = credit_resume_response
    import opto.trainer.algorithms as algorithms
    import opto.optimizers as optimizers

    algorithms.EXP22Trainer = EvidenceTrainer
    optimizers.EXP22Evidence = EvidenceOptimizer
    if MODULE not in task.S._MODULE_REGISTRY:
        task.S.register_module(
            MODULE,
            task.S.ModuleRegistryEntry(
                build=lambda level, resources: EvidencePolicy(
                    level["module"]["config"], resources["llm_clients"].get("forward")
                ),
                snapshot=task.snapshot,
                restore=task.restore,
                validate_artifact=C.validate_record,
                validate_config=task.validate_artifact,
                capabilities=frozenset(
                    {"multi_component", "json_snapshot", "trace_module"}
                ),
            ),
        )
        task.S.register_evaluator(EVALUATOR, evaluate)


def lower_spec(
    policy: dict[str, Any], train: list[dict[str, Any]], seed: int, calls: int
) -> dict[str, Any]:
    """Instantiate the learned instruction/code through explicit control-plane fields."""
    validate_meta(policy)
    spec = task.specification(train, seed=seed, curriculum=False, calls=calls)
    spec["budget"].update(eval_llm_calls=2000, evaluator_runs=2000)
    level = spec["levels"][0]
    level["surface"]["targets"] = ["ranker_source", "answer_instruction"]
    level["module"]["ref"] = MODULE
    level["objective"]["evaluator_ref"] = EVALUATOR
    level["objective"][
        "intent"
    ] = "Improve exact-match using ranker code and reader instruction, with exactly four documents."
    engine = level["engine"]["config"]
    engine["optimizer"] = "EXP22Evidence"
    engine["trainer"] = "EXP22Trainer"
    engine["optimizer_kwargs"] = {
        "objective": policy["update_instruction"],
        "selector_source": policy["selector_source"],
        "log": False,
        "memory_size": 0,
        "initial_var_char_limit": 16000,
    }
    return spec


def lower_fit(
    root: Path,
    policy: dict[str, Any],
    train: list[dict[str, Any]],
    *,
    seed: int,
    calls: int = 6,
    factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run/replay one real lower learning chain under an immutable meta artifact."""
    global EVENTS, CURRENT_BATCH, SELECTOR_FAILURES, EVIDENCE_CALLS
    register()
    name = task.digest(policy)
    folder = root / "chains" / str(seed) / name
    spec = lower_spec(policy, train, seed, calls)
    C.retain(
        folder / "identity.json",
        {
            "policy": policy,
            "train_hash": task.digest(train),
            "seed": seed,
            "calls": calls,
        },
    )
    C.retain(folder / "spec.json", spec)
    if (folder / "result.json").exists():
        return X.read(folder / "result.json")
    GRAPH.clear()
    EVENTS, CURRENT_BATCH, SELECTOR_FAILURES, EVIDENCE_CALLS = [], [], [], 0
    C.ACTIVE = X.Journal(folder, root / "reader_cache" / str(seed), seed, calls)
    C.retain(folder / "artifacts" / f"{task.digest(task.INITIAL)}.json", task.INITIAL)

    def clients(profile: Any, role: str) -> Any:
        """Preserve the existing provider recording, cache and guarded accounting path."""
        return C.ACTIVE.client(
            profile, role, factory(profile, role) if factory else None
        )

    production = task.S.run_spec(spec, resources={"llm_factory": clients}).to_dict()
    terminal_invalid = production.get("error") == "invalid candidate evaluation"
    if (
        C.ACTIVE.failure
        or C.ACTIVE.calls != calls
        or (production.get("error") and not terminal_invalid)
    ):
        incomplete = {
            "calls": C.ACTIVE.calls,
            "failure": C.ACTIVE.failure,
            "production": production,
        }
        C.retain(
            folder / "incomplete_attempts" / f"{task.digest(incomplete)}.json",
            incomplete,
        )
        raise RuntimeError(
            "incomplete EXP22 lower run; preserve receipts and resume only unfinished work"
        )
    report = {
        "seed": seed,
        "name": name,
        "optimizer_responses": C.ACTIVE.calls,
        "batch_events": C.ACTIVE.batches,
        "reader_logical_calls": C.ACTIVE.reader_calls,
        "selector_failures": len(SELECTOR_FAILURES),
        "fallbacks": SELECTOR_FAILURES,
        "train_observations": len(EVENTS),
        "production": production,
    }
    C.retain(folder / "result.json", report)
    return report


def episode_seed(episode: str) -> int:
    """Keep per-episode reader cache and local RNG identities stable and distinct."""
    return 22011 + 2 * EPISODES.index(episode)


def episode_job(args: dict[str, Any]) -> dict[str, Any]:
    """Lock a shared episode/policy before requests, avoiding duplicate baseline generations."""
    root = ROOT / "lower" / args["episode"]
    lock_path = root / "locks" / task.digest(args["policy"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _episode_run(args)


def _episode_run(args: dict[str, Any]) -> dict[str, Any]:
    """Execute one lower episode in a fresh process; measurement is explicitly gated."""
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

    _load_key()
    episode, policy, mode = args["episode"], args["policy"], args.get("mode", "all")
    seed = episode_seed(episode)
    root = ROOT / "lower" / episode
    name = task.digest(policy)
    data = panels()
    if episode.startswith("test"):
        gate = X.read(ROOT / "meta_selection.json")
        if name not in {task.digest(p) for p in gate["policies"].values()}:
            raise ValueError("unselected meta artifact cannot run TEST episodes")
        if mode == "all":
            raise ValueError(
                "TEST measurement requires a separate global prefix freeze"
            )
        if mode == "measure" and not (ROOT / "all_prefixes_frozen.json").exists():
            raise ValueError(
                "global prefix freeze required before any TEST measurement"
            )
    result_path = root / "reports" / f"{name}.json"
    if result_path.exists():
        return X.read(result_path)
    if mode != "measure":
        fitted = lower_fit(root, policy, data[f"{episode}_train"], seed=seed, calls=6)
        selected = X.select(
            root, seed, name, data[f"{episode}_train"], data[f"{episode}_validation"], 6
        )
        if mode == "fit":
            return {
                "episode": episode,
                "name": name,
                "selected": selected,
                "fit": fitted,
            }
    else:
        fitted = X.read(root / "chains" / str(seed) / name / "result.json")
        selected = X.read(root / "selection" / str(seed) / f"{name}.json")
    measured = X.measure(
        root,
        seed,
        selected,
        data[f"{episode}_measure"],
        "test" if episode.startswith("test") else "meta_measure",
    )
    folder = root / "chains" / str(seed) / name
    events = [X.read(p) for p in sorted((folder / "train_events").glob("*.json"))]
    evidence = [X.read(p) for p in sorted((folder / "evidence").glob("*.json"))]
    report = {
        "episode": episode,
        "policy_hash": name,
        "valid": fitted["selector_failures"] == 0,
        "selector_failures": fitted["selector_failures"],
        **measured,
        "train_error_counts": dict(Counter(e["kind"] for e in events)),
        "unique_feedback_questions": len(
            {e["row_key"] for r in evidence for e in r["feedback"]["evidence"]}
        ),
        "lower_optimizer_responses": fitted["optimizer_responses"],
        "final_program_hash": selected["prefixes"][-1],
    }
    C.retain(result_path, report)
    return report


def parallel_episodes(
    jobs: list[dict[str, Any]], workers: int = 8
) -> list[dict[str, Any]]:
    """Bound active subprocesses; finish only the current wave on a provider failure."""
    results = []
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        for offset in range(0, len(jobs), workers):
            futures = [
                pool.submit(episode_job, j) for j in jobs[offset : offset + workers]
            ]
            results.extend(f.result() for f in futures)
    return results


def assess_meta(policy: dict[str, Any], episodes: list[str]) -> dict[str, Any]:
    """Score a genuine optimizer by actual descendant learning, never by a proxy guess."""
    if not episodes or any(e.startswith("test") for e in episodes):
        raise ValueError("discovery/selection cannot read confirmation episodes")
    reports = parallel_episodes(
        [{"episode": e, "policy": policy} for e in episodes], workers=2
    )
    valid = all(r["valid"] for r in reports)
    return {
        "valid": valid,
        "utility": statistics.mean(r["primary"] for r in reports) if valid else None,
        "episodes": {
            r["episode"]: {
                k: r[k]
                for k in (
                    "curve",
                    "final",
                    "selector_failures",
                    "train_error_counts",
                    "unique_feedback_questions",
                )
            }
            for r in reports
        },
    }


def meta_evaluate(
    output: Any, example: Any, context: Mapping[str, Any]
) -> EvaluationResult:
    """Preserve every evaluated O1 candidate, using content-keyed complete child evaluations."""
    policy = dict(getattr(output, "data", output)["components"])
    key = task.digest(policy)
    error = None
    try:
        validate_meta(policy)
        arm = META_CONTEXT["arm"]
        if arm == "P" and policy["selector_source"] != BASE_META["selector_source"]:
            raise ValueError("P may not change code")
        if (
            arm == "C"
            and policy["update_instruction"] != BASE_META["update_instruction"]
        ):
            raise ValueError("C may not change instruction")
    except (ValueError, TypeError) as exc:
        error = type(exc).__name__
    if error:
        assessed = {"valid": False, "utility": None, "error_kind": error}
    else:
        cache = (
            M.ACTIVE.directory.parent
            / "fitness"
            / f'{task.digest([policy,META_CONTEXT["episodes"]])}.json'
        )
        if cache.exists():
            assessed = X.read(cache)
        else:
            assessed = assess_meta(policy, META_CONTEXT["episodes"])
            C.retain(cache, assessed)
    event = {"slot": M.ACTIVE.calls, "policy": policy, "hash": key, **assessed}
    META_EVENTS.append(event)
    C.retain(M.ACTIVE.directory / "evaluations" / f"{len(META_EVENTS):04d}.json", event)
    return EvaluationResult(
        assessed["valid"],
        "ok" if assessed["valid"] else "invalid_optimizer",
        {"score": assessed["utility"]} if assessed["valid"] else {},
        json.dumps({k: v for k, v in event.items() if k != "policy"}),
        artifacts=event,
    )


class MetaOptimizer(OptoPrimeV2):
    """Expose bounded evaluated history only in recursive arms; independent prompts stay fresh."""

    def construct_prompt(
        self, summary: Any, mask: Any = None, *args: Any, **kwargs: Any
    ) -> tuple[str, str]:
        """Separate invariant instructions from five distinct training-episode observations."""
        projected = copy.copy(summary)
        projected.graph = [
            (
                1,
                "meta_artifact -> run production O0 learning -> validation-selected prefix curve on META-TRAIN",
            )
        ]
        projected.documentation = {}
        projected.inputs, projected.others, projected.output = {}, {}, {}
        independent = META_CONTEXT["arm"] == "I-PC"
        unique = {e["hash"]: e for e in META_EVENTS}
        feedback = (
            {
                "independent": True,
                "instruction": "Generate one optimizer from the unchanged seed, without prior candidate outcomes.",
            }
            if independent
            else {
                "evaluated_candidates": [
                    {k: v for k, v in e.items() if k != "policy"}
                    for e in list(unique.values())[-5:]
                ]
            }
        )
        projected.user_feedback = json.dumps(feedback, ensure_ascii=False)
        system, user = super().construct_prompt(projected, mask, *args, **kwargs)
        C.retain(
            M.ACTIVE.directory / "generation" / f"{M.ACTIVE.calls:02d}.json",
            {
                "parent_variables": json.loads(json.dumps(summary.variables)),
                "feedback": feedback,
                "request": X.stable_request(
                    {"messages": [{"role": "user", "content": user}]}
                ),
            },
        )
        return system, user


def meta_run(
    root: Path,
    arm: str,
    *,
    calls: int,
    episodes: list[str],
    factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Use the production engine for iterative search or fresh single-proposal independent runs."""
    global META_CONTEXT, META_EVENTS
    if (
        arm not in ARMS[1:]
        or calls < 1
        or any(
            e not in ["pilot0", "pilot1", "discovery0", "discovery1"] for e in episodes
        )
    ):
        raise ValueError("invalid meta arm, call budget or discovery episodes")
    if arm == "I-PC" and calls > 1:
        reports = [
            meta_run(
                root / f"{i+1:02d}", arm, calls=1, episodes=episodes, factory=factory
            )
            for i in range(calls)
        ]
        events = [
            {**e, "slot": i + 1 if e["slot"] else 0}
            for i, r in enumerate(reports)
            for e in r["events"]
        ]
        result = {
            "arm": arm,
            "completed_responses": sum(r["completed_responses"] for r in reports),
            "events": events,
        }
        C.retain(root / "result.json", result)
        return result
    register()
    import opto.trainer.algorithms as algorithms
    import opto.optimizers as optimizers

    algorithms.EXP22MetaTrainer = M.MetaTrainer
    optimizers.EXP22Meta = MetaOptimizer
    if META_EVALUATOR not in task.S._EVALUATOR_REGISTRY:
        task.S.register_evaluator(META_EVALUATOR, meta_evaluate)
    spec = M.specification(1, "OptoPrimeV2", 0, calls)
    level = spec["levels"][0]
    level["module"]["config"]["components"] = dict(BASE_META)
    level["surface"]["targets"] = (
        ["update_instruction"]
        if arm == "P"
        else ["selector_source"] if arm == "C" else list(BASE_META)
    )
    level["objective"]["evaluator_ref"] = META_EVALUATOR
    level["objective"]["trace_config"].update(max_chars=24000, max_nodes=24)
    level["engine"]["config"].update(optimizer="EXP22Meta", trainer="EXP22MetaTrainer")
    level["engine"]["config"]["optimizer_kwargs"].update(
        objective=META_INSTRUCTION, initial_var_char_limit=16000
    )
    level["datasets"] = {
        "train": [{"suite": "EXP22 meta training episodes"}],
        "validation": [],
        "holdout": [],
    }
    spec["runtime"]["seed"] = 22031
    C.retain(root / "identity.json", {"arm": arm, "episodes": episodes, "calls": calls})
    C.retain(root / "spec.json", spec)
    if (root / "result.json").exists():
        return X.read(root / "result.json")
    GRAPH.clear()
    META_CONTEXT = {"arm": arm, "episodes": episodes}
    META_EVENTS = []
    M.ACTIVE = X.Journal(root, root / "unused_reader_cache", 22031, calls)

    def clients(profile: Any, role: str) -> Any:
        """Use metered canonical role clients; child calls run in independent processes."""
        return M.ACTIVE.client(
            profile, role, factory(profile, role) if factory else None
        )

    production = task.S.run_spec(spec, resources={"llm_factory": clients}).to_dict()
    if M.ACTIVE.calls != calls or production.get("error"):
        C.retain(
            root / "incomplete_production" / f"{task.digest(production)}.json",
            production,
        )
        raise RuntimeError("incomplete EXP22 meta run; preserve evidence and diagnose")
    production_path = root / "production.json"
    if production_path.exists() and X.read(production_path).get("error"):
        production_path = root / "production_completed.json"
    C.retain(production_path, production)
    result = {
        "arm": arm,
        "completed_responses": M.ACTIVE.calls,
        "events": META_EVENTS,
        "production_record": production_path.name,
    }
    C.retain(root / "result.json", result)
    return result


def freeze_meta_selection(root: Path, policies: dict[str, Any]) -> None:
    """Lock all five meta artifacts before any confirmatory lower learning starts."""
    if set(policies) != set(ARMS):
        raise ValueError("complete B/P/C/PC/I-PC selections required")
    for policy in policies.values():
        validate_meta(policy)
    value = {
        "policies": policies,
        "hashes": {a: task.digest(p) for a, p in policies.items()},
    }
    C.retain(root / "meta_selection.json", value)
    if not (root / "meta_selection_time.json").exists():
        C.retain(
            root / "meta_selection_time.json",
            {"utc": datetime.now(timezone.utc).isoformat(), "hash": task.digest(value)},
        )


def meta_job(args: tuple[str, str, int, list[str]]) -> dict[str, Any]:
    """Isolate simultaneous O1 graphs and nested experiments in separate processes."""
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

    _load_key()
    root, arm, calls, episodes = args
    return meta_run(Path(root), arm, calls=calls, episodes=episodes)


def run_search(stage: str) -> dict[str, Any]:
    """Run all search arms without early stopping on poor candidates or scores."""
    pilot_mode = stage == "pilot_meta"
    calls = 1 if pilot_mode else 6
    episodes = ["pilot0", "pilot1"] if pilot_mode else ["discovery0", "discovery1"]
    root = ROOT / stage
    protocol = {
        "stage": stage,
        "calls_per_arm": calls,
        "arms": ARMS[1:],
        "episodes": episodes,
        "source_sha256": source_hashes(),
        "implementation_source": Path(__file__).read_text(),
        "ordering": "four concurrent arm processes; two child episode workers each; sequential dependencies within iterative arms",
        "baseline_meta": BASE_META,
        "instruction": META_INSTRUCTION,
    }
    path = root / "protocol.json"
    if path.exists() and X.read(path) != protocol:
        old = X.read(path)
        amendment = X.read(ROOT / "engineering_resume_amendment.json")
        keys = set(protocol) - {"source_sha256", "implementation_source"}
        if (
            any(old[k] != protocol[k] for k in keys)
            or amendment["before_source_sha256"] != old["source_sha256"]
            or amendment["after_source_sha256"] != protocol["source_sha256"]
        ):
            raise ValueError("unregistered pilot protocol drift")
        C.retain(root / "resume_protocol.json", protocol)
    else:
        C.retain(path, protocol)
    if not pilot_mode:
        verify_freeze()
    if X.read(ROOT / "pilot_diagnostic" / "summary.json")["status"] != "PASS":
        raise ValueError("diagnostic gates must pass before nested live search")
    jobs = [(str(root / arm), arm, calls, episodes) for arm in ARMS[1:]]
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        reports = [f.result() for f in [pool.submit(meta_job, j) for j in jobs]]
    result = {
        "stage": stage,
        "completed_responses": sum(r["completed_responses"] for r in reports),
        "arms": {
            r["arm"]: {
                "responses": r["completed_responses"],
                "eligible_distinct_artifacts": len(
                    {e["hash"] for e in r["events"] if e["valid"]}
                ),
            }
            for r in reports
        },
        "status": "COMPLETE",
    }
    C.retain(root / "summary.json", result)
    return result


def freeze() -> dict[str, Any]:
    """Freeze scientific decisions after the engineering pilot and before discovery."""
    if X.read(ROOT / "pilot_meta" / "summary.json")["status"] != "COMPLETE":
        raise ValueError("complete nested engineering pilot required")
    data = panels()
    value = {
        "experiment": "EXP22",
        "status": "FROZEN",
        "source_sha256": source_hashes(),
        "implementation_source": Path(__file__).read_text(),
        "profiles": A.profiles(),
        "baseline_meta": BASE_META,
        "meta_instruction": META_INSTRUCTION,
        "splits": {
            k: {"ids": [r["id"] for r in v], "hash": task.digest(v)}
            for k, v in data.items()
        },
        "arms": ARMS,
        "outer_discovery_replicates": 1,
        "meta_proposals": 6,
        "lower_proposals": 6,
        "test_episodes": [e for e in EPISODES if e.startswith("test")],
        "seed_map": {e: episode_seed(e) for e in EPISODES},
        "selection": "all eligible candidates evaluated on meta validation; maximum curve, baseline then earliest proposal tie",
        "primary": "mean exact-match of validation-selected O0 prefixes 0..6, higher better",
        "primary_contrast": "PC-B",
        "secondary_contrasts": ["P-B", "C-B", "PC-I-PC"],
        "practical_gain": 0.05,
        "target_EM": 0.7,
        "uncertainty": {
            "method": "paired episode percentile bootstrap",
            "draws": 10000,
            "seed": 22099,
            "indices": [249, 9749],
            "scope": "conditional on selected O1 artifacts, not replicated O1 discovery",
        },
        "evidence": {
            "max_examples": 5,
            "document_chars_each": 800,
            "question_chars": 800,
            "history": "all observations within six-update episode",
            "projection": "native traces archived; prompt data replaced by selected events plus current batch aggregate",
        },
        "transport": "existing role profiles; completed empty outputs consume slots; ambiguous requests require reconciliation",
        "workers": {"meta_arms": 4, "children_each": 2, "confirmation": 8},
        "cache": "exact model/profile/request/episode seed for reader; immutable meta policy + episode + frozen source for child fitness",
        "fallback": "baseline selector on deployment failure; any selector failure makes discovery/selection candidate ineligible",
        "proposal_protocol_sha256": hashlib.sha256(
            (prepare.ROOT / "experiments/recursive_opt/_shared/o1_learning/EXP22.md").read_bytes()
        ).hexdigest(),
        "python": sys.version,
    }
    C.retain(ROOT / "manifest.json", value)
    if not (ROOT / "freeze_time.json").exists():
        C.retain(
            ROOT / "freeze_time.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "manifest_hash": task.digest(value),
            },
        )
    return {"status": "FROZEN", "manifest_hash": task.digest(value)}


def verify_freeze() -> dict[str, Any]:
    """Refuse scientific code drift before issuing or resuming confirmation work."""
    manifest = X.read(ROOT / "manifest.json")
    if manifest["source_sha256"] != source_hashes():
        raise ValueError("frozen scientific source drift")
    return manifest


def select_meta() -> dict[str, Any]:
    """Select on untouched meta validation only after every discovery arm finishes."""
    verify_freeze()
    if X.read(ROOT / "discovery" / "summary.json")["completed_responses"] != 24:
        raise ValueError("all discovery slots required before meta validation")
    menus = {}
    for arm in ARMS[1:]:
        events = X.read(ROOT / "discovery" / arm / "result.json")["events"]
        candidates = {task.digest(BASE_META): {"policy": BASE_META, "slot": 0}}
        for event in events:
            if event["valid"] and event["hash"] not in candidates:
                candidates[event["hash"]] = {
                    "policy": event["policy"],
                    "slot": event["slot"],
                }
        menus[arm] = candidates
    policies = {
        h: item["policy"] for menu in menus.values() for h, item in menu.items()
    }
    reports = parallel_episodes(
        [{"episode": "selection", "policy": p} for p in policies.values()]
    )
    by_hash = {r["policy_hash"]: r for r in reports}
    chosen = {"B": BASE_META}
    for arm, menu in menus.items():
        eligible = [h for h in menu if by_hash[h]["valid"]]
        best = max(eligible, key=lambda h: (by_hash[h]["primary"], -menu[h]["slot"], h))
        chosen[arm] = menu[best]["policy"]
    C.retain(ROOT / "meta_validation.json", {"menus": menus, "reports": by_hash})
    freeze_meta_selection(ROOT, chosen)
    return {
        "status": "SELECTED",
        "hashes": {a: task.digest(p) for a, p in chosen.items()},
    }


def confirm() -> dict[str, Any]:
    """Finish every lower selection before any final measurement, retaining all arms."""
    verify_freeze()
    selected = X.read(ROOT / "meta_selection.json")["policies"]
    unique = {task.digest(p): p for p in selected.values()}
    jobs = [
        {"episode": episode, "policy": policy, "mode": "fit"}
        for episode in EPISODES
        if episode.startswith("test")
        for policy in unique.values()
    ]
    parallel_episodes(jobs)
    selections = {}
    for episode in [e for e in EPISODES if e.startswith("test")]:
        root = ROOT / "lower" / episode
        seed = episode_seed(episode)
        selections[episode] = {
            h: X.read(root / "selection" / str(seed) / f"{h}.json") for h in unique
        }
    C.retain(ROOT / "all_prefixes_frozen.json", selections)
    if not (ROOT / "all_prefixes_time.json").exists():
        C.retain(
            ROOT / "all_prefixes_time.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "hash": task.digest(selections),
            },
        )
    for episode, records in selections.items():
        seed = episode_seed(episode)
        X.freeze_test(
            ROOT / "lower" / episode,
            list(unique),
            [seed],
            {str(seed): records},
            calls=6,
        )
    parallel_episodes([{**j, "mode": "measure"} for j in jobs])
    return analyze()


def accounting() -> dict[str, Any]:
    """Recompute usage and attempted/completed slots from immutable raw receipts."""
    totals = {}
    for role in ["optimizer", "reader"]:
        paths = [
            p
            for p in ROOT.rglob("request.json")
            if ("optimizer" in p.parts) == (role == "optimizer")
        ]
        usage, finish = Counter(), Counter()
        complete = failures = pending = 0
        for path in paths:
            request = X.read(path)
            if request["request_hash"] != task.digest(request["request"]):
                raise ValueError("request hash mismatch")
            response = path.with_name("response.json")
            if response.exists():
                raw = X.read(response)
                if raw["request_hash"] != request["request_hash"]:
                    raise ValueError("response hash mismatch")
                complete += 1
                usage.update(
                    {
                        k: v
                        for k, v in raw["usage"].items()
                        if isinstance(v, (int, float))
                    }
                )
                for choice in raw["response"].get("choices", []):
                    finish[choice.get("finish_reason", "unknown")] += 1
            elif path.with_name("failure.json").exists():
                failures += 1
            else:
                pending += 1
        totals[role] = {
            "completed": complete,
            "failed_requests": failures,
            "unresolved": pending,
            "request_receipts": len(paths),
            "usage": dict(usage),
            "finish_reasons": dict(finish),
        }
    return totals


def analyze() -> dict[str, Any]:
    """Require full coverage and recompute primary contrasts from all retained curves."""
    selected = X.read(ROOT / "meta_selection.json")["policies"]
    episodes = [e for e in EPISODES if e.startswith("test")]
    reports = {}
    for arm in ARMS:
        reports[arm] = []
        for episode in episodes:
            path = (
                ROOT
                / "lower"
                / episode
                / "reports"
                / f"{task.digest(selected[arm])}.json"
            )
            if not path.exists():
                raise ValueError("complete five-arm/six-episode results required")
            report = X.read(path)
            if (
                len(report["curve"]) != 7
                or statistics.mean(report["curve"]) != report["primary"]
            ):
                raise ValueError("inconsistent lower curve")
            reports[arm].append(report)
    values = {a: [r["primary"] for r in rows] for a, rows in reports.items()}
    result = {
        "experiment": "EXP22",
        "status": "COMPLETE",
        "episodes": episodes,
        "arms": {
            a: {
                "primary": values[a],
                "mean": statistics.mean(values[a]),
                "median": statistics.median(values[a]),
                "final": [r["final"] for r in reports[a]],
                "selector_failures": sum(r["selector_failures"] for r in reports[a]),
            }
            for a in ARMS
        },
        "contrasts": {
            f"{a}-{b}": paired(values[a], values[b])
            for a, b in [("P", "B"), ("C", "B"), ("PC", "B"), ("PC", "I-PC")]
        },
        "selected_hashes": {a: task.digest(p) for a, p in selected.items()},
        "accounting": accounting(),
        "scope": "selected-optimizer transfer; one O1 discovery per arm; no general superiority, extra-depth or amortization claim",
    }
    C.retain(ROOT / "results.json", result)
    return result


def credit_status() -> dict[str, Any]:
    """Read financial availability privately; persist only safe numeric fields."""
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

    _load_key()
    value = {"utc": datetime.now(timezone.utc).isoformat()}
    for endpoint, fields in [
        ("credits", ("total_credits", "total_usage")),
        ("key", ("limit", "usage", "limit_remaining")),
    ]:
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/" + endpoint,
            headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)["data"]
        value[endpoint] = {k: data.get(k) for k in fields}
    path = ROOT / "credit" / f'{len(list((ROOT/"credit").glob("*.json"))):03d}.json'
    C.retain(path, value)
    return value


def credit_resume_response(
    folder: Path, profile: Mapping[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    """Reissue only hash-listed explicit key-limit rejections; never replace a completed reply."""
    from . import resume

    path = ROOT / "resume_rejections.json"
    allowed = (
        {r["path"]: r["request_hash"] for r in X.read(path)["requests"]}
        if path.exists()
        else {}
    )
    current = folder
    while (current / "0/request.json").exists():
        attempt = current / "0"
        if (attempt / "response.json").exists():
            break
        try:
            relative = str((attempt / "request.json").relative_to(ROOT))
        except ValueError:
            break
        if relative not in allowed:
            break
        raw = X.read(attempt / "request.json")
        if (
            allowed[relative] != task.digest(request)
            or raw["request_hash"] != allowed[relative]
        ):
            raise ValueError("credit resume request hash mismatch")
        if X.read(attempt / "failure.json")["http_status_codes"] != [403]:
            raise ValueError(
                "credit resume requires the recorded explicit 403 rejection"
            )
        current = current / "credit_resume"
    return resume.ORIGINAL(current, profile, request)


def main() -> None:
    """Explicit resumable stages; no implicit confirmatory or live execution on import."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "diagnostic",
            "pilot",
            "freeze",
            "discover",
            "select",
            "confirm",
            "analyze",
            "credit",
            "accounting",
            "all",
        ],
    )
    args = parser.parse_args()
    if args.stage in ("pilot", "all") and (ROOT / "resume_rejections.json").exists():
        credit = credit_status()
        if (
            credit["key"]["limit_remaining"] is not None
            and credit["key"]["limit_remaining"] <= 0
            or credit["credits"]["total_credits"] <= credit["credits"]["total_usage"]
        ):
            raise RuntimeError(
                "provider key/account depleted; no live resume request issued"
            )
    stages = {
        "prepare": register_pilot,
        "diagnostic": diagnostic,
        "pilot": lambda: run_search("pilot_meta"),
        "freeze": freeze,
        "discover": lambda: run_search("discovery"),
        "select": select_meta,
        "confirm": confirm,
        "analyze": analyze,
        "credit": credit_status,
        "accounting": accounting,
    }
    if args.stage == "all":
        result = {}
        for stage in ["diagnostic", "pilot", "freeze", "discover", "select", "confirm"]:
            result = stages[stage]()
            print(
                json.dumps({"completed_stage": stage, "status": result.get("status")}),
                flush=True,
            )
    else:
        result = stages[args.stage]()
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if args.stage in ("credit", "accounting")
                or k in ("experiment", "status", "EM", "checks", "arms", "contrasts")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
