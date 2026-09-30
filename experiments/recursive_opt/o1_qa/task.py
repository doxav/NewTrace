"""Mixed-surface HotpotQA adapter for Trace; no alternative search loop."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import re
import signal
import string
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from experiments.recursive_opt.multiobjective_reasoning.components import (
    _record_call,
    _response_text,
    _usage_dict,
    _provider_metadata,
)
from opto.features.recursive_opt import spec as S
from opto.trace import bundle, node
from opto.trace.modules import Module
from opto.trainer.objectives import EvaluationResult

MODULE_REF = "o1_qa.module.hotpot@1"
EVALUATOR_REF = "o1_qa.evaluator.hotpot@1"
INITIAL = {
    "ranker_source": (
        "def rank(question, passages):\n"
        "    tokens = set(''.join(c.lower() if c.isalnum() else ' ' for c in question).split())\n"
        "    def score(i):\n"
        "        text = passages[i]['title'] + ' ' + ' '.join(passages[i]['sentences'])\n"
        "        words = set(''.join(c.lower() if c.isalnum() else ' ' for c in text).split())\n"
        "        return len(tokens.intersection(words))\n"
        "    return sorted(range(len(passages)), key=lambda i: (-score(i), i))\n"
    ),
    "answer_instruction": (
        "Answer the question using the supplied documents. Connect relevant facts "
        "across documents when needed. Give a concise answer."
    ),
    "top_k": 4,
    "bridge_expansion": False,
}
DESCRIPTIONS = {
    "ranker_source": "Python rank(question, passages) returns a permutation of passage indices. No imports, I/O or private attributes. Passages have title and sentences.",
    "answer_instruction": "Instructions for the small reader model; this does not change its identity or decoding settings.",
    "top_k": "Integer 2..10: number of ranked documents supplied to the reader.",
    "bridge_expansion": "Boolean: append the first retrieved document to the query and rerank remaining documents once, using the same ranker.",
}


class InvalidPolicy(ValueError):
    """Typed source/execution failure, distinct from a wrong answer."""


def digest(value: Any) -> str:
    """Hash canonical JSON without process-dependent hash()."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def validate_source(source: str) -> None:
    """Reject forbidden operations before subprocess execution; not a security proof."""
    if not isinstance(source, str) or not 1 <= len(source) <= 12000:
        raise InvalidPolicy("ranker source must contain 1..12000 characters")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise InvalidPolicy("ranker syntax error") from error
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise InvalidPolicy("source must define exactly rank(question, passages)")
    fn = tree.body[0]
    if (
        fn.name != "rank"
        or [a.arg for a in fn.args.args] != ["question", "passages"]
        or fn.args.posonlyargs
        or fn.args.kwonlyargs
        or fn.args.vararg
        or fn.args.kwarg
        or fn.args.defaults
        or fn.decorator_list
    ):
        raise InvalidPolicy("required signature: rank(question, passages)")
    forbidden = (
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        ast.ClassDef,
        ast.AsyncFunctionDef,
        ast.With,
        ast.AsyncWith,
    )
    banned_names = {
        "open",
        "eval",
        "exec",
        "compile",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "input",
        "breakpoint",
        "help",
    }
    for item in ast.walk(tree):
        if (
            isinstance(item, forbidden)
            or isinstance(item, ast.Name)
            and (item.id.startswith("_") or item.id in banned_names)
            or isinstance(item, ast.Attribute)
            and item.attr.startswith("_")
        ):
            raise InvalidPolicy("ranker contains prohibited operations")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Enforce the four mixed types without string/int/bool coercion."""
    if not isinstance(artifact, Mapping) or set(artifact) != set(INITIAL):
        raise ValueError(
            "policy requires exactly ranker_source, answer_instruction, top_k, bridge_expansion"
        )
    validate_source(artifact["ranker_source"])
    text = artifact["answer_instruction"]
    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
        raise ValueError("answer_instruction must contain 1..4000 characters")
    if type(artifact["top_k"]) is not int or not 2 <= artifact["top_k"] <= 10:
        raise ValueError("top_k must be an integer from 2 to 10")
    if type(artifact["bridge_expansion"]) is not bool:
        raise TypeError("bridge_expansion must be boolean")


def public_input(row: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist question/documents; labels, IDs, types and supporting facts stay outside."""
    passages = [
        {"title": title, "sentences": list(sentences)}
        for title, sentences in row["context"]
    ]
    if (
        not isinstance(row["question"], str)
        or not row["question"].strip()
        or not 2 <= len(passages) <= 10
        or any(
            not isinstance(p["title"], str)
            or not p["sentences"]
            or not all(isinstance(s, str) for s in p["sentences"])
            for p in passages
        )
    ):
        raise ValueError("question and 2..10 nonempty text passages required")
    return {"question": row["question"], "passages": passages}


def _rank_worker() -> None:
    """Run in a fresh Python interpreter with no credential environment."""
    import builtins
    import json
    import resource
    from pathlib import Path

    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (65536,) * 2)
    names = "abs all any bool dict enumerate float int len list max min range reversed round set sorted str sum tuple zip"
    scope = {"__builtins__": {name: getattr(builtins, name) for name in names.split()}}
    exec(compile(Path("ranker.py").read_text(), "ranker.py", "exec"), scope)
    payload = json.loads(Path("input.json").read_text())
    result = scope["rank"](payload["question"], payload["passages"])
    Path("result.json").write_text(json.dumps(result, allow_nan=False))


def run_ranker(
    source: str, payload: Mapping[str, Any], *, timeout_s: float = 2.0
) -> list[int]:
    """Execute rank code with hard timeout and validate a complete permutation."""
    validate_source(source)
    if timeout_s <= 0 or set(payload) != {"question", "passages"}:
        raise ValueError("positive timeout and public-only ranker payload required")
    with tempfile.TemporaryDirectory(prefix="trace-qa-ranker-") as folder:
        root = Path(folder)
        (root / "ranker.py").write_text(source)
        (root / "input.json").write_text(json.dumps(payload))
        (root / "worker.py").write_text(
            inspect.getsource(_rank_worker) + "\n_rank_worker()\n"
        )
        with subprocess.Popen(
            [sys.executable, "-I", "-S", "worker.py"],
            cwd=root,
            env={"PATH": os.defpath, "LANG": "C.UTF-8"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        ) as process:
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired as error:
                raise InvalidPolicy("ranker timeout") from error
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        try:
            result = json.loads((root / "result.json").read_text())
        except (OSError, ValueError) as error:
            raise InvalidPolicy("ranker execution error") from error
    if (
        not isinstance(result, list)
        or any(type(i) is not int for i in result)
        or sorted(result) != list(range(len(payload["passages"])))
    ):
        raise InvalidPolicy("ranker must return a permutation of passage indices")
    return result


@bundle()
def rank_passages(source: str, payload: dict[str, Any]) -> list[int]:
    """Expose the actual ranked document indices as a semantic Trace node."""
    return run_ranker(source, payload)


@bundle()
def expand_query(
    payload: dict[str, Any], order: list[int], enabled: bool
) -> dict[str, Any]:
    """Expose the optional second-hop query without access to supporting labels."""
    query = payload["question"]
    if enabled:
        first = payload["passages"][order[0]]
        query += " " + first["title"] + " " + " ".join(first["sentences"])[:1600]
    return {**payload, "question": query}


@bundle()
def select_context(
    payload: dict[str, Any],
    first_order: list[int],
    second_order: list[int],
    expanded: dict[str, Any],
    top_k: int,
    expansion: bool,
) -> dict[str, Any]:
    """Retain the first hop and join the remaining ranked documents deterministically."""
    order = (
        first_order[:1] + [i for i in second_order if i != first_order[0]]
        if expansion
        else first_order
    )
    selected = order[:top_k]
    return {
        "initial_order": first_order,
        "selected": selected,
        "query_after_expansion": expanded["question"],
        "ranker_executions": 2 if expansion else 1,
        "documents": [payload["passages"][i] for i in selected],
    }


def retrieve(source: Any, payload: dict[str, Any], top_k: Any, expansion: Any) -> Any:
    """Compose traced operations while keeping actual code execution in the subprocess."""
    first = rank_passages(source, payload)
    expanded = expand_query(payload, first, expansion)
    second = (
        rank_passages(source, expanded)
        if getattr(expansion, "data", expansion)
        else first
    )
    return select_context(payload, first, second, expanded, top_k, expansion)


@bundle()
def make_prompt(
    instruction: str, payload: dict[str, Any], retrieval: dict[str, Any]
) -> str:
    """Construct one small-reader request from selected public documents only."""
    context = "\n\n".join(
        f"[{p['title']}]\n{''.join(p['sentences'])}" for p in retrieval["documents"]
    )
    return (
        f"{instruction}\n\nQuestion: {payload['question']}\n\nDocuments:\n{context}\n\n"
        "Return exactly one final line: FINAL: <short answer>. Do not append an explanation."
    )


@bundle()
def pack_answer(call: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, Any]:
    """Preserve response and retrieval trace; strict missing-format output remains visible."""
    lines = re.findall(r"(?m)^\s*FINAL:\s*(\S[^\n]*)$", call["content"])
    return {
        "answer": lines[-1].strip() if len(lines) == 1 else "",
        "format_valid": len(lines) == 1,
        "call": call,
        "retrieval": retrieval,
    }


class QAPolicy(Module):
    """Four trainable nodes with a fixed, single-call weak reader."""

    def __init__(self, artifact: Mapping[str, Any], reader: Any) -> None:
        """Create genuine code/string/int/bool parameter nodes."""
        validate_artifact(artifact)
        for name, value in artifact.items():
            setattr(
                self,
                name,
                node(value, name=name, trainable=True, description=DESCRIPTIONS[name]),
            )
        self.reader = reader

    def forward(self, example: Any) -> Any:
        """Execute the traced retrieval→prompt→reader→answer workflow."""
        validate_artifact(snapshot(self))
        payload = public_input(getattr(example, "data", example))
        retrieval = retrieve(
            self.ranker_source, payload, self.top_k, self.bridge_expansion
        )
        prompt = make_prompt(self.answer_instruction, payload, retrieval)
        if self.reader is None:
            raise RuntimeError("a forward-role reader is required")
        import time

        started = time.monotonic()
        response = self.reader(messages=[{"role": "user", "content": prompt.data}])
        call = _record_call(
            prompt,
            _response_text(response),
            _usage_dict(response),
            (
                0.0
                if getattr(self.reader, "deterministic_trace", False)
                else time.monotonic() - started
            ),
            _provider_metadata(response),
        )
        return pack_answer(call, retrieval)


def snapshot(module: Module) -> dict[str, Any]:
    """Preserve exact raw code and native parameter types."""
    return {name: getattr(module, name).data for name in INITIAL}


def restore(module: Module, artifact: Mapping[str, Any]) -> None:
    """Validate atomically before restoring all nodes."""
    validate_artifact(artifact)
    for name, value in artifact.items():
        getattr(module, name)._set(value)


def answer_metrics(prediction: str, expected: str) -> tuple[float, float]:
    """Hotpot answer EM and token F1, including the official yes/no/noanswer rule."""

    def normalize(text: str) -> str:
        """Lowercase, remove punctuation/articles and collapse whitespace."""
        text = "".join(c for c in text.lower() if c not in string.punctuation)
        return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())

    p, g = normalize(prediction), normalize(expected)
    em = float(p == g)
    if p != g and ({p, g} & {"yes", "no", "noanswer"}):
        return em, 0.0
    overlap = sum((Counter(p.split()) & Counter(g.split())).values())
    return em, 2 * overlap / (len(p.split()) + len(g.split())) if overlap else 0.0


def evaluate(output: Any, example: Any, context: Mapping[str, Any]) -> EvaluationResult:
    """Score the held-out answer; training-only diagnosis identifies missed documents."""
    data = getattr(output, "data", output)
    em, f1 = answer_metrics(data["answer"], example["answer"])
    selected = {p["title"] for p in data["retrieval"]["documents"]}
    gold = {title for title, _ in example["supporting_facts"]}
    recall = len(gold & selected) / len(gold)
    feedback = (
        f"answer_EM={em}; answer_F1={f1:.3f}; supporting_document_recall={recall:.3f}"
    )
    if context["phase"] == "fit":
        feedback += f"; expected={example['answer']!r}; observed={data['answer']!r}; missing_documents={sorted(gold - selected)!r}"
    return EvaluationResult(
        True,
        "ok",
        {
            "accuracy": em,
            "answer_f1": f1,
            "support_recall": recall,
            "format_valid": float(data["format_valid"]),
        },
        feedback,
    )


def register() -> None:
    """Register task adapters while retaining the production Trace engine."""
    if MODULE_REF not in S._MODULE_REGISTRY:
        S.register_module(
            MODULE_REF,
            S.ModuleRegistryEntry(
                build=lambda level, resources: QAPolicy(
                    level["module"]["config"], resources["llm_clients"].get("forward")
                ),
                snapshot=snapshot,
                restore=restore,
                validate_artifact=validate_artifact,
                validate_config=validate_artifact,
                capabilities=frozenset(
                    {"multi_component", "json_snapshot", "trace_module"}
                ),
            ),
        )
        S.register_evaluator(EVALUATOR_REF, evaluate)


def split_rows(
    rows: list[dict[str, Any]], counts: Mapping[str, int], *, seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Balanced hash-order splits, disjoint by ID, question and supporting-title pair."""
    if any(type(n) is not int or n < 2 or n % 2 for n in counts.values()):
        raise ValueError("split sizes must be positive even integers")
    result = {name: [] for name in counts}
    seen: set[str] = set()
    for kind in ("bridge", "comparison"):
        candidates = sorted(
            (row for row in rows if row["type"] == kind),
            key=lambda r: digest([seed, r["id"]]),
        )
        iterator = iter(candidates)
        for name, size in counts.items():
            while sum(r["type"] == kind for r in result[name]) < size // 2:
                row = next(iterator, None)
                if row is None:
                    raise ValueError("not enough unique examples for requested splits")
                keys = {
                    "id:" + row["id"],
                    "q:" + " ".join(row["question"].lower().split()),
                    "support:"
                    + digest(sorted({p[0] for p in row["supporting_facts"]})),
                }
                if not seen & keys:
                    seen.update(keys)
                    result[name].append(row)
    return result


def specification(
    train: list[dict[str, Any]], *, seed: int, curriculum: bool, calls: int = 6
) -> dict[str, Any]:
    """Return canonical fitting dict; final selection/holdout are deliberately absent."""
    trainer = {
        "batch_size": 6,
        "num_threads": 1,
        "test_frequency": None,
        "selection_score_window": "latest_train_batch",
    }
    if curriculum:
        trainer["curriculum"] = {"history_size": 2, "success_threshold": 1.0}
    return {
        "schema_version": S.SCHEMA_VERSION,
        "kind": S.SPEC_KIND,
        "runtime": {"seed": seed, "offline": False, "test_mode": True},
        "llm_profiles": {
            "reader": {
                "provider": "openrouter",
                "model": "qwen/qwen-2.5-7b-instruct",
                "temperature": 0.0,
                "max_tokens": 192,
                "request_timeout_s": 60,
                "transport_max_attempts": 2,
                "transport_base_delay_s": 2,
                "request_params": {"top_p": 1.0},
            },
            "optimizer": {
                "provider": "openrouter",
                "model": "deepseek/deepseek-v4-flash-0731",
                "temperature": 0.6,
                "max_tokens": 16000,
                "request_timeout_s": 300,
                "transport_max_attempts": 4,
                "transport_base_delay_s": 2,
                "request_params": {
                    "top_p": 1.0,
                    "extra_body": {
                        "reasoning": {"effort": "low"},
                        "provider": {"sort": "throughput"},
                    },
                },
            },
        },
        "budget": {
            "optimizer_llm_calls": calls,
            "eval_llm_calls": 400,
            "evaluator_runs": 400,
            "on_exceed": "raise",
        },
        "levels": [
            {
                "id": "O0",
                "surface": {"kind": "module", "targets": list(INITIAL)},
                "module": {"ref": MODULE_REF, "config": dict(INITIAL), "inputs": {}},
                "engine": {
                    "name": "trace",
                    "config": {
                        "optimizer": "OptoPrimeV2",
                        "trainer": "PrioritySearch",
                        "iterations": calls + 1,
                        "num_candidates": 1,
                        "validation_gate": False,
                        "optimizer_kwargs": {},
                        "trainer_kwargs": trainer,
                    },
                },
                "objective": {
                    "evaluator_ref": EVALUATOR_REF,
                    "intent": "Improve answer exact-match on new questions by changing retrieval code, reader instruction, top_k or bridge_expansion. Keep their declared types. Do not memorize question-answer pairs.",
                    "metrics": {
                        "accuracy": {
                            "direction": "maximize",
                            "source": "evaluation.metrics.accuracy",
                            "aggregate_examples": "mean",
                        }
                    },
                    "selection": {"mode": "scalar", "score_key": "accuracy"},
                    "trace_config": {
                        "mode": "internal",
                        "detail": "full",
                        "credit_horizon": "full",
                        "max_nodes": 24,
                        "max_chars": 6000,
                        "semantic_names": [
                            "rank_passages",
                            "expand_query",
                            "select_context",
                            "make_prompt",
                            "_record_call",
                            "pack_answer",
                        ],
                    },
                },
                "datasets": {"train": train, "validation": [], "holdout": []},
                "llm_roles": {"forward": "reader", "optimizer": "optimizer"},
            }
        ],
    }
