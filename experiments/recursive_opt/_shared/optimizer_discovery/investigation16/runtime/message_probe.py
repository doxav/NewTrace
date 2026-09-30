"""Check exact message transmission by reusing the existing local-only HTTP probe."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.runtime import timeout_probe as P

ROOT = Path(__file__).resolve().parent


def run_request(request: dict[str, Any]) -> dict[str, Any]:
    """Send the declared two-user input only to loopback through the actual client stack."""
    messages = request.get("messages")
    settings = request.get("settings")
    allowed = {
        "temperature",
        "top_p",
        "max_tokens",
        "extra_body",
        "timeout",
        "seed",
        "num_retries",
    }
    if (
        request.get("model") != P.MODEL
        or not isinstance(messages, list)
        or len(messages) != 2
        or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] != "user"
            or not isinstance(message["content"], str)
            or not message["content"]
            for message in messages
        )
        or not isinstance(settings, dict)
        or set(settings) != allowed
        or settings["max_tokens"] != 32000
        or settings["num_retries"] != 0
        or settings["timeout"] != 300
        or settings["extra_body"] != {"reasoning": {"effort": "low"}}
    ):
        raise ValueError(
            "local message probe requires the declared safe two-user request shape"
        )
    original_factory = P.runmode.make_live_llm

    def factory(*args: Any, **kwargs: Any) -> Any:
        """Reuse the real fixed-model factory, changing only its synthetic request input."""
        client = original_factory(*args, **kwargs)

        def send(**call_kwargs: Any) -> Any:
            """Preserve loopback endpoint/placeholder credential and pass exact original messages."""
            call_kwargs.update(copy.deepcopy(settings))
            call_kwargs["messages"] = copy.deepcopy(messages)
            return client(**call_kwargs)

        return send

    with patch.object(P.runmode, "make_live_llm", factory):
        result = P.run_case("fast")
    wire = result["request_bodies"][0] if result["request_bodies"] else {}
    result["messages_exact"] = wire.get("messages") == messages
    result["message_summaries"] = [
        {
            "index": index,
            "role": message["role"],
            "characters": len(message["content"]),
            "utf8_bytes": len(message["content"].encode()),
            "sha256": B.source_hash(message["content"]),
        }
        for index, message in enumerate(messages)
    ]
    if (
        result["outcome"] != "completed"
        or result["requests"] != 1
        or not result["messages_exact"]
    ):
        raise RuntimeError(
            "local message transmission did not preserve the declared request"
        )
    if (
        wire.get("max_tokens") != settings["max_tokens"]
        or wire.get("reasoning") != settings["extra_body"]["reasoning"]
    ):
        raise RuntimeError("local message transmission changed generation settings")
    return result


def main() -> None:
    """Preserve one exact local HTTP replay of a recorded request without contacting a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "message_results.json")
    args = parser.parse_args()
    if E.exists(args.output):
        raise RuntimeError("refusing to replace a completed message-transmission probe")
    request = E.read(args.request)
    result = {
        "kind": "LOCAL_HTTP_MESSAGE_TRANSMISSION_NO_MODEL_CALLS",
        "created_ns": time.time_ns(),
        "source_request": str(args.request),
        "source_request_sha256": B.digest(request),
        "helper_sha256": B.source_hash(Path(__file__).read_text()),
        "reused_probe_sha256": B.source_hash(Path(P.__file__).read_text()),
        "result": run_request(request),
    }
    I.persist(args.output, result)
    print(
        json.dumps(
            {
                key: result["result"][key]
                for key in (
                    "messages_exact",
                    "message_summaries",
                    "requests",
                    "external_connections_blocked",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
