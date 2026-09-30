"""Audit the installed timeout path using only a loopback HTTP responder."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import httpx
import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler

from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from opto.features.recursive_opt import runmode

MODEL = "deepseek/deepseek-v4-flash-0731"
ROOT = Path(__file__).resolve().parent


def run_case(mode: Literal["fast", "drip", "stall"]) -> dict[str, Any]:
    """Run actual wrappers/transport against a synthetic, secret-free loopback response."""
    if mode not in {"fast", "drip", "stall"}:
        raise ValueError("unknown local timeout probe case")
    result: dict[str, Any] = {
        "mode": mode,
        "requests": 0,
        "request_bodies": [],
        "timeout_extensions": [],
        "external_connections_blocked": 0,
    }
    body = json.dumps(
        {
            "id": "local-timeout-diagnostic",
            "object": "chat.completion",
            "created": 1,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "LOCAL SYNTHETIC RESPONSE",
                    },
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        """Return JSON with either continuous progress or a deliberately idle body."""

        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            """Suppress access logs so authorization headers can never be printed."""

        def do_POST(self) -> None:
            """Read synthetic request body only and implement the declared timing case."""
            result["requests"] += 1
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            result["request_bodies"].append(request)
            padding = 48 if mode == "drip" else 0
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body) + padding))
            self.end_headers()
            self.wfile.flush()
            try:
                if mode == "drip":
                    for _ in range(12):
                        time.sleep(0.1)
                        self.wfile.write(b"    ")
                        self.wfile.flush()
                elif mode == "stall":
                    time.sleep(1.0)
                self.wfile.write(body)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The stalled client is expected to close after its read timeout.
                return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    port = server.server_address[1]
    original_connect = socket.socket.connect

    def local_connect(sock: socket.socket, address: Any) -> None:
        """Reject all connections except this probe's exact loopback endpoint."""
        if not isinstance(address, tuple) or address[:2] != ("127.0.0.1", port):
            result["external_connections_blocked"] += 1
            raise RuntimeError("local timeout probe prohibits external connections")
        original_connect(sock, address)

    def capture_timeout(request: httpx.Request) -> None:
        """Record only HTTPX timeout extensions, never request headers or credentials."""
        result["timeout_extensions"].append(dict(request.extensions["timeout"]))

    capture = io.StringIO()
    previous_telemetry = litellm.telemetry
    litellm.telemetry = False
    try:
        with (
            patch.object(socket.socket, "connect", local_connect),
            contextlib.redirect_stdout(capture),
            contextlib.redirect_stderr(capture),
            httpx.Client(
                trust_env=False, event_hooks={"request": [capture_timeout]}
            ) as http_client,
        ):
            transport = HTTPHandler(client=http_client)
            client = runmode.make_live_llm(
                "openrouter/" + MODEL,
                cache=False,
                max_retries=1,
                request_timeout_s=300,
                allow_env_overrides=False,
                empty_response_retries=0,
                budget_resource=None,
            )
            kwargs = {
                "messages": [
                    {"role": "user", "content": "Local transport diagnostic only."}
                ],
                "api_base": f"http://127.0.0.1:{port}/api/v1",
                "api_key": "local-probe-placeholder",
                "client": transport,
                "num_retries": 0,
                "temperature": 0.6,
                "top_p": 1.0,
                "max_tokens": 32000,
                "extra_body": {"reasoning": {"effort": "low"}},
            }
            if mode != "fast":
                kwargs["timeout"] = 0.5
            started = time.monotonic()
            try:
                response = client(**kwargs)
                result.update(
                    {
                        "outcome": "completed",
                        "response_model": response.model,
                        "response_id": response.id,
                    }
                )
            except Exception as error:  # noqa: BLE001 - retain typed transport outcomes
                chain, seen = [], set()
                current: BaseException | None = error
                while current is not None and id(current) not in seen:
                    seen.add(id(current))
                    chain.append(type(current).__name__)
                    current = current.__cause__ or current.__context__
                result.update({"outcome": "error", "error_chain": chain})
            result["elapsed_monotonic_s"] = time.monotonic() - started
    finally:
        litellm.telemetry = previous_telemetry
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    return result


def main() -> None:
    """Preserve one immutable local probe result and exact installed-source provenance."""
    target = ROOT / "results.json"
    if target.exists():
        raise RuntimeError("refusing to replace completed runtime diagnostic")
    files = [
        Path(__file__),
        ROOT / "PROTOCOL.md",
        Path(runmode.__file__),
        Path(inspect.getfile(HTTPHandler)),
    ]
    package_root = Path(litellm.__file__).parent
    files.extend(
        package_root / name
        for name in ("main.py", "utils.py", "llms/custom_httpx/llm_http_handler.py")
    )
    result = {
        "kind": "LOCAL_HTTP_ENGINEERING_DIAGNOSTIC_NO_MODEL_CALLS",
        "created_ns": time.time_ns(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("litellm", "httpx", "httpcore")
        },
        "sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
        "cases": [run_case(mode) for mode in ("fast", "drip", "stall")],
    }
    I.persist(target, result)
    print(json.dumps(result["cases"], indent=2))


if __name__ == "__main__":
    main()
