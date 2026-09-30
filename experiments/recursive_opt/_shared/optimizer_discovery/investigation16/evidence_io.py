"""Exact immutable evidence recording for prospective stages after frozen G1."""

from __future__ import annotations

import gzip
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G

_RECORDING_LOCK = threading.Lock()


def _check_credentials(text: str) -> None:
    """Privately reject actual or recognizable keys without echoing their contents."""
    active_key = os.environ.get("OPENROUTER_API_KEY")
    generic = re.findall(r"\bsk-[A-Za-z0-9_-]{24,}\b", text)
    if (
        (active_key and active_key in text)
        or re.search(r"\bsk-or-v1-[A-Za-z0-9_-]{8,}\b", text)
        or any(
            any(character.isdigit() for character in candidate)
            and any(character.isalpha() for character in candidate[3:])
            for candidate in generic
        )
    ):
        raise RuntimeError("credential-shaped content cannot be persisted")


def persist(path: Path, value: Any) -> None:
    """Preserve exact finite JSON bytes, rejecting secrets and conflicting overwrite."""
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    _check_credentials(text)
    if E.exists(path):
        original = (
            path.read_bytes()
            if path.exists()
            else gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())
        )
        if original.decode("utf-8") != text:
            raise RuntimeError("refusing to overwrite completed evidence")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    if len(payload) > 450000:
        path = path.with_suffix(path.suffix + ".gz")
        payload = gzip.compress(payload, mtime=0)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


class ExactRecorder:
    """Inject the existing read/exists protocol and an exact writer into a slot call."""

    persist = staticmethod(persist)
    read = staticmethod(E.read)
    exists = staticmethod(E.exists)


def complete_slot(
    directory: Path, request: dict[str, Any], client: Any
) -> dict[str, Any]:
    """Reuse G1's transport/retry/resume code with process-local exact recording.

    No frozen file is changed. The recorder is restored on success or error.
    This adapter requires the registered single-call concurrency; concurrent
    direct calls to G.complete_slot in this same process are unsupported. The
    live G1 process has independent module state and retains its frozen recorder.
    """
    if not _RECORDING_LOCK.acquire(blocking=False):
        raise RuntimeError("exact slot recorder requires concurrency one")
    previous = G.E
    try:
        G.E = ExactRecorder
        return G.complete_slot(directory, request, client)
    finally:
        G.E = previous
        _RECORDING_LOCK.release()
