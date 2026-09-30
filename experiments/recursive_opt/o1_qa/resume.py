"""Explicit EXP21 development-only recovery; never modifies completed evidence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from opto.features.recursive_opt.runmode import make_live_llm

from . import campaign as C, pilot, task

ORIGINAL = C.live_response
ROOT = Path()
AUTHORIZED: dict[str, str] = {}


def install(root: Path) -> None:
    """Load the finite signed-by-hash list of interrupted development requests."""
    global ROOT, AUTHORIZED
    ROOT = root
    AUTHORIZED = {}
    for name in ("resume_amendment.json", "resume_credit_amendment.json"):
        path = root / name
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        for item in manifest["requests"]:
            relative = item["path"]
            if not relative.startswith("development/") or ".." in Path(relative).parts:
                raise ValueError("resume authorization must be confined to development")
            raw = json.loads((root / relative).read_text())
            if raw["request_hash"] != item["request_hash"]:
                raise ValueError("resume authorization request mismatch")
            if name == "resume_credit_amendment.json":
                failure = (root / relative).with_name("failure.json")
                if not failure.exists() or json.loads(failure.read_text()).get(
                    "http_status_codes"
                ) != [402]:
                    raise ValueError("credit resume requires a recorded HTTP 402")
            AUTHORIZED[relative] = item["request_hash"]
    C.live_response = live_response


def live_response(
    folder: Path, profile: Mapping[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    """Reissue only listed lost/rejected attempts, recording a separate immutable wave."""
    try:
        prefix = str(folder.relative_to(ROOT)) + "/"
    except ValueError:
        return ORIGINAL(folder, profile, request)
    if not any(path.startswith(prefix) for path in AUTHORIZED):
        return ORIGINAL(folder, profile, request)
    attempt = 0
    while (folder / str(attempt) / "request.json").exists():
        current = folder / str(attempt)
        raw = json.loads((current / "request.json").read_text())
        if raw["request_hash"] != task.digest(request):
            raise ValueError("resumed request differs from the recorded request")
        if (current / "response.json").exists():
            return pilot.request_once(current, request, lambda **_: None)
        path = str((current / "request.json").relative_to(ROOT))
        failure = current / "failure.json"
        explicit_rejection = (
            failure.exists()
            and json.loads(failure.read_text()).get("remote_completion")
            == "explicit_rejection"
        )
        if AUTHORIZED.get(path) != raw["request_hash"] and not explicit_rejection:
            raise RuntimeError("ambiguous remote request requires reconciliation")
        attempt += 1
    if attempt >= 4:
        raise RuntimeError("provider rejection retry waves exhausted")
    current = folder / str(attempt)
    events: list[dict[str, Any]] = []

    def record(event: str, kind: str | None) -> None:
        """Record transport labels using the existing sanitized schema."""
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
            and json.loads(failure.read_text()).get("remote_completion")
            == "explicit_rejection"
            and attempt < 3
        ):
            time.sleep(4 * (2**attempt))
            return live_response(folder, profile, request)
        raise
