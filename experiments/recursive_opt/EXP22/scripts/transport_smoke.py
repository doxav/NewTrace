"""Start S2 with a single exact-routing direct OpenRouter completion."""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preflight import write_json
from src.transport import BASE_URL, EXTRA_BODY, MODEL, SERVING_PROVIDER


def payload() -> dict[str, Any]:
    """Build the tiny smoke request without credentials or an LLM seed."""
    return {
        "model": MODEL, **EXTRA_BODY,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0.7, "max_tokens": 32,
    }


def request_json(path: str, key: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    """Send one request; never persist authorization or raw HTTP exceptions."""
    request = urllib.request.Request(
        BASE_URL + path, data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def main() -> int:
    """Persist sanitized serving evidence and stop if exact routing fails."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required")
    body = payload()
    code, response = request_json("/chat/completions", key, body)
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "outbound_body": body, "http_status": code, "timeout_seconds": 600,
        "generation_id": response.get("id"), "returned_model": response.get("model"),
        "serving_provider": response.get("provider"), "usage": response.get("usage"),
        "error": response.get("error"),
    }
    if code == 200 and not record["serving_provider"] and record["generation_id"]:
        metadata_code, metadata = request_json("/generation?" + urllib.parse.urlencode({"id": record["generation_id"]}), key)
        data = metadata.get("data", {})
        record["metadata_http_status"] = metadata_code
        record["serving_provider"] = data.get("provider_name")
    returned = record["returned_model"] or ""
    record["passed"] = code == 200 and (returned == MODEL or returned.startswith(MODEL + "-")) and record["serving_provider"] == SERVING_PROVIDER
    evidence = {"direct": record, "skydiscover": {"status": "NOT_RUN"}, "trace": {"status": "NOT_RUN"}, "passed": False}
    write_json(ROOT / "artifacts/openrouter_transport_validation.json", evidence)
    if not record["passed"]:
        write_json(ROOT / "artifacts/STOP.json", {
            "status": "STOPPED_PROVIDER", "reason": "Direct OpenRouter exact-routing smoke failed",
            "iteration": 0, "last_valid_score": None, "last_active_policy": None,
            "evidence": "artifacts/openrouter_transport_validation.json",
            "recommended_fix": "Resolve the recorded provider error without changing model/provider/session; rerun S2 before optimizer execution.",
        })
    print(json.dumps({"direct_transport_passed": record["passed"], "http_status": code, "serving_provider": record["serving_provider"]}))
    return 0 if record["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
