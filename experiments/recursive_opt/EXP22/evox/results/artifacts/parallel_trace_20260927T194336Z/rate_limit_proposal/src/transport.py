"""Exact OpenRouter clients using each framework's supported extension points."""

import copy
import os
from collections.abc import Mapping
from typing import Any

import openai
from skydiscover.optimize.config import LLMModelConfig
from skydiscover.optimize.llm.openai import OpenAILLM

MODEL = "z-ai/glm-5.3-flash"
SESSION = "benchmark-PRIMS-SIGNAL-run-001"
BASE_URL = "https://openrouter.ai/api/v1"
PROVIDER = "novita"
SERVING_PROVIDER = "Novita"
REASONING_EFFORT = "low"
EXTRA_BODY = {"provider": {"only": [PROVIDER]}, "session_id": SESSION, "reasoning_effort": REASONING_EFFORT}
HTTP_ROLES: dict[int, str] = {}
HTTP_RECORDS: list[dict[str, Any]] = []
MAX_RECOVERABLE_RATE_LIMITS = 2
RATE_LIMIT_BACKOFF_SECONDS = 30


def is_recoverable_rate_limit(record: dict[str, Any]) -> bool:
    """Recognize only an empty Novita shared-pool refusal of the frozen solution request."""
    body = record.get('outbound_body')
    error = record.get('error')
    metadata = error.get('metadata') if isinstance(error, dict) else None
    return (
        record.get('passed') is False and record.get('role') == 'solution'
        and record.get('http_status') == 429
        and isinstance(metadata, dict) and error.get('code') == 429
        and metadata.get('provider_name') == SERVING_PROVIDER
        and metadata.get('limit_source') == 'upstream_provider_shared_pool'
        and isinstance(body, dict) and body.get('model') == MODEL
        and all(body.get(key) == value for key, value in EXTRA_BODY.items())
        and body.get('temperature') == 0.7 and body.get('max_tokens') == 32000
        and record.get('content_lengths') == [] and record.get('finish_reasons') == []
        and record.get('returned_model') is None and record.get('serving_provider') is None
        and record.get('generation_id') is None
    )


def transport_records_acceptable(records: list[dict[str, Any]]) -> bool:
    """Keep all identity guards, permitting at most two precisely identified refusals."""
    refused = [record for record in records if record.get('passed') is not True]
    return len(refused) <= MAX_RECOVERABLE_RATE_LIMITS and all(is_recoverable_rate_limit(record) for record in refused)


class SkyOpenRouter(OpenAILLM):
    """Preserve stock generation while injecting the mandatory routing body."""

    def __init__(self, model_cfg: LLMModelConfig) -> None:
        """Reject model/endpoint leakage before constructing the stock client."""
        if model_cfg.name != MODEL or model_cfg.api_base != BASE_URL:
            raise ValueError("EXP22 requires the exact GLM OpenRouter identity")
        private = copy.copy(model_cfg)
        private.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not private.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        super().__init__(private)

    async def _call_api(self, params: dict[str, Any]) -> str:
        """Add routing after stock generation assembles the final SDK payload."""
        return await super()._call_api({**params, "extra_body": copy.deepcopy(EXTRA_BODY)})


class TraceOpenRouter:
    """CP-A's explicit non-portable client, supplied through llm_factory."""

    def __init__(self, profile: Mapping[str, Any], role: str) -> None:
        """Bind a role without allowing alternate provider identities."""
        if profile['provider'] != 'openrouter' or profile['model'] != MODEL:
            raise ValueError("EXP22 requires the exact GLM OpenRouter identity")
        self.role = role
        self.client = openai.OpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url=BASE_URL, timeout=600, max_retries=0)
        HTTP_ROLES[id(self.client._client)] = 'meta' if role == 'optimizer' else role

    def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        """Send the guarded role request with the fixed upstream routing."""
        kwargs.pop('model', None)
        kwargs.pop('session_id', None)
        kwargs['extra_body'] = {**kwargs.get('extra_body', {}), **copy.deepcopy(EXTRA_BODY)}
        return self.client.chat.completions.create(model=MODEL, messages=messages, **{key: value for key, value in kwargs.items() if value is not None})


def model_config() -> LLMModelConfig:
    """Construct a secret-free stock model config with the custom client hook."""
    return LLMModelConfig(name=MODEL, api_base=BASE_URL, temperature=0.7, max_tokens=32000, timeout=600, retries=0, retry_delay=1, init_client=SkyOpenRouter)
