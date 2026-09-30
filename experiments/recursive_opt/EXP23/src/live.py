"""Live optimizer LLM (one fixed OpenRouter provider) and an EvoX-style LLM proposer.

The key is read only from the OPENROUTER_API_KEY environment variable; it is never
logged or written. Every call is recorded (latency, served provider, tokens, cost,
finish reason) without prompt or response text.
"""

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

MODEL = 'z-ai/glm-5.3-flash'
BASE_URL = 'https://openrouter.ai/api/v1'
PROVIDERS = ('deepinfra', 'novita', 'inference-net')


class LiveOptimizerLLM:
    """String-returning callable (wrap with DummyLLM for OptoPrimeV2), pinned to one provider."""

    def __init__(self, provider: str, log_path: Path, role: str = 'optimizer', max_tokens: int = 8000) -> None:
        import openai
        if provider not in PROVIDERS:
            raise ValueError(f'provider must be one of {PROVIDERS}')
        key = os.environ.get('OPENROUTER_API_KEY')
        if not key:
            raise RuntimeError('OPENROUTER_API_KEY is not set')
        self.client = openai.OpenAI(api_key=key, base_url=BASE_URL, timeout=600, max_retries=0)
        self.provider, self.role, self.max_tokens, self.log_path = provider, role, max_tokens, log_path
        self.calls, self.prompt_chars, self.lock = 0, [], threading.Lock()

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Bounded retry on 429/5xx/timeouts (same provider, every attempt logged)."""
        import openai
        for attempt in range(4):
            try:
                return self._once(*args, attempt=attempt, **kwargs)
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                if attempt == 3:
                    raise
                time.sleep(10 * 2 ** attempt)
        raise AssertionError('unreachable')

    def _once(self, *args: Any, attempt: int = 0, **kwargs: Any) -> str:
        messages = kwargs.get('messages') or args[0]
        if attempt == 0:
            self.prompt_chars.append(sum(len(str(m.get('content', ''))) for m in messages))
        started = time.time()
        record: dict[str, Any] = {'role': self.role, 'provider_requested': self.provider, 'prompt_chars': self.prompt_chars[-1], 'attempt': attempt}
        try:
            response = self.client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=self.max_tokens, temperature=0.7,
                extra_body={'provider': {'only': [self.provider], 'allow_fallbacks': False}, 'reasoning_effort': 'low', 'usage': {'include': True}})
            usage = response.usage
            text = response.choices[0].message.content or ''
            record.update(ok=True, served_by=getattr(response, 'provider', None), model=response.model, finish=response.choices[0].finish_reason,
                          prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens, cost=getattr(usage, 'cost', None), text_chars=len(text))
            return text
        except Exception as error:
            record.update(ok=False, error_type=type(error).__name__, status=getattr(error, 'status_code', None))
            raise
        finally:
            record['latency_s'] = round(time.time() - started, 2)
            with self.lock:
                self.calls += attempt == 0
                with self.log_path.open('a') as stream:
                    stream.write(json.dumps(record) + '\n')


EVOX_SYSTEM = """You are an expert coder evolving a search algorithm for program optimization (EvoX meta-evolution).
The search algorithm decides which previously evaluated solution (the parent) an LLM mutates next.
Your policy is scored by how much the best solution improves while it is active:
    score = (best_end - best_start) * (1 + log(1 + best_start)) / sqrt(window)
Return ONLY the new policy in a single fenced block (```json for knob policies, ```python for code policies)."""

SURFACE_HELP = {
    'knobs': 'Policy = JSON with exactly {"temperature": 0.02..5, "epsilon": 0..1, "reuse_penalty": 0..3, "elite_k": 1..64}: with probability epsilon pick a uniform random parent; otherwise softmax over the top elite_k members with logit = rank_pct/temperature - reuse_penalty*uses.',
    'code': 'Policy = Python defining select_parent(members, rng) -> index. members: list of dicts {score, rank (0 best), rank_pct (1 best), uses, age}; rng: random.Random. Only builtins and math are available.',
}


def evox_prompt(surface: str, parent: dict, context: list[dict], stats: dict, feedback: bool = True) -> list[dict]:
    """EvoX-shaped meta prompt: parent + scored context policies + search statistics."""
    lines = [SURFACE_HELP[surface], '', '# Parent policy to improve']
    lines.append(parent['text'] + (f"\n(score: {parent['score']:.4f})" if feedback and parent.get('score') is not None else ''))
    if feedback and context:
        lines += ['', '# Other evaluated policies'] + [f"- score {c['score']:.4f}: {c['text']}" for c in context]
    if feedback:
        lines += ['', '# Search statistics', json.dumps(stats)]
    lines += ['', 'Propose one improved policy.']
    return [{'role': 'system', 'content': EVOX_SYSTEM}, {'role': 'user', 'content': '\n'.join(lines)}]


BLOCK = re.compile(r'```(?:json|python)?\s*\n(.*?)```', re.DOTALL)


def extract_policy(text: str) -> str:
    """Take the last fenced block, else the raw text."""
    blocks = BLOCK.findall(text or '')
    return (blocks[-1] if blocks else (text or '')).strip() + ('\n' if blocks and 'def ' in blocks[-1] else '')
