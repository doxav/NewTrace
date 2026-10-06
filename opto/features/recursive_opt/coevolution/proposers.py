"""Meta-level proposers: produce a new policy source from a parent policy and feedback.

``LLMRewriteProposer`` reproduces EvoX: full rewrite from an assembled meta prompt, up to
``retries`` attempts, validator errors optionally fed back (EvoX does not feed them back).
``TraceProposer`` uses one persistent OptoPrimeV2 whose trainable node is the policy
source; its memory keeps previous (policy, feedback) pairs across proposals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from opto.optimizers.optoprime_v2 import OptoPrimeV2
from opto.trace import bundle, node
from opto.utils.llm import DummyLLM

from .feedback import POLICY_CONTRACT
from .operator import parse_full_rewrite
from .policy import UNIFORM_POLICY_SOURCE

LLMText = Callable[[str, str], str]
Validate = Callable[[str], Optional[str]]


@dataclass
class ProposalResult:
    ok: bool
    source: Optional[str]
    attempts: int
    errors: List[str] = field(default_factory=list)


class LLMRewriteProposer:
    def __init__(self, llm: LLMText, retries: int = 3, feed_errors: bool = False, language: str = 'python') -> None:
        self.llm, self.retries, self.feed_errors, self.language = llm, retries, feed_errors, language
        self.prompts: List[Tuple[str, str]] = []

    def propose(self, build_prompt: Callable[[Sequence[str]], Tuple[str, str]], validate: Validate, **_: object) -> ProposalResult:
        errors: List[str] = []
        for attempt in range(1, self.retries + 1):
            system, user = build_prompt(errors if self.feed_errors else ())
            self.prompts.append((system, user))
            try:
                reply = self.llm(system, user)
            except Exception as error:  # noqa: BLE001
                return ProposalResult(False, None, attempt, errors + [f'LLM generation failed: {error}'])
            source = parse_full_rewrite(reply or '', self.language)
            error = validate(source) if source else 'No valid solution found in response'
            if error is None:
                return ProposalResult(True, source, attempt, errors)
            errors.append(error)
        return ProposalResult(False, None, self.retries, errors)


@bundle()
def policy_window(selection_policy, evidence):
    """The deployed selection policy and the measured evidence of its last window:
    per-decision execution trace, window score, population statistics and scored context policies."""
    return evidence


class TraceProposer:
    """Persistent OptoPrimeV2 over the policy source (Trace as the meta-optimizer)."""

    def __init__(self, llm: LLMText, memory_size: int = 5, retries: int = 3, max_tokens: int = 8000, initial_source: str = UNIFORM_POLICY_SOURCE) -> None:
        self.llm, self.memory_size, self.retries = llm, memory_size, retries
        self.node = node(initial_source, trainable=True, name='selection_policy', description='Complete Python source implementing:\n' + POLICY_CONTRACT)
        self.optimizer = OptoPrimeV2([self.node], llm=DummyLLM(self._call), memory_size=memory_size, log=False, max_tokens=max_tokens, initial_var_char_limit=100000)
        self.prompts: List[Tuple[str, str]] = []

    def _call(self, *args: object, **kwargs: object) -> str:
        messages = kwargs.get('messages') or args[0]
        system = '\n'.join(str(m.get('content', '')) for m in messages if m.get('role') == 'system')
        user = '\n'.join(str(m.get('content', '')) for m in messages if m.get('role') != 'system')
        self.prompts.append((system, user))
        return self.llm(system, user)

    def propose(self, parent_source: str, validate: Validate, feedback: str = '', context: Sequence[str] = (), **_: object) -> ProposalResult:
        errors: List[str] = []
        evidence = '\n\n'.join(f'context policy {i}:\n{c}' for i, c in enumerate(context, 1)) or 'no context policies'
        for attempt in range(1, self.retries + 1):
            self.node._data = parent_source
            output = policy_window(self.node, evidence)
            text = feedback + (('\nPrevious proposals failed validation:\n' + '\n'.join(f'- {e}' for e in errors)) if errors else '')
            self.optimizer.zero_feedback()
            self.optimizer.backward(output, text)
            try:
                self.optimizer.step()
            except Exception as error:  # noqa: BLE001
                errors.append(f'optimizer step failed: {type(error).__name__}: {error}')
                continue
            source = str(self.node.data)
            error = 'optimizer proposed no change' if source == parent_source else validate(source)
            if error is None:
                return ProposalResult(True, source, attempt, errors)
            errors.append(error)
        return ProposalResult(False, None, self.retries, errors)
