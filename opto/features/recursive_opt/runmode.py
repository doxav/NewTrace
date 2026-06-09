"""
opto.features.recursive_opt.runmode
===================================

Make the *execution mode* impossible to misread. The whole point of this module
is to remove the "it looked like it worked but it was a stub" trap:

  * ``resolve_live()`` — returns True only if ``--live`` AND a key are present.
    If ``--live`` is requested WITHOUT a key, it RAISES instead of silently
    falling back to the offline stub.
  * ``mode_banner()`` — one loud line stating LIVE vs OFFLINE-STUB and whether
    Trace-Bench / PR #73 are actually in use, plus the efficacy caveat.

Read this once: OFFLINE-STUB mode exercises the *plumbing* (do nodes connect,
does backward reach the trainable parameter, does the loop run) using synthetic
analytic scores. Stub scores are NOT a measure of whether meta-optimization
*works* on real tasks — only a LIVE run with a real LLM measures efficacy.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Optional

_PREFLIGHTED_MODELS: set[str] = set()


def have_key() -> bool:
    return bool(
        os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    )


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _redact_secrets(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-<redacted>", text)
    return re.sub(r"proj_[A-Za-z0-9_-]+", "proj_<redacted>", text)


def uses_completion_token_param(model_name: str) -> bool:
    """Return True for models that reject the legacy ``max_tokens`` parameter."""
    return "gpt-5" in str(model_name).lower()


class CompletionTokenCompatLLM:
    """Translate Trace optimizer token kwargs for GPT-5-style LiteLLM calls."""

    def __init__(self, llm: Any, model_name: str) -> None:
        self._llm = llm
        self.model_name = model_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if uses_completion_token_param(self.model_name):
            kwargs = dict(kwargs)
            if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        return self._llm(*args, **kwargs)


def make_live_llm(
    model: Optional[str] = None,
    *,
    cache: bool = True,
    max_retries: int = 10,
    base_delay: float = 1.0,
) -> Any:
    """Create the LiteLLM backend used by recursive-opt live optimizers."""
    from opto.utils.llm import LiteLLM

    model_name = (
        model
        or os.environ.get("RECURSIVE_OPT_MODEL")
        or os.environ.get("TRACE_LITELLM_MODEL")
        or "gpt-5.4-nano"
    )
    llm = LiteLLM(
        model=model_name,
        cache=cache,
        max_retries=max_retries,
        base_delay=base_delay,
    )
    if uses_completion_token_param(model_name):
        return CompletionTokenCompatLLM(llm, model_name)
    return llm


def preflight_model(model: Optional[str] = None) -> None:
    """Fail fast when the configured live LLM model/key cannot be used."""
    model_name = (
        model
        or os.environ.get("RECURSIVE_OPT_MODEL")
        or os.environ.get("TRACE_LITELLM_MODEL")
        or "gpt-5.4-nano"
    )
    if model_name in _PREFLIGHTED_MODELS:
        return
    try:
        llm = make_live_llm(
            model_name,
            cache=False,
            max_retries=1,
            base_delay=0.1,
        )
        llm(
            messages=[{"role": "user", "content": "Return exactly: ok"}],
            max_tokens=8,
            temperature=0,
        )
        _PREFLIGHTED_MODELS.add(model_name)
    except Exception as exc:
        message = _redact_secrets(str(exc).splitlines()[0] if str(exc) else type(exc).__name__)
        raise SystemExit(
            "\nERROR: live model preflight failed for "
            f"{model_name!r}: {type(exc).__name__}: {message}\n"
            "Set RECURSIVE_OPT_MODEL to an accessible model, or set "
            "RECURSIVE_OPT_SKIP_MODEL_PREFLIGHT=1 only if you intentionally want "
            "to defer provider errors to the optimizer call.\n"
        ) from exc


def resolve_live(argv: Optional[list] = None) -> bool:
    """True iff ``--live`` AND an API key are present.

    Raises SystemExit if ``--live`` is requested but no key is set, so a live
    test can NEVER silently degrade to the offline stub.
    """
    argv = sys.argv if argv is None else argv
    want = "--live" in argv
    if want and not have_key():
        raise SystemExit(
            "\nERROR: --live was requested but no API key is set "
            "(OPENAI_API_KEY / OPENROUTER_API_KEY).\n"
            "Refusing to silently fall back to the offline stub.\n"
            "Set a key for a real LLM run, or drop --live to run the offline "
            "PLUMBING demo (synthetic scores).\n"
        )
    live = want and have_key()
    if live and os.environ.get("RECURSIVE_OPT_MODEL"):
        os.environ["TRACE_LITELLM_MODEL"] = os.environ["RECURSIVE_OPT_MODEL"]
    if live:
        if not _env_flag("RECURSIVE_OPT_SKIP_MODEL_PREFLIGHT", False):
            preflight_model(os.environ.get("TRACE_LITELLM_MODEL"))
        if _env_flag("RECURSIVE_OPT_USE_TRACEBENCH", True):
            from .tracebench import ensure_default_task_adapter

            ensure_default_task_adapter(
                require=_env_flag("RECURSIVE_OPT_REQUIRE_TRACEBENCH", True)
            )
    return live


def tracebench_mode() -> str:
    try:
        from .tracebench import real_mode_status
        return real_mode_status()
    except Exception:
        return "STUB (synthetic analytic scores — tests plumbing, NOT efficacy)"


def pr73_mode() -> str:
    try:
        from .traces import HAVE_PR73
    except Exception:
        HAVE_PR73 = False
    return (
        "AVAILABLE" if HAVE_PR73 else "ABSENT (graph/OTEL/Sysmon paths cannot run here)"
    )


def mode_banner(live: bool) -> str:
    bar = "=" * 72
    if live:
        model = os.environ.get(
            "RECURSIVE_OPT_MODEL", "LiteLLM default (set RECURSIVE_OPT_MODEL)"
        )
        head = f"[MODE] LIVE LLM run  ·  model = {model}"
        caveat = "Scores below reflect a REAL optimizer run."
    else:
        head = "[MODE] OFFLINE STUB run  ·  NO LLM is called"
        caveat = (
            "Scores below are SYNTHETIC (analytic formula). They show the "
            "plumbing runs and the optimization path is wired; they do NOT "
            "measure whether meta-optimization actually improves real tasks. "
            "Use --live with a key for efficacy."
        )
    return (
        f"{bar}\n{head}\n  Trace-Bench: {tracebench_mode()}\n"
        f"  PR #73 graph/OTEL: {pr73_mode()}\n  {caveat}\n{bar}"
    )
