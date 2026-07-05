"""
opto.features.recursive_opt.runmode
===================================

Make the *execution mode* impossible to misread. The whole point of this module
is to remove the "it looked like it worked but it was a stub" trap:

  * ``resolve_live()`` — returns True only if ``--live`` AND a key are present.
    If ``--live`` is requested WITHOUT a key, it RAISES instead of silently
    falling back to a non-live run.
  * ``mode_banner()`` — one loud line stating LIVE vs OFFLINE-STUB and whether
    Trace-Bench / graph telemetry backends are actually in use, plus the
    efficacy caveat.

Read this once: non-live mode does not call an optimizer LLM. Task-scoring
examples require an explicitly registered Trace-Bench adapter; otherwise they
raise instead of inventing synthetic benchmark scores. Only a LIVE run with a
real LLM optimizer measures recursive optimization efficacy.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Optional

from .budget import BudgetResource, budget_status, budgeted_llm

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

    def __init__(self, llm: Any, model_name: str, request_timeout_s: Optional[float] = None) -> None:
        self._llm = llm
        self.model_name = model_name
        self.request_timeout_s = request_timeout_s

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        if self.request_timeout_s is not None:
            kwargs.setdefault("timeout", self.request_timeout_s)
        if uses_completion_token_param(self.model_name):
            if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        return self._llm(*args, **kwargs)


def _positive_float_env(name: str) -> Optional[float]:
    """Read an optional positive float environment variable."""
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _positive_int_env(name: str) -> Optional[int]:
    """Read an optional positive integer environment variable."""
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def make_live_llm(
    model: Optional[str] = None,
    *,
    cache: bool = True,
    max_retries: int = 10,
    base_delay: float = 1.0,
    request_timeout_s: Optional[float] = None,
    budget_resource: Optional[BudgetResource] = "optimizer_llm_calls",
) -> Any:
    """Create the LiteLLM backend used by recursive-opt live optimizers.

    `budget_resource` defaults to optimizer proposal calls. Pass None for
    preflight calls or other probes that should not consume recursive budget.
    """
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
        max_retries=_positive_int_env("RECURSIVE_OPT_LLM_MAX_RETRIES") or max_retries,
        base_delay=_positive_float_env("RECURSIVE_OPT_LLM_BASE_DELAY_S") or base_delay,
    )
    timeout_s = request_timeout_s if request_timeout_s is not None else _positive_float_env("RECURSIVE_OPT_LLM_TIMEOUT_S")
    if uses_completion_token_param(model_name) or timeout_s is not None:
        llm = CompletionTokenCompatLLM(llm, model_name, timeout_s)
    return budgeted_llm(llm, budget_resource)


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
            budget_resource=None,
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
    test can NEVER silently degrade to a non-live run.
    """
    argv = sys.argv if argv is None else argv
    want = "--live" in argv
    if want and not have_key():
        raise SystemExit(
            "\nERROR: --live was requested but no API key is set "
            "(OPENAI_API_KEY / OPENROUTER_API_KEY).\n"
            "Refusing to silently fall back to a non-live run.\n"
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


def trace_io_mode() -> str:
    """Return whether optional graph/telemetry trace backends are importable."""
    try:
        from .traces import HAVE_TRACE_IO
    except Exception:
        HAVE_TRACE_IO = False
    return (
        "AVAILABLE"
        if HAVE_TRACE_IO
        else "UNAVAILABLE (graph/OTEL/Sysmon backends are not importable)"
    )


def _using_real_task_adapter() -> bool:
    """Return True when task scoring is backed by a registered real adapter."""
    try:
        from .tracebench import using_real_tasks

        return using_real_tasks()
    except Exception:
        return False


def mode_banner(live: bool) -> str:
    bar = "=" * 72
    if live:
        model = os.environ.get(
            "RECURSIVE_OPT_MODEL", "LiteLLM default (set RECURSIVE_OPT_MODEL)"
        )
        head = f"[MODE] LIVE LLM run  ·  model = {model}"
        caveat = "Scores below reflect a REAL optimizer run."
    elif _using_real_task_adapter():
        head = "[MODE] OFFLINE real Trace-Bench eval  ·  NO optimizer LLM is called"
        caveat = (
            "Task scores below come from the registered Trace-Bench adapter. "
            "Any search in this section is deterministic/manual; use --live "
            "or the live notebook cells for LLM-driven optimization."
        )
    else:
        head = "[MODE] OFFLINE run  ·  NO optimizer LLM and NO task adapter"
        caveat = (
            "Task-scoring examples will raise until a Trace-Bench adapter is "
            "registered. Use ensure_eval_only_task_adapter(...) for bounded "
            "real eval-only scoring, or --live with a key for LLM-driven "
            "optimization."
        )
    return (
        f"{bar}\n{head}\n  Trace-Bench: {tracebench_mode()}\n"
        f"  Graph/telemetry: {trace_io_mode()}\n"
        f"  Global budget: {budget_status()}\n  {caveat}\n{bar}"
    )
