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
import sys
from typing import Optional


def have_key() -> bool:
    return bool(
        os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    )


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
    return live


def tracebench_mode() -> str:
    try:
        from .tracebench import HAVE_TB
    except Exception:
        HAVE_TB = False
    return (
        "REAL (Trace-Bench installed)"
        if HAVE_TB
        else "STUB (synthetic analytic scores — tests plumbing, NOT efficacy)"
    )


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
