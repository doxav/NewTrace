def _baseline_promotion_policy(self, artifact_report):
    # Heuristic: combine "validation" with a "gain" proxy.
    mean_score = float(artifact_report.get("mean_score", 0.0))
    syntax_ok = bool(artifact_report.get("syntax_ok", True))
    saturated = bool(artifact_report.get("saturated_control", False))

    # Validated means syntactically acceptable and not saturated.
    validated = syntax_ok and (not saturated)

    # Gain proxy: ensure we don't emit negative gain as a reason to promote.
    # (We only promote when mean_score indicates non-negative improvement.)
    gain = max(0.0, mean_score)

    # Decide action:
    # - promote only if validated AND the prior indicates non-negative gain
    # - otherwise do_not_promote
    action = "promote" if (validated and mean_score >= 0.0) else "do_not_promote"

    return (
        f"action: {action}\n"
        f"code: baseline\n"
        f"gain: {gain:.2f}\n"
        f"validated: {str(validated).lower()}\n"
        f"reason: best score"
    )
