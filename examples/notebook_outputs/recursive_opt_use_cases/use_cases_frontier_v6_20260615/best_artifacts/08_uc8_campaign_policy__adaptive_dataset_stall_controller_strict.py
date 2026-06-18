def _baseline_campaign_policy(self, diagnostics):
    """Return action/task/reason from diagnostics; this seed is intentionally weak."""
    task_spec = str(diagnostics.get("task", "")).lower()
    # Heuristic steering to avoid getting stuck on a single internal task.
    mixed_regressed = bool(diagnostics.get("mixed_regressed", False))
    mean_score = float(diagnostics.get("mean_score", 0.0))
    spread = float(diagnostics.get("spread", 0.0))

    # Default parameters
    action = "action: continue"
    max_examples = 8
    reason = "default"

    # If we are regressing or scores are low/flat, try switching/exploring.
    if mixed_regressed or mean_score < 0.2 or spread < 0.01:
        action = "action: switch"
        reason = "switch_on_regression_or_stall"

        if "gsm8k" in task_spec:
            # Feedback often suggests trying bbeh when gsm8k is saturating.
            task = "internal:multiobjective_bbeh"
            max_examples = 6
        elif "qasper" in task_spec:
            task = "hf:qasper"
            max_examples = 6
        else:
            task = "internal:multiobjective_bbeh"
            max_examples = 6
    else:
        # Otherwise exploit lightly, but vary reason.
        action = "action: continue"
        max_examples = 8
        reason = "exploit_with_low_risk"
        if "qasper" in task_spec:
            task = "hf:qasper"
        elif "bbeh" in task_spec:
            task = "internal:multiobjective_bbeh"
        else:
            task = "internal:multiobjective_gsm8k"

    return f"{action}\ntask: {task}\nmax_examples: {max_examples}\nreason: {reason}"
