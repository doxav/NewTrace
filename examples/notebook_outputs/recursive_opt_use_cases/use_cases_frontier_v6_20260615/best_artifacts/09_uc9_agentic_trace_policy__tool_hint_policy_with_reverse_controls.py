def _baseline_agentic_trace_policy(self, signal):
    s = (signal or "").lower()

    # If saturated/control: avoid expensive tool calls.
    if any(k in s for k in ["saturated", "zero gain", "avoid", "control", "stop"]):
        return "tools: stop\nhint: control mode detected; skip expensive tools"

    # Prior failures/family examples -> trace search
    if any(
        k in s
        for k in [
            "prior failures",
            "family examples",
            "family",
            "cold",
            "warm",
            "held-out",
            "heldout",
            "promoting",
            "transfer",
        ]
    ):
        return "tools: trace_search\nhint: collect prior failures/family examples and compare cold vs warm on held-out families"

    # Validate candidate on subset -> run subset
    if any(
        k in s for k in ["validate", "small subset", "subset", "accept", "run_subset"]
    ):
        return "tools: run_subset\nhint: validate candidate prompt/code on a small subset before accepting"

    # Syntax/artifact reuse -> artifact linter
    if any(
        k in s
        for k in [
            "syntax",
            "saved artifact",
            "inspect",
            "artifact",
            "code reuse",
            "current_code",
            "linter",
        ]
    ):
        return "tools: artifact_linter\nhint: inspect saved artifact for syntax and reuse current_code safely"

    # Fallback
    return "tools: note\nhint: observe the feedback and determine required next tool"
