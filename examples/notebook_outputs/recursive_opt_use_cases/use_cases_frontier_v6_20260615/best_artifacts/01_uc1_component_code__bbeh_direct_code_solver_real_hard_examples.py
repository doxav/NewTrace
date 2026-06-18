def _bbeh_direct_solver(self, question):
    """Return True/False for a BBEH boolean expression ending with ' is'."""
    # Extract the boolean expression portion before the trailing " is"
    q = (question or "").strip()
    if q.endswith(" is"):
        expr = q[: -len(" is")].strip()
    else:
        # Fallback: use whole string
        expr = q

    # Very small, safe evaluator for the kinds of expressions in these problems
    # Support: True/False, parentheses, 'not', and whitespace.
    allowed_tokens = set(list("()") + ["not", "True", "False"])
    # Remove characters that are not expected for this restricted grammar
    # (keeps it simple; if something unexpected appears, fail closed)
    for bad in [
        ",",
        ";",
        "[",
        "]",
        "{",
        "}",
        "=",
        "&",
        "|",
        "<",
        ">",
        "+",
        "-",
        "*",
        "/",
    ]:
        if bad in expr:
            return "False"

    try:
        # Evaluate only within an empty builtins environment
        val = eval(expr, {"__builtins__": {}}, {})
        return "True" if bool(val) else "False"
    except Exception:
        # If parsing/evaluation fails, return a default.
        return "False"
