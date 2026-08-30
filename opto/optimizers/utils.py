from typing import Dict, Any

def print_color(message, color=None, logger=None):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
    }
    print(
        f"{colors.get(color, '')}{message}\033[0m"
    )  # Default to no color if invalid color is provided

    if logger is not None:
        logger.log(message)


def truncate_expression(value, limit):
    # https://stackoverflow.com/questions/1436703/what-is-the-difference-between-str-and-repr
    value = str(value)
    if len(value) > limit:
        return value[:limit] + "...(skipped due to length limit)"
    return value


def extract_top_level_blocks(text: str, tag: str):
    """Extract all top-level <tag>...</tag> blocks from text."""
    blocks = []
    start_tag = f'<{tag}>'
    end_tag = f'</{tag}>'
    stack = []
    start = None
    i = 0
    while i < len(text):
        if text.startswith(start_tag, i):
            if not stack:
                start = i + len(start_tag)
            stack.append(i)
            i += len(start_tag)
        elif text.startswith(end_tag, i):
            if stack:
                stack.pop()
                if not stack and start is not None:
                    blocks.append(text[start:i])
                    start = None
            i += len(end_tag)
        else:
            i += 1
    return blocks


def extract_first_top_level_block(text: str, tag: str):
    blocks = extract_top_level_blocks(text, tag)
    return blocks[0] if blocks else None


def strip_nested_blocks(text: str, tag: str) -> str:
    """Remove all nested <tag>...</tag> blocks from text, leaving only the top-level text."""
    result = ''
    start_tag = f'<{tag}>'
    end_tag = f'</{tag}>'
    stack = []
    i = 0
    last = 0
    while i < len(text):
        if text.startswith(start_tag, i):
            if not stack:
                result += text[last:i]
            stack.append(i)
            i += len(start_tag)
        elif text.startswith(end_tag, i):
            if stack:
                stack.pop()
                if not stack:
                    last = i + len(end_tag)
            i += len(end_tag)
        else:
            i += 1
    if not stack:
        result += text[last:]
    return result.strip()


def extract_reasoning_and_remainder(text: str, tag: str = "reasoning"):
    """Extract reasoning and the remainder of the text after reasoning block (if closed). Strip whitespace only if properly closed."""
    start_tag = f'<{tag}>'
    end_tag = f'</{tag}>'
    start = text.find(start_tag)
    if start == -1:
        return '', text
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        # If not properly closed, don't strip whitespace to preserve original formatting
        return text[start:], ''
    return text[start:end].strip(), text[end + len(end_tag):]


def extract_xml_like_data(text: str, reasoning_tag: str = "reasoning",
                          improved_variable_tag: str = "variable",
                          name_tag: str = "name",
                          value_tag: str = "value") -> Dict[str, Any]:
    """
    Extract thinking content and improved variables from text containing XML-like tags.

    Args:
        text (str): Text containing <reasoning> and <variable> tags

    Returns:
        Dict containing:
        - 'reasoning': content of <reasoning> element
        - 'variables': dict mapping variable names to their values
    """
    result = {
        'reasoning': '',
        'variables': {}
    }

    # Extract reasoning and the remainder of the text
    reasoning, remainder = extract_reasoning_and_remainder(text, reasoning_tag)
    result['reasoning'] = reasoning

    # Only parse variables from the remainder (i.e., after a closed reasoning tag)
    variable_blocks = extract_top_level_blocks(remainder, improved_variable_tag)
    for var_block in variable_blocks:
        name_block = extract_first_top_level_block(var_block, name_tag)
        value_block = extract_first_top_level_block(var_block, value_tag)
        # Only add if both name and value tags are present and name is non-empty after stripping
        if name_block is not None and value_block is not None:
            var_name = name_block.strip()
            var_value = value_block.strip() if value_block is not None else ''
            if var_name:  # Only require name to be non-empty, value can be empty
                result['variables'][var_name] = var_value
    return result


class LLMEmptyResponseError(RuntimeError):
    """A provider returned a completion with no message content.

    This is a normal provider outcome, not a programming error: the completion can be
    truncated at ``max_tokens`` before any text is emitted, filtered, or dropped while
    the provider is shedding load. Returning that ``None`` to callers produced
    ``TypeError: argument of type 'NoneType' is not iterable`` several frames later, in
    code that simply asked ``if "TERMINATE" in response``. The exception says what
    happened and, where the provider reports it, why.
    """


def extract_response_content(response: Any, *, context: str = "LLM") -> str:
    """Return a completion's message content, failing loudly if there is none.

    Raises
    ------
    LLMEmptyResponseError
        If the response carries no choices, or its first choice has no content. The
        provider's ``finish_reason`` is included when present because it distinguishes
        the causes that matter: ``length`` means the token budget was too small,
        ``content_filter`` means the request was refused.
    """
    # Some backends hand back the text directly rather than a completion object.
    if isinstance(response, str):
        content, finish_reason = response, None
    else:
        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMEmptyResponseError(
                f"{context} returned no choices; the provider produced no completion"
            )
        choice = choices[0]
        content = getattr(getattr(choice, "message", None), "content", None)
        finish_reason = getattr(choice, "finish_reason", None)
    # Whitespace-only content is as unusable as None, and reaches callers the same way.
    if not isinstance(content, str) or not content.strip():
        detail = f" (finish_reason={finish_reason!r})" if finish_reason else ""
        raise LLMEmptyResponseError(
            f"{context} returned a completion with no usable content{detail}. This usually "
            "means the response was truncated at max_tokens before any text, was filtered, "
            "or the provider shed load; retry or raise max_tokens."
        )
    return content
