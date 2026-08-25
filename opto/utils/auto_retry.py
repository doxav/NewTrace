"""Bounded retries for provider rate limits and transient transport failures."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import TypeVar


_T = TypeVar("_T")
_RETRYABLE_MESSAGES = (
    "rate limit",
    "timeout",
    "temporary",
    "service unavailable",
    "internal server error",
    "bad gateway",
    "service temporarily unavailable",
    "too many requests",
    "quota",
    "overloaded",
    "resource has been exhausted",
    "resource_exhausted",
    "ratelimiterror",
    "quotaexceedederror",
    "connection error",
    "connection reset",
    "connection aborted",
    "connection closed",
    "server disconnected",
    "broken pipe",
    "network is unreachable",
)
_RETRYABLE_TYPES = (
    "ratelimiterror",
    "timeouterror",
    "connectionreseterror",
    "connectionabortederror",
    "brokenpipeerror",
    "apiconnectionerror",
    "serviceunavailableerror",
    "internalservererror",
    "remoteprotocolerror",
    "protocolerror",
    "connecterror",
    "readerror",
    "readtimeout",
    "pooltimeout",
)
_RETRYABLE_HTTP_CODES = ('code": 429', 'code": 503', 'code": 502', 'code": 500')


class TransportRetryError(RuntimeError):
    """Report exhaustion of a bounded transient provider retry policy."""


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield one exception and its explicit or implicit causal chain once."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_retryable(error: BaseException) -> bool:
    """Return whether an exception chain identifies a transient provider failure."""
    for item in _exception_chain(error):
        message = str(item).lower()
        error_type = type(item).__name__.lower()
        if (
            any(fragment in message for fragment in _RETRYABLE_MESSAGES)
            or any(fragment in error_type for fragment in _RETRYABLE_TYPES)
            or any(code in message for code in _RETRYABLE_HTTP_CODES)
        ):
            return True
    return False


def _transport_failure_kind(error: BaseException) -> str:
    """Classify only the transport diagnostics required by live experiments."""
    chain = list(_exception_chain(error))
    messages = " ".join(str(item).lower() for item in chain)
    types = " ".join(type(item).__name__.lower() for item in chain)
    if "connection reset" in messages or "connectionreseterror" in types:
        return "connection_reset"
    if "server disconnected" in messages or "remoteprotocolerror" in types:
        return "server_disconnected"
    return "other"


def retry_with_exponential_backoff(
    func: Callable[[], _T],
    max_retries: int = 10,
    base_delay: float = 1.0,
    operation_name: str = "operation",
    retry_event_callback: Callable[[str, str | None], None] | None = None,
) -> _T:
    """Retry an identical call only for bounded transient provider failures."""
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 1:
        raise ValueError("max_retries must be a positive integer")
    if not isinstance(base_delay, (int, float)) or isinstance(base_delay, bool) or base_delay < 0:
        raise ValueError("base_delay must be a non-negative number")
    retried = False
    for retry_attempt in range(max_retries):
        try:
            result = func()
            if retried and retry_event_callback is not None:
                retry_event_callback("recovered", None)
            return result
        except Exception as error:
            if not _is_retryable(error):
                raise
            failure_kind = _transport_failure_kind(error)
            if retry_event_callback is not None:
                retry_event_callback("transient_failure", failure_kind)
            if retry_attempt == max_retries - 1:
                if retry_event_callback is not None:
                    retry_event_callback("exhausted", failure_kind)
                raise TransportRetryError(
                    f"{operation_name}: transient transport failure after "
                    f"{max_retries} attempts: {error}"
                ) from error
            message = str(error).lower()
            error_type = type(error).__name__.lower()
            rate_limited = (
                "rate limit" in message
                or "ratelimiterror" in error_type
                or "quota" in message
                or "resource has been exhausted" in message
                or 'code": 429' in message
            )
            delay = (
                2 * (retry_attempt + 1) ** 2 + retry_attempt
                if rate_limited
                else base_delay * (2**retry_attempt) + (0.1 * retry_attempt)
            )
            retried = True
            if retry_event_callback is not None:
                retry_event_callback("retry", failure_kind)
            time.sleep(delay)
    raise AssertionError("retry loop exhausted without returning or raising")
