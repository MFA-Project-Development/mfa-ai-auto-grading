"""Assessment-API client exceptions.

Wrapping ``httpx`` errors in our own hierarchy keeps route handlers and
other services free from SDK-specific imports and gives logs a clear,
searchable failure taxonomy.
"""

from __future__ import annotations


class AssessmentAPIError(Exception):
    """Base class for all assessment-API failures."""


class AssessmentAPIRequestError(AssessmentAPIError):
    """Network-level failure (DNS, connection refused, TLS, etc.)."""


class AssessmentAPITimeoutError(AssessmentAPIError):
    """Upstream did not respond within the configured timeout."""


class AssessmentAPIStatusError(AssessmentAPIError):
    """Upstream replied with a non-2xx status code.

    Exposes ``status_code``, ``url``, and the decoded ``body`` (either
    parsed JSON or raw text) so callers can branch on the failure
    without reparsing the response.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        url: str,
        body: object | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url
        self.body = body


class AssessmentAPIDecodeError(AssessmentAPIError):
    """The response body could not be decoded as expected (e.g. bad JSON)."""
