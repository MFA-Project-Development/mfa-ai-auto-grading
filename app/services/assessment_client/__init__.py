"""Assessment-API integration package.

A thin, reusable async client for the Phase-One Assessment API
(``https://phase-one-api.dara-it.site/api/v1/``). Prefer this over
ad-hoc ``httpx`` usage so retries, timeouts, auth, and exception
wrapping stay consistent across the codebase.
"""

from app.services.assessment_client.client import (
    AssessmentAPIClient,
    AssessmentAPIClientSync,
    JSONType,
    get_assessment_client,
)
from app.services.assessment_client.exceptions import (
    AssessmentAPIDecodeError,
    AssessmentAPIError,
    AssessmentAPIRequestError,
    AssessmentAPIStatusError,
    AssessmentAPITimeoutError,
)

__all__ = [
    "AssessmentAPIClient",
    "AssessmentAPIClientSync",
    "JSONType",
    "get_assessment_client",
    "AssessmentAPIError",
    "AssessmentAPIRequestError",
    "AssessmentAPITimeoutError",
    "AssessmentAPIStatusError",
    "AssessmentAPIDecodeError",
]
