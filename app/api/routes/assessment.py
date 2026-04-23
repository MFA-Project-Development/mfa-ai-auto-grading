"""Assessment-API passthrough routes.

These endpoints proxy selected reads from the Phase-One Assessment API
(``https://phase-one-api.dara-it.site/api/v1/``). The caller's bearer
token is forwarded as-is to the upstream; since both systems share the
same Keycloak realm, no extra exchange is required.

Router-level ``Depends(get_current_user)`` (applied in ``app/main.py``)
still validates the incoming token first, so we never hand an unchecked
token to the upstream.
"""

from __future__ import annotations

import logging
import uuid
from typing import Literal

import json as json_lib

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field, field_validator

from app.core.security import get_bearer_token
from app.services.assessment_client import (
    AssessmentAPIClient,
    AssessmentAPIError,
    AssessmentAPIStatusError,
    get_assessment_client,
)


# --- request schemas --------------------------------------------------------
# Kept inline (same pattern used by ``app/api/routes/auth.py``) because
# these DTOs are only consumed by this router.


class BulkAnswerUpdate(BaseModel):
    """One entry in the ``PUT /submissions/{id}/answers/bulk`` payload."""

    answerId: uuid.UUID
    pointsAwarded: float = Field(ge=0)
    paperId: uuid.UUID


class AnnotationCreate(BaseModel):
    """Body for ``POST /answers/{id}/annotations``.

    ``contentJson`` is an opaque *serialized JSON* blob - the upstream
    re-parses it server-side and responds ``500`` when it isn't valid
    JSON (observed empirically: the Swagger placeholder ``"string"``
    fails, ``"{}"`` succeeds). Validating here converts what would be
    an upstream ``500`` into a client-friendly ``422``.
    """

    contentJson: str = Field(min_length=1)

    @field_validator("contentJson")
    @classmethod
    def _must_be_json(cls, value: str) -> str:
        try:
            json_lib.loads(value)
        except ValueError as exc:
            raise ValueError(
                "contentJson must be a JSON-serialized string "
                "(e.g. '{}', '[]', or a stringified editor document)."
            ) from exc
        return value


class FeedbackCreate(BaseModel):
    """Body for ``POST /answers/{id}/feedbacks``.

    ``annotationId`` is required by the upstream; pair the feedback
    with the annotation it refers to so the grader UI can highlight
    them together.
    """

    comment: str = Field(min_length=1)
    annotationId: uuid.UUID

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/assessment", tags=["Assessment API"])

# Response headers worth forwarding from the upstream on binary
# responses. ``content-length`` is deliberately excluded - Starlette
# will set it from the returned body, and forwarding a stale value can
# confuse clients. Hop-by-hop headers (``connection``,
# ``transfer-encoding``, etc.) are also filtered out.
_FORWARDABLE_BINARY_HEADERS: frozenset[str] = frozenset(
    {
        "content-type",
        "content-disposition",
        "cache-control",
        "etag",
        "last-modified",
        "expires",
    }
)


# ------------------------------------------------------------------ helpers
async def _proxy_json(
    client: AssessmentAPIClient,
    method: str,
    path: str,
    token: str,
    *,
    params: dict[str, object] | None = None,
    json: object | None = None,
) -> object:
    """Forward a JSON-in/JSON-out call to the upstream.

    Centralises the ``AssessmentAPIError`` -> ``HTTPException`` mapping
    so every verb handler above stays a four-liner. The upstream
    status code and body are re-emitted verbatim on non-2xx replies so
    the caller can surface them directly in the UI.
    """
    try:
        return await client.request_json(
            method,
            path,
            params=params,
            json=json,
            headers={"Authorization": f"Bearer {token}"},
        )
    except AssessmentAPIStatusError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.body) from exc
    except AssessmentAPIError as exc:
        logger.exception("Assessment API call failed: %s %s", method, path)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
