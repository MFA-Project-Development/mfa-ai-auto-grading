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


@router.get("/classes")
async def list_classes(
    page: int = Query(default=1, ge=1, description="1-based page index."),
    size: int = Query(default=10, ge=1, le=100, description="Page size."),
    property: Literal["NAME", "CREATED_AT", "UPDATED_AT"] = Query(
        default="NAME",
        description="Field to sort by (upstream vocabulary).",
    ),
    direction: Literal["ASC", "DESC"] = Query(
        default="ASC", description="Sort direction."
    ),
    token: str = Depends(get_bearer_token),
    client: AssessmentAPIClient = Depends(get_assessment_client),
) -> object:
    """Proxy ``GET /classes`` on the Phase-One Assessment API.

    Mirrors::

        curl -X GET \\
          'https://phase-one-api.dara-it.site/api/v1/classes?page=1&size=10&property=NAME&direction=ASC' \\
          -H 'Authorization: Bearer <token>'

    Upstream failures are mapped to HTTP status codes the front-end can
    surface directly:

    * Non-2xx replies are re-raised with the upstream status and body.
    * Network / timeout / decode errors collapse to ``502 Bad Gateway``.
    """
    params = {
        "page": page,
        "size": size,
        "property": property,
        "direction": direction,
    }

    try:
        return await client.get_json(
            "classes",
            params=params,
            # Per-call header wins over the client's default Authorization.
            # We forward the caller's token verbatim so the upstream sees
            # the same identity.
            headers={"Authorization": f"Bearer {token}"},
        )
    except AssessmentAPIStatusError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.body) from exc
    except AssessmentAPIError as exc:
        logger.exception("Assessment API call failed: GET /classes")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get(
    "/assessments/{assessment_id}/submissions/{submission_id}/papers",
)
async def list_submission_papers(
    assessment_id: uuid.UUID,
    submission_id: uuid.UUID,
    token: str = Depends(get_bearer_token),
    client: AssessmentAPIClient = Depends(get_assessment_client),
) -> object:
    """Proxy ``GET /assessments/{id}/submissions/{id}/papers`` on the upstream.

    Mirrors::

        curl -X GET \\
          'https://phase-one-api.dara-it.site/api/v1/assessments/{assessment_id}/submissions/{submission_id}/papers' \\
          -H 'Authorization: Bearer <token>'

    Typed as ``uuid.UUID`` on both path params so malformed ids are
    rejected with ``422`` before any upstream round-trip.
    """
    path = (
        f"assessments/{assessment_id}"
        f"/submissions/{submission_id}/papers"
    )

    try:
        return await client.get_json(
            path,
            headers={"Authorization": f"Bearer {token}"},
        )
    except AssessmentAPIStatusError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.body) from exc
    except AssessmentAPIError as exc:
        logger.exception("Assessment API call failed: GET %s", path)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/files/preview/{filename}")
async def preview_file(
    filename: str,
    token: str = Depends(get_bearer_token),
    client: AssessmentAPIClient = Depends(get_assessment_client),
) -> Response:
    """Proxy ``GET /files/preview/{filename}`` on the upstream.

    Mirrors::

        curl -X GET \\
          'https://phase-one-api.dara-it.site/api/v1/files/preview/<filename>' \\
          -H 'Authorization: Bearer <token>'

    The upstream returns the raw file bytes (e.g. ``image/jpeg``) with
    an ``inline`` ``Content-Disposition``. We pass the body through
    unchanged and mirror the relevant response headers so browsers can
    render the preview directly from this endpoint.
    """
    # ``filename`` comes from the URL path; reject anything that would
    # let the caller escape the ``files/preview/`` prefix on the
    # upstream (``..``, slashes, backslashes). The upstream uses flat
    # UUID-like filenames so this is a safe, conservative check.
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    path = f"files/preview/{filename}"

    try:
        upstream = await client.request(
            "GET",
            path,
            headers={"Authorization": f"Bearer {token}"},
        )
    except AssessmentAPIStatusError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.body) from exc
    except AssessmentAPIError as exc:
        logger.exception("Assessment API call failed: GET %s", path)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    forwarded: dict[str, str] = {
        name: value
        for name, value in upstream.headers.items()
        if name.lower() in _FORWARDABLE_BINARY_HEADERS
    }
    media_type = upstream.headers.get("content-type", "application/octet-stream")

    return Response(
        content=upstream.content,
        media_type=media_type,
        headers=forwarded,
    )


# ============================================================ submissions
@router.get("/submissions/{submission_id}/answers")
async def list_submission_answers(
    submission_id: uuid.UUID,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1, le=100),
    property: Literal["CREATED_AT", "UPDATED_AT"] = Query(default="CREATED_AT"),
    direction: Literal["ASC", "DESC"] = Query(default="DESC"),
    token: str = Depends(get_bearer_token),
    client: AssessmentAPIClient = Depends(get_assessment_client),
) -> object:
    """Proxy ``GET /submissions/{id}/answers`` on the upstream."""
    path = f"submissions/{submission_id}/answers"
    params = {
        "page": page,
        "size": size,
        "property": property,
        "direction": direction,
    }
    return await _proxy_json(
        client, "GET", path, token, params=params
    )


@router.put("/submissions/{submission_id}/answers/bulk")
async def bulk_update_submission_answers(
    submission_id: uuid.UUID,
    payload: list[BulkAnswerUpdate],
    token: str = Depends(get_bearer_token),
    client: AssessmentAPIClient = Depends(get_assessment_client),
) -> object:
    """Proxy ``PUT /submissions/{id}/answers/bulk`` on the upstream.

    The request body is a JSON array; each entry carries an
    ``answerId`` (the row being graded), its ``pointsAwarded`` score
    and the ``paperId`` the grader was viewing when they assigned the
    score. FastAPI validates each element as a :class:`BulkAnswerUpdate`
    before any network call.
    """
    path = f"submissions/{submission_id}/answers/bulk"
    body = [item.model_dump(mode="json") for item in payload]
    return await _proxy_json(client, "PUT", path, token, json=body)


# =============================================================== answers
@router.post("/answers/{answer_id}/annotations")
async def create_answer_annotation(
    answer_id: uuid.UUID,
    payload: AnnotationCreate,
    token: str = Depends(get_bearer_token),
    client: AssessmentAPIClient = Depends(get_assessment_client),
) -> object:
    """Proxy ``POST /answers/{id}/annotations`` on the upstream."""
    path = f"answers/{answer_id}/annotations"
    return await _proxy_json(
        client, "POST", path, token, json=payload.model_dump(mode="json")
    )


@router.post("/answers/{answer_id}/feedbacks")
async def create_answer_feedback(
    answer_id: uuid.UUID,
    payload: FeedbackCreate,
    token: str = Depends(get_bearer_token),
    client: AssessmentAPIClient = Depends(get_assessment_client),
) -> object:
    """Proxy ``POST /answers/{id}/feedbacks`` on the upstream."""
    path = f"answers/{answer_id}/feedbacks"
    return await _proxy_json(
        client, "POST", path, token, json=payload.model_dump(mode="json")
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
