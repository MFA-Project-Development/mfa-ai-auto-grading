"""Auto-grading HTTP endpoints.

Exposes the auto-detect grading routes
(``POST /api/v1/grading/submit-by-image`` and
``POST /api/v1/grading/submit-multi-by-image``) that accept a handwritten
answer image plus a ``file_id`` and return machine-generated grades
produced by Qwen2.5-VL-7B-Instruct (see
:mod:`app.services.grading_service`).

Authorisation
-------------

Every endpoint is restricted to callers holding ``ROLE_INSTRUCTOR`` or
``ROLE_ADMIN``. Students may **not** call these endpoints - the
:func:`app.core.security.require_role` dependency rejects any token
without one of those roles with HTTP 403.

Error mapping
-------------

=================================================  ==================
Condition                                          HTTP status
=================================================  ==================
Empty image / unreadable bytes                     400
Unsupported content-type (not jpg/png/webp)        400
Image payload exceeds ``_MAX_IMAGE_BYTES``         400
Model not loaded / CUDA OOM at startup             503
Generation exceeds the configured timeout          504
=================================================  ==================
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.models import CurrentUser
from app.core.security import require_role
from app.db.session import get_db
from app.services.grading_service import (
    GradingBusy,
    GradingServiceUnavailable,
    GradingTimeout,
    grade_answer,
    identify_all_questions,
    identify_question,
)
from app.services.question_resolver import (
    AmbiguousMatchError,
    build_answer_key_from_item,
    resolve_question,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/grading", tags=["Grading"])


# --------------------------------------------------------------------- constants

# 10 MiB upper bound on inbound images. Qwen2.5-VL tokenises images into
# variable-length visual token sequences; oversized inputs can easily
# blow past the model's context window *and* the GPU allocator.
_MAX_IMAGE_BYTES = 10 * 1024 * 1024

# Whitelist of inbound content types. We intentionally do NOT trust the
# file extension - browsers routinely send generic ``application/octet-stream``
# for drag-and-drop uploads, so we also sniff the magic bytes below.
_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }
)

# Minimal magic-byte sniffer. Mirrors the whitelist above; used as a
# belt-and-suspenders check when the client-declared content-type is
# generic or obviously wrong.
_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"RIFF", "image/webp"),  # followed by 'WEBP' at offset 8
)

# ``/submit-by-image`` does two Qwen passes (identify then grade) and
# needs a generous end-to-end budget. The identify pass is cheap
# (~2-5s) but the grading pass can still take 30-60s on a cold GPU;
# 180s gives both rooms without being so long that stuck requests back
# up the executor pool. ``/submit-multi-by-image`` reuses the same
# per-question budget, capped by ``_MAX_QUESTIONS_PER_IMAGE``.
_GRADING_BY_IMAGE_TIMEOUT_SECONDS = float(
    os.getenv("GRADING_BY_IMAGE_TIMEOUT_SECONDS", "180")
)

# Safety cap on how many per-question grading passes a single request
# can trigger. Matches the grading_service cap on identify_all_questions.
# Prevents a pathological page (or hallucinated question list) from
# exhausting the GPU budget.
_MAX_QUESTIONS_PER_IMAGE = int(
    os.getenv("GRADING_MAX_QUESTIONS_PER_IMAGE", "5")
)


# --------------------------------------------------------------------- response model


class StepResult(BaseModel):
    """A single step extracted from the student's handwritten solution."""

    step_number: int = Field(
        ..., description="1-based position of this step in the student's written work."
    )
    student_wrote: str = Field(
        ..., description="Exact text the model transcribed from the handwriting for this step."
    )
    expected: str = Field(
        ..., description="What this step should look like according to the answer key."
    )
    is_correct: bool = Field(
        ..., description="True when the student's step matches the expected step."
    )
    error: str | None = Field(
        default=None,
        description="Specific description of the mistake on this step, or null when correct.",
    )


class GradingResponse(BaseModel):
    """Base grading envelope returned by the grading endpoints.

    The response is structured around the student's solution **steps** so
    frontends can highlight exactly which line went wrong, rather than
    just surfacing a final score. The auto-detect endpoints extend this
    shape with a ``match`` block (see :class:`GradingByImageResponse`).
    """

    student_score: int = Field(
        ..., ge=0, description="Awarded score, clamped to [0, max_score]."
    )
    max_score: int = Field(
        ..., ge=1, description="Echo of the request's ``max_score`` field."
    )
    extracted_steps: list[StepResult] = Field(
        default_factory=list,
        description="Per-step breakdown of the student's work as read from the image.",
    )
    first_error_step: int | None = Field(
        default=None,
        description="1-based index of the first step that introduced an error, or null if all correct.",
    )
    error_summary: str | None = Field(
        default=None,
        description="One-sentence description of the root mistake (null on full-credit answers).",
    )
    method_correct: bool = Field(
        ..., description="True when the overall approach is valid, even if arithmetic is off."
    )
    feedback: str = Field(
        ..., description="Human-readable, step-by-step breakdown of marks awarded and deducted."
    )
    is_correct: bool = Field(
        ..., description="True when the model awarded the full ``max_score``."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Model's self-reported confidence based on handwriting legibility."
    )
    score_breakdown: str = Field(
        ..., description='Quick step tally, e.g. "3/4 steps correct".'
    )
    graded_by: str = Field(
        ..., description="Identifier of the model that produced this grade."
    )
    safety_net_engaged: bool = Field(
        default=False,
        description=(
            "True when the final-answer safety net had to upgrade this "
            "response to full marks - i.e. the VLM flagged one or more "
            "intermediate steps but the student's last-line transcription "
            "matched the expected final answer, so we awarded credit for "
            "the correct result despite per-step OCR uncertainty. "
            "Frontends should render a 'teacher review recommended' badge "
            "when this is True."
        ),
    )


# --------------------------------------------------------------------- helpers


def _sniff_content_type(head: bytes) -> str | None:
    """Return a MIME type inferred from the leading magic bytes, or None."""
    for prefix, mime in _MAGIC_SIGNATURES:
        if head.startswith(prefix):
            # Extra WebP guard: RIFF container must carry the WEBP tag.
            if mime == "image/webp" and head[8:12] != b"WEBP":
                continue
            return mime
    return None


def _validate_image(upload: UploadFile, data: bytes) -> None:
    """Enforce size + format constraints on the incoming upload.

    Raises
    ------
    HTTPException
        ``400`` if the payload is empty, too large, or not a supported
        image format.
    """
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image payload is empty.",
        )
    if len(data) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Image too large: {len(data)} bytes "
                f"(limit {_MAX_IMAGE_BYTES} bytes)."
            ),
        )

    declared = (upload.content_type or "").lower().split(";", 1)[0].strip()
    sniffed = _sniff_content_type(data[:16])

    # Accept the upload if *either* the declared type or the magic-byte
    # sniff says it's a supported image. This is lenient enough for
    # octet-stream clients while still rejecting e.g. PDFs or text files.
    if declared not in _ALLOWED_CONTENT_TYPES and sniffed is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unsupported image format. Allowed: jpg, png, webp. "
                f"Received content-type={declared!r}."
            ),
        )


# ====================================================================== auto-detect endpoint


class MatchInfo(BaseModel):
    """How the system resolved the student's image to an answer-key item."""

    matched_item_id: str = Field(
        ..., description="``AnswerKeyItem.id`` (UUID) that was graded against."
    )
    matched_question_no: str = Field(
        ...,
        description=(
            "Canonical ``question_no`` of the matched item as stored in "
            "``answer_key_items``. May differ from what the student wrote "
            "(e.g. student wrote ``Q3`` but DB stores ``0003``)."
        ),
    )
    match_method: Literal["question_number", "semantic"] = Field(
        ...,
        description=(
            "``question_number`` when resolved via exact SQL lookup on "
            "``(file_id, question_no)``; ``semantic`` when resolved via "
            "bge-m3 + Chroma similarity search."
        ),
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "``1.0`` for exact question-number matches, ``1 - cosine_distance`` "
            "for semantic matches. Always in ``[0, 1]``."
        ),
    )
    read_confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description=(
            "How legible the identify-pass found the top of the page. "
            "Echoed here so frontends can warn when grading a low-confidence "
            "read."
        ),
    )


class GradingByImageResponse(GradingResponse):
    """Response envelope for ``POST /api/v1/grading/submit-by-image``.

    Same shape as :class:`GradingResponse` plus a ``match`` block that
    tells the frontend *which* question the auto-detector chose and
    how it got there.
    """

    match: MatchInfo = Field(
        ..., description="Resolver metadata explaining which item was graded."
    )


class AmbiguousCandidate(BaseModel):
    """One candidate returned to the client on a 422 ambiguous match."""

    item_id: str = Field(..., description="``AnswerKeyItem.id`` (UUID).")
    question_no: str = Field(..., description="Canonical ``question_no`` for display.")
    problem_text: str = Field(
        ...,
        description="Problem statement preview (truncated) so the UI can label the candidate.",
    )
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="``1 - cosine_distance`` for this candidate."
    )


class NoMatchDetail(BaseModel):
    """Body of the HTTP 422 when the auto-detector cannot confidently match.

    The frontend should present the ``candidates`` list (if any) to the
    student and surface the ambiguity so they can retry with a clearer
    image or a legible question label.
    """

    reason: Literal["ambiguous_match", "no_question_identifier"] = Field(
        ...,
        description=(
            "``no_question_identifier`` when neither question number nor "
            "problem text was detected; ``ambiguous_match`` when semantic "
            "search was inconclusive."
        ),
    )
    message: str = Field(..., description="Human-readable explanation for logs / UI.")
    candidates: list[AmbiguousCandidate] = Field(
        default_factory=list,
        description="Up to top-K alternatives the student can pick from.",
    )


# ====================================================================== multi-question auto-detect endpoint


class PerQuestionResult(BaseModel):
    """One outcome entry inside a :class:`MultiGradingResponse`.

    The ``status`` field tells the frontend how to render this row:

    - ``"graded"``: ``grade`` is populated with a full
      :class:`GradingByImageResponse`; ``error`` / ``candidates`` are
      empty.
    - ``"not_found"``: the student wrote a label but there was no
      matching ``AnswerKeyItem`` in this ``file_id`` (and no semantic
      fallback fired). ``grade`` is ``null``; ``error`` carries the
      resolver message.
    - ``"ambiguous"``: semantic resolver was inconclusive. ``grade``
      is ``null``; ``candidates`` holds up to top-K alternatives the
      user can pick from.
    - ``"error"``: grading itself failed (timeout, GPU OOM, unexpected
      exception). ``grade`` is ``null``; ``error`` carries a short
      message.
    """

    detected_question_number: str = Field(
        ...,
        description=(
            "Question label the identify pass read from the page, as "
            "written by the student (may differ in format from the "
            "canonical ``AnswerKeyItem.question_no``)."
        ),
    )
    status: Literal["graded", "not_found", "ambiguous", "error"] = Field(
        ..., description="Outcome for this specific question."
    )
    read_confidence: Literal["high", "medium", "low"] = Field(
        default="low",
        description="Identify pass's confidence that this label is legible.",
    )
    grade: GradingByImageResponse | None = Field(
        default=None,
        description="Full grading envelope when ``status == 'graded'``; null otherwise.",
    )
    error: str | None = Field(
        default=None,
        description="Human-readable explanation when ``status != 'graded'``.",
    )
    candidates: list[AmbiguousCandidate] = Field(
        default_factory=list,
        description=(
            "Top-K similarity candidates when ``status == 'ambiguous'``; "
            "empty otherwise. Frontend should surface this list so the "
            "student can retake the image with a clearer question label."
        ),
    )


class MultiGradingResponse(BaseModel):
    """Envelope returned by ``POST /api/v1/grading/submit-multi-by-image``.

    Contains one :class:`PerQuestionResult` per question the identify
    pass detected on the student's page, in top-to-bottom order.
    """

    file_id: str = Field(
        ..., description="Echo of the request's ``file_id`` parameter."
    )
    total_detected: int = Field(
        ...,
        ge=0,
        description=(
            "Number of distinct question labels the identify pass read "
            "from the page (post-dedup, capped at "
            "``GRADING_MAX_QUESTIONS_PER_IMAGE``)."
        ),
    )
    total_graded: int = Field(
        ..., ge=0, description="Number of ``results`` entries with ``status == 'graded'``."
    )
    total_failed: int = Field(
        ..., ge=0, description="Number of ``results`` entries with ``status != 'graded'``."
    )
    results: list[PerQuestionResult] = Field(
        default_factory=list,
        description="One entry per detected question, in top-to-bottom order.",
    )


def _resolve_or_capture(
    db: Session,
    *,
    file_uuid: uuid.UUID,
    detection: dict[str, object],
) -> tuple[object | None, PerQuestionResult | None]:
    """Call ``resolve_question`` and translate exceptions to a result row.

    Returns ``(match_result, None)`` on success, or
    ``(None, PerQuestionResult)`` on a structured failure (the failure
    row is ready to append to the response list).
    """
    qn = str(detection.get("question_number") or "")
    rc_raw = str(detection.get("confidence") or "low")
    rc: Literal["high", "medium", "low"] = (
        rc_raw if rc_raw in {"high", "medium", "low"} else "low"  # type: ignore[assignment]
    )

    try:
        match_result = resolve_question(
            db=db,
            file_id=file_uuid,
            detected_number=qn or None,
            detected_text=detection.get("problem_text_preview") or None,  # type: ignore[arg-type]
            read_confidence=rc,
        )
        return match_result, None
    except AmbiguousMatchError as exc:
        candidates = [
            AmbiguousCandidate(
                item_id=c.item_id,
                question_no=c.question_no,
                problem_text=c.problem_text,
                similarity_score=c.similarity_score,
            )
            for c in exc.candidates
        ]
        status_value: Literal["not_found", "ambiguous"] = (
            "not_found" if exc.reason == "no_question_identifier" else "ambiguous"
        )
        return None, PerQuestionResult(
            detected_question_number=qn,
            status=status_value,
            read_confidence=rc,
            grade=None,
            error=exc.message,
            candidates=candidates,
        )


