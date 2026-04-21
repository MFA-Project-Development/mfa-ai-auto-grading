"""Auto-grading HTTP endpoint.

Exposes a single ``POST /api/v1/grading/submit`` route that accepts a
handwritten answer image plus the question / answer-key metadata and
returns a machine-generated grade produced by Qwen2.5-VL-7B-Instruct
(see :mod:`app.services.grading_service`).

Authorisation
-------------

The endpoint is restricted to callers holding ``ROLE_INSTRUCTOR`` or
``ROLE_ADMIN``. Students may **not** call this endpoint - the
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
Generation exceeds ``_GRADING_TIMEOUT_SECONDS``    504
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
    grade_answer,
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

# Hard wall-clock budget for a single grading call.
#
# Cold-start generous budget: on the very first request after process boot
# the vision encoder has to JIT-compile its CUDA kernels (especially on
# newer GPUs like the RTX 50-series where cuDNN has to autotune from
# scratch), which can easily take 30-60s. Subsequent calls complete in
# 5-15s. We default to 120s so the first call doesn't reliably 504.
#
# Override at deploy time with the ``GRADING_TIMEOUT_SECONDS`` env var.
_GRADING_TIMEOUT_SECONDS = float(os.getenv("GRADING_TIMEOUT_SECONDS", "120"))

# ``/submit-by-image`` does two Qwen passes (identify then grade) so it
# needs a larger end-to-end budget than the manual-override ``/submit``
# endpoint. The identify pass is cheap (~2-5s) but the grading pass
# can still take 30-60s on a cold GPU; 180s gives both rooms without
# being so long that stuck requests back up the executor pool.
_GRADING_BY_IMAGE_TIMEOUT_SECONDS = float(
    os.getenv("GRADING_BY_IMAGE_TIMEOUT_SECONDS", "180")
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
    """JSON envelope returned by ``POST /api/v1/grading/submit``.

    The response is structured around the student's solution **steps** so
    frontends can highlight exactly which line went wrong, rather than
    just surfacing a final score.
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


# --------------------------------------------------------------------- endpoint


@router.post(
    "/submit",
    response_model=GradingResponse,
    summary="Auto-grade a student's handwritten answer image.",
)
async def submit_for_grading(
    image: UploadFile = File(
        ..., description="Student's handwritten answer (jpg/png/webp)."
    ),
    question: str = Form(..., description="The original question text."),
    answer_key: str = Form(
        ...,
        description=(
            "FULL worked solution (step-by-step), not just the final answer. "
            'Example: "Step 1: ...  Step 2: ...  Step 3: final answer". '
            "The grader scores each student step against this reference."
        ),
    ),
    max_score: int = Form(10, ge=1, le=1000, description="Maximum score for this question."),
    subject: str = Form("math", description="Subject hint used as grading context."),
    user: CurrentUser = Depends(require_role("ROLE_INSTRUCTOR", "ROLE_ADMIN")),
) -> GradingResponse:
    """Grade one handwritten answer image with Qwen2.5-VL-7B-Instruct.

    The image is read fully into memory (never persisted to disk),
    validated, then handed to :func:`app.services.grading_service.grade_answer`
    which runs the VLM in a thread-pool executor so the event loop stays
    unblocked. The wall-clock timeout is controlled by
    ``GRADING_TIMEOUT_SECONDS`` (default 120s) and maps to HTTP 504.

    Unlike a plain final-answer check, this endpoint grades
    **step-by-step**: the model transcribes every visible line from the
    handwriting, matches each line against the worked solution in
    ``answer_key``, and flags the first step that introduced an error.
    Callers MUST therefore pass a complete worked solution - a single
    final-answer string will yield poor partial-credit and weak
    feedback. Example ``answer_key``::

        Step 1: (2x^2 - xy + 3y^2) + (2x^2 + 2xy + y^2)
        Step 2: Group like terms: (2x^2 + 2x^2) + (-xy + 2xy) + (3y^2 + y^2)
        Step 3: = 4x^2 + xy + 4y^2
    """
    data = await image.read()
    _validate_image(image, data)

    logger.info(
        "grading: user=%s submitting image bytes=%d subject=%s max_score=%d",
        user.preferred_username or user.sub,
        len(data),
        subject,
        max_score,
    )

    try:
        result = await asyncio.wait_for(
            grade_answer(
                image_bytes=data,
                question=question,
                answer_key=answer_key,
                max_score=max_score,
                subject=subject,
            ),
            timeout=_GRADING_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "grading: user=%s request timed out after %.1fs",
            user.preferred_username or user.sub,
            _GRADING_TIMEOUT_SECONDS,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                f"Grading exceeded the {_GRADING_TIMEOUT_SECONDS:.0f}s budget. "
                "Try again with a smaller or clearer image."
            ),
        )
    except GradingServiceUnavailable as exc:
        logger.error("grading: service unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Grading service unavailable",
        )
    except GradingBusy as exc:
        logger.warning("grading: busy (GPU occupied): %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Another grading request is currently running on the GPU. "
                "Please retry in a few seconds."
            ),
            headers={"Retry-After": "10"},
        )
    except ValueError as exc:
        # Raised by grading_service._decode_image on corrupt bytes.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except RuntimeError as exc:
        # Most commonly CUDA OOM mid-generation. Treat as transient 503.
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg:
            logger.exception("grading: CUDA/runtime failure during generation")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Grading service unavailable",
            )
        logger.exception("grading: unexpected runtime error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected grading failure.",
        )

    return GradingResponse(**result)


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
    student and let them pick, then re-submit via ``/submit`` (manual
    override) with the chosen ``answer_key``.
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


@router.post(
    "/submit-by-image",
    response_model=GradingByImageResponse,
    responses={422: {"model": NoMatchDetail}},
    summary=(
        "Auto-grade a student's handwritten answer image, auto-detecting "
        "the question from the top of the page."
    ),
)
async def submit_by_image(
    image: UploadFile = File(
        ..., description="Student's handwritten answer (jpg/png/webp)."
    ),
    file_id: str = Form(
        ...,
        description=(
            "UUID of the ``AnswerKeyFile`` the student is being graded "
            "against. Search for the matching question is scoped to this "
            "file only."
        ),
    ),
    max_score: int = Form(10, ge=1, le=1000, description="Maximum score for this question."),
    subject: str = Form("math", description="Subject hint used as grading context."),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role("ROLE_INSTRUCTOR", "ROLE_ADMIN")),
) -> GradingByImageResponse:
    """Auto-detect which question the student answered, then grade it.

    Pipeline:

    1. Short Qwen2.5-VL pass reads the top of the image and extracts
       either a question number or a copied problem statement.
    2. :func:`~app.services.question_resolver.resolve_question` picks
       the matching ``AnswerKeyItem`` via SQL-first / Chroma-fallback.
       On low confidence this raises 422 with candidate alternatives.
    3. The full grading Qwen pass runs with the resolved item's
       ``problem_text`` / ``solution_steps`` / ``final_answer`` stitched
       into an ``answer_key`` string, exactly as ``/submit`` does.

    Unlike ``/submit`` the client does not supply ``question`` or
    ``answer_key`` - they're resolved server-side from ``file_id``
    plus whatever the student wrote at the top of their page.
    """
    data = await image.read()
    _validate_image(image, data)

    try:
        file_uuid = uuid.UUID(file_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file_id: {file_id!r} is not a UUID.",
        )

    logger.info(
        "grading-by-image: user=%s file_id=%s bytes=%d subject=%s max_score=%d",
        user.preferred_username or user.sub,
        file_uuid,
        len(data),
        subject,
        max_score,
    )

    # ---------------------------------------------------------------- stage 1: identify
    try:
        identification = await asyncio.wait_for(
            identify_question(image_bytes=data),
            timeout=_GRADING_BY_IMAGE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "grading-by-image: identify stage timed out after %.1fs",
            _GRADING_BY_IMAGE_TIMEOUT_SECONDS,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                f"Identification exceeded the {_GRADING_BY_IMAGE_TIMEOUT_SECONDS:.0f}s "
                "budget. Try again with a smaller or clearer image."
            ),
        )
    except GradingServiceUnavailable as exc:
        logger.error("grading-by-image: service unavailable (identify): %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Grading service unavailable",
        )
    except GradingBusy as exc:
        logger.warning("grading-by-image: busy at identify: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Another grading request is currently running on the GPU. "
                "Please retry in a few seconds."
            ),
            headers={"Retry-After": "10"},
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    # ---------------------------------------------------------------- stage 2: resolve
    try:
        match_result = resolve_question(
            db=db,
            file_id=file_uuid,
            detected_number=identification.get("question_number"),
            detected_text=identification.get("problem_text"),
            read_confidence=identification.get("read_confidence", "low"),
        )
    except AmbiguousMatchError as exc:
        candidates = [
            AmbiguousCandidate(
                item_id=c.item_id,
                question_no=c.question_no,
                problem_text=c.problem_text,
                similarity_score=c.similarity_score,
            ).model_dump()
            for c in exc.candidates
        ]
        logger.info(
            "grading-by-image: ambiguous match reason=%s candidates=%d",
            exc.reason,
            len(candidates),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "reason": exc.reason,
                "message": exc.message,
                "candidates": candidates,
            },
        )

    # ---------------------------------------------------------------- stage 3: grade
    question_text, answer_key_text = build_answer_key_from_item(match_result.item)

    logger.info(
        "grading-by-image: resolved item_id=%s question_no=%r method=%s score=%.4f",
        match_result.item.id,
        match_result.item.question_no,
        match_result.match_method,
        match_result.similarity_score,
    )

    # Prefer the normalised answer for the safety-net comparison; fall
    # back to the raw ``final_answer`` column when normalisation wasn't
    # run at ingestion time.
    expected_final_answer = (
        match_result.item.normalized_answer or match_result.item.final_answer
    )

    try:
        grade_result = await asyncio.wait_for(
            grade_answer(
                image_bytes=data,
                question=question_text,
                answer_key=answer_key_text,
                max_score=max_score,
                subject=subject,
                # Known question label - feeds the prompt's
                # "strip this exact label from the top of the page"
                # hint so the VLM does not transcribe e.g. "0003."
                # as part of step 1's math expression.
                question_label=match_result.item.question_no,
                # Known final answer - powers the server-side safety
                # net. If the student's transcribed work contains this
                # string anywhere, grading is upgraded to full marks
                # regardless of intermediate OCR noise.
                expected_final_answer=expected_final_answer,
            ),
            timeout=_GRADING_BY_IMAGE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "grading-by-image: grade stage timed out after %.1fs",
            _GRADING_BY_IMAGE_TIMEOUT_SECONDS,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                f"Grading exceeded the {_GRADING_BY_IMAGE_TIMEOUT_SECONDS:.0f}s budget. "
                "Try again with a smaller or clearer image."
            ),
        )
    except GradingServiceUnavailable as exc:
        logger.error("grading-by-image: service unavailable (grade): %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Grading service unavailable",
        )
    except GradingBusy as exc:
        logger.warning("grading-by-image: busy at grade: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Another grading request is currently running on the GPU. "
                "Please retry in a few seconds."
            ),
            headers={"Retry-After": "10"},
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg:
            logger.exception("grading-by-image: CUDA/runtime failure during grading")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Grading service unavailable",
            )
        logger.exception("grading-by-image: unexpected runtime error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected grading failure.",
        )

    grade_result["match"] = MatchInfo(
        matched_item_id=str(match_result.item.id),
        matched_question_no=match_result.item.question_no,
        match_method=match_result.match_method,  # type: ignore[arg-type]
        similarity_score=float(match_result.similarity_score),
        read_confidence=match_result.read_confidence,  # type: ignore[arg-type]
    ).model_dump()

    return GradingByImageResponse(**grade_result)
