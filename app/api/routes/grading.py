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

# ``/submit-multi-by-image`` performs one identify pass plus one
# grading pass PER detected question. Each grading pass can take
# 30-60s on a cold GPU; for the 5-question cap the wall-clock can
# reach ~5 minutes. Default 300s gives comfortable headroom without
# encouraging truly-hung requests to linger forever.
_GRADING_MULTI_TIMEOUT_SECONDS = float(
    os.getenv("GRADING_MULTI_TIMEOUT_SECONDS", "300")
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
    safety_net_engaged: bool = Field(
        default=False,
        description=(
            "True when the final-answer safety net had to upgrade this "
            "response to full marks - i.e. the VLM flagged one or more "
            "intermediate steps but the student's last-line transcription "
            "matched the expected final answer, so we awarded credit for "
            "the correct result despite per-step OCR uncertainty. "
            "Frontends should render a 'teacher review recommended' badge "
            "when this is True. False on clean VLM output and on the "
            "manual ``/submit`` path (which has no expected final answer "
            "to compare against)."
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
            "empty otherwise. Frontend should let the student pick one "
            "and re-submit via ``/submit`` (manual override) or a future "
            "``/submit-by-item`` endpoint."
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


@router.post(
    "/submit-multi-by-image",
    response_model=MultiGradingResponse,
    responses={422: {"model": NoMatchDetail}},
    summary=(
        "Auto-grade a page containing MULTIPLE student answers. Runs one "
        "identify pass to list every question on the page, then grades "
        "each one independently against the same ``file_id``."
    ),
)
async def submit_multi_by_image(
    image: UploadFile = File(
        ..., description="Student's handwritten page with work for 1-5 questions."
    ),
    file_id: str = Form(
        ...,
        description=(
            "UUID of the ``AnswerKeyFile`` all detected questions will "
            "be graded against. Every detected question must belong to "
            "this file."
        ),
    ),
    max_score: int = Form(
        10,
        ge=1,
        le=1000,
        description=(
            "Maximum score per question - applied uniformly to every "
            "detected question. If questions in your exam have different "
            "max scores, grade them one at a time via ``/submit-by-image``."
        ),
    ),
    subject: str = Form("math", description="Subject hint used as grading context."),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role("ROLE_INSTRUCTOR", "ROLE_ADMIN")),
) -> MultiGradingResponse:
    """Grade every question the student answered on a single page.

    Pipeline:

    1. One short Qwen2.5-VL pass enumerates up to
       ``GRADING_MAX_QUESTIONS_PER_IMAGE`` (default 5) distinct question
       labels on the page.
    2. For each detected label, resolve the matching ``AnswerKeyItem``
       via :func:`~app.services.question_resolver.resolve_question`
       (SQL-first, semantic fallback).
    3. For each resolved item, run the full grading pass with
       ``other_questions_on_page`` set so the VLM only grades this
       question's lines and ignores the others.

    Per-question failures (unresolved label, ambiguous match, grading
    error) are reported inline in ``results[i].status`` instead of
    failing the whole request. The overall endpoint only returns 422
    when zero questions could be detected, or 422 / 504 when the
    identify pass itself fails.
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
        "grading-multi: user=%s file_id=%s bytes=%d subject=%s max_score=%d",
        user.preferred_username or user.sub,
        file_uuid,
        len(data),
        subject,
        max_score,
    )

    # ---------------------------------------------------------------- stage 1: identify all
    try:
        detections = await asyncio.wait_for(
            identify_all_questions(image_bytes=data),
            timeout=_GRADING_BY_IMAGE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "grading-multi: identify_all timed out after %.1fs",
            _GRADING_BY_IMAGE_TIMEOUT_SECONDS,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                f"Multi-question identification exceeded the "
                f"{_GRADING_BY_IMAGE_TIMEOUT_SECONDS:.0f}s budget. Try "
                "again with a smaller or clearer image."
            ),
        )
    except GradingServiceUnavailable as exc:
        logger.error("grading-multi: service unavailable (identify): %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Grading service unavailable",
        )
    except GradingBusy as exc:
        logger.warning("grading-multi: busy at identify_all: %s", exc)
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

    if not detections:
        logger.info("grading-multi: no questions detected on page for file_id=%s", file_uuid)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "reason": "no_question_identifier",
                "message": (
                    "No question labels could be read from the page. "
                    "Ensure the student wrote a clear question number "
                    "(e.g. '0003.', 'Q3') at the top of each answer."
                ),
                "candidates": [],
            },
        )

    # Hard cap even if the underlying service is misconfigured.
    detections = detections[:_MAX_QUESTIONS_PER_IMAGE]
    all_labels = [str(d.get("question_number") or "") for d in detections]
    logger.info(
        "grading-multi: detected %d question(s): %r",
        len(detections),
        all_labels,
    )

    # ---------------------------------------------------------------- stage 2: per-question resolve + grade
    results: list[PerQuestionResult] = []

    for idx, detection in enumerate(detections):
        qn = str(detection.get("question_number") or "")
        rc_raw = str(detection.get("confidence") or "low")
        rc: Literal["high", "medium", "low"] = (
            rc_raw if rc_raw in {"high", "medium", "low"} else "low"  # type: ignore[assignment]
        )
        others = [label for label in all_labels if label and label != qn]

        logger.info(
            "grading-multi: %d/%d resolving label=%r (others_on_page=%r)",
            idx + 1, len(detections), qn, others,
        )

        # --- 2a: resolve -------------------------------------------
        match_result, failure_row = _resolve_or_capture(
            db, file_uuid=file_uuid, detection=detection
        )
        if failure_row is not None:
            results.append(failure_row)
            logger.info(
                "grading-multi: %d/%d resolve failed status=%s reason=%r",
                idx + 1, len(detections), failure_row.status, failure_row.error,
            )
            continue

        assert match_result is not None  # mypy
        question_text, answer_key_text = build_answer_key_from_item(match_result.item)
        expected_final_answer = (
            match_result.item.normalized_answer or match_result.item.final_answer
        )

        # --- 2b: grade ---------------------------------------------
        try:
            grade_result = await asyncio.wait_for(
                grade_answer(
                    image_bytes=data,
                    question=question_text,
                    answer_key=answer_key_text,
                    max_score=max_score,
                    subject=subject,
                    question_label=match_result.item.question_no,
                    expected_final_answer=expected_final_answer,
                    other_questions_on_page=others,
                ),
                timeout=_GRADING_BY_IMAGE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "grading-multi: %d/%d grade timeout for label=%r",
                idx + 1, len(detections), qn,
            )
            results.append(
                PerQuestionResult(
                    detected_question_number=qn,
                    status="error",
                    read_confidence=rc,
                    grade=None,
                    error=(
                        f"Grading exceeded the "
                        f"{_GRADING_BY_IMAGE_TIMEOUT_SECONDS:.0f}s budget."
                    ),
                )
            )
            continue
        except GradingServiceUnavailable as exc:
            # Model state is now bad for ALL remaining questions;
            # record this one and bail out of the loop so the client
            # sees the rest as untried rather than spuriously errored.
            logger.error("grading-multi: service unavailable mid-loop: %s", exc)
            results.append(
                PerQuestionResult(
                    detected_question_number=qn,
                    status="error",
                    read_confidence=rc,
                    grade=None,
                    error="Grading service unavailable",
                )
            )
            break
        except GradingBusy as exc:
            logger.warning("grading-multi: busy on label=%r: %s", qn, exc)
            results.append(
                PerQuestionResult(
                    detected_question_number=qn,
                    status="error",
                    read_confidence=rc,
                    grade=None,
                    error=(
                        "GPU busy, this question was not graded. "
                        "Retry this question individually."
                    ),
                )
            )
            continue
        except ValueError as exc:
            # Image decode issues should have been caught at stage 0;
            # a ValueError here is exceptional.
            logger.exception("grading-multi: unexpected ValueError for %r", qn)
            results.append(
                PerQuestionResult(
                    detected_question_number=qn,
                    status="error",
                    read_confidence=rc,
                    grade=None,
                    error=str(exc),
                )
            )
            continue
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda" in msg:
                logger.exception(
                    "grading-multi: CUDA/runtime failure on label=%r", qn
                )
                results.append(
                    PerQuestionResult(
                        detected_question_number=qn,
                        status="error",
                        read_confidence=rc,
                        grade=None,
                        error="GPU out of memory",
                    )
                )
                continue
            logger.exception(
                "grading-multi: unexpected runtime error on label=%r", qn
            )
            results.append(
                PerQuestionResult(
                    detected_question_number=qn,
                    status="error",
                    read_confidence=rc,
                    grade=None,
                    error="Unexpected grading failure",
                )
            )
            continue

        # --- 2c: attach match info + record success ---------------
        grade_result["match"] = MatchInfo(
            matched_item_id=str(match_result.item.id),
            matched_question_no=match_result.item.question_no,
            match_method=match_result.match_method,  # type: ignore[arg-type]
            similarity_score=float(match_result.similarity_score),
            read_confidence=match_result.read_confidence,  # type: ignore[arg-type]
        ).model_dump()

        try:
            grade_obj = GradingByImageResponse(**grade_result)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "grading-multi: response-model validation failed for label=%r", qn
            )
            results.append(
                PerQuestionResult(
                    detected_question_number=qn,
                    status="error",
                    read_confidence=rc,
                    grade=None,
                    error=f"Response validation failed: {exc}",
                )
            )
            continue

        results.append(
            PerQuestionResult(
                detected_question_number=qn,
                status="graded",
                read_confidence=rc,
                grade=grade_obj,
                error=None,
            )
        )
        logger.info(
            "grading-multi: %d/%d graded label=%r item=%s score=%d/%d",
            idx + 1, len(detections), qn, match_result.item.id,
            grade_obj.student_score, grade_obj.max_score,
        )

    total_graded = sum(1 for r in results if r.status == "graded")
    total_failed = len(results) - total_graded

    # Zero success -> 422 so the client gets a clear failure rather
    # than a 200 full of errors.
    if total_graded == 0:
        logger.warning(
            "grading-multi: all %d detected question(s) failed; returning 422",
            len(detections),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "reason": "ambiguous_match",
                "message": (
                    "Detected questions on the page but none of them could "
                    "be graded. Review ``results`` below for per-question "
                    "reasons."
                ),
                "candidates": [],
                "results": [r.model_dump() for r in results],
            },
        )

    logger.info(
        "grading-multi: file_id=%s done detected=%d graded=%d failed=%d",
        file_uuid, len(detections), total_graded, total_failed,
    )

    return MultiGradingResponse(
        file_id=str(file_uuid),
        total_detected=len(detections),
        total_graded=total_graded,
        total_failed=total_failed,
        results=results,
    )
