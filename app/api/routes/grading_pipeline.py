"""End-to-end auto-grading pipeline for a single submission.

Exposes ``POST /api/v1/grading/pipeline/run``, an orchestration endpoint
that chains the seven upstream assessment-API calls together with the
local Qwen2.5-VL grading engine into a single flow::

    1. GET  /assessments/{aid}/submissions/{sid}/papers         (upstream)
    2. GET  /files/preview/{filename}  (per paper)              (upstream)
    3. GET  /submissions/{sid}/answers                          (upstream)
    4. <local VLM grading: identify + resolve + grade_answer>
    5. POST /answers/{aid}/annotations (per answer)             (upstream)
    6. POST /answers/{aid}/feedbacks   (per answer)             (upstream)
    7. PUT  /submissions/{sid}/answers/bulk                     (upstream)

Design notes
------------

* The caller's bearer token is forwarded to every upstream call so the
  identity/role checks performed by the assessment API keep working.
* Local VLM grading is serialised on the GPU (``grade_answer`` enforces
  this internally); paper downloads and post-grading annotation /
  feedback creation are parallelised with ``asyncio.gather``.
* Per-paper and per-answer failures are reported inline in the response
  trace instead of aborting the whole pipeline. The only hard-abort
  paths are: (a) upstream papers/answers listing failed, or (b) the
  final bulk update failed.

Authorisation is restricted to ``ROLE_INSTRUCTOR`` / ``ROLE_ADMIN``:
grading is an instructor-side operation and students must never be
able to mutate their own points via this route.
"""

from __future__ import annotations

import asyncio
import json as json_lib
import logging
import os
import re
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.models import CurrentUser
from app.core.security import get_bearer_token, require_role
from app.db.session import get_db
from app.services.assessment_client import (
    AssessmentAPIClient,
    AssessmentAPIError,
    AssessmentAPIStatusError,
    get_assessment_client,
)
from app.services.grading_service import (
    GradingBusy,
    GradingServiceUnavailable,
    GradingTimeout,
    grade_answer,
    identify_all_questions,
)
from app.services.question_resolver import (
    AmbiguousMatchError,
    build_answer_key_from_item,
    resolve_question,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/grading/pipeline", tags=["Grading Pipeline"])


# --------------------------------------------------------------------- constants

# End-to-end budget for the ENTIRE pipeline. Individual stages have
# their own sub-timeouts below; this one just guarantees the caller's
# connection does not hang indefinitely if the GPU wedges.
_PIPELINE_TOTAL_TIMEOUT_SECONDS = 60 * 20  # 20 minutes

# Per-paper budget for the (identify + grade) VLM pass. Matches the
# limit used by ``/submit-multi-by-image``.
_PER_PAPER_GRADING_TIMEOUT_SECONDS = 180.0

# Cap how many questions we will grade per paper image. Mirrors
# ``GRADING_MAX_QUESTIONS_PER_IMAGE`` used elsewhere.
_MAX_QUESTIONS_PER_PAPER = 5

# Feedback strings from the VLM can be verbose; the upstream
# ``POST /feedbacks`` comment field almost certainly has a column limit
# we don't know, so cap it defensively. Override via
# ``GRADING_FEEDBACK_MAX_CHARS`` if the upstream rejects even 2000-char
# bodies (empirically some endpoints cap at 255 or 500).
_MAX_FEEDBACK_COMMENT_CHARS = int(os.getenv("GRADING_FEEDBACK_MAX_CHARS", "2000"))

# Upstream feedback endpoints that do naive string interpolation (or
# cross-service JSON decoding) sometimes choke on ASCII apostrophes,
# backticks, or control chars inside the ``comment`` field. We can't
# fix their bug, but we can swap the problematic characters for visually
# identical Unicode equivalents so the student-facing UI still renders
# correctly. Disable via ``GRADING_FEEDBACK_SANITIZE=0`` once the
# upstream is known to be safe again.
_SANITIZE_FEEDBACK = os.getenv("GRADING_FEEDBACK_SANITIZE", "1") == "1"

# ASCII control chars (0x00-0x1F except \t \n \r, plus DEL 0x7F). Strip
# these unconditionally - no upstream UI wants raw control bytes.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_feedback_comment(text: str) -> str:
    """Make ``text`` safe to POST to the upstream ``comment`` field.

    Two behaviours, both toggled as a group by
    ``GRADING_FEEDBACK_SANITIZE``:

    * ASCII ``'`` (apostrophe) → U+2019 ``’`` (right single quotation
      mark). Visually identical in every student-facing UI, but doesn't
      trip naive backends that interpolate the value into SQL or shell.
    * ASCII ``\``` (backtick) → U+2018 ``‘`` (left single quotation
      mark). Covers shell-injection-flavoured bugs.
    * Control characters are stripped (``\t``, ``\n``, ``\r`` preserved).

    Idempotent and cheap: returns the string unchanged when sanitisation
    is disabled or the input is already clean.
    """
    if not _SANITIZE_FEEDBACK or not text:
        return text
    out = text.replace("'", "\u2019").replace("`", "\u2018")
    out = _CONTROL_CHARS_RE.sub("", out)
    return out

# Fallback max score used when a question is detected on a paper but
# has no matching upstream answer row (and therefore no declared
# ``maxPoints``). The real max is always the upstream
# ``questionResponse.points`` when we have it; this constant only
# applies to "orphan" detections that never get committed anyway.
_DEFAULT_MAX_SCORE_FALLBACK = 10

# Subject hint forwarded to the grading VLM. The underlying
# ``grade_answer`` function already defaults to ``"math"`` but we pin
# the value here so the pipeline behaviour is explicit and auditable
# without opening the service module.
_GRADING_SUBJECT = "math"

# Pulls the first zero-or-more-padded question number out of an HTML
# answer-text snippet like ``<p>0003</p><p>(3x^2+...)</p>``. We look
# for a <p>-wrapped digit run first, then fall back to any 1-4 digit
# token so the regex copes with plain-text or lightly-tagged bodies.
_Q_NO_FROM_HTML = re.compile(r"<p[^>]*>\s*0*(\d{1,4})\s*</p>", re.IGNORECASE)
_Q_NO_FALLBACK = re.compile(r"\b0*(\d{1,4})\b")


# --------------------------------------------------------------------- schemas
# Request ----------------------------------------------------------------


class GradingPipelineRequest(BaseModel):
    """Inputs for ``POST /api/v1/grading/pipeline/run``.

    ``file_id`` points at the :class:`~app.db.models.answer_key_file.AnswerKeyFile`
    that holds the ground-truth answer key (same semantics as the
    ``file_id`` field on ``/submit-by-image``).

    Per-question max scores are NOT configurable: they come from the
    upstream ``answer.questionResponse.points`` field on each answer
    row. Forcing a single request-level ``max_score`` was causing the
    bulk update to trip a ``pointsAwarded > maxPoints`` check on the
    upstream, so the API now treats the upstream's declared max as the
    authority.
    """

    assessment_id: uuid.UUID
    submission_id: uuid.UUID
    file_id: uuid.UUID = Field(
        ...,
        description="AnswerKeyFile UUID holding the local answer-key for this assessment.",
    )


# Response --------------------------------------------------------------


class GradedQuestionTrace(BaseModel):
    """One question graded from one paper image."""

    detected_question_number: str
    matched_question_no: str | None = None
    matched_item_id: str | None = None
    match_method: Literal["question_number", "semantic"] | None = None
    similarity_score: float | None = None
    read_confidence: Literal["high", "medium", "low"] = "low"

    student_score: int | None = None
    max_score: int | None = None
    method_correct: bool | None = None
    is_correct: bool | None = None
    first_error_step: int | None = None
    error_summary: str | None = None
    feedback_preview: str | None = None

    status: Literal["graded", "not_found", "ambiguous", "error"] = "error"
    error: str | None = None


class PipelinePaperResult(BaseModel):
    """Outcome for a single paper image."""

    paper_id: uuid.UUID
    filename: str
    page: int | None = None
    download_status: Literal["ok", "error"] = "ok"
    download_error: str | None = None
    graded_questions: list[GradedQuestionTrace] = Field(default_factory=list)


class PipelineAnswerResult(BaseModel):
    """Outcome for a single answer row on the submission."""

    answer_id: uuid.UUID
    matched_paper_id: uuid.UUID | None = None
    matched_question_no: str | None = None
    points_awarded: float | None = None
    annotation_id: uuid.UUID | None = None
    feedback_id: uuid.UUID | None = None
    status: Literal[
        "graded",
        "no_grade_matched",
        "annotation_failed",
        "feedback_failed",
    ] = "no_grade_matched"
    error: str | None = None


class GradingPipelineResponse(BaseModel):
    """Full trace of one pipeline run."""

    assessment_id: uuid.UUID
    submission_id: uuid.UUID
    file_id: uuid.UUID

    papers_fetched: int
    papers_downloaded: int
    questions_detected: int
    questions_graded: int

    answers_fetched: int
    answers_matched: int
    annotations_created: int
    feedbacks_created: int
    bulk_update_applied: bool

    papers: list[PipelinePaperResult] = Field(default_factory=list)
    answers: list[PipelineAnswerResult] = Field(default_factory=list)


# --------------------------------------------------------------------- helpers


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _extract_question_no(answer_text: str | None) -> str | None:
    """Return the canonical 4-digit ``question_no`` embedded in an answer.

    Upstream answers store the question's prompt as HTML with the
    question number wrapped in its own ``<p>`` block, e.g.
    ``<p>0003</p><p>(3x^2+...)</p>``. We extract that leading integer,
    strip padding, and re-pad to four digits to match the canonical
    ``AnswerKeyItem.question_no`` format used elsewhere in the project.
    """
    if not answer_text:
        return None
    m = _Q_NO_FROM_HTML.search(answer_text)
    if m is None:
        m = _Q_NO_FALLBACK.search(answer_text)
    if m is None:
        return None
    return m.group(1).zfill(4)


def _normalise_question_no(value: str | None) -> str | None:
    """Strip leading zeros and re-pad to 4 digits, or return ``None``."""
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if not digits:
        return None
    return digits.lstrip("0").zfill(4) or "0000"


async def _download_paper(
    client: AssessmentAPIClient,
    filename: str,
    token: str,
) -> tuple[bytes | None, str | None]:
    """Return ``(bytes, None)`` on success or ``(None, error_message)``."""
    try:
        resp = await client.request(
            "GET", f"files/preview/{filename}", headers=_bearer(token)
        )
        return resp.content, None
    except AssessmentAPIStatusError as exc:
        return None, f"upstream {exc.status_code}: {str(exc.body)[:200]}"
    except AssessmentAPIError as exc:
        return None, f"{type(exc).__name__}: {exc}"


async def _grade_paper(
    *,
    db: Session,
    file_id: uuid.UUID,
    image_bytes: bytes,
    max_points_by_qno: dict[str, int],
) -> list[GradedQuestionTrace]:
    """Run identify + resolve + grade over one paper image.

    ``max_points_by_qno`` is a lookup of canonical 4-digit question
    number -> upstream ``maxPoints``. It is derived from the answers
    list and used as the authoritative cap for ``grade_answer`` so the
    VLM never returns a score the upstream would reject on bulk
    update. Questions that cannot be looked up (e.g. present on the
    paper but missing from the answers list) fall back to
    :data:`_DEFAULT_MAX_SCORE_FALLBACK`.

    Each detected question produces one :class:`GradedQuestionTrace`
    entry. Failures are recorded inline; the caller keeps going.
    """
    try:
        detections = await asyncio.wait_for(
            identify_all_questions(image_bytes=image_bytes),
            timeout=_PER_PAPER_GRADING_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return [
            GradedQuestionTrace(
                detected_question_number="",
                status="error",
                error=(
                    f"identify_all_questions timed out after "
                    f"{_PER_PAPER_GRADING_TIMEOUT_SECONDS:.0f}s"
                ),
            )
        ]
    except (GradingBusy, GradingServiceUnavailable, GradingTimeout) as exc:
        return [
            GradedQuestionTrace(
                detected_question_number="",
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
        ]
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("pipeline: identify_all_questions crashed")
        return [
            GradedQuestionTrace(
                detected_question_number="",
                status="error",
                error=f"identify crashed: {exc}",
            )
        ]

    if not detections:
        return []

    detections = detections[:_MAX_QUESTIONS_PER_PAPER]
    all_labels = [str(d.get("question_number") or "") for d in detections]

    traces: list[GradedQuestionTrace] = []
    for detection in detections:
        qn = str(detection.get("question_number") or "")
        rc_raw = str(detection.get("confidence") or "low")
        rc: Literal["high", "medium", "low"] = (
            rc_raw if rc_raw in {"high", "medium", "low"} else "low"  # type: ignore[assignment]
        )

        # --- resolve
        try:
            match_result = resolve_question(
                db=db,
                file_id=file_id,
                detected_number=qn or None,
                detected_text=detection.get("problem_text_preview") or None,
                read_confidence=rc,
            )
        except AmbiguousMatchError as exc:
            status_value: Literal["not_found", "ambiguous"] = (
                "not_found"
                if exc.reason == "no_question_identifier"
                else "ambiguous"
            )
            traces.append(
                GradedQuestionTrace(
                    detected_question_number=qn,
                    read_confidence=rc,
                    status=status_value,
                    error=exc.message,
                )
            )
            continue

        question_text, answer_key_text = build_answer_key_from_item(match_result.item)
        expected_final_answer = (
            match_result.item.normalized_answer or match_result.item.final_answer
        )
        others = [label for label in all_labels if label and label != qn]

        # Per-question upstream max. Falls back to a safe default for
        # detections that never match an upstream answer row (those
        # entries won't reach the bulk update anyway).
        matched_key = _normalise_question_no(match_result.item.question_no) or ""
        per_q_max = max_points_by_qno.get(matched_key, _DEFAULT_MAX_SCORE_FALLBACK)

        # --- grade
        try:
            grade_result = await asyncio.wait_for(
                grade_answer(
                    image_bytes=image_bytes,
                    question=question_text,
                    answer_key=answer_key_text,
                    max_score=per_q_max,
                    subject=_GRADING_SUBJECT,
                    question_label=match_result.item.question_no,
                    expected_final_answer=expected_final_answer,
                    other_questions_on_page=others,
                    # Scope the optional secondary RAG retrieval (controlled
                    # by GRADING_RAG_EXTRA_TOP_K in grading_service) to the
                    # answer-key file the student is being graded against.
                    file_id=str(match_result.item.file_id),
                ),
                timeout=_PER_PAPER_GRADING_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            traces.append(
                GradedQuestionTrace(
                    detected_question_number=qn,
                    matched_question_no=match_result.item.question_no,
                    matched_item_id=str(match_result.item.id),
                    match_method=match_result.match_method,  # type: ignore[arg-type]
                    similarity_score=float(match_result.similarity_score),
                    read_confidence=rc,
                    status="error",
                    error=(
                        f"grade_answer timed out after "
                        f"{_PER_PAPER_GRADING_TIMEOUT_SECONDS:.0f}s"
                    ),
                )
            )
            continue
        except (GradingBusy, GradingServiceUnavailable, GradingTimeout) as exc:
            traces.append(
                GradedQuestionTrace(
                    detected_question_number=qn,
                    matched_question_no=match_result.item.question_no,
                    matched_item_id=str(match_result.item.id),
                    match_method=match_result.match_method,  # type: ignore[arg-type]
                    similarity_score=float(match_result.similarity_score),
                    read_confidence=rc,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("pipeline: grade_answer crashed for qn=%r", qn)
            traces.append(
                GradedQuestionTrace(
                    detected_question_number=qn,
                    matched_question_no=match_result.item.question_no,
                    matched_item_id=str(match_result.item.id),
                    match_method=match_result.match_method,  # type: ignore[arg-type]
                    similarity_score=float(match_result.similarity_score),
                    read_confidence=rc,
                    status="error",
                    error=f"grade crashed: {exc}",
                )
            )
            continue

        feedback_text = str(grade_result.get("feedback") or "")
        raw_score = int(grade_result.get("student_score") or 0)
        # Belt-and-suspenders clamp: even though ``grade_answer`` itself
        # enforces ``max_score``, any model hiccup that produces a
        # higher number would fail the upstream bulk update with a
        # ``pointsAwarded cannot be greater than maxPoints`` 400.
        clamped_score = max(0, min(raw_score, per_q_max))
        traces.append(
            GradedQuestionTrace(
                detected_question_number=qn,
                matched_question_no=match_result.item.question_no,
                matched_item_id=str(match_result.item.id),
                match_method=match_result.match_method,  # type: ignore[arg-type]
                similarity_score=float(match_result.similarity_score),
                read_confidence=rc,
                student_score=clamped_score,
                max_score=int(grade_result.get("max_score") or per_q_max),
                method_correct=bool(grade_result.get("method_correct")),
                is_correct=bool(grade_result.get("is_correct")),
                first_error_step=grade_result.get("first_error_step"),
                error_summary=grade_result.get("error_summary"),
                feedback_preview=feedback_text[:400] or None,
                status="graded",
            )
        )
    return traces


def _build_annotation_content(
    *,
    paper_id: uuid.UUID,
    trace: GradedQuestionTrace,
    raw_grade: dict[str, Any] | None,
) -> str:
    """Serialise the grading result into a JSON string for ``contentJson``.

    The upstream re-parses this field server-side, so it MUST be valid
    JSON (we verified this empirically - passing free text yields an
    upstream 500). Frontends can re-parse the same shape to render a
    rich per-step breakdown next to the paper image.
    """
    body: dict[str, Any] = {
        "type": "auto_grading_result",
        "version": 1,
        "paper_id": str(paper_id),
        "matched_question_no": trace.matched_question_no,
        "matched_item_id": trace.matched_item_id,
        "match_method": trace.match_method,
        "similarity_score": trace.similarity_score,
        "read_confidence": trace.read_confidence,
        "student_score": trace.student_score,
        "max_score": trace.max_score,
        "method_correct": trace.method_correct,
        "is_correct": trace.is_correct,
        "first_error_step": trace.first_error_step,
        "error_summary": trace.error_summary,
    }
    if raw_grade is not None:
        body["extracted_steps"] = raw_grade.get("extracted_steps", [])
        body["confidence"] = raw_grade.get("confidence")
        body["safety_net_engaged"] = bool(raw_grade.get("safety_net_engaged"))
        body["graded_by"] = raw_grade.get("graded_by")
    return json_lib.dumps(body, ensure_ascii=False)


# =================================================================== endpoint


@router.post(
    "/run",
    response_model=GradingPipelineResponse,
    summary=(
        "Run the full auto-grading pipeline for one submission: "
        "fetch papers, grade each image against file_id, then write "
        "annotations + feedbacks + bulk-update points upstream."
    ),
)
async def run_pipeline(
    payload: GradingPipelineRequest,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role("ROLE_INSTRUCTOR", "ROLE_ADMIN")),
    token: str = Depends(get_bearer_token),
    client: AssessmentAPIClient = Depends(get_assessment_client),
) -> GradingPipelineResponse:
    """Grade every paper on a submission and commit the results upstream.

    Returns a detailed :class:`GradingPipelineResponse` trace so the
    caller can see exactly which papers were downloaded, which
    questions matched, which answers were updated, and where each
    failure (if any) occurred. The pipeline is best-effort per paper
    and per answer - a single failure does not abort the whole run.
    """
    logger.info(
        "pipeline: start user=%s assessment=%s submission=%s file=%s",
        user.preferred_username or user.sub,
        payload.assessment_id,
        payload.submission_id,
        payload.file_id,
    )

    try:
        return await asyncio.wait_for(
            _run(payload=payload, db=db, token=token, client=client),
            timeout=_PIPELINE_TOTAL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "pipeline: overall budget of %.0fs exhausted for submission=%s",
            _PIPELINE_TOTAL_TIMEOUT_SECONDS,
            payload.submission_id,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                f"Pipeline exceeded the {_PIPELINE_TOTAL_TIMEOUT_SECONDS:.0f}s "
                "budget. Re-run against a smaller submission or retry later."
            ),
        )


async def _run(
    *,
    payload: GradingPipelineRequest,
    db: Session,
    token: str,
    client: AssessmentAPIClient,
) -> GradingPipelineResponse:
    """Inner orchestration; wrapped by ``run_pipeline`` for the overall timeout."""
    # --------------------------------------------------------------- step 1+3: list papers + answers (parallel)
    papers_task = client.get_json(
        (
            f"assessments/{payload.assessment_id}"
            f"/submissions/{payload.submission_id}/papers"
        ),
        headers=_bearer(token),
    )
    answers_task = client.get_json(
        f"submissions/{payload.submission_id}/answers",
        params={
            "page": 1,
            "size": 200,
            "property": "CREATED_AT",
            "direction": "ASC",
        },
        headers=_bearer(token),
    )

    try:
        papers_resp, answers_resp = await asyncio.gather(papers_task, answers_task)
    except AssessmentAPIStatusError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.body) from exc
    except AssessmentAPIError as exc:
        logger.exception("pipeline: failed to fetch papers/answers")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    papers = _unwrap_list(papers_resp)
    answers = _unwrap_items(answers_resp)
    logger.info(
        "pipeline: fetched %d paper(s) and %d answer(s)", len(papers), len(answers)
    )

    if not papers:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Submission has no papers attached; nothing to grade.",
        )

    # Build a canonical ``question_no -> max_points`` lookup from the
    # upstream answers. The upstream enforces ``pointsAwarded <=
    # maxPoints`` on the bulk update, so this is the authoritative cap
    # we feed into the VLM for every matched question.
    max_points_by_qno: dict[str, int] = {}
    for a in answers:
        qresp = a.get("questionResponse") or {}
        qkey = _extract_question_no(qresp.get("text"))
        if not qkey:
            continue
        try:
            max_points = int(float(qresp.get("points") or 0))
        except (TypeError, ValueError):
            max_points = _DEFAULT_MAX_SCORE_FALLBACK
        if max_points > 0:
            max_points_by_qno[qkey] = max_points

    # --------------------------------------------------------------- step 2: download each paper image (parallel)
    download_coros = [
        _download_paper(client, str(p.get("name") or ""), token) for p in papers
    ]
    downloads = await asyncio.gather(*download_coros)

    paper_results: list[PipelinePaperResult] = []
    paper_bytes_map: dict[uuid.UUID, bytes] = {}
    for p, (body, err) in zip(papers, downloads):
        paper_id = uuid.UUID(str(p["paperId"]))
        result = PipelinePaperResult(
            paper_id=paper_id,
            filename=str(p.get("name") or ""),
            page=p.get("page"),
            download_status="ok" if body is not None else "error",
            download_error=err,
        )
        if body is not None:
            paper_bytes_map[paper_id] = body
        paper_results.append(result)

    downloaded_count = sum(1 for r in paper_results if r.download_status == "ok")
    logger.info(
        "pipeline: %d/%d paper images downloaded", downloaded_count, len(papers)
    )

    # --------------------------------------------------------------- step 4: grade each image (SERIAL - GPU)
    # Keeps the raw grade dicts around so we can embed extracted_steps
    # inside the upstream annotation body.
    raw_grades_by_paper: dict[uuid.UUID, list[dict[str, Any]]] = {}
    for result in paper_results:
        if result.download_status != "ok":
            continue
        image_bytes = paper_bytes_map[result.paper_id]
        logger.info(
            "pipeline: grading paper=%s filename=%s (%d bytes)",
            result.paper_id,
            result.filename,
            len(image_bytes),
        )
        # The helper already catches its own exceptions, so this never
        # raises past us - no try/except needed.
        traces = await _grade_paper(
            db=db,
            file_id=payload.file_id,
            image_bytes=image_bytes,
            max_points_by_qno=max_points_by_qno,
        )
        result.graded_questions = traces
        raw_grades_by_paper[result.paper_id] = [t.model_dump() for t in traces]

    questions_detected = sum(len(r.graded_questions) for r in paper_results)
    questions_graded = sum(
        1
        for r in paper_results
        for t in r.graded_questions
        if t.status == "graded"
    )
    logger.info(
        "pipeline: %d question(s) graded out of %d detected",
        questions_graded,
        questions_detected,
    )

    # --------------------------------------------------------------- step 5+6: match graded results to answers, then annotate + feedback
    # Build a lookup of (question_no -> (paper_id, trace)). When the
    # same question is detected on multiple papers we keep the FIRST
    # graded one (top-to-bottom page order) which matches how the
    # frontend presents submission papers.
    graded_lookup: dict[str, tuple[uuid.UUID, GradedQuestionTrace]] = {}
    for pr in paper_results:
        for trace in pr.graded_questions:
            if trace.status != "graded" or not trace.matched_question_no:
                continue
            key = _normalise_question_no(trace.matched_question_no)
            if key and key not in graded_lookup:
                graded_lookup[key] = (pr.paper_id, trace)

    # Build answer result rows (one per upstream answer).
    #
    # Per-answer annotation tasks are issued in parallel; feedback
    # creation is serialised per-answer (needs the just-minted
    # annotationId). Pydantic validation on the ``AssessmentAPIClient``
    # response is intentionally light - the upstream shape is already
    # vetted by the separate /api/v1/assessment/* routes.
    answer_results: list[PipelineAnswerResult] = []
    for a in answers:
        try:
            answer_id = uuid.UUID(str(a["answerId"]))
        except Exception:  # pragma: no cover - defensive
            continue

        row = PipelineAnswerResult(answer_id=answer_id, status="no_grade_matched")
        qno = _extract_question_no(
            (a.get("questionResponse") or {}).get("text")
        )
        if qno and qno in graded_lookup:
            paper_id, trace = graded_lookup[qno]
            row.matched_paper_id = paper_id
            row.matched_question_no = qno
            # Clamp once here; the final ``scored_rows`` builder below
            # clamps again just before the bulk update.
            cap = max_points_by_qno.get(qno, _DEFAULT_MAX_SCORE_FALLBACK)
            row.points_awarded = max(
                0.0, min(float(trace.student_score or 0), float(cap))
            )
        answer_results.append(row)

    matched_count = sum(1 for r in answer_results if r.points_awarded is not None)
    logger.info(
        "pipeline: matched %d/%d answers to a graded question",
        matched_count,
        len(answer_results),
    )

    annotations_created = 0
    feedbacks_created = 0

    # Resolve raw grade dicts keyed by (paper_id, matched_question_no)
    # so we can attach extracted_steps to the annotation body.
    raw_by_match: dict[tuple[uuid.UUID, str], dict[str, Any]] = {}
    for pr in paper_results:
        for raw in raw_grades_by_paper.get(pr.paper_id, []):
            key = (pr.paper_id, str(raw.get("matched_question_no") or ""))
            raw_by_match[key] = raw

    async def _annotate_and_feedback(row: PipelineAnswerResult) -> None:
        nonlocal annotations_created, feedbacks_created

        if row.points_awarded is None or row.matched_paper_id is None:
            return

        qkey = _normalise_question_no(row.matched_question_no) or ""
        _, trace = graded_lookup[qkey]

        # "Student got it right" shortcut: skip the annotation + feedback
        # API calls entirely when there is nothing instructive to tell
        # the student. Two triggers:
        #
        #   a) ``points_awarded >= cap`` - full marks. The score is
        #      the verdict; no commentary needed.
        #   b) ``trace.is_correct == True`` - the VLM explicitly judged
        #      the answer correct (final answer right, all steps right).
        #      Even if the score ended up below the cap for some
        #      accounting reason, there is nothing to comment on and no
        #      reason to risk an upstream 500 on a "well done" string.
        #
        # Row still gets ``status="graded"`` + ``points_awarded`` so the
        # final bulk update commits the score.
        cap = max_points_by_qno.get(qkey, _DEFAULT_MAX_SCORE_FALLBACK)
        if row.points_awarded >= float(cap) or trace.is_correct is True:
            logger.info(
                "pipeline: skipping annotation+feedback for answer=%s "
                "(question_no=%s points=%.1f/%s is_correct=%s) - "
                "student got it right; nothing to annotate.",
                row.answer_id,
                row.matched_question_no,
                row.points_awarded,
                cap,
                trace.is_correct,
            )
            row.status = "graded"
            return

        raw_grade = raw_by_match.get(
            (row.matched_paper_id, trace.matched_question_no or "")
        )
        content_json = _build_annotation_content(
            paper_id=row.matched_paper_id,
            trace=trace,
            raw_grade=raw_grade,
        )

        try:
            ann_resp = await client.post_json(
                f"answers/{row.answer_id}/annotations",
                json={"contentJson": content_json},
                headers=_bearer(token),
            )
        except AssessmentAPIError as exc:
            row.status = "annotation_failed"
            row.error = f"annotation: {exc}"
            return

        payload_dict = (ann_resp or {}).get("payload") if isinstance(ann_resp, dict) else None
        annotation_id_raw = None
        if isinstance(payload_dict, dict):
            annotation_id_raw = payload_dict.get("annotationId") or payload_dict.get("id")
        if not annotation_id_raw:
            row.status = "annotation_failed"
            row.error = "annotation response did not include an annotationId"
            return

        try:
            row.annotation_id = uuid.UUID(str(annotation_id_raw))
        except Exception:
            row.status = "annotation_failed"
            row.error = f"malformed annotationId: {annotation_id_raw!r}"
            return
        annotations_created += 1

        raw_comment = (
            trace.feedback_preview or trace.error_summary or "Auto-graded"
        )[:_MAX_FEEDBACK_COMMENT_CHARS]
        comment = _sanitize_feedback_comment(raw_comment)
        try:
            fb_resp = await client.post_json(
                f"answers/{row.answer_id}/feedbacks",
                json={
                    "comment": comment,
                    "annotationId": str(row.annotation_id),
                },
                headers=_bearer(token),
            )
        except AssessmentAPIStatusError as exc:
            # Log the exact request + response so upstream 5xx/4xx can be
            # diagnosed without re-running the grade. The comment string
            # is the only part that differs between successful and failing
            # feedback calls; 99% of the time the upstream is rejecting
            # either its length or a specific character it can't handle
            # (apostrophes, backticks, control chars).
            logger.warning(
                "pipeline: feedback POST failed status=%d answer_id=%s "
                "annotation_id=%s comment_len=%d comment_preview=%r "
                "upstream_body=%r",
                exc.status_code,
                row.answer_id,
                row.annotation_id,
                len(comment),
                comment[:200],
                exc.body if hasattr(exc, "body") else None,
            )
            row.status = "feedback_failed"
            row.error = f"feedback: {exc}"
            return
        except AssessmentAPIError as exc:
            logger.warning(
                "pipeline: feedback POST errored answer_id=%s "
                "comment_len=%d comment_preview=%r err=%s",
                row.answer_id,
                len(comment),
                comment[:200],
                exc,
            )
            row.status = "feedback_failed"
            row.error = f"feedback: {exc}"
            return

        fb_payload = (fb_resp or {}).get("payload") if isinstance(fb_resp, dict) else None
        if isinstance(fb_payload, dict):
            fb_id = fb_payload.get("feedbackId") or fb_payload.get("id")
            if fb_id:
                try:
                    row.feedback_id = uuid.UUID(str(fb_id))
                except Exception:
                    pass
        feedbacks_created += 1
        row.status = "graded"

    await asyncio.gather(*(_annotate_and_feedback(r) for r in answer_results))

    # --------------------------------------------------------------- step 7: bulk update points
    # Only include answers that successfully received an annotation
    # (otherwise we'd be writing points the grader UI can't show a
    # rationale for). ``annotation_failed`` / ``feedback_failed`` are
    # excluded intentionally.
    #
    # ``pointsAwarded`` is clamped one more time against the upstream's
    # per-answer ``maxPoints`` as a defence-in-depth guard. The VLM and
    # per-paper clamp already enforce this, but the upstream returns a
    # ``400 pointsAwarded (X) cannot be greater than maxPoints (Y)`` on
    # violation which would abort the whole bulk call for everyone.
    scored_rows: list[dict[str, Any]] = []
    for r in answer_results:
        if (
            r.status != "graded"
            or r.points_awarded is None
            or r.matched_paper_id is None
        ):
            continue
        cap = max_points_by_qno.get(
            _normalise_question_no(r.matched_question_no) or "",
            _DEFAULT_MAX_SCORE_FALLBACK,
        )
        capped = max(0.0, min(float(r.points_awarded), float(cap)))
        r.points_awarded = capped
        scored_rows.append(
            {
                "answerId": str(r.answer_id),
                "pointsAwarded": capped,
                "paperId": str(r.matched_paper_id),
            }
        )

    bulk_update_applied = False
    if scored_rows:
        try:
            await client.put_json(
                f"submissions/{payload.submission_id}/answers/bulk",
                json=scored_rows,
                headers=_bearer(token),
            )
            bulk_update_applied = True
            logger.info(
                "pipeline: bulk-updated points for %d answer(s)", len(scored_rows)
            )
        except AssessmentAPIStatusError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.body) from exc
        except AssessmentAPIError as exc:
            logger.exception("pipeline: bulk update failed")
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return GradingPipelineResponse(
        assessment_id=payload.assessment_id,
        submission_id=payload.submission_id,
        file_id=payload.file_id,
        papers_fetched=len(papers),
        papers_downloaded=downloaded_count,
        questions_detected=questions_detected,
        questions_graded=questions_graded,
        answers_fetched=len(answers),
        answers_matched=matched_count,
        annotations_created=annotations_created,
        feedbacks_created=feedbacks_created,
        bulk_update_applied=bulk_update_applied,
        papers=paper_results,
        answers=answer_results,
    )


# --------------------------------------------------------------------- payload helpers


def _unwrap_list(resp: Any) -> list[dict[str, Any]]:
    """Extract ``payload`` from an ``{message, payload: [...], status}`` envelope."""
    if isinstance(resp, dict):
        payload = resp.get("payload")
        if isinstance(payload, list):
            return payload
    if isinstance(resp, list):
        return resp  # upstream sometimes returns the array directly
    return []


def _unwrap_items(resp: Any) -> list[dict[str, Any]]:
    """Extract ``payload.items`` from a paginated envelope."""
    if isinstance(resp, dict):
        payload = resp.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            return payload["items"]
        if isinstance(payload, list):
            return payload
    return []
