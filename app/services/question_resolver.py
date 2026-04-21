"""Hybrid question resolution for the auto-detect grading endpoint.

Given the output of :func:`app.services.grading_service.identify_question`
(a detected question-number and/or copied problem-statement from the
top of the student's handwritten page), this module picks the
corresponding :class:`~app.db.models.answer_key_item.AnswerKeyItem`
from the instructor's uploaded answer key.

Resolution is a three-level waterfall:

1. **Exact SQL lookup** on ``(file_id, question_no)`` - when the
   student clearly wrote a question number (e.g. "Q3", "1.", "2.a").
   We try a handful of normalisation variants so a student who writes
   ``"3"`` still matches a row stored as ``"0003"`` or ``"Q3"``.
2. **Semantic search** over the Chroma collection filtered by
   ``file_id`` - when the student copied the problem statement at the
   top of the page, or when step 1 came up empty. We compute a bge-m3
   embedding on the detected text and take the top hit *only* if it's
   confident enough (cosine distance below a threshold AND clearly
   better than the runner-up).
3. **Give up and raise** :class:`AmbiguousMatchError` with top-3
   candidates so the route layer can return an HTTP 422 and let the
   frontend prompt the student "Which question is this?".

This module does **not** embed anything itself at top-of-function
scope - it imports :mod:`app.services.embedding_service` and
:mod:`app.services.chroma_service` lazily inside the resolver so that
units tests exercising only the normaliser don't pay the
``SentenceTransformer("BAAI/bge-m3")`` load cost.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.db.models.answer_key_item import AnswerKeyItem
from app.repositories.answer_key_item_repository import AnswerKeyItemRepository

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- tunables
#
# These are cosine distances on normalised bge-m3 vectors (so lower =
# more similar, range [0, 2]). Numbers below came from the design
# discussion; override via env vars in production once you've calibrated
# on real traffic.
#
# - ``_SIM_DISTANCE_THRESHOLD``: top hit must be at least this close
#   (i.e. distance below this value) to be accepted as a confident match.
# - ``_SIM_MARGIN``: top hit's distance must beat the runner-up by at
#   least this much. Without a margin check, two near-duplicate problems
#   in the same PDF would each "win" half the time.
# - ``_TOP_K``: number of candidates returned to the client on a 422.

import os  # noqa: E402  (kept below module docstring for readability)

_SIM_DISTANCE_THRESHOLD = float(
    os.getenv("GRADING_SIMILARITY_DISTANCE_THRESHOLD", "0.25")
)
_SIM_MARGIN = float(os.getenv("GRADING_SIMILARITY_MARGIN", "0.08"))
_TOP_K = int(os.getenv("GRADING_RESOLVER_TOP_K", "3"))


# --------------------------------------------------------------------- public types


@dataclass
class MatchResult:
    """Successful resolution: which item, and how we found it."""

    item: AnswerKeyItem
    match_method: str  # "question_number" | "semantic"
    similarity_score: float  # 1.0 for exact number; (1 - distance) for semantic
    read_confidence: str  # echoed from identify_question's "read_confidence"


@dataclass
class Candidate:
    """One row in the 422 ``candidates`` list shown to the frontend."""

    item_id: str
    question_no: str
    problem_text: str
    similarity_score: float


class AmbiguousMatchError(Exception):
    """Raised when no confident match could be found.

    Carries a machine-readable ``reason`` plus a ``candidates`` list
    that the route layer surfaces in the 422 response body so the UI
    can ask the student to pick the correct question manually.
    """

    def __init__(
        self,
        *,
        reason: str,
        message: str,
        candidates: list[Candidate] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.candidates = candidates or []


# --------------------------------------------------------------------- normaliser


_STRIP_PREFIXES_RE = re.compile(
    r"^\s*(?:question|q(?:uestion)?|no\.?|num(?:ber)?|#)\s*[:.\-]?\s*",
    re.IGNORECASE,
)


def _normalize_variants(raw: str) -> list[str]:
    """Return ordered candidate ``question_no`` strings to try in SQL.

    Real-world data is messy: the student might scrawl ``"Q3"`` while
    the answer-key PDF was parsed into rows with ``question_no="3"``
    (or ``"0003"``, or ``"3."``). Rather than force a canonical shape
    everywhere, we try a short list of variants in priority order and
    accept the first one that hits the ``(file_id, question_no)``
    unique index.

    Dedup preserves insertion order so higher-confidence variants are
    tried first.
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []

    # Strip any leading "Q", "Question ", "#", "No." prefix so we can
    # re-attach canonical variants below.
    core = _STRIP_PREFIXES_RE.sub("", s).strip().rstrip(".").strip()
    if not core:
        core = s  # fall back to the raw string if stripping ate everything

    variants: list[str] = []
    variants.append(s)  # raw, as-written
    variants.append(core)  # without prefix, e.g. "3" from "Q3"
    variants.append(f"Q{core}")  # "Q3"
    variants.append(f"{core}.")  # "3."

    # Zero-padded forms - only meaningful for purely numeric cores.
    if core.isdigit():
        n = int(core)
        variants.append(f"{n:04d}")  # "0003"
        variants.append(f"Q{n:02d}")  # "Q03"

    # Dedup preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        v_clean = v.strip()
        if not v_clean:
            continue
        if v_clean in seen:
            continue
        seen.add(v_clean)
        out.append(v_clean)
    return out


# --------------------------------------------------------------------- helpers


def _as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    """Parse ``value`` as UUID, raising ``ValueError`` on bad input."""
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _build_candidates_from_chroma(result: dict[str, Any]) -> list[Candidate]:
    """Shape Chroma's nested query result into a flat ``list[Candidate]``."""
    ids_outer = result.get("ids") or []
    distances_outer = result.get("distances") or []
    metadatas_outer = result.get("metadatas") or []
    documents_outer = result.get("documents") or []

    ids = ids_outer[0] if ids_outer else []
    distances = distances_outer[0] if distances_outer else []
    metadatas = metadatas_outer[0] if metadatas_outer else []
    documents = documents_outer[0] if documents_outer else []

    candidates: list[Candidate] = []
    for i, doc_id in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) else {}
        meta = meta or {}
        dist = float(distances[i]) if i < len(distances) else 1.0
        doc = documents[i] if i < len(documents) else ""
        candidates.append(
            Candidate(
                item_id=str(doc_id),
                question_no=str(meta.get("question_no") or ""),
                problem_text=str(
                    meta.get("problem_text") or doc or ""
                )[:500],
                similarity_score=round(max(0.0, 1.0 - dist), 4),
            )
        )
    return candidates


# --------------------------------------------------------------------- resolver


def resolve_question(
    db: Session,
    file_id: str | uuid.UUID,
    detected_number: str | None,
    detected_text: str | None,
    read_confidence: str = "low",
) -> MatchResult:
    """Resolve an :class:`AnswerKeyItem` from the identify-pass hints.

    Parameters
    ----------
    db:
        Open SQLAlchemy session (route-scoped, from ``Depends(get_db)``).
    file_id:
        UUID of the :class:`~app.db.models.answer_key_file.AnswerKeyFile`
        the student is being graded against. Search is always scoped
        to this file.
    detected_number:
        Question identifier read off the top of the page, if any
        (e.g. ``"3"``, ``"Q5"``, ``"2.a"``). ``None`` when the model
        saw no identifier.
    detected_text:
        Problem statement the student copied from the exam paper, if
        any. Used for semantic fallback.
    read_confidence:
        ``"high" | "medium" | "low"`` - echoed onto the successful
        :class:`MatchResult` so the route layer can surface it.

    Raises
    ------
    AmbiguousMatchError
        No confident match was found. ``exc.reason`` is one of
        ``"no_question_identifier"`` (neither number nor text was
        detected) or ``"ambiguous_match"`` (semantic search was
        inconclusive). ``exc.candidates`` holds up to ``_TOP_K``
        alternatives the frontend can offer the student.
    """
    file_uuid = _as_uuid(file_id)
    repo = AnswerKeyItemRepository(db)

    # --- Level 1: exact number lookup --------------------------------
    if detected_number:
        variants = _normalize_variants(detected_number)
        logger.info(
            "resolver: trying question-number lookup file_id=%s variants=%r",
            file_uuid,
            variants,
        )
        for variant in variants:
            item = repo.get_by_question_no(file_uuid, variant)
            if item is not None:
                logger.info(
                    "resolver: matched by question-number file_id=%s variant=%r "
                    "item_id=%s question_no=%r",
                    file_uuid,
                    variant,
                    item.id,
                    item.question_no,
                )
                return MatchResult(
                    item=item,
                    match_method="question_number",
                    similarity_score=1.0,
                    read_confidence=read_confidence,
                )
        logger.info(
            "resolver: question-number lookup miss for file_id=%s raw=%r",
            file_uuid,
            detected_number,
        )

    # --- Level 2: semantic fallback ----------------------------------
    if detected_text:
        # Lazy imports so test code exercising only _normalize_variants
        # doesn't trigger the bge-m3 model load.
        from app.services.chroma_service import search_documents
        from app.services.embedding_service import get_embedding

        embedding = get_embedding(detected_text)
        search_result = search_documents(
            query_embedding=embedding,
            top_k=_TOP_K,
            where={"file_id": str(file_uuid)},
        )

        candidates = _build_candidates_from_chroma(search_result)
        if candidates:
            distances_outer = search_result.get("distances") or []
            distances = distances_outer[0] if distances_outer else []
            top_distance = float(distances[0]) if distances else 1.0
            second_distance = (
                float(distances[1]) if len(distances) > 1 else 1.0
            )
            margin = second_distance - top_distance

            confident = (
                top_distance < _SIM_DISTANCE_THRESHOLD
                and margin >= _SIM_MARGIN
            )

            logger.info(
                "resolver: semantic top1=%.4f second=%.4f margin=%.4f "
                "threshold=%.2f margin_req=%.2f confident=%s",
                top_distance,
                second_distance,
                margin,
                _SIM_DISTANCE_THRESHOLD,
                _SIM_MARGIN,
                confident,
            )

            if confident:
                top_id = candidates[0].item_id
                try:
                    item = repo.get(_as_uuid(top_id))
                except (ValueError, TypeError):
                    item = None
                if item is None:
                    # Chroma/SQL drift: the vector exists but the row
                    # was deleted. Surface as ambiguous so the frontend
                    # shows candidates.
                    logger.warning(
                        "resolver: chroma hit %s has no matching SQL row; "
                        "treating as ambiguous",
                        top_id,
                    )
                else:
                    return MatchResult(
                        item=item,
                        match_method="semantic",
                        similarity_score=round(
                            max(0.0, 1.0 - top_distance), 4
                        ),
                        read_confidence=read_confidence,
                    )

            raise AmbiguousMatchError(
                reason="ambiguous_match",
                message=(
                    "Could not confidently identify the question from the "
                    "copied problem statement. Please pick the correct "
                    "question below."
                ),
                candidates=candidates,
            )

        # Semantic search returned nothing - empty file or Chroma out
        # of sync. Same 422 path, just with zero candidates so the
        # frontend knows to show the full question list.
        raise AmbiguousMatchError(
            reason="ambiguous_match",
            message=(
                "Could not identify the question: semantic search returned "
                "no candidates for this answer-key file."
            ),
            candidates=[],
        )

    # --- Level 3: neither signal present -----------------------------
    raise AmbiguousMatchError(
        reason="no_question_identifier",
        message=(
            "Could not identify the question: the student neither wrote a "
            "question number nor copied the problem statement at the top "
            "of the page."
        ),
        candidates=[],
    )


# --------------------------------------------------------------------- answer-key builder


def build_answer_key_from_item(item: AnswerKeyItem) -> tuple[str, str]:
    """Format an :class:`AnswerKeyItem` into ``(question, answer_key)`` strings.

    The grading VLM expects two separate inputs: the original question
    text and the step-by-step worked solution. We build both from the
    structured projection columns (``problem_text`` + ``solution_steps``
    + ``final_answer``) and fall back to the raw parser ``content`` if
    the structured pass wasn't run for this row.

    Step numbering (important for VLM alignment)
    --------------------------------------------

    Students almost always write the **starting expression** as their
    first line, then one line per transformation. The instructor's
    ``solution_steps`` typically do NOT include the starting expression
    (it's redundant with ``problem_text``), so naive positional
    labelling puts the first transformation at "Step 1" - which then
    collides with the student's "Step 1" (the starting expression) in
    the grading prompt.

    We fix this by emitting the starting expression as an explicit
    "Step 1" in the answer-key text and shifting ``solution_steps``
    down by one. That way the student's line-N aligns with answer-key
    step-N and the VLM does not compare the student's distribution
    line against the answer key's final answer.
    """
    question = (item.problem_text or "").strip()
    if not question:
        question = (item.content or "").strip()

    parts: list[str] = ["Expected step-by-step solution:"]

    # Step 1 = starting expression (copied from the question). Aligns
    # with what the student writes on line 1 of their page.
    if question:
        parts.append(
            f"Step 1 (matches the starting expression from the question): "
            f"{question}"
        )

    # Steps 2, 3, ... = the instructor's solution_steps, shifted down
    # by 1 so they follow Step 1 positionally.
    steps = item.solution_steps or []
    cleaned_steps = [s.strip() for s in steps if s and s.strip()]
    next_step_num = 2 if question else 1
    for step in cleaned_steps:
        parts.append(f"Step {next_step_num}: {step}")
        next_step_num += 1

    if item.final_answer and item.final_answer.strip():
        parts.append(f"Final answer: {item.final_answer.strip()}")

    # Defensive fallback: if nothing structured was available, drop in
    # the raw parser output so the grader has *something*.
    if len(parts) == 1:  # only the header
        raw = (item.content or "").strip()
        parts.append(raw or "(no worked solution available)")

    answer_key = "\n".join(parts)
    return question, answer_key
