"""Choose and execute a parsing strategy for a set of extracted PDF pages.

The selector is the single entry point used by the upload pipeline. It knows
about every concrete parser and applies robust fallback so callers never need
to handle parsing errors themselves.

Selection is **quality-aware**: structured parsers (``question_number`` and
``heading``) are only picked when the prepared page texts are clean enough
to chunk safely. When text quality is low - typical for OCR'd image PDFs
where the OCR engine lands the digits but corrupts the math - we fall back
to the page parser instead of emitting corrupted structured chunks.

Gates:

* ``question_number_parser``:
    quality >= QUALITY_GATES[mode]["question_number"]["quality"]
    AND suspicious <= QUALITY_GATES[mode]["question_number"]["suspicious"]
    AND question_count >= MIN_QUESTION_ANCHORS_FOR_STRUCTURED
* ``heading_parser``:
    quality >= QUALITY_GATES[mode]["heading"]["quality"]
    AND heading_count >= MIN_HEADING_ANCHORS_FOR_HEADING
* otherwise:
    ``page_fallback_parser``

The mode thresholds are tightened for ``image_pdf`` (stricter quality,
stricter suspicious) so OCR-derived structure must look nearly pristine
before we trust it.
"""

from __future__ import annotations

import logging
from typing import Literal

from app.models.schemas import QuestionChunk
from app.services.parsers import heading_parser as heading_parser
from app.services.parsers import page_fallback_parser as page_fallback_parser
from app.services.parsers import question_number_parser as question_number_parser
from app.services.parsers.base import (
    QUESTION_NUMBER_PATTERN,
    build_combined_text,
    normalize_whitespace,
)
from app.services.pdf_service import (
    PdfMode,
    average_text_quality,
    readability_ratio,
    suspicious_ratio,
)

logger = logging.getLogger(__name__)

ParseMode = Literal["auto", "question_number", "heading", "page"]

MIN_QUESTION_ANCHORS_FOR_STRUCTURED = 3
MIN_HEADING_ANCHORS_FOR_HEADING = 3
MIN_TEXT_CHARS_FOR_STRUCTURED = 200

# Gate thresholds per PDF mode. The ``text_pdf`` row matches the user spec
# exactly (quality >= 0.65, suspicious < 0.15, heading >= 0.55). For
# ``image_pdf`` we tighten both rules because OCR tends to produce Korean
# look-alike glyphs (``텅``, ``딥``, ``밉`` ...) that are hard to detect as
# garbage character-by-character but destroy the math semantics.
QUALITY_GATES: dict[str, dict[str, dict[str, float]]] = {
    "text_pdf": {
        "question_number": {"quality": 0.65, "suspicious": 0.15},
        "heading": {"quality": 0.55, "suspicious": 0.30},
    },
    "image_pdf": {
        "question_number": {"quality": 0.75, "suspicious": 0.10},
        "heading": {"quality": 0.65, "suspicious": 0.20},
    },
}

DEFAULT_PDF_MODE: PdfMode = "text_pdf"


def detect_strategy(
    pages: list[dict],
    *,
    pdf_mode: PdfMode | None = None,
) -> str:
    """Pick a parser name based on structure + text-quality heuristics.

    Args:
        pages: non-empty page dicts after OCR routing.
        pdf_mode: result of :func:`pdf_service.detect_pdf_mode`. Defaults to
            ``"text_pdf"`` when not supplied. ``"image_pdf"`` selects the
            stricter gate row in :data:`QUALITY_GATES`.

    Returns:
        The parser name to run first. Always a value in :data:`_PARSERS`.
    """
    full_text, _, _ = build_combined_text(pages)
    total_chars = len(full_text)

    if total_chars == 0:
        return page_fallback_parser.PARSER_NAME

    mode = pdf_mode or DEFAULT_PDF_MODE
    gates = QUALITY_GATES[mode]

    quality = readability_ratio(full_text)
    suspicious = suspicious_ratio(full_text)
    question_count = len(QUESTION_NUMBER_PATTERN.findall(full_text))
    heading_count = heading_parser.count_heading_markers(full_text)

    logger.info(
        "parser_selector: stats pdf_mode=%s total_chars=%d "
        "quality=%.2f suspicious=%.2f question_count=%d heading_count=%d",
        mode,
        total_chars,
        quality,
        suspicious,
        question_count,
        heading_count,
    )

    qn_gate = gates["question_number"]
    if (
        question_count >= MIN_QUESTION_ANCHORS_FOR_STRUCTURED
        and total_chars >= MIN_TEXT_CHARS_FOR_STRUCTURED
        and quality >= qn_gate["quality"]
        and suspicious <= qn_gate["suspicious"]
    ):
        return question_number_parser.PARSER_NAME

    if question_count >= MIN_QUESTION_ANCHORS_FOR_STRUCTURED:
        logger.warning(
            "parser_selector: %d question anchors seen but "
            "quality=%.2f (need>=%.2f) suspicious=%.2f (need<=%.2f) "
            "pdf_mode=%s; refusing question_number parser",
            question_count,
            quality,
            qn_gate["quality"],
            suspicious,
            qn_gate["suspicious"],
            mode,
        )

    heading_gate = gates["heading"]
    if (
        heading_count >= MIN_HEADING_ANCHORS_FOR_HEADING
        and quality >= heading_gate["quality"]
        and suspicious <= heading_gate["suspicious"]
    ):
        return heading_parser.PARSER_NAME

    return page_fallback_parser.PARSER_NAME


def parse_with_selected_strategy(
    pages: list[dict],
    source_file: str,
    parse_mode: ParseMode | str = "auto",
    pdf_mode: PdfMode | None = None,
) -> list[QuestionChunk]:
    """Run the requested (or auto-detected) parser with guaranteed fallback.

    Args:
        pages: Page dicts (already routed through OCR if applicable).
        source_file: Original filename, propagated to each chunk.
        parse_mode: ``"auto"`` to let heuristics decide, or an explicit
            parser name (``"question_number"`` / ``"heading"`` / ``"page"``).
        pdf_mode: Upstream classification from
            :func:`pdf_service.detect_pdf_mode`. Used to pick the gate row
            in :data:`QUALITY_GATES`.

    Returns:
        A non-empty list of :class:`QuestionChunk` objects, unless the PDF
        had no readable text at all (in which case an empty list).

    If the chosen parser returns zero chunks we retry with
    :mod:`page_fallback_parser` so the upload never silently yields nothing.
    """
    non_empty = [p for p in pages if normalize_whitespace(p.get("text", ""))]
    if not non_empty:
        logger.warning("parser_selector: no non-empty pages in %s", source_file)
        return []

    quality = average_text_quality(non_empty)
    primary = _resolve_primary_parser(parse_mode, non_empty, pdf_mode=pdf_mode)

    logger.info(
        "parser_selector: selected=%s parse_mode=%s pdf_mode=%s file=%s "
        "pages=%d avg_quality=%.2f",
        primary,
        parse_mode,
        pdf_mode or "unknown",
        source_file,
        len(non_empty),
        quality,
    )

    chunks = _safe_parse(primary, non_empty, source_file)
    if chunks:
        logger.info(
            "parser_selector: parser=%s produced %d chunks for %s",
            primary,
            len(chunks),
            source_file,
        )
        return chunks

    if primary != page_fallback_parser.PARSER_NAME:
        logger.warning(
            "parser_selector: parser=%s returned 0 chunks, falling back to "
            "page parser for %s",
            primary,
            source_file,
        )
        chunks = _safe_parse(page_fallback_parser.PARSER_NAME, non_empty, source_file)
        logger.info(
            "parser_selector: fallback page parser produced %d chunks for %s",
            len(chunks),
            source_file,
        )

    return chunks


_PARSE_MODE_ALIASES = {
    "page": page_fallback_parser.PARSER_NAME,
    "page_fallback": page_fallback_parser.PARSER_NAME,
    "question_number": question_number_parser.PARSER_NAME,
    "heading": heading_parser.PARSER_NAME,
}


def _resolve_primary_parser(
    parse_mode: str,
    pages: list[dict],
    *,
    pdf_mode: PdfMode | None = None,
) -> str:
    if parse_mode == "auto":
        return detect_strategy(pages, pdf_mode=pdf_mode)

    resolved = _PARSE_MODE_ALIASES.get(parse_mode)
    if resolved is not None:
        return resolved

    logger.warning(
        "parser_selector: unknown parse_mode=%r, falling back to auto-detection",
        parse_mode,
    )
    return detect_strategy(pages, pdf_mode=pdf_mode)


def _safe_parse(parser_name: str, pages: list[dict], source_file: str) -> list[QuestionChunk]:
    """Run a parser by name, swallowing exceptions so the caller can fall back."""
    parser_fn = _PARSERS[parser_name]
    try:
        return parser_fn(pages, source_file)
    except Exception:
        logger.exception(
            "parser_selector: parser=%s raised while parsing %s",
            parser_name,
            source_file,
        )
        return []


_PARSERS = {
    question_number_parser.PARSER_NAME: question_number_parser.parse,
    heading_parser.PARSER_NAME: heading_parser.parse,
    page_fallback_parser.PARSER_NAME: page_fallback_parser.parse,
}
