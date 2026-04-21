"""Parser for structured PDFs that use 4-digit question numbers (e.g. ``0001``)."""

from __future__ import annotations

from app.models.schemas import QuestionChunk
from app.services.parsers.base import (
    QUESTION_NUMBER_PATTERN,
    attach_structured_answer,
    build_combined_text,
    extract_answer_text,
    infer_page_numbers_for_span,
    normalize_whitespace,
    resolve_chapter_for_pages,
)

PARSER_NAME = "question_number"


def parse(pages: list[dict], source_file: str) -> list[QuestionChunk]:
    """Split pages into chunks using 4-digit question-number anchors.

    Each match of ``^\\d{4}`` starts a new question; text between matches is
    treated as that question's content. Returns an empty list when no anchors
    are found so the selector can fall back to a different strategy.
    """
    full_text, page_offsets, page_chapter_map = build_combined_text(pages)
    if not full_text:
        return []

    matches = list(QUESTION_NUMBER_PATTERN.finditer(full_text))
    if not matches:
        return []

    chunks: list[QuestionChunk] = []
    for idx, match in enumerate(matches):
        question_no = match.group(1)
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)

        raw_chunk = normalize_whitespace(full_text[start:end].strip())
        chunk_pages = infer_page_numbers_for_span(start, end, page_offsets)

        chunk = QuestionChunk(
            question_no=question_no,
            content=raw_chunk,
            page_numbers=chunk_pages,
            chapter=resolve_chapter_for_pages(chunk_pages, page_chapter_map),
            answer_text=extract_answer_text(raw_chunk),
            source_file=source_file,
            parser_used=PARSER_NAME,
        )
        chunks.append(attach_structured_answer(chunk))

    return chunks
