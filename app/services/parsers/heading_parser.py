"""Parser for PDFs structured by human-readable headings.

Handles documents whose question boundaries are markers like ``Question 1``,
``Problem 2``, ``Exercise 3``, ``Example 4``, ``Q5`` or ``No. 6`` rather than
numeric 4-digit codes. Matching is case-insensitive and anchored to line
starts to limit false positives from incidental prose.
"""

from __future__ import annotations

import re

from app.models.schemas import QuestionChunk
from app.services.parsers.base import (
    attach_structured_answer,
    build_combined_text,
    extract_answer_text,
    infer_page_numbers_for_span,
    normalize_whitespace,
    resolve_chapter_for_pages,
)

PARSER_NAME = "heading"

HEADING_PATTERN = re.compile(
    r"""
    (?mi)                                    # multiline + case-insensitive
    ^[\s\u00b7\-\*\u2022]*                   # leading whitespace/bullets
    (?P<label>
        Question | Problem | Exercise | Example | Q | No
    )
    \s*\.?\s*                                # optional dot + spacing
    (?P<num>\d{1,4})
    \b                                       # avoid matching partial digits
    """,
    re.VERBOSE,
)


def parse(pages: list[dict], source_file: str) -> list[QuestionChunk]:
    """Split pages into chunks using heading markers as boundaries.

    Returns an empty list when no headings are detected so the selector can
    fall back to another strategy.
    """
    full_text, page_offsets, page_chapter_map = build_combined_text(pages)
    if not full_text:
        return []

    matches = list(HEADING_PATTERN.finditer(full_text))
    if not matches:
        return []

    used_ids: set[str] = set()
    chunks: list[QuestionChunk] = []

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)

        raw_chunk = normalize_whitespace(full_text[start:end].strip())
        if not raw_chunk:
            continue

        heading_text = match.group(0).strip()
        number_token = match.group("num")
        question_no = _make_unique_question_no(number_token, used_ids)
        chunk_pages = infer_page_numbers_for_span(start, end, page_offsets)

        chunk = QuestionChunk(
            question_no=question_no,
            content=raw_chunk,
            page_numbers=chunk_pages,
            chapter=resolve_chapter_for_pages(chunk_pages, page_chapter_map),
            answer_text=extract_answer_text(raw_chunk),
            source_file=source_file,
            parser_used=PARSER_NAME,
            heading_text=heading_text,
        )
        chunks.append(attach_structured_answer(chunk))

    return chunks


def count_heading_markers(text: str) -> int:
    """Return how many heading markers appear in ``text``.

    Useful for the parser selector's heuristic without re-running the full
    parse pipeline.
    """
    return len(HEADING_PATTERN.findall(text))


def _make_unique_question_no(number_token: str, used_ids: set[str]) -> str:
    """Build a stable, unique ``heading_{num}`` id, de-duplicating on collision."""
    base = f"heading_{number_token}"
    if base not in used_ids:
        used_ids.add(base)
        return base

    suffix = 2
    while True:
        candidate = f"{base}_{suffix}"
        if candidate not in used_ids:   
            used_ids.add(candidate)
            return candidate
        suffix += 1
