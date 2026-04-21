"""Fallback parser that emits one chunk per non-empty page.

Used when the document has no consistent question-number markers, or when the
structured parser fails. It guarantees the pipeline can always produce at
least one embeddable chunk from any readable PDF.
"""

from __future__ import annotations

from app.models.schemas import QuestionChunk
from app.services.parsers.base import (
    attach_structured_answer,
    extract_answer_text,
    extract_page_local_chapter,
    normalize_whitespace,
)

PARSER_NAME = "page_fallback"


def parse(pages: list[dict], source_file: str) -> list[QuestionChunk]:
    """Return one ``QuestionChunk`` per non-empty page.

    The synthetic question number uses the pattern ``page_{n}`` so chunks are
    still uniquely identifiable in ChromaDB.
    """
    chunks: list[QuestionChunk] = []
    current_chapter: str | None = None

    for page in pages:
        text = normalize_whitespace(page.get("text", ""))
        if not text:
            continue

        # Only trust chapter candidates found near the TOP of a page so we
        # don't promote a random "01 ..." line buried in body text into a
        # chapter header for the whole run.
        local_chapter = extract_page_local_chapter(text)
        if local_chapter:
            current_chapter = local_chapter

        page_number = page["page_number"]
        chunk = QuestionChunk(
            question_no=f"page_{page_number}",
            content=text,
            page_numbers=[page_number],
            chapter=current_chapter,
            answer_text=extract_answer_text(text),
            source_file=source_file,
            parser_used=PARSER_NAME,
        )
        chunks.append(attach_structured_answer(chunk))

    return chunks
