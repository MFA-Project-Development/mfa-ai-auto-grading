from __future__ import annotations

import re

from app.models.schemas import QuestionChunk
from app.services.answer_extraction import extract_structured_answer

# Strict chapter pattern:
#
#   * Two digits, leading zero required for 01-09 (so ``69`` from a page
#     number is NEVER matched). Allowed range 01-29.
#   * At least one space / dot / tab.
#   * A Korean-led title between 2 and 40 chars. Titles that start with a
#     digit (page numbers, question numbers) are rejected.
#   * Pattern is line-anchored in multiline mode so stray "01" tokens mid-
#     paragraph don't trigger.
#
# Designed to match textbook chapter lines like ``01 다항식의 연산`` and to
# reject OCR noise like ``69\n순열과조합`` (digits + newline) or ``0069 텅``.
CHAPTER_PATTERN = re.compile(
    r"(?m)^\s*(0[1-9]|1\d|2\d)\s+([\uac00-\ud7a3][^\n]{2,40})\s*$"
)

ANSWER_PATTERN = re.compile(r"답\s*(.+)")
QUESTION_NUMBER_PATTERN = re.compile(r"(?m)^(\d{4})\b")

# How many lines from the top of a page we're willing to accept as the
# chapter header. Beyond that, a "chapter" candidate is almost certainly a
# false positive (body text that happens to start with two digits).
_CHAPTER_TOP_LINES = 8


def normalize_whitespace(text: str) -> str:
    """Collapse blank lines and trim each line."""
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line]
    return "\n".join(cleaned_lines)


def extract_chapter_candidates(page_text: str) -> list[str]:
    """Return chapter header candidates found anywhere in a page's text.

    Uses the strict :data:`CHAPTER_PATTERN` so page numbers and question
    numbers are not mistaken for chapters. The returned list is normalized to
    ``"NN  TITLE"`` form (single space, no trailing whitespace).
    """
    candidates: list[str] = []
    for match in CHAPTER_PATTERN.finditer(page_text):
        number, title = match.group(1), match.group(2).strip()
        if not title or title[0].isdigit():
            continue
        candidates.append(f"{number} {title}")
    return candidates


def extract_page_local_chapter(
    page_text: str,
    *,
    top_lines: int = _CHAPTER_TOP_LINES,
) -> str | None:
    """Return the chapter header found near the TOP of ``page_text``, if any.

    This is the preferred extractor for per-page chapter assignment: it
    refuses to pick up a chapter-looking line that appears deep in the page
    (which is almost always body text, not a real header).
    """
    lines = (page_text or "").splitlines()
    if not lines:
        return None

    head = "\n".join(lines[:top_lines])
    candidates = extract_chapter_candidates(head)
    return candidates[0] if candidates else None


def extract_answer_text(chunk_text: str) -> str | None:
    """Pick up the `답 ...` line (Korean answer marker) if present."""
    match = ANSWER_PATTERN.search(chunk_text)
    if not match:
        return None

    answer = match.group(1).strip()
    return answer[:500] if answer else None


def build_combined_text(
    pages: list[dict],
) -> tuple[str, list[tuple[int, int, int]], dict[int, str | None]]:
    """Concatenate normalized page texts and compute per-page metadata.

    Returns:
        full_text: concatenated, normalized text.
        page_offsets: list of ``(start, end, page_number)`` spans into
            ``full_text``.
        page_chapter_map: ``{page_number: chapter_or_None}`` where each page's
            chapter is the most recent page-local header seen at or before
            that page. Callers should look up a chunk's chapter by its first
            page; this prevents a late TOC page from poisoning earlier
            chunks' chapter values.
    """
    combined_parts: list[str] = []
    page_offsets: list[tuple[int, int, int]] = []
    page_chapter_map: dict[int, str | None] = {}
    cursor = 0
    current_chapter: str | None = None

    for page in pages:
        page_text = normalize_whitespace(page.get("text", ""))
        if not page_text:
            continue

        local_chapter = extract_page_local_chapter(page_text)
        if local_chapter:
            current_chapter = local_chapter

        page_number = page["page_number"]
        page_chapter_map[page_number] = current_chapter

        block = f"\n{page_text}\n"
        combined_parts.append(block)

        start = cursor
        cursor += len(block)
        end = cursor

        page_offsets.append((start, end, page_number))

    full_text = "".join(combined_parts).strip()
    return full_text, page_offsets, page_chapter_map


def resolve_chapter_for_pages(
    page_numbers: list[int],
    page_chapter_map: dict[int, str | None],
) -> str | None:
    """Pick the best chapter for a chunk given the pages it covers.

    Prefers the chapter active on the chunk's FIRST page, falling back to the
    most recent chapter seen at or before any of the chunk's pages.
    """
    if not page_numbers:
        return None

    first_page = page_numbers[0]
    direct = page_chapter_map.get(first_page)
    if direct:
        return direct

    previous = [
        ch for pg, ch in page_chapter_map.items()
        if ch and pg <= first_page
    ]
    return previous[-1] if previous else None


def infer_page_numbers_for_span(
    start: int,
    end: int,
    page_offsets: list[tuple[int, int, int]],
) -> list[int]:
    """Map a [start, end) span in the combined text to the page numbers it covers."""
    pages: list[int] = []
    for page_start, page_end, page_number in page_offsets:
        if start < page_end and end > page_start:
            pages.append(page_number)
    return pages


def attach_structured_answer(chunk: QuestionChunk) -> QuestionChunk:
    """Enrich ``chunk`` with the structured, comparison-ready projection.

    Central glue between the parsers (which only produce raw chunks) and
    :mod:`app.services.answer_extraction`. Every concrete parser should
    pipe its emitted :class:`QuestionChunk` through this helper before
    returning so the downstream upload pipeline always sees populated
    ``problem_text`` / ``solution_steps`` / ``final_answer`` /
    ``normalized_answer`` / ``answer_type`` / ``formula_list`` fields.

    Safe to call twice - any non-null existing field is preserved. The
    function never raises: on malformed input the chunk is returned with
    whatever partial structured fields could be extracted.
    """
    try:
        structured = extract_structured_answer(
            chunk.content,
            explicit_answer_text=chunk.answer_text,
            question_no=chunk.question_no,
        )
    except Exception:
        return chunk

    return chunk.model_copy(
        update={
            "problem_text": chunk.problem_text or structured.problem_text,
            "solution_steps": chunk.solution_steps or structured.solution_steps,
            "final_answer": chunk.final_answer or structured.final_answer,
            "normalized_answer": (
                chunk.normalized_answer or structured.normalized_answer
            ),
            "answer_type": chunk.answer_type or structured.answer_type.value,
            "formula_list": chunk.formula_list or structured.formula_list,
        }
    )

