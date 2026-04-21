"""Turn a raw answer-key chunk into a structured, comparison-ready item.

A chunk emitted by the parser pipeline is still mostly a block of
question-number text; the fields we care about (problem statement,
intermediate work, final answer) sit on separate lines that a human can
read but that machine-matching cannot easily consume.

This module is responsible for that secondary pass:

* Peel off the leading question-number token (``0001`` / ``Q1`` / ``page_3``).
* Detect multi-part items (``⑴``, ``⑵`` …) and pass them through as-is in
  ``problem_text`` / ``final_answer`` so the caller still has access to the
  original groupings; a future iteration can split each part into its own
  :class:`StructuredAnswer`.
* Pick up the first non-``=``-leading line as ``problem_text``.
* Collect lines that start with ``=`` as ``solution_steps``.
* Pick up whatever follows the Korean ``답`` marker as ``final_answer``, with
  a fallback to the last ``=`` step when no ``답`` marker is present.
* Run :func:`normalize_answer_text` over the final answer.
* Collect every ``= <expr>`` fragment into ``formula_list`` for partial-
  match / step-level grading.
* Classify the item with :func:`classify_answer_type`.

The function is intentionally tolerant: answer keys vary wildly in layout,
so on any ambiguous input we prefer returning ``None`` fields over making
up data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.db.enums import AnswerType
from app.services.answer_extraction.classifier import (
    PARENTHESIZED_MARKER_RE,
    classify_answer_type,
)
from app.services.answer_extraction.normalizer import normalize_answer_text

# The Korean answer marker. The parser pipeline already pulls the ``답 ...``
# line out once via :func:`app.services.parsers.base.extract_answer_text`;
# we still scan here so this module is usable standalone.
_ANSWER_MARKER_RE = re.compile(r"(?m)^\s*답\s*[:：]?\s*(.+)$")

# Matches lines that are pure "solution steps" (``= <expr>``).
_STEP_LINE_RE = re.compile(r"(?m)^\s*=\s*(.+?)\s*$")

# Matches the question-number prefix the parser writes into each chunk
# (``^\d{4}`` or heading-based like ``Question 12``). Used to strip it so
# the first "real" line of the chunk is the problem statement.
_QUESTION_PREFIX_RE = re.compile(
    r"""
    ^\s*
    (?:
        \d{1,4}                          # 0001 / 42
        |
        (?:Question|Problem|Exercise|Example|Q|No)\s*\.?\s*\d+
    )
    \s*[:.\-)]?\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Matches an ``=``-joined expression anywhere on a line. Used when a
# formula is written on a single line (``(x+1)(x-2) = x^2 - x - 2``) so we
# can collect both halves into ``formula_list``.
_INLINE_EQUATION_RE = re.compile(r"([^\n=]+?=[^\n=]+)")


@dataclass(slots=True)
class StructuredAnswer:
    """Comparison-ready projection of a single answer-key chunk.

    Attributes:
        problem_text: The original problem statement, or ``None`` if the
            chunk only contained an answer.
        solution_steps: Ordered list of intermediate ``=`` steps shown in
            the answer key.
        final_answer: Raw final answer text (what appeared after ``답``)
            before normalization.
        normalized_answer: Output of :func:`normalize_answer_text`.
        answer_type: Rule-based classification.
        formula_list: Every distinct formula fragment found in the chunk -
            includes both step lines and inline ``lhs = rhs`` expressions.
            Useful for partial step matching against a student's work.
        sub_part_count: Number of ``⑴``/``⑵``... markers detected; the
            classifier uses this to tag ``MULTIPLE_PART_ANSWER``.
    """

    problem_text: str | None = None
    solution_steps: list[str] = field(default_factory=list)
    final_answer: str | None = None
    normalized_answer: str | None = None
    answer_type: AnswerType = AnswerType.UNKNOWN
    formula_list: list[str] = field(default_factory=list)
    sub_part_count: int = 0


def extract_structured_answer(
    content: str,
    *,
    explicit_answer_text: str | None = None,
    question_no: str | None = None,
) -> StructuredAnswer:
    """Parse ``content`` into a :class:`StructuredAnswer`.

    Args:
        content: Full chunk text (what the parsers store as
            ``QuestionChunk.content``).
        explicit_answer_text: The ``answer_text`` value the parser already
            extracted (via :func:`extract_answer_text`). When provided we
            prefer it over re-scanning ``content`` for the ``답`` marker.
        question_no: Optional question number; if present we ensure the
            leading token is stripped from the problem statement even when
            the parser left it embedded.

    Returns:
        A fully populated :class:`StructuredAnswer`. All fields may be
        ``None``/empty for unusable input - the function never raises.
    """
    result = StructuredAnswer()
    if not content or not content.strip():
        return result

    stripped = content.strip()
    result.sub_part_count = len(PARENTHESIZED_MARKER_RE.findall(stripped))

    problem_text, step_lines = _split_problem_and_steps(
        stripped, question_no=question_no
    )
    result.problem_text = problem_text or None
    result.solution_steps = step_lines

    final_raw = explicit_answer_text
    if not final_raw:
        final_raw = _find_answer_marker(stripped)
    if not final_raw and step_lines:
        final_raw = step_lines[-1]
    result.final_answer = final_raw.strip() if final_raw else None

    result.normalized_answer = normalize_answer_text(result.final_answer)

    result.formula_list = _collect_formula_list(stripped, step_lines)

    result.answer_type = classify_answer_type(
        result.normalized_answer,
        raw_content=stripped,
        sub_part_count=result.sub_part_count,
    )
    return result


# ---------------------------------------------------------------------- internals
def _split_problem_and_steps(
    text: str, *, question_no: str | None
) -> tuple[str, list[str]]:
    """Split a chunk into ``(problem_text, [step1, step2, ...])``.

    Algorithm:
        1. Tokenize by line.
        2. Drop the first line's question-number prefix (``0001`` / ``Q1`` /
           ``page_N``). If ``question_no`` is known, also strip that exact
           token from the very start.
        3. The first non-empty line that does NOT start with ``=`` and is
           NOT an answer marker line becomes ``problem_text``.
        4. All subsequent lines starting with ``=`` become solution steps.
        5. Stop collecting steps once we hit the ``답`` marker.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", []

    first = lines[0]
    first = _QUESTION_PREFIX_RE.sub("", first, count=1).strip()
    if question_no and first.startswith(question_no):
        first = first[len(question_no):].lstrip(" :.-)\t").strip()
    if first:
        lines[0] = first
    else:
        lines.pop(0)
        if not lines:
            return "", []

    problem_text: str | None = None
    step_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("답"):
            break
        if line.startswith("="):
            step_match = _STEP_LINE_RE.match(line)
            if step_match:
                step_lines.append(step_match.group(1).strip())
            continue

        if problem_text is None:
            problem_text = line

    return problem_text or "", step_lines


def _find_answer_marker(text: str) -> str | None:
    match = _ANSWER_MARKER_RE.search(text)
    if not match:
        return None
    return match.group(1).strip() or None


def _collect_formula_list(text: str, step_lines: list[str]) -> list[str]:
    """Gather every formula-looking fragment.

    * All captured ``= <expr>`` step lines (already stripped of the leading
      ``=``) are included verbatim.
    * Any line that contains an inline ``lhs = rhs`` equation is added too.
    * Duplicates are removed while preserving first-seen order so the list
      is stable across reingestions.
    """
    seen: set[str] = set()
    formulas: list[str] = []

    for step in step_lines:
        normalized = normalize_answer_text(step) or step
        if normalized and normalized not in seen:
            seen.add(normalized)
            formulas.append(normalized)

    for line in text.splitlines():
        for match in _INLINE_EQUATION_RE.finditer(line):
            fragment = match.group(1).strip()
            if not fragment or fragment.startswith("답"):
                continue
            normalized = normalize_answer_text(fragment) or fragment
            if normalized and normalized not in seen:
                seen.add(normalized)
                formulas.append(normalized)

    return formulas
