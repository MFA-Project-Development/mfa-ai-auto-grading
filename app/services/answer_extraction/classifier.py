"""Rule-based classification of answer-key items.

The matcher used by the grader can pick a different strategy per
:class:`AnswerType` (e.g. compare numeric answers with a tolerance, compare
polynomial expressions after normalization, treat equation solutions as a
set). This module makes that initial classification cheap and dependency-
free so ingestion stays fast.

We intentionally avoid a heavy algebra CAS here; the classifier works by
pattern-matching on the already-normalized final answer and on whether
the chunk carries sub-part markers like ``⑴`` ``⑵``.
"""

from __future__ import annotations

import re

from app.db.enums import AnswerType

# Detects sub-part markers commonly used in Korean math answer keys.
# Covers parenthesised digits ``⑴``-``⒇`` (U+2474-U+2487), negative
# circled digits ``⓵``-``⓾`` (U+24F5-U+24FE), plain circled digits
# ``①``-``⑳`` (U+2460-U+2473), and the dingbat-style circles
# ``❶``-``➓`` (U+2776-U+2793) so a single regex works on a wide variety
# of OCR'd answer keys.
PARENTHESIZED_MARKER_RE = re.compile(
    r"[\u2460-\u2473\u2474-\u2487\u24f5-\u24fe\u2776-\u2793]"
)

# Pure-number answers: integers, decimals, signed, simple fractions.
# Accepts ``42`` / ``-3.14`` / ``1/2`` / ``-7/2``.
NUMERIC_RE = re.compile(r"^[+\-]?\d+(?:\.\d+)?(?:\s*/\s*[+\-]?\d+(?:\.\d+)?)?$")

# Factored expression: looks like one or more parenthesised groups
# concatenated (optionally with leading coefficient / sign). Examples:
# ``(x+1)(x-2)``, ``-2(x+1)^2(x-3)``, ``(x - y)(x + y)``.
FACTORED_RE = re.compile(
    r"^[+\-]?\s*\d*\s*"
    r"(?:\([^()]+\)(?:\^\d+)?\s*){2,}$"
)

# Explicit equation / solution set: the normalized answer contains an ``=``
# sign (``x = 3``) or a solution-set shape (``x = 1, 2``).
EQUATION_RE = re.compile(r"=")

# Polynomial-ish: contains a letter (variable) and at least one arithmetic
# operator. We don't try to validate the expression; anything that looks
# like "letters + operators" and isn't already factored/equation lands
# here.
POLYNOMIAL_RE = re.compile(r"[A-Za-z]")
ARITHMETIC_RE = re.compile(r"[+\-*/^]")


def classify_answer_type(
    normalized_answer: str | None,
    *,
    raw_content: str | None = None,
    sub_part_count: int = 0,
) -> AnswerType:
    """Return the best :class:`AnswerType` guess for an answer-key item.

    Args:
        normalized_answer: Output of
            :func:`app.services.answer_extraction.normalizer.normalize_answer_text`.
            Passing the normalized form lets the regexes stay simple.
        raw_content: Optional full chunk content. Used to detect multi-part
            answers when the final answer itself does not carry the markers.
        sub_part_count: Precomputed count of detected ``⑴`` / ``⑵`` markers,
            supplied by the structured parser. When ``>= 2`` the item is
            classified as :attr:`AnswerType.MULTIPLE_PART_ANSWER` regardless
            of shape (an item with e.g. both ``⑴ 3x+1`` and ``⑵ x=2``
            cannot be cleanly placed in a single-value type).
        raw_content: if supplied and markers are present we bump
            ``sub_part_count`` automatically so callers can pass only the
            raw chunk for a quick classification.

    Returns:
        The matching :class:`AnswerType`. Falls back to
        :attr:`AnswerType.UNKNOWN` when nothing plausible matches.
    """
    effective_parts = sub_part_count
    if raw_content and effective_parts < 2:
        effective_parts = max(
            effective_parts,
            len(PARENTHESIZED_MARKER_RE.findall(raw_content)),
        )

    if effective_parts >= 2:
        return AnswerType.MULTIPLE_PART_ANSWER

    if not normalized_answer:
        return AnswerType.UNKNOWN

    stripped = normalized_answer.strip()

    if NUMERIC_RE.match(stripped):
        return AnswerType.NUMERIC

    if FACTORED_RE.match(stripped):
        return AnswerType.FACTORED_EXPRESSION

    if EQUATION_RE.search(stripped):
        return AnswerType.EQUATION_SOLUTION

    if POLYNOMIAL_RE.search(stripped) and ARITHMETIC_RE.search(stripped):
        return AnswerType.POLYNOMIAL_EXPRESSION

    return AnswerType.UNKNOWN
