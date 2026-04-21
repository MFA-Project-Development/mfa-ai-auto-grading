"""Helpers for turning raw answer-key chunks into comparison-ready items.

The upload pipeline reads scanned or native PDFs, chunks them by question,
and then passes those chunks through this package so each answer-key item
carries structured fields that student answers can be matched against:

* ``problem_text``       : the original problem statement (first expression).
* ``solution_steps``     : intermediate ``=`` lines that show the work.
* ``final_answer``       : the raw answer text that followed the ``답`` marker.
* ``normalized_answer``  : whitespace/operator-normalized version of the final
  answer, suitable for direct string comparison with a normalized student
  answer.
* ``answer_type``        : simple rule-based classification (numeric,
  polynomial_expression, factored_expression, equation_solution,
  multiple_part_answer).
* ``formula_list``       : every distinct expression (``...= ...``) that appears
  in the chunk, useful for partial / step-level matching later.

Re-exports:
    :class:`StructuredAnswer` – pydantic-free dataclass carrying all of the
    above, ready to be attached onto a :class:`QuestionChunk`.
    :func:`extract_structured_answer` – top-level entry point.
    :func:`normalize_answer_text`     – pure normalization helper.
    :func:`classify_answer_type`      – pure classification helper.
"""

from __future__ import annotations

from app.services.answer_extraction.classifier import classify_answer_type
from app.services.answer_extraction.normalizer import normalize_answer_text
from app.services.answer_extraction.structured_parser import (
    StructuredAnswer,
    extract_structured_answer,
)

__all__ = [
    "StructuredAnswer",
    "classify_answer_type",
    "extract_structured_answer",
    "normalize_answer_text",
]
