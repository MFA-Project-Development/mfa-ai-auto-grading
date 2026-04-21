"""Embedding generation for answer-key chunks.

The embedding text feeds ChromaDB's semantic search. Historically it was a
best-effort concatenation of the raw chunk plus the ``답 ...`` line, which
worked but leaned heavily on noisy OCR output.

Now that each chunk carries a structured projection
(:class:`app.services.answer_extraction.StructuredAnswer`), we prefer to
embed the clean ``problem_text`` / ``final_answer`` / ``solution_steps``
fields. The raw ``content`` is only used as a fallback for legacy callers
that haven't run structured extraction yet.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache
def get_model() -> SentenceTransformer:
    return SentenceTransformer("BAAI/bge-m3")


def build_embedding_text(
    question_no: str,
    content: str,
    answer_text: str | None,
    chapter: str | None,
    problem_text: str | None = None,
    final_answer: str | None = None,
    solution_steps: list[str] | None = None,
) -> str:
    """Build a single string to embed for a chunk.

    Precedence is:

    * ``problem_text`` / ``final_answer`` / ``solution_steps`` when the
      structured projection is available.
    * Raw ``content`` + ``answer_text`` as a fallback, preserving the
      previous behaviour for callers that haven't switched yet.

    Output is capped at 5000 characters so no single chunk can blow up the
    model context window.
    """
    parts: list[str] = [f"Question number: {question_no}"]

    if chapter:
        parts.append(f"Chapter: {chapter}")

    has_structured = bool(
        (problem_text and problem_text.strip())
        or (final_answer and final_answer.strip())
        or solution_steps
    )

    if has_structured:
        if problem_text and problem_text.strip():
            parts.append("Problem:")
            parts.append(" ".join(problem_text.split()))

        if solution_steps:
            cleaned_steps = [
                " ".join(step.split()) for step in solution_steps if step.strip()
            ]
            if cleaned_steps:
                parts.append("Steps:")
                parts.extend(f"= {step}" for step in cleaned_steps)

        if final_answer and final_answer.strip():
            parts.append(f"Answer: {' '.join(final_answer.split())}")
    else:
        parts.append("Content:")
        parts.append(" ".join(content.split()))
        if answer_text:
            parts.append(f"Answer: {' '.join(answer_text.split())}")

    text = "\n".join(parts)
    return text[:5000]


def get_embedding(text: str) -> list[float]:
    model = get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()
