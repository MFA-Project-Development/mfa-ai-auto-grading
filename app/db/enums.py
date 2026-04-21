"""Enum types for the SQL layer.

Kept in one module so both SQLAlchemy models and service/repository code
share a single source of truth for valid values. We use Python ``str``
enums so values serialise cleanly through FastAPI / Pydantic and compare
directly against the strings stored in Postgres ``VARCHAR`` columns.
"""

from __future__ import annotations

from enum import Enum


class IngestionStatus(str, Enum):
    """Lifecycle of an uploaded answer-key file.

    Flow:
        UPLOADING -> UPLOADED -> PROCESSING -> COMPLETED
    with FAILED and DELETED as terminal states that can be reached from
    any step.
    """

    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class StorageProvider(str, Enum):
    """Where the binary file for an ``AnswerKeyFile`` lives."""

    MINIO = "minio"
    LOCAL = "local"


class SourceType(str, Enum):
    """How the user supplied the answer key."""

    PDF = "pdf"
    IMAGE = "image"


class PdfMode(str, Enum):
    """Outcome of the PDF text-vs-image classifier (nullable for images)."""

    TEXT_PDF = "text_pdf"
    IMAGE_PDF = "image_pdf"


class ChunkType(str, Enum):
    """Which parser produced the chunk row."""

    QUESTION = "question"
    HEADING = "heading"
    PAGE_FALLBACK = "page_fallback"


class AnswerType(str, Enum):
    """Classification of a parsed answer so graders can pick a matcher.

    * ``NUMERIC``              : plain numeric value (``42``, ``3.14``, ``-7/2``).
    * ``POLYNOMIAL_EXPRESSION``: algebraic expression in expanded form
      (``3x^2 - xy + 4y^2``).
    * ``FACTORED_EXPRESSION``  : expression written as a product of factors
      (``(x+1)(x-2)``).
    * ``EQUATION_SOLUTION``    : answer contains an ``=`` sign or an explicit
      solution set (``x = 3``, ``x = 1, 2``).
    * ``MULTIPLE_PART_ANSWER`` : item carries ``⑴``/``⑵``/... sub-parts.
    * ``UNKNOWN``              : nothing above matched.
    """

    NUMERIC = "numeric"
    POLYNOMIAL_EXPRESSION = "polynomial_expression"
    FACTORED_EXPRESSION = "factored_expression"
    EQUATION_SOLUTION = "equation_solution"
    MULTIPLE_PART_ANSWER = "multiple_part_answer"
    UNKNOWN = "unknown"


# Convenience sets for validation / filtering.
INGESTION_STATUS_VALUES: tuple[str, ...] = tuple(s.value for s in IngestionStatus)
SOURCE_TYPE_VALUES: tuple[str, ...] = tuple(s.value for s in SourceType)
PDF_MODE_VALUES: tuple[str, ...] = tuple(s.value for s in PdfMode)
CHUNK_TYPE_VALUES: tuple[str, ...] = tuple(s.value for s in ChunkType)
STORAGE_PROVIDER_VALUES: tuple[str, ...] = tuple(s.value for s in StorageProvider)
ANSWER_TYPE_VALUES: tuple[str, ...] = tuple(s.value for s in AnswerType)
