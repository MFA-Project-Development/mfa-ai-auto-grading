"""Pydantic DTOs.

Kept separate from the SQLAlchemy models so the API surface can evolve
independently of the storage schema. ``QuestionChunk`` is the internal
representation produced by the parser pipeline; everything ``Response`` is
what the HTTP layer emits.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class QuestionChunk(BaseModel):
    question_no: str = Field(..., description="Question number like 0001")
    content: str = Field(..., description="Full chunk text for this question")
    page_numbers: list[int] = Field(default_factory=list)
    chapter: str | None = None
    answer_text: str | None = None
    source_file: str
    parser_used: str | None = Field(
        default=None,
        description=(
            "Name of the parser that produced this chunk "
            "(e.g. question_number, heading, page_fallback)"
        ),
    )
    heading_text: str | None = Field(
        default=None,
        description="Original heading text captured by heading-based parsers",
    )

    # -------------------------------------------------------------- structured projection
    # Populated by :mod:`app.services.answer_extraction` after raw parsing.
    # All of these fields are optional so callers that haven't (yet) run
    # the structured pass keep working unchanged.
    problem_text: str | None = Field(
        default=None,
        description="First non-'=' line after the question number: the problem statement.",
    )
    solution_steps: list[str] = Field(
        default_factory=list,
        description="Ordered '=' step lines extracted from the answer-key chunk.",
    )
    final_answer: str | None = Field(
        default=None,
        description="Raw final answer text (what followed the '답' marker).",
    )
    normalized_answer: str | None = Field(
        default=None,
        description="Whitespace/operator-normalized final answer for direct matching.",
    )
    answer_type: str | None = Field(
        default=None,
        description="Rule-based AnswerType value (numeric, polynomial_expression, ...).",
    )
    formula_list: list[str] = Field(
        default_factory=list,
        description="Every formula fragment found in the chunk, used for step matching.",
    )


class UploadResponse(BaseModel):
    success: bool
    label: str
    fileId: uuid.UUID
    ingestionStatus: str
    parserUsed: str | None = None
    totalQuestions: int


class AnswerKeyFileRead(BaseModel):
    """File-level response shape used by list/detail endpoints."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    label: str = Field(validation_alias="file_name")
    original_file_name: str
    storage_path: str | None
    mime_type: str | None
    file_size: int | None
    source_type: str
    pdf_mode: str | None
    parser_used: str | None
    ingestion_status: str
    total_pages: int | None
    total_chunks: int | None
    subject: str | None
    grade: str | None
    language: str | None
    error_message: str | None
    # Object storage metadata (MinIO / S3-compatible).
    storage_provider: str | None
    bucket_name: str | None
    object_name: str | None
    object_etag: str | None
    stored_file_name: str | None
    created_at: datetime
    updated_at: datetime


class PresignedUrlResponse(BaseModel):
    """Response payload for the presigned download endpoint."""

    success: bool = True
    file_id: uuid.UUID
    object_name: str
    bucket_name: str
    url: str
    expires_in_seconds: int


class AnswerKeyFileListResponse(BaseModel):
    success: bool = True
    count: int
    results: list[AnswerKeyFileRead]


class AnswerKeyItemRead(BaseModel):
    """Item-level response shape used by list/detail endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    file_id: uuid.UUID
    question_no: str
    heading_text: str | None
    chapter: str | None
    content: str
    answer_text: str | None
    # Structured, comparison-ready projection of the chunk. May be absent
    # on rows ingested before the structured extraction pass existed.
    problem_text: str | None = None
    solution_steps: list[str] | None = None
    final_answer: str | None = None
    normalized_answer: str | None = None
    answer_type: str | None = None
    formula_list: list[str] | None = None
    page_start: int | None
    page_end: int | None
    page_numbers: str | None
    parser_used: str | None
    chunk_type: str
    vector_id: str | None
    created_at: datetime
    updated_at: datetime


class AnswerKeyItemListResponse(BaseModel):
    success: bool = True
    count: int
    results: list[AnswerKeyItemRead]
