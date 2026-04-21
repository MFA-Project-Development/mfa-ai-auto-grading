"""SQLAlchemy model for a parsed answer-key chunk / question row.

One row per chunk produced by the parser pipeline (question number,
heading, or whole-page fallback). Each row carries a ``vector_id``
pointing into ChromaDB so SQL stays the source of truth for structured
metadata while vectors continue to live in the vector store.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.answer_key_file import AnswerKeyFile


class AnswerKeyItem(Base):
    """A single parsed chunk belonging to an ``AnswerKeyFile``."""

    __tablename__ = "answer_key_items"
    __table_args__ = (
        # Question numbers are unique per file (``question_no`` is a string
        # like "0001", "Q1", "2.a", etc.). Uniqueness at the DB level keeps
        # reingestion idempotent.
        UniqueConstraint(
            "file_id",
            "question_no",
            name="uq_answer_key_items_file_id_question_no",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    file_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("answer_key_files.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ------------------------------------------------------------------ chunk payload
    question_no: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    heading_text: Mapped[str | None] = mapped_column(String(512), nullable=True)
    chapter: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # ``content`` is the ORIGINAL, unmodified chunk produced by the parser.
    # We deliberately keep it intact alongside the structured projection
    # below so we can always re-run extraction / reclassification on the
    # raw text without reingesting the PDF.
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Legacy-ish: the "답 ..." line the parser picked up with a simple
    # regex. Superseded by ``final_answer``; retained for backwards
    # compatibility with existing rows and consumers.
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ------------------------------------------------------------------ structured projection
    # These columns hold the cleaned-up, comparison-ready projection of a
    # chunk. Each is nullable so rows ingested before the structured
    # extraction pass existed continue to load cleanly.
    problem_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    solution_steps_json: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True
    )
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    # String rather than native ENUM so we can introduce new answer types
    # without a schema migration. Values come from
    # :class:`app.db.enums.AnswerType`.
    answer_type: Mapped[str | None] = mapped_column(
        String(32), nullable=True, index=True
    )
    formula_list_json: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True
    )

    # ------------------------------------------------------------------ page tracking
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Stored as comma-separated string ("1,2,3") to stay aligned with the
    # existing Chroma metadata shape. Could be swapped for JSONB later.
    page_numbers: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ------------------------------------------------------------------ provenance
    parser_used: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chunk_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # ChromaDB document id. Nullable at insert time; filled in once the
    # vector is actually written to Chroma.
    vector_id: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        unique=True,
        index=True,
    )

    # ------------------------------------------------------------------ timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    file: Mapped["AnswerKeyFile"] = relationship(
        "AnswerKeyFile",
        back_populates="items",
    )

    # ------------------------------------------------------------------ convenience accessors
    # Pydantic's ``AnswerKeyItemRead`` schema surfaces the structured
    # projection as plain list[str] fields. The storage columns end with
    # ``_json`` (reflecting their JSONB type) so the read-only properties
    # below give the schema a stable attribute name to pull from.
    @property
    def solution_steps(self) -> list[str] | None:
        return self.solution_steps_json

    @property
    def formula_list(self) -> list[str] | None:
        return self.formula_list_json

    def __repr__(self) -> str:  # pragma: no cover - debug helper only
        return (
            f"<AnswerKeyItem id={self.id} file_id={self.file_id} "
            f"question_no={self.question_no!r}>"
        )
