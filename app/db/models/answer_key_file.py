"""SQLAlchemy model for an uploaded answer-key file.

One row per upload. Acts as the authoritative record for file-level
metadata, ingestion lifecycle, and the parent side of the
``AnswerKeyFile -> AnswerKeyItem`` relationship. Per-chunk data lives on
:class:`app.db.models.answer_key_item.AnswerKeyItem`, and actual embedding
vectors still live in ChromaDB.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.enums import IngestionStatus

if TYPE_CHECKING:
    from app.db.models.answer_key_item import AnswerKeyItem


class AnswerKeyFile(Base):
    """An uploaded answer-key PDF or image and its ingestion metadata."""

    __tablename__ = "answer_key_files"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # ------------------------------------------------------------------ file
    file_name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    original_file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    # Legacy / optional: filesystem path when files are mirrored locally.
    # Kept for backwards compatibility; MinIO is now the primary store.
    storage_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # ------------------------------------------------------------------ object storage (MinIO)
    # ``storage_provider``: "minio" | "local" - see StorageProvider enum.
    # ``bucket_name``     : MinIO bucket the object was placed in.
    # ``object_name``     : Fully-qualified object key inside the bucket
    #                       (e.g. "answer-keys/<uuid>/original.pdf").
    # ``object_etag``     : ETag returned by MinIO on PUT; useful for
    #                       integrity checks during reprocessing.
    # ``stored_file_name``: The canonical on-disk filename inside the
    #                       object (e.g. "original.pdf"). Separate from
    #                       ``object_name`` so we can display something
    #                       user-friendly without parsing the full key.
    storage_provider: Mapped[str | None] = mapped_column(String(16), nullable=True)
    bucket_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    object_name: Mapped[str | None] = mapped_column(String(512), nullable=True, index=True)
    object_etag: Mapped[str | None] = mapped_column(String(128), nullable=True)
    stored_file_name: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ------------------------------------------------------------------ classification
    # ``source_type``  : "pdf" | "image"
    # ``pdf_mode``     : "text_pdf" | "image_pdf" | NULL (non-PDF uploads)
    # ``parser_used``  : filled in after parser_selector runs
    source_type: Mapped[str] = mapped_column(String(16), nullable=False)
    pdf_mode: Mapped[str | None] = mapped_column(String(16), nullable=True)
    parser_used: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # ------------------------------------------------------------------ lifecycle
    ingestion_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=IngestionStatus.UPLOADED.value,
        server_default=IngestionStatus.UPLOADED.value,
        index=True,
    )
    total_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ------------------------------------------------------------------ optional metadata
    subject: Mapped[str | None] = mapped_column(String(128), nullable=True)
    grade: Mapped[str | None] = mapped_column(String(64), nullable=True)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)

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

    # ------------------------------------------------------------------ relationships
    items: Mapped[list["AnswerKeyItem"]] = relationship(
        "AnswerKeyItem",
        back_populates="file",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper only
        return (
            f"<AnswerKeyFile id={self.id} name={self.file_name!r} "
            f"status={self.ingestion_status}>"
        )
