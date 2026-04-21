"""Repository for :class:`AnswerKeyFile`.

All file-level CRUD / lifecycle updates funnel through this class so
endpoints stay small and business logic stays testable. The repository
never commits implicitly - the caller decides transaction boundaries.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.enums import IngestionStatus, PdfMode, SourceType, StorageProvider
from app.db.models.answer_key_file import AnswerKeyFile


class AnswerKeyFileRepository:
    """CRUD helper for ``answer_key_files`` rows."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------ create
    def create(
        self,
        *,
        file_name: str,
        original_file_name: str,
        source_type: SourceType | str,
        mime_type: str | None = None,
        file_size: int | None = None,
        storage_path: str | None = None,
        pdf_mode: PdfMode | str | None = None,
        subject: str | None = None,
        grade: str | None = None,
        language: str | None = None,
        ingestion_status: IngestionStatus | str = IngestionStatus.UPLOADED,
    ) -> AnswerKeyFile:
        """Insert a new file row and flush so ``id`` is available."""
        record = AnswerKeyFile(
            file_name=file_name,
            original_file_name=original_file_name,
            source_type=_as_value(source_type),
            mime_type=mime_type,
            file_size=file_size,
            storage_path=storage_path,
            pdf_mode=_as_value(pdf_mode) if pdf_mode is not None else None,
            subject=subject,
            grade=grade,
            language=language,
            ingestion_status=_as_value(ingestion_status),
        )
        self.db.add(record)
        self.db.flush()
        return record

    # ------------------------------------------------------------------ read
    def get(self, file_id: uuid.UUID | str) -> AnswerKeyFile | None:
        """Return a file by id, or ``None`` when it does not exist."""
        return self.db.get(AnswerKeyFile, _as_uuid(file_id))

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: IngestionStatus | str | None = None,
    ) -> list[AnswerKeyFile]:
        """Return a page of files, newest first, optionally filtered by status."""
        stmt = select(AnswerKeyFile).order_by(AnswerKeyFile.created_at.desc())
        if status is not None:
            stmt = stmt.where(AnswerKeyFile.ingestion_status == _as_value(status))
        stmt = stmt.offset(offset).limit(limit)
        return list(self.db.scalars(stmt).all())

    # ------------------------------------------------------------------ update
    def update_status(
        self,
        record: AnswerKeyFile,
        status: IngestionStatus | str,
        *,
        error_message: str | None = None,
    ) -> AnswerKeyFile:
        """Update lifecycle status. Clears ``error_message`` on non-failed states."""
        record.ingestion_status = _as_value(status)
        if _as_value(status) == IngestionStatus.FAILED.value:
            record.error_message = error_message
        else:
            record.error_message = None
        self.db.flush()
        return record

    def update_parser_used(self, record: AnswerKeyFile, parser_used: str) -> AnswerKeyFile:
        record.parser_used = parser_used
        self.db.flush()
        return record

    def update_totals(
        self,
        record: AnswerKeyFile,
        *,
        total_pages: int | None = None,
        total_chunks: int | None = None,
    ) -> AnswerKeyFile:
        if total_pages is not None:
            record.total_pages = total_pages
        if total_chunks is not None:
            record.total_chunks = total_chunks
        self.db.flush()
        return record

    def set_pdf_mode(self, record: AnswerKeyFile, pdf_mode: PdfMode | str) -> AnswerKeyFile:
        record.pdf_mode = _as_value(pdf_mode)
        self.db.flush()
        return record

    def set_storage_metadata(
        self,
        record: AnswerKeyFile,
        *,
        provider: StorageProvider | str,
        bucket_name: str,
        object_name: str,
        stored_file_name: str | None = None,
        object_etag: str | None = None,
    ) -> AnswerKeyFile:
        """Persist the result of a successful MinIO/object-storage upload."""
        record.storage_provider = _as_value(provider)
        record.bucket_name = bucket_name
        record.object_name = object_name
        record.stored_file_name = stored_file_name
        record.object_etag = object_etag
        self.db.flush()
        return record

    def clear_storage_metadata(self, record: AnswerKeyFile) -> AnswerKeyFile:
        """Wipe object-storage fields (used after a MinIO delete)."""
        record.storage_provider = None
        record.bucket_name = None
        record.object_name = None
        record.stored_file_name = None
        record.object_etag = None
        self.db.flush()
        return record

    def set_error(self, record: AnswerKeyFile, message: str) -> AnswerKeyFile:
        """Convenience for the failure path: mark failed + persist message."""
        record.ingestion_status = IngestionStatus.FAILED.value
        record.error_message = message
        self.db.flush()
        return record

    def update_fields(self, record: AnswerKeyFile, **fields: Any) -> AnswerKeyFile:
        """Bulk field setter. Unknown attributes are ignored on purpose."""
        for key, value in fields.items():
            if hasattr(record, key):
                setattr(record, key, value)
        self.db.flush()
        return record

    # ------------------------------------------------------------------ delete
    def delete(self, record: AnswerKeyFile) -> None:
        """Delete the file row. Child items cascade via the FK."""
        self.db.delete(record)
        self.db.flush()


# ---------------------------------------------------------------------- helpers
def _as_value(value: Any) -> str:
    """Coerce either an Enum or raw string to its stored string value."""
    return value.value if hasattr(value, "value") else str(value)


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
