"""File- and item-level detail/list endpoints.

Reads go through SQL (the metadata source of truth). ChromaDB is still
available for semantic search via :mod:`app.api.routes.search`.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.enums import IngestionStatus
from app.db.session import get_db
from app.models.schemas import (
    AnswerKeyFileListResponse,
    AnswerKeyFileRead,
    AnswerKeyItemListResponse,
    AnswerKeyItemRead,
    PresignedUrlResponse,
)
from app.repositories import AnswerKeyFileRepository, AnswerKeyItemRepository
from app.services.storage import StorageError, get_storage_service

router = APIRouter(prefix="/api/v1/answer-keys", tags=["Answer Key Detail"])


# ====================================================================== FILES
@router.get("/files", response_model=AnswerKeyFileListResponse)
def list_answer_key_files(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> AnswerKeyFileListResponse:
    """List uploaded files (newest first), optionally filtered by status."""
    _validate_status(status)
    repo = AnswerKeyFileRepository(db)
    records = repo.list(limit=limit, offset=offset, status=status)
    return AnswerKeyFileListResponse(
        count=len(records),
        results=[AnswerKeyFileRead.model_validate(r) for r in records],
    )


@router.get("/files/{file_id}", response_model=AnswerKeyFileRead)
def get_answer_key_file(
    file_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> AnswerKeyFileRead:
    """Return a single file record by id."""
    record = AnswerKeyFileRepository(db).get(file_id)
    if record is None:
        raise HTTPException(status_code=404, detail="File not found")
    return AnswerKeyFileRead.model_validate(record)


@router.get(
    "/files/{file_id}/download-url",
    response_model=PresignedUrlResponse,
)
def get_answer_key_file_download_url(
    file_id: uuid.UUID,
    expires: int = Query(
        default=3600,
        ge=60,
        le=7 * 24 * 3600,
        description="Validity of the presigned URL in seconds (1 min - 7 days).",
    ),
    db: Session = Depends(get_db),
) -> PresignedUrlResponse:
    """Generate a time-limited MinIO GET URL for the stored file.

    Returns 404 if the file row is unknown, 409 if the row exists but no
    MinIO object is attached (e.g. the upload failed before storage), and
    502 if MinIO refuses to mint the URL.
    """
    record = AnswerKeyFileRepository(db).get(file_id)
    if record is None:
        raise HTTPException(status_code=404, detail="File not found")
    if not record.object_name or not record.bucket_name:
        raise HTTPException(
            status_code=409,
            detail="File has no associated object in storage",
        )

    storage = get_storage_service()
    try:
        url = storage.get_presigned_url(record.object_name, expires_seconds=expires)
    except StorageError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not generate download URL: {exc}",
        ) from exc

    return PresignedUrlResponse(
        file_id=record.id,
        object_name=record.object_name,
        bucket_name=record.bucket_name,
        url=url,
        expires_in_seconds=expires,
    )


# ====================================================================== ITEMS
@router.get(
    "/files/{file_id}/items",
    response_model=AnswerKeyItemListResponse,
)
def list_items_for_file(
    file_id: uuid.UUID,
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> AnswerKeyItemListResponse:
    """Return every parsed chunk belonging to ``file_id``."""
    file_repo = AnswerKeyFileRepository(db)
    if file_repo.get(file_id) is None:
        raise HTTPException(status_code=404, detail="File not found")

    item_repo = AnswerKeyItemRepository(db)
    items = item_repo.list_by_file(file_id, limit=limit, offset=offset)
    return AnswerKeyItemListResponse(
        count=len(items),
        results=[AnswerKeyItemRead.model_validate(i) for i in items],
    )


@router.get("/items/{item_id}", response_model=AnswerKeyItemRead)
def get_item_by_id(
    item_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> AnswerKeyItemRead:
    """Return a single parsed chunk by its SQL id."""
    record = AnswerKeyItemRepository(db).get(item_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return AnswerKeyItemRead.model_validate(record)


@router.get(
    "/files/{file_id}/items/by-question/{question_no}",
    response_model=AnswerKeyItemRead,
)
def get_item_by_question_no(
    file_id: uuid.UUID,
    question_no: str,
    db: Session = Depends(get_db),
) -> AnswerKeyItemRead:
    """Return the chunk with ``question_no`` inside the given file."""
    record = AnswerKeyItemRepository(db).get_by_question_no(file_id, question_no)
    if record is None:
        raise HTTPException(status_code=404, detail="Question not found in file")
    return AnswerKeyItemRead.model_validate(record)


# ---------------------------------------------------------------------- helpers
def _validate_status(status: str | None) -> None:
    if status is None:
        return
    allowed = {s.value for s in IngestionStatus}
    if status not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Allowed: {sorted(allowed)}",
        )
