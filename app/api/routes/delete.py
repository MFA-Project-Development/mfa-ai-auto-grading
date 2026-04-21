"""Delete endpoints.

Deletions cascade across all three stores:

    MinIO:    object is removed from the bucket using the stored
              ``object_name`` (failures surface as HTTP 502 rather than
              being silently ignored).
    ChromaDB: every vector associated with the file (explicit
              ``vector_id`` values from SQL, plus a ``file_id`` metadata
              sweep to catch stragglers).
    SQL:      ``AnswerKeyItem`` rows cascade via the FK; the
              ``AnswerKeyFile`` row is then dropped.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.models.answer_key_file import AnswerKeyFile
from app.db.session import get_db
from app.repositories import AnswerKeyFileRepository, AnswerKeyItemRepository
from app.services.chroma_service import (
    delete_document_by_id,
    delete_documents_by_file_id,
    delete_documents_by_ids,
)
from app.services.storage import StorageError, get_storage_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/answer-keys", tags=["Answer Key Delete"])


@router.delete("/files/{file_id}")
def delete_answer_key_file(
    file_id: uuid.UUID,
    ignore_storage_errors: bool = Query(
        default=False,
        description=(
            "When true, continue deleting SQL/Chroma data even if the "
            "MinIO object removal fails. Use for cleaning up orphaned rows."
        ),
    ),
    db: Session = Depends(get_db),
) -> dict:
    """Delete a file across MinIO, ChromaDB, and PostgreSQL."""
    file_repo = AnswerKeyFileRepository(db)
    item_repo = AnswerKeyItemRepository(db)

    record = file_repo.get(file_id)
    if record is None:
        raise HTTPException(status_code=404, detail="File not found")

    # 1) MinIO: remove the stored object FIRST. Doing it before the SQL
    #    drop means a storage outage does not leave a dangling binary
    #    whose pointer we just erased.
    storage_deleted = _delete_storage_object(
        record=record,
        ignore_errors=ignore_storage_errors,
    )

    # 2) ChromaDB: explicit vector_ids plus a residual sweep by file_id.
    items = item_repo.list_by_file(file_id)
    known_vector_ids = [i.vector_id for i in items if i.vector_id]
    deleted_by_id = delete_documents_by_ids(known_vector_ids) if known_vector_ids else 0
    residual = delete_documents_by_file_id(str(file_id))
    deleted_from_chroma = deleted_by_id + residual["deleted_count"]

    # 3) PostgreSQL: FK cascade drops items with the parent, but we call
    #    delete_by_file explicitly so the returned count is authoritative.
    item_count = item_repo.delete_by_file(file_id)
    file_repo.delete(record)
    db.commit()

    logger.info(
        "delete: file_id=%s items_deleted=%d vectors_deleted=%d storage_deleted=%s",
        file_id, item_count, deleted_from_chroma, storage_deleted,
    )

    return {
        "success": True,
        "fileId": str(file_id),
        "deletedItems": item_count,
        "deletedVectors": deleted_from_chroma,
        "storageDeleted": storage_deleted,
    }


@router.delete("/items/{item_id}")
def delete_answer_key_item(
    item_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> dict:
    """Delete a single item row and its Chroma vector."""
    item_repo = AnswerKeyItemRepository(db)
    record = item_repo.get(item_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Item not found")

    vector_id = record.vector_id
    vector_removed = False
    if vector_id:
        vector_removed = delete_document_by_id(vector_id)

    item_repo.delete(record)
    db.commit()

    return {
        "success": True,
        "itemId": str(item_id),
        "vectorDeleted": vector_removed,
    }


def _delete_storage_object(
    *,
    record: AnswerKeyFile,
    ignore_errors: bool,
) -> bool:
    """Remove the MinIO object referenced by ``record``.

    Returns ``True`` when something was actually removed. Returns
    ``False`` when there was nothing to delete (e.g. the upload failed
    and the row has no ``object_name``). Raises HTTP 502 when MinIO
    rejects the delete and ``ignore_errors`` is ``False``.
    """
    if not record.object_name:
        logger.info(
            "delete: no MinIO object for file_id=%s (object_name is null)",
            record.id,
        )
        return False

    storage = get_storage_service()
    try:
        return storage.delete_object(record.object_name)
    except StorageError as exc:
        logger.exception(
            "delete: MinIO delete failed file_id=%s object=%s",
            record.id, record.object_name,
        )
        if ignore_errors:
            return False
        raise HTTPException(
            status_code=502,
            detail=f"Object storage delete failed: {exc}",
        ) from exc
