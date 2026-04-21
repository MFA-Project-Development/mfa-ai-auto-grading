"""Repository for :class:`AnswerKeyItem`.

Handles bulk insertion of parsed chunks, lookup by question number, and
cascade-friendly deletion scoped to a single file.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.enums import ChunkType
from app.db.models.answer_key_item import AnswerKeyItem


class AnswerKeyItemRepository:
    """CRUD helper for ``answer_key_items`` rows."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------ create
    def create(
        self,
        *,
        file_id: uuid.UUID,
        question_no: str,
        content: str,
        chunk_type: ChunkType | str,
        heading_text: str | None = None,
        chapter: str | None = None,
        answer_text: str | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
        page_numbers: str | None = None,
        parser_used: str | None = None,
        vector_id: str | None = None,
        item_id: uuid.UUID | None = None,
        problem_text: str | None = None,
        solution_steps: list[str] | None = None,
        final_answer: str | None = None,
        normalized_answer: str | None = None,
        answer_type: str | None = None,
        formula_list: list[str] | None = None,
    ) -> AnswerKeyItem:
        """Insert a single item row and return the flushed ORM instance.

        The ``problem_text`` / ``solution_steps`` / ``final_answer`` /
        ``normalized_answer`` / ``answer_type`` / ``formula_list`` args are
        the structured projection produced by
        :mod:`app.services.answer_extraction`. All are optional so callers
        that haven't enabled the structured pass keep working.
        """
        record = AnswerKeyItem(
            id=item_id or uuid.uuid4(),
            file_id=file_id,
            question_no=question_no,
            heading_text=heading_text,
            chapter=chapter,
            content=content,
            answer_text=answer_text,
            page_start=page_start,
            page_end=page_end,
            page_numbers=page_numbers,
            parser_used=parser_used,
            chunk_type=_as_value(chunk_type),
            vector_id=vector_id,
            problem_text=problem_text,
            solution_steps_json=_clean_list(solution_steps),
            final_answer=final_answer,
            normalized_answer=normalized_answer,
            answer_type=answer_type,
            formula_list_json=_clean_list(formula_list),
        )
        self.db.add(record)
        self.db.flush()
        return record

    # ------------------------------------------------------------------ read
    def get(self, item_id: uuid.UUID | str) -> AnswerKeyItem | None:
        return self.db.get(AnswerKeyItem, _as_uuid(item_id))

    def list_by_file(
        self,
        file_id: uuid.UUID | str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[AnswerKeyItem]:
        """Return all items for a file ordered by ``question_no``."""
        stmt = (
            select(AnswerKeyItem)
            .where(AnswerKeyItem.file_id == _as_uuid(file_id))
            .order_by(AnswerKeyItem.question_no.asc(), AnswerKeyItem.created_at.asc())
        )
        if offset:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.db.scalars(stmt).all())

    def get_by_question_no(
        self,
        file_id: uuid.UUID | str,
        question_no: str,
    ) -> AnswerKeyItem | None:
        """Return one item identified by ``(file_id, question_no)``."""
        stmt = select(AnswerKeyItem).where(
            AnswerKeyItem.file_id == _as_uuid(file_id),
            AnswerKeyItem.question_no == question_no,
        )
        return self.db.scalars(stmt).first()

    # ------------------------------------------------------------------ delete
    def delete(self, record: AnswerKeyItem) -> None:
        self.db.delete(record)
        self.db.flush()

    def delete_by_file(self, file_id: uuid.UUID | str) -> int:
        """Delete every item for ``file_id``. Returns the row count."""
        stmt = delete(AnswerKeyItem).where(AnswerKeyItem.file_id == _as_uuid(file_id))
        result = self.db.execute(stmt)
        self.db.flush()
        return int(result.rowcount or 0)


# ---------------------------------------------------------------------- helpers
def _as_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _clean_list(value: list[str] | None) -> list[str] | None:
    """Return ``None`` for empty/None input so JSONB columns stay NULL.

    Storing ``[]`` technically works but it forces consumers to treat
    "empty list" and "not computed" as different states. Collapsing to
    ``NULL`` at write time keeps the metadata easier to reason about.
    """
    if not value:
        return None
    cleaned = [item for item in value if item and item.strip()]
    return cleaned or None
