"""ChromaDB integration.

Vectors live in Chroma; PostgreSQL owns the structured metadata and
lifecycle. Every document written here carries ``source_file``,
``file_id`` (``AnswerKeyFile.id`` UUID as string) and ``question_no`` in
its metadata so we can audit, delete, or sweep by SQL primary key.

Callers are expected to supply an explicit ``doc_id`` (typically
``AnswerKeyItem.id`` stringified) so the same UUID lives in both stores.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings as app_settings

_client = chromadb.PersistentClient(
    path=app_settings.chroma_path,
    settings=ChromaSettings(anonymized_telemetry=False),
)

_collection = _client.get_or_create_collection(
    name=app_settings.chroma_collection,
    metadata={"hnsw:space": "cosine", "embedding_model": "BAAI/bge-m3"},
)


def add_question_document(
    source_file: str,
    question_no: str,
    document: str,
    embedding: list[float],
    page_numbers: list[int],
    chapter: str | None,
    answer_text: str | None,
    file_id: str,
    doc_id: str,
    parser_used: str | None = None,
    heading_text: str | None = None,
    problem_text: str | None = None,
    final_answer: str | None = None,
    normalized_answer: str | None = None,
    answer_type: str | None = None,
) -> str:
    """Upsert one question document into Chroma.

    Args:
        file_id: ``AnswerKeyFile.id`` as string. Stored in metadata so we
            can delete / query by SQL primary key later.
        doc_id: Chroma id. Pass the ``AnswerKeyItem.id`` UUID so both
            stores share a single identifier.
        problem_text / final_answer / normalized_answer / answer_type:
            structured projection produced by
            :mod:`app.services.answer_extraction`. Stored on the Chroma
            metadata so search results can be filtered/compared by the
            cleaned-up fields without an extra SQL round-trip.

    Returns:
        The id actually written to Chroma (always equal to ``doc_id``).
    """
    metadata: dict[str, str] = {
        "source_file": source_file,
        "question_no": question_no,
        "page_numbers": ",".join(map(str, page_numbers)),
        "chapter": chapter or "",
        "answer_text": answer_text or "",
        "parser_used": parser_used or "",
        "heading_text": heading_text or "",
        "file_id": file_id,
        "problem_text": problem_text or "",
        "final_answer": final_answer or "",
        "normalized_answer": normalized_answer or "",
        "answer_type": answer_type or "",
        "type": "answer_key_question",
    }

    _collection.upsert(
        ids=[doc_id],
        documents=[document],
        embeddings=[embedding],
        metadatas=[metadata],
    )

    return doc_id


def search_documents(
    query_embedding: list[float],
    top_k: int = 5,
    where: dict | None = None,
) -> dict:
    """Top-k semantic search over the answer-key collection.

    ``where`` is an optional Chroma metadata filter, e.g.
    ``{"file_id": "<uuid>"}`` to restrict results to a single uploaded
    answer-key PDF. When omitted the search runs across the whole
    collection (original behaviour).
    """
    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return _collection.query(**kwargs)


def get_documents_by_file_id(file_id: str, limit: int = 10000) -> dict:
    """Fetch every vector for an ``AnswerKeyFile.id`` (SQL PK)."""
    return _collection.get(
        where={"file_id": file_id},
        include=["metadatas"],
        limit=limit,
    )


def delete_document_by_id(doc_id: str) -> bool:
    """Remove a single vector. Returns ``True`` if it existed."""
    existing = _collection.get(ids=[doc_id], include=[])
    ids = existing.get("ids", [])

    if not ids:
        return False

    _collection.delete(ids=[doc_id])
    return True


def delete_documents_by_ids(ids: list[str]) -> int:
    """Delete a specific set of Chroma ids. Returns how many existed."""
    if not ids:
        return 0
    existing = _collection.get(ids=ids, include=[])
    existing_ids = existing.get("ids", []) or []
    if not existing_ids:
        return 0
    _collection.delete(ids=existing_ids)
    return len(existing_ids)


def delete_documents_by_file_id(file_id: str) -> dict:
    """Delete every vector whose metadata ``file_id`` matches."""
    existing = get_documents_by_file_id(file_id=file_id)
    ids = existing.get("ids", []) or []

    if not ids:
        return {"deleted_count": 0, "deleted_ids": []}

    _collection.delete(ids=ids)
    return {"deleted_count": len(ids), "deleted_ids": ids}
