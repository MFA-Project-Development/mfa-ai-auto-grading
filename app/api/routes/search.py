from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.chroma_service import search_documents
from app.services.embedding_service import get_embedding

router = APIRouter(prefix="/api/v1/answer-keys", tags=["Answer Key Search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


@router.post("/search")
def search_answer_keys(payload: SearchRequest) -> dict:
    """Semantic search over answer-key chunks via ChromaDB."""
    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_embedding = get_embedding(query_text)
    results = search_documents(query_embedding=query_embedding, top_k=payload.top_k)

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    items = []
    for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        items.append({
            "docId": doc_id,
            "questionNo": metadata.get("question_no"),
            "sourceFile": metadata.get("source_file"),
            "fileId": metadata.get("file_id"),
            "pages": metadata.get("page_numbers"),
            "chapter": metadata.get("chapter"),
            "answerText": metadata.get("answer_text"),
            "score": float(1 - distance) if distance is not None else None,
            "document": document,
        })

    return {
        "success": True,
        "query": query_text,
        "count": len(items),
        "results": items,
    }
