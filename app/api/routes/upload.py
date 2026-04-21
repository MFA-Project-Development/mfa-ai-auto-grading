"""Upload endpoint for answer-key ingestion.

Three-store ingestion flow:

    file upload
        -> create ``AnswerKeyFile`` row (status=uploading)
        -> upload bytes to MinIO (bucket + object name derived from SQL id)
        -> update ``AnswerKeyFile`` with storage metadata (status=processing)
        -> extract text / OCR / parser selection
        -> parse chunks
        -> for each chunk:
               insert vector into ChromaDB
               insert ``AnswerKeyItem`` row with ``vector_id``
        -> mark ``AnswerKeyFile`` completed with totals + parser_used

Responsibility split:

* PostgreSQL is the source of truth for file/item metadata and lifecycle.
* MinIO stores the uploaded binary so reprocessing never requires a
  re-upload.
* ChromaDB stores vector embeddings.

Failure handling:

* MinIO upload failure -> file row marked ``failed`` with the storage
  error; no Chroma / item rows exist yet.
* Parser / embedding failure after MinIO upload -> file row marked
  ``failed`` and any Chroma vectors written so far are cleaned up. The
  MinIO object is deliberately **retained** so the operator can retry
  ingestion without re-uploading.
"""

from __future__ import annotations

import logging
import uuid

import fitz  # PyMuPDF
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.enums import ChunkType, IngestionStatus, PdfMode, SourceType, StorageProvider
from app.db.models.answer_key_file import AnswerKeyFile
from app.db.session import get_db
from app.models.schemas import QuestionChunk, UploadResponse
from app.repositories import AnswerKeyFileRepository, AnswerKeyItemRepository
from app.services.chroma_service import (
    add_question_document,
    delete_documents_by_ids,
)
from app.services.embedding_service import build_embedding_text, get_embedding
from app.services.ocr_service import (
    extract_text_from_image_file,
    extract_text_with_ocr,
    is_ocr_available,
)
from app.services.parsers.parser_selector import parse_with_selected_strategy
from app.services.pdf_service import (
    average_text_quality,
    classify_upload,
    detect_pdf_mode,
    extract_pages_from_pdf,
    page_needs_ocr,
    parse_page_spec,
    render_page_to_png_bytes,
    score_text_quality,
    suspicious_ratio,
)
from app.services.storage import (
    StorageError,
    get_storage_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/answer-keys", tags=["Answer Keys"])


# ---------------------------------------------------------------------- PARSER -> CHUNK TYPE
# Map parser_selector output names to our ChunkType enum.
_PARSER_TO_CHUNK_TYPE: dict[str, ChunkType] = {
    "question_number": ChunkType.QUESTION,
    "heading": ChunkType.HEADING,
    "page_fallback": ChunkType.PAGE_FALLBACK,
}


# ====================================================================== ENDPOINT
@router.post("/upload", response_model=UploadResponse)
async def upload_answer_key(
    file: UploadFile = File(...),
    parse_mode: str = Form("auto"),
    ocr_mode: str = Form("auto"),
    page_range: str = Form(""),
    subject: str | None = Form(None),
    grade: str | None = Form(None),
    language: str | None = Form(None),
    db: Session = Depends(get_db),
) -> UploadResponse:
    """Ingest an answer-key PDF or image.

    SQL is updated transactionally at each stage so the ``AnswerKeyFile``
    row always reflects the true ingestion state, even on failure.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    kind = classify_upload(file.filename, file.content_type)
    if kind == "unsupported":
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: PDF, PNG, JPG, JPEG, WEBP",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    logger.info(
        "upload: file=%s kind=%s content_type=%s size=%d page_range=%r",
        file.filename,
        kind,
        file.content_type,
        len(raw_bytes),
        page_range,
    )

    file_repo = AnswerKeyFileRepository(db)
    item_repo = AnswerKeyItemRepository(db)
    storage = get_storage_service()

    # Step 1: create the parent SQL row in UPLOADING state. Commit so the
    # row is observable to other transactions (monitoring, retries, etc.)
    # even if MinIO takes a while or fails outright.
    file_record = file_repo.create(
        file_name=file.filename,
        original_file_name=file.filename,
        source_type=SourceType.PDF if kind == "pdf" else SourceType.IMAGE,
        mime_type=file.content_type,
        file_size=len(raw_bytes),
        subject=subject,
        grade=grade,
        language=language,
        ingestion_status=IngestionStatus.UPLOADING,
    )
    db.commit()
    file_id = file_record.id

    # Step 2: push the original bytes to MinIO. We do this before any
    # expensive OCR / parsing work so a storage outage fails fast and
    # never leaves a half-ingested record around.
    try:
        _upload_to_storage(
            raw_bytes=raw_bytes,
            content_type=file.content_type,
            file_record=file_record,
            file_repo=file_repo,
            storage=storage,
        )
        db.commit()
    except StorageError as exc:
        logger.exception(
            "upload: MinIO upload failed file=%s file_id=%s", file.filename, file_id
        )
        _mark_storage_failed(
            db=db, file_repo=file_repo, file_record=file_record, message=str(exc)
        )
        raise HTTPException(
            status_code=502,
            detail=f"Object storage unavailable: {exc}",
        ) from exc

    written_vector_ids: list[str] = []

    try:
        # Step 3: extract pages (PDF or image) and determine pdf_mode.
        if kind == "pdf":
            pages, pdf_mode = _extract_pdf_pages(
                raw_bytes,
                ocr_mode=ocr_mode,
                filename=file.filename,
                page_range=page_range,
            )
        else:
            if page_range.strip() and page_range.strip().lower() != "all":
                logger.info(
                    "upload: ignoring page_range=%r for image upload %s",
                    page_range,
                    file.filename,
                )
            pages = _extract_image_pages(raw_bytes, filename=file.filename)
            pdf_mode = PdfMode.IMAGE_PDF.value

        if not pages or not any(p["text"].strip() for p in pages):
            raise HTTPException(
                status_code=400,
                detail="No readable text could be extracted from the file",
            )

        file_repo.set_pdf_mode(file_record, pdf_mode)
        file_repo.update_totals(file_record, total_pages=len(pages))
        db.commit()

        # Step 4: parser pipeline.
        chunks = parse_with_selected_strategy(
            pages,
            source_file=file.filename,
            parse_mode=parse_mode,
            pdf_mode=pdf_mode,
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="No question chunks found")

        parser_used = chunks[0].parser_used or "unknown"
        logger.info(
            "upload: file=%s chunks=%d parser=%s parse_mode=%s",
            file.filename,
            len(chunks),
            parser_used,
            parse_mode,
        )
        file_repo.update_parser_used(file_record, parser_used)
        db.commit()

        # Step 5: embed each chunk, write to Chroma, then record in SQL.
        inserted_ids, previews = _embed_and_store(
            chunks=chunks,
            file_record=file_record,
            item_repo=item_repo,
            written_vector_ids=written_vector_ids,
        )

        # Step 6: final bookkeeping.
        file_repo.update_totals(file_record, total_chunks=len(inserted_ids))
        file_repo.update_status(file_record, IngestionStatus.COMPLETED)
        db.commit()

        return UploadResponse(
            success=True,
            fileName=file.filename,
            fileId=file_id,
            ingestionStatus=file_record.ingestion_status,
            parserUsed=file_record.parser_used,
            totalQuestions=len(inserted_ids),
            insertedIds=inserted_ids,
            previews=previews,
        )

    except HTTPException as exc:
        _mark_failed(
            db=db,
            file_repo=file_repo,
            file_record=file_record,
            message=f"{exc.status_code}: {exc.detail}",
            written_vector_ids=written_vector_ids,
        )
        raise
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("upload: unexpected failure for file=%s", file.filename)
        _mark_failed(
            db=db,
            file_repo=file_repo,
            file_record=file_record,
            message=str(exc),
            written_vector_ids=written_vector_ids,
        )
        raise HTTPException(status_code=500, detail="Upload failed") from exc


# ====================================================================== HELPERS
def _mark_failed(
    *,
    db: Session,
    file_repo: AnswerKeyFileRepository,
    file_record: AnswerKeyFile,
    message: str,
    written_vector_ids: list[str],
) -> None:
    """Handle parse/embedding failure *after* a successful MinIO upload.

    Rolls back any uncommitted SQL work, marks the file FAILED with the
    error message, and purges any ChromaDB vectors written before the
    error. The MinIO object is intentionally retained so the operator
    can retry ingestion without re-uploading.
    """
    try:
        db.rollback()
    except Exception:  # pragma: no cover
        logger.exception("upload: rollback failed")

    # Reattach the file row (the rollback detached it) and flip its status.
    try:
        reattached = file_repo.get(file_record.id)
        if reattached is not None:
            file_repo.set_error(reattached, message)
            db.commit()
    except Exception:  # pragma: no cover
        logger.exception("upload: failed to persist failure state")

    if written_vector_ids:
        try:
            removed = delete_documents_by_ids(written_vector_ids)
            logger.info(
                "upload: cleaned up %d/%d orphan vectors after failure",
                removed,
                len(written_vector_ids),
            )
        except Exception:  # pragma: no cover
            logger.exception("upload: Chroma cleanup after failure failed")


def _upload_to_storage(
    *,
    raw_bytes: bytes,
    content_type: str | None,
    file_record: AnswerKeyFile,
    file_repo: AnswerKeyFileRepository,
    storage,
) -> None:
    """Push the uploaded bytes to MinIO and persist the storage metadata.

    Called after the ``AnswerKeyFile`` row is created in UPLOADING state.
    On success, the row moves to PROCESSING and carries ``bucket_name``,
    ``object_name``, ``object_etag``, ``stored_file_name``, and
    ``storage_provider``. On failure, the exception bubbles up and the
    caller is responsible for marking the row FAILED.
    """
    storage.ensure_bucket_exists()

    object_name = storage.build_object_name(
        file_id=file_record.id,
        original_filename=file_record.original_file_name,
    )
    stored_file_name = object_name.rsplit("/", 1)[-1]

    uploaded = storage.upload_file_bytes(
        object_name=object_name,
        data=raw_bytes,
        content_type=content_type,
        extra_metadata={
            "x-amz-meta-file-id": str(file_record.id),
            "x-amz-meta-original-filename": file_record.original_file_name,
        },
    )

    file_repo.set_storage_metadata(
        file_record,
        provider=StorageProvider.MINIO,
        bucket_name=uploaded.bucket_name,
        object_name=uploaded.object_name,
        stored_file_name=stored_file_name,
        object_etag=uploaded.etag,
    )
    file_repo.update_status(file_record, IngestionStatus.PROCESSING)

    logger.info(
        "upload: MinIO upload ok file_id=%s bucket=%s object=%s size=%d etag=%s",
        file_record.id, uploaded.bucket_name, uploaded.object_name,
        uploaded.size, uploaded.etag,
    )


def _mark_storage_failed(
    *,
    db: Session,
    file_repo: AnswerKeyFileRepository,
    file_record: AnswerKeyFile,
    message: str,
) -> None:
    """Flip the file row to FAILED after a MinIO upload failure.

    Rolls back any uncommitted work, re-attaches the row (since the
    rollback detached the in-memory instance), then commits the new
    terminal state so callers / monitoring see a consistent record.
    """
    try:
        db.rollback()
    except Exception:  # pragma: no cover
        logger.exception("upload: rollback failed after MinIO error")

    try:
        reattached = file_repo.get(file_record.id)
        if reattached is not None:
            file_repo.set_error(reattached, message)
            db.commit()
    except Exception:  # pragma: no cover
        logger.exception("upload: failed to persist storage failure state")


def _extract_pdf_pages(
    pdf_bytes: bytes,
    ocr_mode: str,
    filename: str,
    page_range: str = "",
) -> tuple[list[dict], str]:
    """Return ``(pages, pdf_mode)`` for a PDF with OCR-aware routing."""
    try:
        pages = extract_pages_from_pdf(pdf_bytes)
    except (fitz.FileDataError, RuntimeError) as exc:
        logger.warning("upload: corrupted PDF file=%s err=%s", filename, exc)
        raise HTTPException(status_code=400, detail="Could not read PDF") from exc

    if not pages:
        raise HTTPException(status_code=400, detail="Could not read PDF")

    pages = _apply_page_range(pages, page_range, filename=filename)

    pdf_mode = detect_pdf_mode(pages)
    native_quality = average_text_quality(pages)
    logger.info(
        "upload: pdf_mode=%s file=%s pages=%d native_avg_quality=%.2f ocr_mode=%s",
        pdf_mode,
        filename,
        len(pages),
        native_quality,
        ocr_mode,
    )

    if ocr_mode == "off":
        logger.info("upload: OCR disabled by ocr_mode=off for %s", filename)
        return pages, pdf_mode

    if not is_ocr_available():
        logger.info(
            "upload: OCR engine unavailable; using native text only for %s",
            filename,
        )
        return pages, pdf_mode

    if pdf_mode == "image_pdf":
        _apply_pdf_ocr_first(pages, pdf_bytes, filename=filename)
    else:
        _apply_pdf_ocr_fallback(pages, pdf_bytes, ocr_mode=ocr_mode, filename=filename)

    return pages, pdf_mode


def _apply_pdf_ocr_first(
    pages: list[dict],
    pdf_bytes: bytes,
    filename: str,
) -> None:
    """OCR every page and REPLACE its text unconditionally.

    Used for true image PDFs where native extraction yields empty strings.
    Per-page errors are swallowed so a single bad page cannot fail the whole
    upload.
    """
    ocr_success = 0

    logger.info(
        "upload: OCR-first MODE file=%s pages=%d (image PDF detected)",
        filename,
        len(pages),
    )

    for page in pages:
        page_no = page["page_number"]
        native_text = page["text"]

        try:
            image_bytes = render_page_to_png_bytes(pdf_bytes, page_no)
        except Exception:
            logger.exception(
                "upload: failed to render page=%d for %s", page_no, filename
            )
            continue

        try:
            ocr_text = extract_text_with_ocr(image_bytes)
        except Exception:
            logger.exception(
                "upload: OCR threw while processing page=%d for %s", page_no, filename
            )
            ocr_text = ""

        if not ocr_text.strip():
            logger.warning(
                "upload: OCR-first page=%d file=%s produced empty text "
                "(native_len=%d); keeping native",
                page_no, filename, len(native_text),
            )
            continue

        quality = score_text_quality(ocr_text)
        logger.info(
            "upload: OCR-first page=%d file=%s native_len=%d ocr_len=%d ocr_quality=%.2f",
            page_no, filename, len(native_text), len(ocr_text), quality,
        )

        page["text"] = ocr_text
        ocr_success += 1

    logger.info(
        "upload: OCR-first SUMMARY file=%s pages_ocred=%d/%d",
        filename, ocr_success, len(pages),
    )


def _apply_page_range(
    pages: list[dict],
    page_range: str,
    filename: str,
) -> list[dict]:
    """Filter ``pages`` down to those requested by ``page_range``."""
    try:
        wanted = parse_page_spec(page_range, total_pages=len(pages))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid page_range: {exc}") from exc

    if len(wanted) == len(pages):
        return pages

    wanted_set = set(wanted)
    filtered = [p for p in pages if p["page_number"] in wanted_set]

    logger.info(
        "upload: page_range applied file=%s requested=%s kept=%d/%d",
        filename, wanted, len(filtered), len(pages),
    )
    return filtered


def _apply_pdf_ocr_fallback(
    pages: list[dict],
    pdf_bytes: bytes,
    ocr_mode: str,
    filename: str,
) -> None:
    """Mutate ``pages`` in place, substituting OCR text only when genuinely
    better. See the previous implementation notes (BUG 5 / BUG 6 fixes)."""
    ocr_pages: list[int] = []
    ocr_invocations = 0
    skipped_text_pages = 0
    force = ocr_mode == "always"

    logger.info(
        "upload: OCR fallback ARMED file=%s ocr_mode=%s total_pages=%d",
        filename, ocr_mode, len(pages),
    )

    for page in pages:
        page_no = page["page_number"]
        native_text = page["text"]
        page_kind = page.get("pdf_kind", "unknown")

        if page_kind == "text" and not force:
            skipped_text_pages += 1
            logger.debug(
                "upload: OCR fallback SKIPPED page=%d file=%s reason=pdf_kind_text",
                page_no, filename,
            )
            continue

        if not force and not page_needs_ocr(native_text):
            logger.debug(
                "upload: OCR fallback SKIPPED page=%d file=%s reason=native_ok "
                "native_len=%d",
                page_no, filename, len(native_text),
            )
            continue

        ocr_invocations += 1
        logger.info(
            "upload: OCR fallback TRIGGERED page=%d file=%s reason=%s "
            "pdf_kind=%s native_len=%d",
            page_no, filename,
            "forced" if force else "weak_native_text",
            page_kind, len(native_text),
        )

        try:
            image_bytes = render_page_to_png_bytes(pdf_bytes, page_no)
        except Exception:
            logger.exception(
                "upload: failed to render page=%d for %s", page_no, filename
            )
            continue

        try:
            ocr_text = extract_text_with_ocr(image_bytes)
        except Exception:
            logger.exception(
                "upload: OCR threw while processing page=%d for %s", page_no, filename
            )
            ocr_text = ""

        if not ocr_text.strip():
            logger.info(
                "upload: OCR page=%d file=%s ocr_len=0 selected=native",
                page_no, filename,
            )
            continue

        ocr_suspicious = suspicious_ratio(ocr_text)
        if ocr_suspicious > 0.15:
            logger.info(
                "upload: OCR page=%d file=%s REJECTED ocr_suspicious=%.2f "
                "> 0.15 selected=native",
                page_no, filename, ocr_suspicious,
            )
            continue

        native_quality = score_text_quality(native_text)
        ocr_quality = score_text_quality(ocr_text)

        logger.info(
            "upload: OCR page=%d file=%s native_len=%d ocr_len=%d "
            "native_quality=%.2f ocr_quality=%.2f ocr_suspicious=%.2f",
            page_no, filename,
            len(native_text), len(ocr_text),
            native_quality, ocr_quality, ocr_suspicious,
        )

        if ocr_quality > native_quality:
            page["text"] = ocr_text
            ocr_pages.append(page_no)
            logger.info(
                "upload: OCR page=%d file=%s selected=ocr", page_no, filename
            )
        else:
            logger.info(
                "upload: OCR page=%d file=%s selected=native (native_quality >= ocr_quality)",
                page_no, filename,
            )

    logger.info(
        "upload: OCR fallback SUMMARY file=%s invoked=%d/%d "
        "replaced=%d skipped_text_pages=%d pages_replaced=%s",
        filename, ocr_invocations, len(pages),
        len(ocr_pages), skipped_text_pages, ocr_pages or "[]",
    )


def _extract_image_pages(image_bytes: bytes, filename: str) -> list[dict]:
    """Turn an image upload into the same page-dict shape the parsers expect."""
    if not is_ocr_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "OCR engine is not available on the server; image uploads "
                "require PaddleOCR to be installed"
            ),
        )

    text = extract_text_from_image_file(image_bytes)
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="OCR could not extract any text from the image",
        )

    logger.info("upload: OCR extracted %d chars from image %s", len(text), filename)
    return [{"page_number": 1, "text": text}]


def _embed_and_store(
    chunks: list[QuestionChunk],
    file_record: AnswerKeyFile,
    item_repo: AnswerKeyItemRepository,
    written_vector_ids: list[str],
) -> tuple[list[str], list[dict]]:
    """Embed each chunk, write it to Chroma, and persist the SQL row.

    The SQL row is the source of truth - ``vector_id`` links it to the
    ChromaDB document. We pre-generate the item UUID and reuse it as the
    Chroma id so the two stores share a single traceable identifier.
    """
    inserted_ids: list[str] = []
    previews: list[dict] = []

    for chunk in chunks:
        if not chunk.content.strip():
            logger.warning(
                "upload: skipping empty chunk question_no=%s file=%s",
                chunk.question_no, chunk.source_file,
            )
            continue

        # The parsers now attach a structured projection onto every
        # chunk; use it to build a cleaner embedding and to feed both
        # Chroma metadata and the SQL row. Fallback to raw content is
        # handled inside ``build_embedding_text``.
        embedding_text = build_embedding_text(
            question_no=chunk.question_no,
            content=chunk.content,
            answer_text=chunk.answer_text,
            chapter=chunk.chapter,
            problem_text=chunk.problem_text,
            final_answer=chunk.final_answer,
            solution_steps=chunk.solution_steps,
        )
        embedding = get_embedding(embedding_text)

        # Shared id: UUID that acts as both AnswerKeyItem.id and Chroma doc id.
        item_id = uuid.uuid4()
        vector_id = str(item_id)

        doc_id = add_question_document(
            source_file=chunk.source_file,
            question_no=chunk.question_no,
            document=embedding_text,
            embedding=embedding,
            page_numbers=chunk.page_numbers,
            chapter=chunk.chapter,
            answer_text=chunk.answer_text,
            parser_used=chunk.parser_used,
            heading_text=chunk.heading_text,
            file_id=str(file_record.id),
            doc_id=vector_id,
            problem_text=chunk.problem_text,
            final_answer=chunk.final_answer,
            normalized_answer=chunk.normalized_answer,
            answer_type=chunk.answer_type,
        )
        written_vector_ids.append(doc_id)

        parser_name = chunk.parser_used or file_record.parser_used or "unknown"
        chunk_type = _PARSER_TO_CHUNK_TYPE.get(parser_name, ChunkType.QUESTION)

        page_start, page_end = _page_bounds(chunk.page_numbers)
        page_numbers_str = (
            ",".join(str(p) for p in chunk.page_numbers)
            if chunk.page_numbers
            else None
        )

        item_repo.create(
            item_id=item_id,
            file_id=file_record.id,
            question_no=chunk.question_no,
            content=chunk.content,
            chunk_type=chunk_type,
            heading_text=chunk.heading_text,
            chapter=chunk.chapter,
            answer_text=chunk.answer_text,
            page_start=page_start,
            page_end=page_end,
            page_numbers=page_numbers_str,
            parser_used=parser_name,
            vector_id=doc_id,
            problem_text=chunk.problem_text,
            solution_steps=chunk.solution_steps,
            final_answer=chunk.final_answer,
            normalized_answer=chunk.normalized_answer,
            answer_type=chunk.answer_type,
            formula_list=chunk.formula_list,
        )

        inserted_ids.append(doc_id)

        if len(previews) < 10:
            previews.append({
                "docId": doc_id,
                "itemId": str(item_id),
                "questionNo": chunk.question_no,
                "pages": chunk.page_numbers,
                "chapter": chunk.chapter,
                "answerText": chunk.answer_text,
                "parserUsed": parser_name,
                "headingText": chunk.heading_text,
                "preview": chunk.content[:200],
                "problemText": chunk.problem_text,
                "solutionSteps": chunk.solution_steps,
                "finalAnswer": chunk.final_answer,
                "normalizedAnswer": chunk.normalized_answer,
                "answerType": chunk.answer_type,
                "formulaList": chunk.formula_list,
            })

    return inserted_ids, previews


def _page_bounds(page_numbers: list[int] | None) -> tuple[int | None, int | None]:
    if not page_numbers:
        return None, None
    return min(page_numbers), max(page_numbers)
