"""PyMuPDF-based PDF extraction and low-level file-type helpers.

Responsibilities:

1. Extract native text from a PDF page-by-page with layout awareness:
   superscripts (``x²``) reconstructed as ``x^2``, image "stencil" answer
   boxes converted into ``답 ...`` text markers, and chapter headings
   detected from the top 15 % of each page bbox AND the left margin band
   (never from body / TOC).
2. Classify each page as ``"text"``, ``"image"`` or ``"mixed"`` using
   span-level signals WITH superscript-aware joining so math-heavy pages
   are never mis-classified as image pages.
3. Score text quality (readable ratio, artifact ratio, suspicious-glyph
   ratio) so the parser selector can gate structured parsers.
4. Classify uploaded files (``pdf`` / ``image`` / ``unsupported``) and
   expose a few small helpers (``render_page_to_png_bytes``,
   ``parse_page_spec``).

Bug fixes applied (v2):
- BUG 1: _classify_page_from_dict now joins spans using _render_span logic
  so superscript digits are attached as ^N before math detection runs.
  Previously x² split into "x" + "2" never matched _MATH_SIGNAL_RE, causing
  math-heavy pages to be classified as "image" and sent to OCR needlessly.
- BUG 2: _MATH_SIGNAL_RE widened to catch lone ^N, numeric expressions
  (3+1), and variable letters without requiring an adjacent digit on the
  same span.
- BUG 3: _render_line stencil matching now uses vertical midpoint comparison
  with a wider 12pt tolerance instead of y0-to-y0 with 5pt, preventing
  answer boxes from being missed when the stencil is vertically centred on
  a text line.
- BUG 4: _detect_chapter_from_dict also scans the left margin band (left
  12 % of page width) in addition to the top 15 % band, so chapter labels
  printed in the page margin are captured instead of falling back to TOC
  text from elsewhere in the document.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Iterable, Literal

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

FileKind = Literal["pdf", "image", "unsupported"]
PdfMode = Literal["text_pdf", "image_pdf"]
PageKind = Literal["text", "image", "mixed"]

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

PDF_MIME_TYPES = {"application/pdf"}
IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}

# Legacy empty-ish threshold kept for pages that lack a ``pdf_kind`` tag
# (e.g. when a caller hands us a hand-built page list without spans).
_EMPTYISH_NATIVE_CHARS = 30

# Fraction of image-like pages above which a document is treated as an
# image PDF (OCR-first) rather than a text PDF (OCR-as-fallback).
_IMAGE_PDF_EMPTY_RATIO = 0.60

# Per-page classification thresholds.
#   readable_chars >= 100 OR has_math  -> "text"
#   readable_chars < 30  AND not math  -> "image"
#   otherwise                          -> "mixed"
_PAGE_TEXT_MIN_READABLE = 100
_PAGE_IMAGE_MAX_READABLE = 30

# Readable = ASCII alnum + Hangul syllables.
_READABLE_CHAR_RE = re.compile(r"[A-Za-z0-9\uac00-\ud7a3]")

# FIX BUG 2: widened to catch:
#   - algebraic expressions: x^2, y3, A2, a+b
#   - lone superscripts after joining: ^2, ^3
#   - numeric expressions: 3+1, 5-2, 2=2
#   - standard operators: +, -, =, ×, ÷
#   - equals with space: a = b
_MATH_SIGNAL_RE = re.compile(
    r"[x-zX-ZA-Ca-c]\^?\d"   # algebraic variable + optional ^ + digit
    r"|\^[0-9]"               # lone superscript token: ^2, ^3
    r"|[0-9][+\-=][0-9]"      # numeric expression: 3+1, 5-2
    r"|[+\-=×÷]"              # any operator on its own
)

# Known broken-OCR glyphs that OCR engines (PaddleOCR historically, GLM-OCR
# today) emit when they misread stencil strokes.
_SUSPICIOUS_PATTERN = re.compile(r"[텅딥밉법봐릉\ufffd]|\?{2,}|-{3,}")

# Simple OCR-artifact chars used by score_text_quality.
_OCR_ARTIFACT_CHARS = "?\ufffd"

# PyMuPDF span flags.
_SUPERSCRIPT_FLAG = 2
_BOLD_FLAG = 16

# FIX BUG 3: increased from 5.0 to 12.0 pts, and matching now uses the
# vertical midpoint of both the stencil and the text line so a stencil
# that is vertically centred on a line (not flush with its top) still
# matches reliably.
_STENCIL_Y_TOLERANCE = 12.0

# Fraction of page height considered the "header band" for chapter detection.
_CHAPTER_TOP_FRACTION = 0.15

# FIX BUG 4: also scan this fraction of page WIDTH as a left-margin band.
# Korean textbooks often print the chapter label ("02 나머지정리") rotated or
# inset in the left margin rather than at the very top of the page.
_CHAPTER_LEFT_FRACTION = 0.12

# Chapter pattern: 01-29 followed by a Hangul-led title.
_CHAPTER_SPAN_RE = re.compile(
    r"^\s*(0[1-9]|1\d|2\d)\s+([\uac00-\ud7a3][^\n]{2,40})\s*$"
)

# Fallback answer markers used when a stencil is not present on a line.
_ANSWER_FALLBACK_RE = re.compile(r"[■□目圖]")

# Canonical answer marker recognised by downstream parsers.
_ANSWER_MARKER = "답"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_pdf_text(text: str) -> str:
    """Strip problematic invisible characters from PyMuPDF output."""
    return (
        text.replace("\u00ad", "")   # soft hyphen
            .replace("\ufeff", "")   # BOM
            .replace("\xa0", " ")    # non-breaking space
            .replace("\t", " ")
    )


def extract_pages_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """Return one structured dict per page of the PDF.

    Each dict contains:

    * ``page_number`` - 1-based page index.
    * ``text``        - layout-aware extraction: superscript digits attached
      with ``^``, image answer stencils converted to ``답 ...`` lines, and
      the page chapter (if detected) prepended as the first line.
    * ``pdf_kind``    - :class:`PageKind` (``"text"`` / ``"image"`` /
      ``"mixed"``) produced from span signals with superscript-aware joining.
    * ``chapter``     - chapter string detected for this page, or ``None``.

    Failures on a single page never abort the extraction: the problem page
    falls back to plain ``get_text("text")`` and is tagged ``pdf_kind =
    "image"`` so the OCR pipeline can pick it up.
    """
    pages: list[dict] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for index, page in enumerate(doc):
            page_number = index + 1

            try:
                data = page.get_text("dict") or {}
            except Exception:
                logger.exception(
                    "pdf_service: get_text('dict') failed on page=%d", page_number
                )
                data = {}

            try:
                stencils = _find_stencil_bboxes(page)
            except Exception:
                logger.exception(
                    "pdf_service: stencil detection failed on page=%d", page_number
                )
                stencils = []

            try:
                text, chapter = _render_page(data, stencils)
            except Exception:
                logger.exception(
                    "pdf_service: structured render failed on page=%d", page_number
                )
                text = page.get_text("text") or ""
                chapter = None

            try:
                # FIX BUG 1: use superscript-aware classification
                kind = _classify_page_from_dict(data) if data else "image"
            except Exception:
                logger.exception(
                    "pdf_service: page classification failed on page=%d", page_number
                )
                kind = "mixed"

            pages.append({
                "page_number": page_number,
                "text": clean_pdf_text(text),
                "pdf_kind": kind,
                "chapter": chapter,
            })

    return pages


def render_page_to_png_bytes(pdf_bytes: bytes, page_number: int, dpi: int = 200) -> bytes:
    """Rasterize a single PDF page to PNG bytes at ``dpi`` resolution."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_number - 1]
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return pix.tobytes("png")


def score_page_text(text: str) -> float:
    """Return a 0.0 - 1.0 confidence that the text is a real extraction.

    FIX: Math-heavy lines like "5x^3-3x^2+x-12" are short but real.
    Old logic scored anything under 40 chars as 0.1, causing valid math
    lines to trigger unnecessary OCR. Now we check for math signals first
    and give short-but-math text a passing score so OCR is not triggered.

    * empty                                                -> 0.0
    * short with NO math signals                          -> 0.1
    * short WITH math signals (operators, ^, digits)      -> 0.6
    * full pages (>= 400 chars + digits/symbols)          -> 1.0
    """
    stripped = text.strip()
    if not stripped:
        return 0.0

    length = len(stripped)
    digit_count = sum(ch.isdigit() for ch in stripped)
    symbol_count = sum(ch in "+-=/^()[]{}" for ch in stripped)
    has_math_signals = digit_count >= 2 or symbol_count >= 2

    if length < 40:
        # Short but has math content → don't trigger OCR
        return 0.6 if has_math_signals else 0.1

    length_score = min(length / 400.0, 1.0)
    signal_score = 1.0 if (digit_count >= 3 or symbol_count >= 3) else 0.6
    return round(length_score * signal_score, 3)


def page_needs_ocr(text: str, threshold: float = 0.4) -> bool:
    """True when a page's native text is weak enough to warrant OCR."""
    return score_page_text(text) < threshold


def score_text_quality(text: str) -> float:
    """Return a 0.0 - 1.0 "is this text clean enough to parse" score.

    Combines readable-char ratio with an OCR-artifact penalty. Structured
    parsers should only run above ~0.55 on text PDFs and ~0.75 on image
    PDFs; otherwise prefer the page-fallback parser.
    """
    stripped = (text or "").strip()
    if not stripped:
        return 0.0

    n = len(stripped)
    readable = len(_READABLE_CHAR_RE.findall(stripped))
    artifacts = sum(stripped.count(ch) for ch in _OCR_ARTIFACT_CHARS)

    readable_ratio = readable / n
    artifact_ratio = min(artifacts / n, 0.5)

    return round(max(0.0, readable_ratio - artifact_ratio), 3)


def average_text_quality(pages: list[dict]) -> float:
    """Mean :func:`score_text_quality` across non-empty pages (0.0 if none)."""
    non_empty = [p for p in pages if (p.get("text") or "").strip()]
    if not non_empty:
        return 0.0
    total = sum(score_text_quality(p["text"]) for p in non_empty)
    return round(total / len(non_empty), 3)


def readability_ratio(text: str) -> float:
    """Fraction of chars that are readable (ASCII alnum + Hangul).

    Exported for :mod:`parser_selector`, which needs a purer legibility
    signal without the artifact penalty baked into :func:`score_text_quality`.
    """
    stripped = (text or "").strip()
    if not stripped:
        return 0.0
    valid = len(_READABLE_CHAR_RE.findall(stripped))
    return round(valid / len(stripped), 3)


def suspicious_ratio(text: str) -> float:
    """Fraction of the text occupied by known OCR-garbage patterns.

    Counts matches of :data:`_SUSPICIOUS_PATTERN` (Korean look-alike glyphs
    OCR engines emit for stencil strokes, runs of ``?`` / ``-``, unicode
    replacement character) divided by total char count.

    The parser selector refuses ``question_number_parser`` when this exceeds
    ~0.15 because the resulting chunks are always corrupted math.
    """
    stripped = (text or "").strip()
    if not stripped:
        return 0.0
    hits = sum(len(m.group(0)) for m in _SUSPICIOUS_PATTERN.finditer(stripped))
    return round(hits / len(stripped), 3)


def detect_pdf_mode(
    pages: list[dict],
    *,
    image_ratio: float = _IMAGE_PDF_EMPTY_RATIO,
) -> PdfMode:
    """Classify the document as ``"text_pdf"`` or ``"image_pdf"``.

    Aggregates each page's ``pdf_kind`` tag (set during extraction with the
    now-fixed superscript-aware classifier):

    * ``>= image_ratio`` of pages tagged ``"image"`` -> ``"image_pdf"``
      (upload pipeline will OCR every page and replace native text).
    * Otherwise -> ``"text_pdf"`` (upload pipeline keeps native text and
      OCRs only pages that look weak).

    Pages without a ``pdf_kind`` tag (legacy / hand-built inputs) are
    re-classified from text length so this function stays usable on page
    lists not produced by :func:`extract_pages_from_pdf`.

    An empty page list is treated as ``"image_pdf"`` so callers attempt OCR
    rather than silently producing zero chunks.
    """
    if not pages:
        return "image_pdf"

    image_like = 0
    for page in pages:
        kind = page.get("pdf_kind")
        if kind is None:
            stripped = (page.get("text") or "").strip()
            kind = "image" if len(stripped) < _EMPTYISH_NATIVE_CHARS else "text"
        if kind == "image":
            image_like += 1

    ratio = image_like / len(pages)
    mode: PdfMode = "image_pdf" if ratio >= image_ratio else "text_pdf"

    logger.debug(
        "pdf_service: detect_pdf_mode pages=%d image_like=%d ratio=%.2f mode=%s",
        len(pages),
        image_like,
        ratio,
        mode,
    )
    return mode


def classify_upload(filename: str | None, content_type: str | None) -> FileKind:
    """Classify an upload by extension (preferred) or MIME type as fallback."""
    ext = _file_extension(filename)

    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in IMAGE_EXTENSIONS:
        return "image"

    if content_type:
        normalized = content_type.lower().split(";")[0].strip()
        if normalized in PDF_MIME_TYPES:
            return "pdf"
        if normalized in IMAGE_MIME_TYPES:
            return "image"

    return "unsupported"


def parse_page_spec(spec: str | None, total_pages: int) -> list[int]:
    """Turn a human-friendly page spec into a sorted list of 1-based page numbers.

    Supported syntax (whitespace tolerated, case-insensitive):

    * empty / ``None`` / ``"all"`` - every page (``1..total_pages``)
    * ``"3"``              - single page
    * ``"1-5"``            - inclusive range
    * ``"1,3,5"``          - multiple pages
    * ``"1-3,5,7-9"``      - mixed ranges and singles

    Raises:
        ValueError: on malformed input or out-of-range page numbers.
    """
    if total_pages <= 0:
        raise ValueError("Document has no pages")

    all_pages = list(range(1, total_pages + 1))
    if spec is None:
        return all_pages

    normalized = spec.strip().lower()
    if not normalized or normalized == "all":
        return all_pages

    wanted: set[int] = set()
    for raw_part in normalized.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" in part:
            start_str, _, end_str = part.partition("-")
            start = _parse_page_number(start_str, original=part)
            end = _parse_page_number(end_str, original=part)
            if start > end:
                raise ValueError(
                    f"Invalid page range '{part}': start ({start}) is after end ({end})"
                )
            wanted.update(range(start, end + 1))
        else:
            wanted.add(_parse_page_number(part, original=part))

    out_of_range = sorted(p for p in wanted if p < 1 or p > total_pages)
    if out_of_range:
        raise ValueError(
            f"Requested pages {out_of_range} are outside the document (1-{total_pages})"
        )

    return sorted(wanted)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _file_extension(filename: str | None) -> str:
    if not filename:
        return ""
    return os.path.splitext(filename)[1].lower()


def _parse_page_number(token: str, original: str) -> int:
    token = token.strip()
    if not token or not token.isdigit():
        raise ValueError(f"Invalid page token '{token}' in '{original}'")
    value = int(token)
    if value < 1:
        raise ValueError(f"Page numbers are 1-based; got '{token}' in '{original}'")
    return value


def _classify_kind(readable_chars: int, has_math: bool) -> PageKind:
    """Map the two primitive signals to a :data:`PageKind`."""
    if readable_chars >= _PAGE_TEXT_MIN_READABLE or has_math:
        return "text"
    if readable_chars < _PAGE_IMAGE_MAX_READABLE and not has_math:
        return "image"
    return "mixed"


def _classify_page_from_dict(data: dict) -> PageKind:
    """Classify a page from its PyMuPDF dict payload.

    FIX BUG 1: spans are now joined using the same superscript-aware logic
    as :func:`_render_span` before the math signal regex runs. Previously,
    superscript digits were separate spans (e.g. "x" then "2") and
    ``_MATH_SIGNAL_RE`` never matched across the span boundary, causing
    math-heavy answer-key pages to be mis-classified as ``"image"`` and
    sent to the OCR engine unnecessarily.
    """
    parts: list[str] = []
    for block in data.get("blocks", []) or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                parts.append(_render_span(span))

    joined = "".join(parts)
    readable = len(_READABLE_CHAR_RE.findall(joined))
    has_math = bool(_MATH_SIGNAL_RE.search(joined))

    kind = _classify_kind(readable, has_math)
    logger.debug(
        "pdf_service: _classify_page_from_dict readable=%d has_math=%s kind=%s",
        readable,
        has_math,
        kind,
    )
    return kind


def _render_span(span: dict) -> str:
    """Apply superscript-digit handling to a single span.

    PyMuPDF flags a superscript with ``span["flags"] & 2``. Such digit spans
    are emitted as ``^N`` so ``"x"`` + ``"2"`` (two separate spans where the
    second is superscript) round-trips to ``"x^2"`` in the assembled text.
    """
    text = span.get("text", "") or ""
    flags = span.get("flags", 0) or 0
    if flags & _SUPERSCRIPT_FLAG and text.strip().isdigit():
        return f"^{text.strip()}"
    return text


def _find_stencil_bboxes(page: Any) -> list[fitz.Rect]:
    """Return bboxes for every image drawn on the page.

    Answer boxes in Korean textbooks are typically CCITT stencil images
    stamped onto the page rather than glyphs, so they appear here rather
    than in ``get_text("dict")``. Missing / malformed entries are skipped.
    """
    stencils: list[fitz.Rect] = []
    try:
        images = page.get_images(full=True)
    except Exception:
        return stencils

    for img in images:
        try:
            bbox = page.get_image_bbox(img)
        except Exception:
            continue
        if bbox is None:
            continue
        rect = fitz.Rect(bbox)
        if rect.is_empty or rect.is_infinite:
            continue
        stencils.append(rect)
    return stencils


def _iter_text_lines(data: dict) -> Iterable[tuple[tuple[float, float, float, float], list[dict]]]:
    """Yield ``(line_bbox, spans)`` for every text line in reading order."""
    for block in data.get("blocks", []) or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []) or []:
            bbox = line.get("bbox")
            spans = line.get("spans") or []
            if bbox and spans:
                yield bbox, spans


def _render_line(
    line_bbox: tuple[float, float, float, float],
    spans: list[dict],
    stencils: list[fitz.Rect],
) -> tuple[str | None, str | None]:
    """Render one line into ``(body_text, answer_text)``.

    FIX BUG 3: stencil matching now compares the VERTICAL MIDPOINT of the
    stencil against the vertical midpoint of the text line, with a tolerance
    of 12 pt (increased from 5 pt). The old y0-to-y0 comparison with 5 pt
    frequently missed answer boxes that were vertically centred on a line
    (stencil.y0 == line.y0 + ~6 pt), leaving answerText as null.

    If a stencil matches: spans left of the stencil become body text, spans
    right of ``stencil.x1`` become the answer.

    If no stencil matches: fall back to text answer markers (■ □ 目 圖);
    anything right of the first such glyph is treated as the answer.

    Returns ``(None, None)`` when the line is empty after rendering.
    """
    # Vertical midpoint of this text line.
    line_mid = (line_bbox[1] + line_bbox[3]) / 2

    # FIX BUG 3: match on midpoint of stencil vs midpoint of line.
    stencil = next(
        (
            s for s in stencils
            if abs((s.y0 + s.y1) / 2 - line_mid) < _STENCIL_Y_TOLERANCE
        ),
        None,
    )

    if stencil is not None:
        left_spans = [s for s in spans if (s.get("bbox") or [0, 0, 0, 0])[2] <= stencil.x1]
        right_spans = [s for s in spans if (s.get("bbox") or [0, 0, 0, 0])[0] >= stencil.x1]
        body = "".join(_render_span(s) for s in left_spans).strip() or None
        answer = "".join(_render_span(s) for s in right_spans).strip() or None
        return body, answer

    joined = "".join(_render_span(s) for s in spans)
    match = _ANSWER_FALLBACK_RE.search(joined)
    if match:
        body = joined[: match.start()].strip() or None
        answer = joined[match.end():].strip() or None
        return body, answer

    return (joined.strip() or None), None


def _detect_chapter_from_dict(data: dict) -> str | None:
    """Return the chapter string for a page from its header and margin bands.

    FIX BUG 4: in addition to the top 15 % band, this function now also
    scans spans whose LEFT edge is within the left 12 % of the page width.
    Korean math textbooks commonly print the chapter label in the left margin
    (rotated or inset) rather than at the very top of the content area, so
    the old top-only search returned None and the pipeline fell back to TOC
    text from elsewhere in the document, producing wrong chapter names like
    ``"69\\n순열과조합"``.

    Detection priority:
    1. Strict pattern ``NN 한글제목`` (01-29 + Hangul title) on any top/
       left-margin span, or on the merged string of all such spans.
    2. First bold, non-digit-led span in the header/margin band.
    3. ``None`` — never guess.
    """
    page_height = data.get("height") or 0
    page_width = data.get("width") or 0
    if page_height <= 0:
        return None

    top_limit = page_height * _CHAPTER_TOP_FRACTION
    left_limit = page_width * _CHAPTER_LEFT_FRACTION

    top_spans: list[dict] = []
    for _, spans in _iter_text_lines(data):
        for span in spans:
            bbox = span.get("bbox")
            if not bbox:
                continue
            # FIX BUG 4: include spans in top band OR left margin band.
            in_top_band = bbox[1] < top_limit
            in_left_band = page_width > 0 and bbox[0] < left_limit
            if in_top_band or in_left_band:
                top_spans.append(span)

    if not top_spans:
        return None

    # Pass 1: strict per-span match.
    for span in top_spans:
        text = (span.get("text") or "").strip()
        match = _CHAPTER_SPAN_RE.match(text)
        if match:
            number, title = match.group(1), match.group(2).strip()
            return f"{number} {title}"

    # Pass 2: merge all header/margin spans and try again.
    # Handles chapters split across multiple small spans like
    # "01" + " " + "다항식의 연산".
    merged = " ".join((s.get("text") or "").strip() for s in top_spans).strip()
    merged = re.sub(r"\s+", " ", merged)
    match = _CHAPTER_SPAN_RE.match(merged)
    if match:
        return f"{match.group(1)} {match.group(2).strip()}"

    # Pass 3: bold non-digit-led span as a loose fallback.
    for span in top_spans:
        text = (span.get("text") or "").strip()
        if not text or text[0].isdigit():
            continue
        flags = span.get("flags", 0) or 0
        if flags & _BOLD_FLAG:
            return text[:50]

    return None


def _render_page(data: dict, stencils: list[fitz.Rect]) -> tuple[str, str | None]:
    """Render a full page into ``(text, chapter)``.

    Chapter (if detected) is prepended as the first line so
    :func:`app.services.parsers.base.extract_page_local_chapter` picks it
    up without needing access to this module's metadata.
    """
    chapter = _detect_chapter_from_dict(data)

    rendered: list[str] = []
    for line_bbox, spans in _iter_text_lines(data):
        body, answer = _render_line(line_bbox, spans, stencils)
        if body:
            rendered.append(body)
        if answer:
            rendered.append(f"{_ANSWER_MARKER} {answer}")

    body_text = "\n".join(rendered).strip()

    if chapter:
        first_line = body_text.split("\n", 1)[0] if body_text else ""
        if first_line.strip() != chapter:
            body_text = f"{chapter}\n{body_text}" if body_text else chapter

    return body_text, chapter
