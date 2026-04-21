"""PaddleOCR-backed OCR service.

Provides a small, testable surface that the rest of the pipeline can call:

* :func:`extract_text_with_ocr` - OCR on an image passed as raw bytes (used by
  the PDF page-level fallback after PyMuPDF renders a page to PNG).
* :func:`extract_text_from_image_file` - OCR on an uploaded image file.
* :func:`normalize_ocr_text` - whitespace cleanup on the raw OCR output.
* :func:`postprocess_ocr_text` - math/noise normalization layered on top of
  ``normalize_ocr_text`` to turn raw PaddleOCR strings into parser-friendly
  text (e.g. ``a+b`` -> ``a + b``, ``--`` -> ``-``).
* :func:`choose_best_text` - decide whether to keep native PDF text or a page's
  OCR output, based on a cheap readability score. Prevents OCR from degrading
  already-good native extraction.
* :func:`is_ocr_available` - lets callers degrade gracefully when the engine
  is not installed.

Design notes:
    * PaddleOCR is imported lazily and the engine is memoised in a
      module-level singleton so a model download + init only happens once per
      process (``functools.lru_cache``).
    * If ``paddleocr`` or ``paddlepaddle`` is not installed, the service logs
      a warning and returns an empty string. Callers are expected to keep
      their native text when OCR yields nothing.
    * Low-confidence detections are dropped *before* post-processing so noisy
      fragments (broken math, ``?``, ``--``) never reach the parser.
    * All exceptions are caught and logged so an OCR failure never crashes
      the ingestion request.
"""

from __future__ import annotations

import io
import logging
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OCR_LANG = "korean"

# Drop PaddleOCR detections whose per-box confidence is below this threshold.
# 0.5 is conservative: PaddleOCR is usually very confident (>0.9) on real text,
# so most garbage fragments fall well below this line.
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Math / punctuation symbols PaddleOCR frequently emits in unicode form that
# the downstream parsers and embedder are happier seeing in ASCII.
_MATH_SYMBOL_REPLACEMENTS: dict[str, str] = {
    "\u2014": "-",   # em dash
    "\u2013": "-",   # en dash
    "\u2212": "-",   # minus sign
    "\u00d7": "*",   # multiplication sign
    "\u2715": "*",   # multiplication X
    "\u2a2f": "*",   # vector/cross product
    "\u00f7": "/",   # division sign
}

# Operator spacing: only add spaces when both sides are "word-like" so we do
# not mangle unary minus (``-5``), signed exponents (``1e-3``) or parenthesised
# negatives (``(-x)``). ``\w`` in Python matches Korean characters too.
_OPERATOR_SPACING_RE = re.compile(r"(?<=[\w\)])([+\-*/=])(?=[\w\(])")

# Characters considered "meaningful" when scoring how readable a string is.
# Covers ASCII letters/digits and the Hangul syllables block.
_READABLE_CHAR_RE = re.compile(r"[A-Za-z0-9\uac00-\ud7a3]")


@lru_cache(maxsize=4)
def _get_ocr_engine(lang: str = DEFAULT_OCR_LANG) -> Any | None:
    """Return a memoised PaddleOCR instance for ``lang``, or ``None`` if unavailable.

    The engine is created lazily on first use so application startup stays fast
    even when OCR is never invoked.
    """
    try:
        from paddleocr import PaddleOCR  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "ocr_service: paddleocr is not installed; OCR fallback disabled"
        )
        return None

    try:
        engine = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        logger.info("ocr_service: PaddleOCR engine initialized lang=%s", lang)
        return engine
    except Exception:
        logger.exception("ocr_service: failed to initialize PaddleOCR lang=%s", lang)
        return None


def is_ocr_available(lang: str = DEFAULT_OCR_LANG) -> bool:
    """Quick probe used by callers that want to branch on OCR availability."""
    return _get_ocr_engine(lang) is not None


def get_ocr_status(
    lang: str = DEFAULT_OCR_LANG,
    *,
    probe_engine: bool = False,
) -> dict[str, Any]:
    """Return a diagnostic snapshot describing whether PaddleOCR is usable.

    Keys:
        * ``installed``     - ``paddleocr`` importable in this process.
        * ``initialized``   - engine for ``lang`` has been successfully built
          (only populated when ``probe_engine=True`` or the cache is already
          warm from an earlier call).
        * ``available``     - ``True`` when the pipeline will actually run
          OCR. When ``probe_engine`` is ``False`` this is optimistic: it
          reflects "installed + not known to be broken" and the first real
          OCR call is still where the engine is built.
        * ``lang``          - language probed.
        * ``paddleocr_version`` - package version string if importable, else
          ``None``.
        * ``reason``        - short human-readable summary, handy for logs
          and the ``/health/ocr`` endpoint.

    ``probe_engine=False`` is the cheap, side-effect-free variant used at
    startup so we do NOT trigger PaddleOCR's (expensive, logger-mutating)
    engine init just to print a banner. Set ``probe_engine=True`` from the
    ``/health/ocr`` endpoint or a diagnostic command when you actually want
    to force the model download + init now.

    This function never raises: any unexpected error is swallowed and
    surfaced in ``reason`` so health checks stay 200 even when OCR is broken.
    """
    status: dict[str, Any] = {
        "installed": False,
        "initialized": False,
        "available": False,
        "lang": lang,
        "paddleocr_version": None,
        "reason": "",
    }

    try:
        import paddleocr  # type: ignore[import-not-found]
        status["installed"] = True
        status["paddleocr_version"] = getattr(paddleocr, "__version__", None)
    except ImportError:
        status["reason"] = "paddleocr is not installed; OCR fallback disabled"
        return status
    except Exception as exc:
        status["reason"] = f"paddleocr import failed: {exc}"
        return status

    if not probe_engine:
        status["available"] = True
        status["reason"] = (
            f"paddleocr v{status['paddleocr_version']} importable "
            f"(engine will init on first use, lang={lang})"
        )
        return status

    engine = _get_ocr_engine(lang)
    if engine is None:
        status["reason"] = (
            f"paddleocr installed (v{status['paddleocr_version']}) "
            f"but engine init failed for lang={lang}"
        )
        return status

    status["initialized"] = True
    status["available"] = True
    status["reason"] = (
        f"PaddleOCR v{status['paddleocr_version']} ready (lang={lang})"
    )
    return status


def log_ocr_status(
    lang: str = DEFAULT_OCR_LANG,
    *,
    probe_engine: bool = False,
) -> dict[str, Any]:
    """Log and return the OCR status.

    Defaults to ``probe_engine=False`` so calling this at FastAPI startup
    does NOT trigger PaddleOCR engine init (which would stall boot AND
    hijack the root logger, silencing our INFO logs everywhere else).
    """
    status = get_ocr_status(lang, probe_engine=probe_engine)
    if status["available"]:
        logger.info(
            "ocr_service: OCR fallback ENABLED - %s",
            status["reason"],
        )
    else:
        logger.warning(
            "ocr_service: OCR fallback DISABLED - %s",
            status["reason"],
        )
    return status


def extract_text_with_ocr(
    image_bytes: bytes,
    lang: str = DEFAULT_OCR_LANG,
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> str:
    """OCR an image supplied as raw bytes and return cleaned plain text.

    The returned string has already been confidence-filtered, whitespace
    normalized, and run through :func:`postprocess_ocr_text`, so callers can
    feed it directly into the parser pipeline.

    Returns an empty string on any failure (engine missing, decode error,
    OCR exception, empty result). Callers should treat ``""`` as "no OCR
    text available" and keep their native extraction.
    """
    if not image_bytes:
        return ""

    engine = _get_ocr_engine(lang)
    if engine is None:
        return ""

    image = _decode_image_to_numpy(image_bytes)
    if image is None:
        return ""

    try:
        result = engine.ocr(image, cls=True)
    except Exception:
        logger.exception("ocr_service: OCR inference failed")
        return ""

    formatted = _format_ocr_result(result, min_confidence=min_confidence)
    cleaned = postprocess_ocr_text(formatted)
    logger.debug(
        "ocr_service: OCR produced len=%d (min_confidence=%.2f)",
        len(cleaned),
        min_confidence,
    )
    return cleaned


def extract_text_from_image_file(
    file_bytes: bytes,
    lang: str = DEFAULT_OCR_LANG,
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> str:
    """OCR an uploaded image file (PNG / JPG / JPEG / WEBP, ...).

    Thin wrapper over :func:`extract_text_with_ocr` kept as its own function
    so callers can express intent clearly at the endpoint layer.
    """
    return extract_text_with_ocr(file_bytes, lang=lang, min_confidence=min_confidence)


def normalize_ocr_text(text: str) -> str:
    """Collapse blank lines and trim each line of raw OCR output."""
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def postprocess_ocr_text(text: str) -> str:
    """Clean up normalized OCR text so it is safe to feed into parsers.

    Applied in order:

    1. Normalize math/punctuation symbols (em/en dashes -> ``-``, ``×`` -> ``*``,
       ``÷`` -> ``/``, etc.).
    2. Collapse repeated noise (``--`` -> ``-``, ``??`` -> ``?``).
    3. Insert spaces around binary arithmetic operators so ``a+b`` becomes
       ``a + b`` without touching unary ``-5`` or ``(-x)``.
    4. Collapse runs of spaces/tabs on each line while preserving line breaks.

    The function is idempotent on already-clean text and safe to run even when
    OCR yielded an empty string.
    """
    if not text:
        return ""

    for src, dst in _MATH_SYMBOL_REPLACEMENTS.items():
        if src in text:
            text = text.replace(src, dst)

    text = re.sub(r"-{2,}", "-", text)
    text = re.sub(r"\?{2,}", "?", text)

    text = _OPERATOR_SPACING_RE.sub(r" \1 ", text)

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    # Keep paragraph breaks but kill >2 consecutive blank lines.
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def readability_score(text: str) -> float:
    """Return the fraction of ``text`` made up of letters / digits / Hangul.

    Used by :func:`choose_best_text` to compare native vs OCR quality. A page
    full of ``?`` or box-drawing garbage scores near 0; a clean paragraph
    typically scores above ~0.75.
    """
    stripped = (text or "").strip()
    if not stripped:
        return 0.0
    valid = len(_READABLE_CHAR_RE.findall(stripped))
    return valid / len(stripped)


def choose_best_text(native: str, ocr: str) -> str:
    """Return whichever of ``native`` / ``ocr`` is more readable.

    Rules:
        * empty OCR  -> keep native
        * empty native -> use OCR
        * otherwise pick the one with the higher :func:`readability_score`;
          native wins ties so we never replace good native text with comparable
          OCR output.

    This is the safe default for the PDF pipeline: callers who also need to
    know *which* source won (for logging / metrics) can call
    :func:`readability_score` directly or use the ``source``-returning helper
    in :mod:`app.services.pdf_service`.
    """
    native = native or ""
    ocr = ocr or ""

    if not ocr.strip():
        return native
    if not native.strip():
        return ocr

    native_score = readability_score(native)
    ocr_score = readability_score(ocr)

    if ocr_score > native_score:
        return ocr
    return native


def _decode_image_to_numpy(image_bytes: bytes):
    """Decode ``image_bytes`` into an RGB numpy array suitable for PaddleOCR.

    Returns ``None`` if the image cannot be read or Pillow/numpy are missing.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        logger.warning(
            "ocr_service: Pillow/numpy not available; cannot decode image for OCR"
        )
        return None

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return np.array(img.convert("RGB"))
    except Exception:
        logger.exception("ocr_service: failed to decode image bytes")
        return None


def _format_ocr_result(
    result: Any,
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> str:
    """Convert PaddleOCR's nested output into clean multi-line text.

    PaddleOCR returns ``list[list[[bbox, (text, confidence)]]]`` (outer list =
    pages, inner list = detected boxes). We preserve detection order, which
    PaddleOCR sorts top-to-bottom, left-to-right, and drop any box whose
    confidence is below ``min_confidence``.
    """
    if not result:
        return ""

    kept = 0
    dropped = 0
    lines: list[str] = []

    for page in result:
        if not page:
            continue
        for item in page:
            text, confidence = _extract_text_from_item(item)
            if not text:
                continue
            if confidence < min_confidence:
                dropped += 1
                continue
            lines.append(text)
            kept += 1

    if dropped:
        logger.debug(
            "ocr_service: filtered low-confidence boxes kept=%d dropped=%d threshold=%.2f",
            kept,
            dropped,
            min_confidence,
        )

    return normalize_ocr_text("\n".join(lines))


def _extract_text_from_item(item: Any) -> tuple[str, float]:
    """Pull ``(text, confidence)`` out of one PaddleOCR detection entry.

    Confidence defaults to ``1.0`` when the underlying payload doesn't carry
    one (e.g. older PaddleOCR versions or a plain string payload) so such
    entries are never filtered out by the confidence threshold.
    """
    if not item or len(item) < 2:
        return "", 0.0

    payload = item[1]

    if isinstance(payload, (list, tuple)) and payload:
        text = str(payload[0]).strip()
        confidence = 1.0
        if len(payload) >= 2:
            try:
                confidence = float(payload[1])
            except (TypeError, ValueError):
                confidence = 1.0
        return text, confidence

    if isinstance(payload, str):
        return payload.strip(), 1.0

    return "", 0.0
