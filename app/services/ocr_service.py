"""GLM-OCR-backed OCR service (vLLM OpenAI-compatible client).

Provides a small, testable surface that the rest of the pipeline can call:

* :func:`extract_text_with_ocr` - OCR on an image passed as raw bytes (used by
  the PDF page-level fallback after PyMuPDF renders a page to PNG).
* :func:`extract_text_from_image_file` - OCR on an uploaded image file.
* :func:`transcribe_image` - OCR on an in-memory ``PIL.Image`` with an
  optional custom prompt. Used by the grading service after its opencv
  preprocessing pipeline.
* :func:`extract_text_with_ocr_async`, :func:`extract_text_from_image_file_async`,
  :func:`transcribe_image_async` - async siblings that POST through
  ``httpx.AsyncClient``. Prefer these from async handlers so multiple
  in-flight OCR requests can be fused by vLLM's continuous batcher (with
  ``--speculative-config mtp`` this gives a sizeable throughput win over
  running the sync version in an executor).
* :func:`normalize_ocr_text` - whitespace cleanup on the raw OCR output.
* :func:`postprocess_ocr_text` - math/noise normalization layered on top of
  ``normalize_ocr_text`` to turn raw OCR strings into parser-friendly text
  (e.g. ``a+b`` -> ``a + b``, ``--`` -> ``-``).
* :func:`is_ocr_available` - lets callers degrade gracefully when the engine
  is not reachable.

Design notes:
    * The engine is **no longer loaded in-process via ``transformers``**.
      Instead the module talks to a vLLM server that serves
      ``zai-org/GLM-OCR`` behind the OpenAI-compatible chat completions API
      (default ``http://localhost:8080/v1``). This offloads the multi-GB
      weights + CUDA kernels to a dedicated server process and lets us
      benefit from vLLM's continuous batching, paged KV cache, and
      speculative decoding (``--speculative-config '{"method": "mtp",
      "num_speculative_tokens": 3}'``).
    * Images are sent inline as ``data:image/png;base64,...`` URLs inside
      an ``image_url`` content block, which is the standard OpenAI
      multimodal payload shape that vLLM accepts.
    * The HTTP client is memoised in a module-level singleton
      (``functools.lru_cache``) so we keep one connection pool per process
      rather than tearing down TLS / TCP state between requests.
    * If ``httpx`` is not installed the service logs a warning and every
      OCR call returns an empty string; callers are expected to keep their
      native text when OCR yields nothing.
    * GLM-OCR does not return per-box confidences, so the legacy
      ``min_confidence`` parameter is accepted but unused - it exists only so
      the function signature stays stable for existing callers.
    * All exceptions are caught and logged so an OCR failure never crashes
      the ingestion or grading request.

Environment variables:
    * ``GLM_OCR_BASE_URL`` - vLLM OpenAI endpoint. Default
      ``http://localhost:8080/v1``.
    * ``GLM_OCR_MODEL_NAME`` - name under which vLLM serves the model (i.e.
      the value you passed to ``--served-model-name``). Default ``glm-ocr``.
    * ``GLM_OCR_API_KEY`` - sent as ``Authorization: Bearer <key>``. vLLM
      ignores it unless started with ``--api-key``. Default ``EMPTY``.
    * ``GLM_OCR_PROMPT`` - default task prompt. ``"Text Recognition:"``.
    * ``GLM_OCR_MAX_NEW_TOKENS`` - default ``max_tokens`` per request (2048).
    * ``GLM_OCR_TEMPERATURE`` - sampling temperature. OCR wants greedy-ish
      output, so default is ``0.0``.
    * ``GLM_OCR_REQUEST_TIMEOUT`` - per-request timeout seconds (180).
    * ``GLM_OCR_HEALTH_TIMEOUT`` - timeout for the ``/models`` health probe (5).
    * ``GLM_OCR_MAX_IMAGE_SIDE`` - cap in pixels on the longest image side
      before we send the image to vLLM. Protects against oversized scans
      that would exceed ``--max-model-len``. Default 2000; set to 0 to
      disable downscaling.
    * ``GLM_OCR_MODEL_PATH`` - **deprecated**; only used by
      :func:`get_ocr_status` to populate the informational ``model`` field
      when ``GLM_OCR_MODEL_NAME`` is not set.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# GLM-OCR is multilingual; the ``lang`` argument is kept on the public API
# purely for backwards compatibility with earlier PaddleOCR-based callers.
DEFAULT_OCR_LANG = "multilingual"

# Retained for API compatibility. GLM-OCR does not emit per-detection
# confidence scores, so this threshold is a no-op today.
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# vLLM OpenAI-compatible endpoint. The ``/v1`` suffix is required because
# httpx resolves relative URLs ("chat/completions", "models") against the
# base_url. See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
GLM_OCR_BASE_URL = os.environ.get("GLM_OCR_BASE_URL", "http://localhost:8080/v1")

# Name under which vLLM serves the model. Must match the value passed to
# ``vllm serve --served-model-name``. Our deployment uses ``glm-ocr``.
GLM_OCR_MODEL_NAME = os.environ.get("GLM_OCR_MODEL_NAME", "glm-ocr")

# Optional API key. vLLM only validates it when started with ``--api-key``;
# otherwise any non-empty string (including ``"EMPTY"``) works. We always
# send the header so the config is uniform across environments.
GLM_OCR_API_KEY = os.environ.get("GLM_OCR_API_KEY", "EMPTY")

GLM_OCR_PROMPT = os.environ.get("GLM_OCR_PROMPT", "Text Recognition:")

# Mapped to OpenAI's ``max_tokens``. 2048 is enough for typical handwritten
# student pages; raise via env var for long answer-key PDFs.
GLM_OCR_MAX_NEW_TOKENS = int(os.environ.get("GLM_OCR_MAX_NEW_TOKENS", "2048"))

# OCR should be effectively deterministic. 0.0 + no top_p override = greedy
# decoding on vLLM.
GLM_OCR_TEMPERATURE = float(os.environ.get("GLM_OCR_TEMPERATURE", "0.0"))

# 3-minute per-request ceiling. Speculative decoding keeps typical pages
# under 5s, but first-request warm-up on a cold vLLM server can spike.
GLM_OCR_REQUEST_TIMEOUT = float(os.environ.get("GLM_OCR_REQUEST_TIMEOUT", "180"))

# Short timeout for the health probe so /health/ocr never blocks the UI.
GLM_OCR_HEALTH_TIMEOUT = float(os.environ.get("GLM_OCR_HEALTH_TIMEOUT", "5"))

# Cap the longest side (in pixels) of any image we send to vLLM. A 1200×1600
# scan consumes ~3-5k vision tokens; a raw 3000×4000 phone photo would blow
# past the server's ``--max-model-len 16384`` budget and get truncated or
# rejected. 2000px on the long side keeps us well under that ceiling while
# preserving enough resolution for handwriting OCR. Set to 0 to disable.
GLM_OCR_MAX_IMAGE_SIDE = int(os.environ.get("GLM_OCR_MAX_IMAGE_SIDE", "2000"))

# Deprecated: kept so ``get_ocr_status`` can still populate the ``model``
# field when only the old env var is set. Not used for inference anymore.
GLM_OCR_MODEL_PATH = os.environ.get("GLM_OCR_MODEL_PATH", "zai-org/GLM-OCR")

# Math / punctuation symbols OCR frequently emits in unicode form that the
# downstream parsers and embedder are happier seeing in ASCII.
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

# Catch any residual chat special tokens the server leaves behind, e.g.
# ``<|endoftext|>`` / ``<|user|>``. vLLM strips them by default but we
# belt-and-suspenders it here in case a future model config changes that.
_GLM_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


@dataclass
class _GLMOcrEngine:
    """Small wrapper around the vLLM HTTP client.

    Holds a memoised ``httpx.Client`` plus the served model name so call
    sites don't need to know where the endpoint lives. Named ``_GLMOcrEngine``
    (rather than ``_GLMOcrClient``) so we don't need to rename anything in
    the callers' logs/docstrings.
    """

    client: Any
    model: str
    base_url: str


# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_ocr_engine(lang: str = DEFAULT_OCR_LANG) -> _GLMOcrEngine | None:
    """Return a memoised vLLM HTTP client, or ``None`` if ``httpx`` is missing.

    The client is created lazily on first use so application startup stays
    fast. We deliberately do **not** probe the server here - a slow or
    temporarily-down vLLM process must not break the whole import path or
    turn every first request into a hard failure. Health probing lives in
    :func:`get_ocr_status` with ``probe_engine=True``.

    ``lang`` is accepted for API compatibility; GLM-OCR is multilingual.
    """
    del lang  # GLM-OCR is multilingual; parameter is kept for compatibility.

    try:
        import httpx
    except ImportError:
        logger.warning(
            "ocr_service: httpx not installed; vLLM OCR client disabled"
        )
        return None

    try:
        headers = {
            "Authorization": f"Bearer {GLM_OCR_API_KEY}",
            "Content-Type": "application/json",
        }
        # A single long-lived Client keeps the TCP + TLS connection warm,
        # which matters because the OCR image bodies are large (base64 PNG).
        client = httpx.Client(
            base_url=GLM_OCR_BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(
                GLM_OCR_REQUEST_TIMEOUT,
                connect=min(10.0, GLM_OCR_REQUEST_TIMEOUT),
            ),
        )
        logger.info(
            "ocr_service: vLLM OCR client initialised base_url=%s model=%s",
            GLM_OCR_BASE_URL,
            GLM_OCR_MODEL_NAME,
        )
        return _GLMOcrEngine(
            client=client,
            model=GLM_OCR_MODEL_NAME,
            base_url=GLM_OCR_BASE_URL,
        )
    except Exception:
        logger.exception(
            "ocr_service: failed to initialise vLLM OCR client base_url=%s",
            GLM_OCR_BASE_URL,
        )
        return None


@lru_cache(maxsize=1)
def _get_ocr_engine_async(lang: str = DEFAULT_OCR_LANG) -> _GLMOcrEngine | None:
    """Return a memoised async vLLM client for the high-throughput path.

    Parallel to :func:`_get_ocr_engine` but wraps an ``httpx.AsyncClient``
    so callers inside an asyncio event loop can fire many OCR requests
    concurrently and let vLLM's continuous batching amortise the vision
    encoder cost across them. One instance per process - ``httpx.AsyncClient``
    is safe to share across coroutines on the same event loop, which matches
    the FastAPI deployment model (one loop per worker).

    Do NOT call this from a non-async context: awaiting the returned client
    outside a loop will raise. Use :func:`_get_ocr_engine` instead for
    synchronous call sites.
    """
    del lang  # GLM-OCR is multilingual; parameter is kept for compatibility.

    try:
        import httpx
    except ImportError:
        logger.warning(
            "ocr_service: httpx not installed; async vLLM OCR client disabled"
        )
        return None

    try:
        headers = {
            "Authorization": f"Bearer {GLM_OCR_API_KEY}",
            "Content-Type": "application/json",
        }
        client = httpx.AsyncClient(
            base_url=GLM_OCR_BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(
                GLM_OCR_REQUEST_TIMEOUT,
                connect=min(10.0, GLM_OCR_REQUEST_TIMEOUT),
            ),
        )
        logger.info(
            "ocr_service: async vLLM OCR client initialised base_url=%s model=%s",
            GLM_OCR_BASE_URL,
            GLM_OCR_MODEL_NAME,
        )
        return _GLMOcrEngine(
            client=client,
            model=GLM_OCR_MODEL_NAME,
            base_url=GLM_OCR_BASE_URL,
        )
    except Exception:
        logger.exception(
            "ocr_service: failed to initialise async vLLM OCR client "
            "base_url=%s",
            GLM_OCR_BASE_URL,
        )
        return None


def is_ocr_available(lang: str = DEFAULT_OCR_LANG) -> bool:
    """Quick probe used by callers that want to branch on OCR availability.

    "Available" here means "the HTTP client is constructable" - it does
    **not** guarantee the vLLM server is up. Use :func:`get_ocr_status`
    with ``probe_engine=True`` for a live reachability check.
    """
    return _get_ocr_engine(lang) is not None


def get_ocr_status(
    lang: str = DEFAULT_OCR_LANG,
    *,
    probe_engine: bool = False,
) -> dict[str, Any]:
    """Return a diagnostic snapshot describing whether GLM-OCR is usable.

    Keys:
        * ``installed``      - ``httpx`` importable in this process.
        * ``initialized``    - vLLM server responded to ``GET /models`` and
          exposes a model matching ``GLM_OCR_MODEL_NAME`` (only populated
          when ``probe_engine=True`` or a previous probe succeeded).
        * ``available``      - ``True`` when OCR calls should succeed. When
          ``probe_engine`` is ``False`` this is optimistic: it means
          "httpx importable + config looks fine".
        * ``lang``           - language probed (informational only).
        * ``engine``         - always ``"glm-ocr-vllm"``.
        * ``model``          - served-model-name the client will request.
        * ``base_url``       - vLLM endpoint this client targets.
        * ``httpx_version``  - package version string if importable, else ``None``.
        * ``reason``         - short human-readable summary, handy for logs
          and the ``/health/ocr`` endpoint.

    ``probe_engine=False`` is the cheap, side-effect-free variant used at
    startup. Set ``probe_engine=True`` from the ``/health/ocr`` endpoint
    when you actually want to verify the vLLM server is reachable.

    This function never raises: any unexpected error is swallowed and
    surfaced in ``reason`` so health checks stay 200 even when OCR is broken.
    """
    status: dict[str, Any] = {
        "installed": False,
        "initialized": False,
        "available": False,
        "lang": lang,
        "engine": "glm-ocr-vllm",
        "model": GLM_OCR_MODEL_NAME or GLM_OCR_MODEL_PATH,
        "base_url": GLM_OCR_BASE_URL,
        "httpx_version": None,
        "reason": "",
    }

    try:
        import httpx  # type: ignore[import-not-found]
        status["installed"] = True
        status["httpx_version"] = getattr(httpx, "__version__", None)
    except ImportError:
        status["reason"] = "httpx is not installed; vLLM OCR client disabled"
        return status
    except Exception as exc:
        status["reason"] = f"httpx import failed: {exc}"
        return status

    engine = _get_ocr_engine(lang)
    if engine is None:
        status["reason"] = (
            f"httpx installed (v{status['httpx_version']}) but vLLM OCR "
            f"client init failed for base_url={GLM_OCR_BASE_URL}"
        )
        return status

    if not probe_engine:
        status["available"] = True
        status["reason"] = (
            f"httpx v{status['httpx_version']} importable; vLLM OCR will "
            f"connect to {GLM_OCR_BASE_URL} on first use "
            f"(model={GLM_OCR_MODEL_NAME})"
        )
        return status

    # Live probe: ask vLLM for its served models and make sure ours is listed.
    try:
        response = engine.client.get("/models", timeout=GLM_OCR_HEALTH_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        status["reason"] = (
            f"vLLM OCR probe failed: {exc.__class__.__name__}: {exc} "
            f"(base_url={GLM_OCR_BASE_URL})"
        )
        return status

    served_ids = {
        entry.get("id")
        for entry in (payload.get("data") or [])
        if isinstance(entry, dict)
    }
    if GLM_OCR_MODEL_NAME not in served_ids:
        status["reason"] = (
            f"vLLM reachable at {GLM_OCR_BASE_URL} but model "
            f"'{GLM_OCR_MODEL_NAME}' is not served (found: "
            f"{sorted(s for s in served_ids if s) or 'none'})"
        )
        return status

    status["initialized"] = True
    status["available"] = True
    status["reason"] = (
        f"vLLM OCR ready (model={GLM_OCR_MODEL_NAME}, "
        f"base_url={GLM_OCR_BASE_URL}, httpx v{status['httpx_version']})"
    )
    return status


def log_ocr_status(
    lang: str = DEFAULT_OCR_LANG,
    *,
    probe_engine: bool = False,
) -> dict[str, Any]:
    """Log and return the OCR status.

    Defaults to ``probe_engine=False`` so calling this at FastAPI startup
    does NOT hit the network.
    """
    status = get_ocr_status(lang, probe_engine=probe_engine)
    if status["available"]:
        logger.info(
            "ocr_service: OCR client ENABLED - %s",
            status["reason"],
        )
    else:
        logger.warning(
            "ocr_service: OCR client DISABLED - %s",
            status["reason"],
        )
    return status


# ---------------------------------------------------------------------------
# Public OCR entry points
# ---------------------------------------------------------------------------


def extract_text_with_ocr(
    image_bytes: bytes,
    lang: str = DEFAULT_OCR_LANG,
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,  # noqa: ARG001
) -> str:
    """OCR an image supplied as raw bytes and return cleaned plain text.

    The returned string has been normalised and run through
    :func:`postprocess_ocr_text`, so callers can feed it directly into the
    parser pipeline.

    Returns an empty string on any failure (engine missing, decode error,
    network error, HTTP error, empty model response). Callers should treat
    ``""`` as "no OCR text available" and keep their native extraction.

    ``min_confidence`` is kept for backwards compatibility and ignored:
    GLM-OCR does not emit per-detection confidence scores.
    """
    if not image_bytes:
        return ""

    engine = _get_ocr_engine(lang)
    if engine is None:
        return ""

    data_url = _image_bytes_to_data_url(image_bytes)
    if data_url is None:
        return ""

    try:
        raw_text = _run_glm_ocr(engine, data_url)
    except Exception:
        logger.exception("ocr_service: GLM-OCR inference failed")
        raw_text = ""

    cleaned = postprocess_ocr_text(_strip_model_tokens(raw_text))
    logger.debug("ocr_service: GLM-OCR produced len=%d", len(cleaned))
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


async def extract_text_with_ocr_async(
    image_bytes: bytes,
    lang: str = DEFAULT_OCR_LANG,
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,  # noqa: ARG001
) -> str:
    """Async sibling of :func:`extract_text_with_ocr`.

    Use from async handlers (e.g. FastAPI routes) when you want to OCR
    many images concurrently - vLLM's continuous batching will fuse
    the requests into one forward pass, which is a significant win over
    the sync/executor path when grading a batch of pages.
    """
    if not image_bytes:
        return ""

    engine = _get_ocr_engine_async(lang)
    if engine is None:
        return ""

    data_url = _image_bytes_to_data_url(image_bytes)
    if data_url is None:
        return ""

    try:
        raw_text = await _run_glm_ocr_async(engine, data_url)
    except Exception:
        logger.exception("ocr_service: async GLM-OCR inference failed")
        raw_text = ""

    cleaned = postprocess_ocr_text(_strip_model_tokens(raw_text))
    logger.debug("ocr_service: async GLM-OCR produced len=%d", len(cleaned))
    return cleaned


async def extract_text_from_image_file_async(
    file_bytes: bytes,
    lang: str = DEFAULT_OCR_LANG,
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> str:
    """Async sibling of :func:`extract_text_from_image_file`."""
    return await extract_text_with_ocr_async(
        file_bytes, lang=lang, min_confidence=min_confidence
    )


def transcribe_image(
    pil_image: Any,
    prompt: str = GLM_OCR_PROMPT,
    *,
    max_new_tokens: int | None = None,
    lang: str = DEFAULT_OCR_LANG,
) -> str:
    """Run GLM-OCR on an in-memory PIL image with a custom prompt.

    Intended for callers that already hold a decoded + preprocessed
    :class:`PIL.Image.Image` (for example the grading service, which
    runs its own opencv-based denoise / deskew / binarise pipeline
    before sending the page through OCR).

    Reuses the module-level vLLM HTTP client (``@lru_cache``-memoised)
    so we keep one connection pool per process even when both the
    answer-key ingestion path (``extract_text_with_ocr``) and the
    student-handwriting path (this function) are active in the same
    request flow.

    Parameters
    ----------
    pil_image:
        Decoded image ready for OCR. Must be a ``PIL.Image.Image``; any
        mode that Pillow can ``convert("RGB")`` is accepted.
    prompt:
        Task prompt sent along with the image. Defaults to
        :data:`GLM_OCR_PROMPT` (``"Text Recognition:"``) which asks GLM-OCR
        for plain transcription. Callers can pass a custom prompt to ask
        for structured extraction, but GLM-OCR is an OCR model - reasoning
        tasks belong in a downstream LLM.
    max_new_tokens:
        Optional override for the generation budget (mapped to OpenAI's
        ``max_tokens``). Defaults to :data:`GLM_OCR_MAX_NEW_TOKENS`.
    lang:
        Accepted for API compatibility; GLM-OCR is multilingual.

    Returns
    -------
    str
        Cleaned, post-processed transcription. Empty string on any
        failure (engine missing, encode error, network error).
    """
    if pil_image is None:
        return ""

    engine = _get_ocr_engine(lang)
    if engine is None:
        return ""

    data_url = _pil_to_data_url(pil_image)
    if data_url is None:
        return ""

    effective_max_new_tokens = (
        GLM_OCR_MAX_NEW_TOKENS if max_new_tokens is None else int(max_new_tokens)
    )

    try:
        raw_text = _run_glm_ocr(
            engine,
            data_url,
            prompt=prompt,
            max_new_tokens=effective_max_new_tokens,
        )
    except Exception:
        logger.exception("ocr_service: GLM-OCR transcription failed")
        raw_text = ""

    cleaned = postprocess_ocr_text(_strip_model_tokens(raw_text))
    logger.debug(
        "ocr_service: GLM-OCR transcription produced len=%d (prompt_head=%r)",
        len(cleaned),
        prompt[:60],
    )
    return cleaned


async def transcribe_image_async(
    pil_image: Any,
    prompt: str = GLM_OCR_PROMPT,
    *,
    max_new_tokens: int | None = None,
    lang: str = DEFAULT_OCR_LANG,
) -> str:
    """Async sibling of :func:`transcribe_image`.

    Preferred over :func:`transcribe_image` from async call sites. The
    grading service currently offloads the sync version to the default
    executor; switching hot paths to this coroutine lets vLLM's
    continuous batching see multiple in-flight OCR requests at once,
    which is where the speculative-decoding + paged-KV throughput wins
    actually show up.

    Identical semantics, identical return shape - empty string on any
    failure. See :func:`transcribe_image` for parameter docs.
    """
    if pil_image is None:
        return ""

    engine = _get_ocr_engine_async(lang)
    if engine is None:
        return ""

    data_url = _pil_to_data_url(pil_image)
    if data_url is None:
        return ""

    effective_max_new_tokens = (
        GLM_OCR_MAX_NEW_TOKENS if max_new_tokens is None else int(max_new_tokens)
    )

    try:
        raw_text = await _run_glm_ocr_async(
            engine,
            data_url,
            prompt=prompt,
            max_new_tokens=effective_max_new_tokens,
        )
    except Exception:
        logger.exception("ocr_service: async GLM-OCR transcription failed")
        raw_text = ""

    cleaned = postprocess_ocr_text(_strip_model_tokens(raw_text))
    logger.debug(
        "ocr_service: async GLM-OCR transcription produced len=%d (prompt_head=%r)",
        len(cleaned),
        prompt[:60],
    )
    return cleaned


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


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_ocr_payload(
    model: str,
    image_data_url: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Build the OpenAI chat-completions request body for one OCR call.

    Uses the standard OpenAI multimodal payload shape that vLLM accepts:

        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }]

    ``max_new_tokens`` is mapped onto OpenAI's ``max_tokens``. Temperature
    defaults to :data:`GLM_OCR_TEMPERATURE` (0.0) for OCR determinism.
    """
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": int(max_new_tokens),
        "temperature": GLM_OCR_TEMPERATURE,
        "stream": False,
    }


def _parse_ocr_response(response: Any) -> str:
    """Extract the OCR text from a vLLM chat-completions HTTP response.

    Handles 4xx/5xx, non-JSON bodies, empty ``choices``, missing
    ``content``, and the structured list-of-parts content variant vLLM
    occasionally returns. Never raises - always returns a string
    (possibly empty, which the caller treats as "no OCR output").
    """
    if response.status_code >= 400:
        # Body can be large; truncate so we don't flood the log with a
        # 50 KB base64 echo on validation errors.
        body_preview = (response.text or "")[:500]
        logger.warning(
            "ocr_service: vLLM OCR HTTP %d: %s",
            response.status_code,
            body_preview,
        )
        return ""

    try:
        data = response.json()
    except ValueError:
        logger.warning(
            "ocr_service: vLLM OCR returned non-JSON body (len=%d)",
            len(response.content or b""),
        )
        return ""

    choices = data.get("choices") or []
    if not choices:
        logger.warning("ocr_service: vLLM OCR response had no choices: %r", data)
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        logger.warning(
            "ocr_service: vLLM OCR response had no content: %r", choices[0]
        )
        return ""

    # Chat completions content is usually a string, but vLLM can emit the
    # structured list-of-parts form when returning tool/image output. Flatten
    # text parts so downstream post-processing sees a single string.
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        content = "".join(parts)

    return content or ""


def _run_glm_ocr(
    engine: _GLMOcrEngine,
    image_data_url: str,
    *,
    prompt: str = GLM_OCR_PROMPT,
    max_new_tokens: int = GLM_OCR_MAX_NEW_TOKENS,
) -> str:
    """Synchronously POST one OCR request and return the raw text."""
    import httpx  # local import so this module stays importable without httpx

    payload = _build_ocr_payload(
        engine.model, image_data_url, prompt, max_new_tokens
    )

    try:
        response = engine.client.post("/chat/completions", json=payload)
    except httpx.TimeoutException:
        logger.warning(
            "ocr_service: vLLM OCR request timed out after %.1fs (base_url=%s)",
            GLM_OCR_REQUEST_TIMEOUT,
            engine.base_url,
        )
        return ""
    except httpx.RequestError as exc:
        logger.warning(
            "ocr_service: vLLM OCR network error: %s: %s (base_url=%s)",
            exc.__class__.__name__,
            exc,
            engine.base_url,
        )
        return ""

    return _parse_ocr_response(response)


async def _run_glm_ocr_async(
    engine: _GLMOcrEngine,
    image_data_url: str,
    *,
    prompt: str = GLM_OCR_PROMPT,
    max_new_tokens: int = GLM_OCR_MAX_NEW_TOKENS,
) -> str:
    """Async sibling of :func:`_run_glm_ocr`.

    Awaits on ``httpx.AsyncClient.post`` so multiple concurrent callers
    can all be in-flight at the vLLM server, where the scheduler fuses
    them into a single continuous batch. Error handling mirrors the
    sync path byte-for-byte - any failure returns ``""``.
    """
    import httpx

    payload = _build_ocr_payload(
        engine.model, image_data_url, prompt, max_new_tokens
    )

    try:
        response = await engine.client.post("/chat/completions", json=payload)
    except httpx.TimeoutException:
        logger.warning(
            "ocr_service: async vLLM OCR request timed out after %.1fs "
            "(base_url=%s)",
            GLM_OCR_REQUEST_TIMEOUT,
            engine.base_url,
        )
        return ""
    except httpx.RequestError as exc:
        logger.warning(
            "ocr_service: async vLLM OCR network error: %s: %s (base_url=%s)",
            exc.__class__.__name__,
            exc,
            engine.base_url,
        )
        return ""

    return _parse_ocr_response(response)


def _strip_model_tokens(text: str) -> str:
    """Strip residual ``<|...|>`` chat special tokens if any slipped through."""
    if not text:
        return ""
    stripped = _GLM_SPECIAL_TOKEN_RE.sub("", text)
    return normalize_ocr_text(stripped)


def _maybe_downscale(pil_image: Any) -> Any:
    """Shrink ``pil_image`` in place-ish if its longest side exceeds the cap.

    Guards against pathological inputs (4000x5000 phone scans, 6000x8000
    archival TIFFs) that would otherwise balloon the base64 payload and
    exceed vLLM's ``--max-model-len``. Uses LANCZOS resampling so
    handwriting stays readable after the shrink.

    Returns the input unchanged when:

    * ``GLM_OCR_MAX_IMAGE_SIDE`` <= 0 (downscaling disabled),
    * the longest side already fits, or
    * Pillow's ``resize`` raises (we never want an encoding quirk to
      kill OCR - the raw image is still a valid fallback, and vLLM will
      either smart-resize it internally or return a clear 400 that we
      log and translate to ``""``).
    """
    if GLM_OCR_MAX_IMAGE_SIDE <= 0:
        return pil_image

    try:
        from PIL import Image  # re-imported here so the helper works standalone
    except ImportError:
        return pil_image

    try:
        width, height = pil_image.size
    except Exception:
        return pil_image

    longest = max(width, height)
    if longest <= GLM_OCR_MAX_IMAGE_SIDE:
        return pil_image

    scale = GLM_OCR_MAX_IMAGE_SIDE / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    try:
        resized = pil_image.resize(new_size, Image.LANCZOS)
    except Exception:
        logger.warning(
            "ocr_service: resize failed for %sx%s -> %s; sending original",
            width,
            height,
            new_size,
        )
        return pil_image

    logger.debug(
        "ocr_service: downscaled image %sx%s -> %sx%s (cap=%d px)",
        width,
        height,
        new_size[0],
        new_size[1],
        GLM_OCR_MAX_IMAGE_SIDE,
    )
    return resized


def _image_bytes_to_data_url(image_bytes: bytes) -> str | None:
    """Decode ``image_bytes`` and encode them as a base64 PNG data URL.

    Re-encoding through Pillow guarantees a valid PNG regardless of the
    caller's source format (JPEG / WEBP / ...), which avoids surprises on
    the vLLM side where the multimodal tokenizer validates mime type.
    Oversized images are downscaled via :func:`_maybe_downscale` before
    encoding so we never exceed vLLM's token budget.
    Returns ``None`` if Pillow is missing or the image cannot be decoded.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning(
            "ocr_service: Pillow not available; cannot encode image for OCR"
        )
        return None

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB")
            resized = _maybe_downscale(rgb)
            buf = io.BytesIO()
            resized.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
    except Exception:
        logger.exception("ocr_service: failed to decode/encode image bytes for GLM-OCR")
        return None


def _pil_to_data_url(pil_image: Any) -> str | None:
    """Encode an already-decoded PIL image as a base64 PNG data URL.

    Mirrors :func:`_image_bytes_to_data_url` but skips the initial decode so
    callers that already hold a preprocessed PIL image (e.g. the grading
    service after its opencv denoise/deskew pipeline) don't re-encode →
    re-decode → re-encode just to hand the image to
    :func:`transcribe_image`. Oversized images are downscaled via
    :func:`_maybe_downscale` before encoding.
    """
    if pil_image is None:
        return None
    try:
        rgb = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
        resized = _maybe_downscale(rgb)
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        logger.exception("ocr_service: failed to encode PIL image for GLM-OCR")
        return None
