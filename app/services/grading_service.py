"""Qwen2.5-VL-7B-Instruct grading service (bitsandbytes 4-bit NF4).

This module owns the vision-language model used to auto-grade a
student's handwritten step-by-step solution. It is tuned for an
RTX 5080 (16 GB VRAM, CUDA 12.8) and designed to stay under a ~14 GB
peak allocation covering both the model weights and the per-request
inference activations.

Architecture
------------

- **Model**: ``Qwen/Qwen2.5-VL-7B-Instruct`` loaded with
  :class:`~transformers.BitsAndBytesConfig` in 4-bit NF4 (double
  quantisation, fp16 compute dtype). We load via ``bitsandbytes``
  rather than AWQ because ``autoawq`` has no supported wheel on
  Windows + Python 3.11 + torch 2.11 / CUDA 12.8 (Blackwell sm_120):
  its ``triton`` dependency lacks a Windows wheel and the last
  ``autoawq`` release predates torch 2.11. ``bitsandbytes`` 0.49+
  ships native Windows wheels with Blackwell kernels, so NF4 4-bit
  is the drop-in equivalent here. Weights land in VRAM at roughly
  ~5-6 GB (comparable to the AWQ build).
- **Resolution cap**: the processor is pinned to the range
  ``[256*28*28, 1024*28*28]`` pixels (200,704 - 802,816). This single
  knob is the biggest lever on peak VRAM - a 12 MP phone photo would
  otherwise balloon the visual-token activations into the 6 GB range.
  The upper bound was raised from ``512*28*28`` so that cursive
  character pairs like ``+`` vs ``-`` and ``3`` vs ``8`` get enough
  pixels to be disambiguated on legible handwriting (adds roughly
  ~2 GB of activation headroom, still well under the 14 GB budget).
- **Fragmentation mitigation**: ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
  is set *before* torch is imported so the allocator can grow/shrink
  segments instead of fragmenting the reserved pool across requests.
- **Single inference pass**: the prompt asks the model to read the
  handwriting AND verify the math in one response; a second pass is
  only performed when the first output fails JSON parsing.
- **Per-request cleanup**: after each ``model.generate`` call we
  explicitly ``del inputs`` and ``torch.cuda.empty_cache()`` so the
  next request sees a freshly-compacted allocator.

Concurrency
-----------

Only one generation may run at a time on a single GPU. We guard
inference with a non-blocking :class:`threading.Lock`; if a previous
(likely timed-out) request is still holding the GPU, we raise
:class:`GradingBusy` so the caller sees an immediate 503 with a
``Retry-After`` header rather than stacking up executor threads.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import threading
import time
from typing import Any

# CRITICAL: this env var must be set BEFORE the first ``import torch``
# anywhere in the process, otherwise the allocator ignores it. We set
# it at the very top of this module (the grading service is the only
# torch consumer in the app) so the mitigation actually takes effect.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- constants

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_DISPLAY_NAME = "Qwen2.5-VL-7B-Instruct (bnb-4bit)"

# Image-resolution bounds in pixels. Qwen2.5-VL tiles the image into
# 28x28 patches; clamping both ends of the range keeps the visual
# token sequence predictable and keeps activations in VRAM bounded.
#
# Upper bound was raised from 512*28*28 (401 k) to 1024*28*28 (802 k)
# after we observed the 7B model confusing cursive ``+``/``-`` and
# ``3``/``8`` on real student handwriting at the lower cap. The higher
# cap roughly doubles the visual-token budget (~128 -> ~256 merged
# tokens) which translates to +1-2 GB of activation memory during
# inference - still well inside the 14 GB VRAM budget on the RTX 5080.
# Override with env vars if you need to pin a different range.
_MIN_PIXELS = int(os.getenv("GRADING_MIN_PIXELS", str(256 * 28 * 28)))  # 200,704
_MAX_PIXELS = int(os.getenv("GRADING_MAX_PIXELS", str(1024 * 28 * 28)))  # 802,816

# Max tokens the model may emit in a single pass. Enough headroom for
# ~20 step JSON with prose feedback; override via env for edge cases.
_MAX_NEW_TOKENS_PASS = int(os.getenv("GRADING_MAX_NEW_TOKENS", "1024"))


# --------------------------------------------------------------------- exceptions


class GradingServiceUnavailable(RuntimeError):
    """Raised when the grading model is not loaded.

    Covers both the cold-start failure modes (no CUDA, dependencies
    missing, weights download failure) and the terminal case where
    loading the model triggered an unrecoverable CUDA error. The
    route layer translates this into HTTP 503.
    """


class GradingBusy(RuntimeError):
    """Raised when another grading call already holds the GPU lock.

    A previous request may have timed out client-side while its CUDA
    generation is still running - Python cannot cancel a native
    ``generate`` call mid-flight. Rather than queue another blocking
    thread we fast-fail with 503 + ``Retry-After``.
    """


# --------------------------------------------------------------------- prompt

_SYSTEM_PROMPT = (
    "You are a math teacher grading a handwritten student solution. "
    "You will READ the handwriting AND verify the math in ONE response."
)

# Double-braces ({{ / }}) are str.format escapes for literal { / }.
# Single-braced {question}, {answer_key}, {max_score}, {subject},
# {label_hint} are the real substitution points. ``{label_hint}`` is
# filled by ``_build_messages`` - either an empty string or a line
# telling the model the exact question-number label to strip (we know
# it when the request came through ``/submit-by-image``).
_USER_PROMPT_TEMPLATE = """Grade this handwritten math solution.

Question: {question}
Full correct solution (answer key):
{answer_key}
Maximum score: {max_score}
Subject: {subject}
{label_hint}
STEP 1 - READ the handwriting and TRANSCRIBE the math:
For EACH line of mathematical working the student wrote, emit ONE
entry in ``extracted_steps`` with its ``student_wrote`` field set
to the math content of that line.

DO NOT skip any lines. In particular, ALWAYS include the student's
LAST line (the final answer they arrived at) as its own entry in
``extracted_steps``. Missing the final line is a common failure;
count the visible "=" signs on the page and make sure you emitted
at least that many step entries plus one for the starting expression.

CRITICAL - Question labels are NOT math. Strip them before writing
``student_wrote``:
Question labels like "1.", "Q3", "0003.", "Question 5:", "(3)" at
the very start of the page are METADATA. They MUST NOT appear in
``student_wrote``. A label's presence is NEVER grounds for
``is_correct: false`` - strip it, then compare.

Example - this is the exact failure mode to avoid:
  Page starts with:  "0003. (3x^2+2xy-y^2)-(x^2-5xy-4y^2)"

  CORRECT output:
    {{"step_number": 1,
      "student_wrote": "(3x^2+2xy-y^2)-(x^2-5xy-4y^2)",
      "expected":      "(3x^2+2xy-y^2)-(x^2-5xy-4y^2)",
      "is_correct": true, "error": null}}

  WRONG output (do NOT produce this):
    {{"step_number": 1,
      "student_wrote": "0003. (3x^2+2xy-y^2)-(x^2-5xy-4y^2)",
      "is_correct": false,
      "error": "includes question label"}}

If the student copied the starting expression correctly, step 1
is correct regardless of any label on the page.

STEP 2 - VERIFY each step:
The answer key lists the expected steps in order as "Step 1",
"Step 2", ... Match student line N against the answer key's
Step N (same number). In the ``expected`` field for each student
step, put the answer key's Step N text. Do NOT put the final
answer there unless this IS the final-answer step.

CRITICAL VERIFICATION RULES:
- Rule 0: Strip question labels before comparing. A leading "Q3.",
  "0003.", "(3)", "Question 5:" at the top of the student's page is
  a question NUMBER, not a coefficient or part of the math. Never
  flag it as a coefficient mismatch against the expected expression.
- Rule 1: After stripping the label, step 1 must show the EXACT
  expression from the question (this is also the answer key's
  Step 1 "starting expression"). Check every coefficient and sign
  individually. (2x^2-xy+3y^2) != (x^2+xy+3y^2) -> step 1 is WRONG.
- Rule 2: Each subsequent student step must (a) follow mathematically
  from the previous student step AND (b) match the answer key's step
  with the same number, character by character on coefficients and
  signs.
- Rule 3: If step N is wrong, steps after it that use the wrong
  result are 'affected by step N error' - note dependency but
  do not penalise the method again if approach is still valid.
- Rule 4: Award marks proportionally per correct step.
- Rule 5 (FINAL ANSWER PRIORITY): If the student's LAST step
  contains the expected "Final answer" (ignoring a leading "=" and
  whitespace), the student reached the correct result. In that case
  award full marks regardless of uncertainty in any intermediate
  step - the correct final answer is strong evidence that the
  intermediate work is mathematically valid.
- Rule 6 (HANDWRITING AMBIGUITY): Cursive digits can be ambiguous
  (3 vs 8, 1 vs 7, 0 vs 6, 4 vs 9). When a character is unclear,
  use mathematical consistency to resolve it: if reading it one way
  makes the transformation follow logically from the previous step
  AND reading the other way does not, choose the reading that makes
  the math work. Never mark a step wrong because of an ambiguous
  character that has a correct reading.

OUTPUT RULES (strictly follow):
- Output RAW JSON only
- No markdown fences, no backticks, no ```json
- Start your response with {{ and end with }}
- Do not put JSON inside any field value

{{
  "extracted_steps": [
    {{
      "step_number": 1,
      "student_wrote": "exact transcription of this line",
      "expected": "what this line should be per answer key",
      "is_correct": true,
      "error": null
    }},
    {{
      "step_number": 2,
      "student_wrote": "exact transcription",
      "expected": "correct version",
      "is_correct": false,
      "error": "specific error: wrong coefficient of x^2, student wrote 2 but should be 3"
    }}
  ],
  "first_error_step": 2,
  "error_summary": "Student copied wrong starting expression in step 1",
  "method_correct": true,
  "student_score": 7,
  "max_score": {max_score},
  "feedback": "Step 1: WRONG - starting expression differs from question. Step 2: correct method but affected by step 1 error.",
  "is_correct": false,
  "confidence": "high"
}}"""

# Appended to the user prompt on a retry when the first response was
# not parseable. Intentionally short and blunt.
_STRICT_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Your previous response could not be parsed.\n"
    "Output ONLY a valid JSON object. Start with { end with }.\n"
    "No other text."
)


# --------------------------------------------------------------------- module globals
# torch/transformers imports are lazy so that `import app.services.grading_service`
# does not force a multi-second CUDA init (and does not fail on CI
# machines without a GPU).

_model: Any = None
_processor: Any = None
_device: str | None = None
_load_lock = threading.Lock()
_load_attempted = False
# Serialises GPU access across request threads dispatched by the
# FastAPI executor pool. Non-blocking acquire in ``_grade_sync`` so we
# can surface 503/GradingBusy instead of queueing up blocked threads.
_generate_lock = threading.Lock()


# --------------------------------------------------------------------- VRAM helpers


def _log_vram_usage(tag: str) -> None:
    """Log currently-allocated and reserved VRAM on the active CUDA device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev) / (1024**3)
        reserved = torch.cuda.memory_reserved(dev) / (1024**3)
        total = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
        logger.info(
            "VRAM[%s] device=%d allocated=%.2f GiB reserved=%.2f GiB total=%.2f GiB",
            tag,
            dev,
            alloc,
            reserved,
            total,
        )
    except Exception:  # pragma: no cover - diagnostics only
        logger.exception("VRAM usage probe failed")


def _vram_used_mb() -> int:
    """Return currently-allocated VRAM in MiB, or 0 if CUDA is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.memory_allocated() // (1024**2))
    except Exception:
        return 0


def _is_oom(exc: BaseException) -> bool:
    """Heuristic: does *exc* look like a CUDA out-of-memory error?"""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# --------------------------------------------------------------------- model load


def _load_model() -> None:
    """Populate the module-level ``_model`` / ``_processor`` singletons.

    Idempotent and thread-safe: concurrent first-touch callers share
    one load attempt via :data:`_load_lock`. On failure the singletons
    stay ``None`` and :func:`grade_answer` raises
    :class:`GradingServiceUnavailable`.

    The processor is pinned to ``[_MIN_PIXELS, _MAX_PIXELS]`` so every
    incoming image is automatically resized into that band before it
    reaches the visual encoder - this is the single largest knob for
    per-request VRAM usage.
    """
    global _model, _processor, _device, _load_attempted

    with _load_lock:
        if _load_attempted:
            return
        _load_attempted = True

        try:
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )
        except Exception:
            logger.exception(
                "grading: transformers/torch not installed; grading endpoint will 503"
            )
            return

        if not torch.cuda.is_available():
            logger.error(
                "grading: CUDA is not available on this host; "
                "Qwen2.5-VL bnb-4bit requires a GPU. Grading endpoint will 503."
            )
            return

        try:
            import bitsandbytes  # noqa: F401
        except Exception:
            logger.exception(
                "grading: bitsandbytes not installed; "
                "grading endpoint will 503. Install via `pip install bitsandbytes`."
            )
            return

        _device = "cuda"

        # NF4 4-bit with double-quant and fp16 compute dtype:
        # - ~5-6 GB of VRAM for the 7B weights (vs ~14 GB in fp16)
        # - compute in fp16 keeps activation math in the same dtype the
        #   vision encoder already uses, avoiding a bf16<->fp16 hop.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        try:
            logger.info(
                "grading: loading %s with bnb 4-bit NF4 on %s...",
                MODEL_ID,
                _device,
            )
            _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                quantization_config=quant_config,
                dtype=torch.float16,
                device_map=_device,
            )
            _processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                min_pixels=_MIN_PIXELS,
                max_pixels=_MAX_PIXELS,
            )
            _model.eval()
            logger.info(
                "grading: model loaded (min_pixels=%d max_pixels=%d max_new_tokens=%d)",
                _MIN_PIXELS,
                _MAX_PIXELS,
                _MAX_NEW_TOKENS_PASS,
            )
            _log_vram_usage("after-load")
        except Exception as exc:
            if _is_oom(exc):
                logger.error(
                    "grading: CUDA OOM loading %s even in AWQ/fp16 form. "
                    "Grading endpoint will 503. (%s)",
                    MODEL_ID,
                    exc,
                )
            else:
                logger.exception(
                    "grading: load failed for %s; grading endpoint will 503",
                    MODEL_ID,
                )
            _model = None
            _processor = None
            try:
                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass


def is_ready() -> bool:
    """Return ``True`` iff the model + processor are loaded and usable."""
    return _model is not None and _processor is not None


# --------------------------------------------------------------------- image helpers


def _decode_image(image_bytes: bytes) -> Image.Image:
    """Decode raw bytes into a PIL ``RGB`` image.

    We do **not** resize here - the processor applies its own
    ``[_MIN_PIXELS, _MAX_PIXELS]`` smart-resize downstream, which is
    what the visual encoder is trained against.

    Raises
    ------
    ValueError
        If ``image_bytes`` cannot be decoded. The route translates
        this to HTTP 400.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _estimate_pixels_after_resize(image: Image.Image) -> int:
    """Return the pixel count the processor will actually feed the encoder.

    Uses Qwen's own ``smart_resize`` helper when available so the
    number reported matches what the model sees. Falls back to the
    original image dimensions if the helper cannot be imported.
    """
    try:  # pragma: no cover - depends on qwen_vl_utils internals
        from qwen_vl_utils.vision_process import smart_resize  # type: ignore

        h, w = smart_resize(
            image.height,
            image.width,
            min_pixels=_MIN_PIXELS,
            max_pixels=_MAX_PIXELS,
        )
        return int(h * w)
    except Exception:
        return int(image.width * image.height)


# --------------------------------------------------------------------- parsing


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Turn the model's raw output into a validated dict.

    Defensive against three common failure modes:

    1. Markdown-wrapped JSON (``` ```json ... ``` ```) - stripped with
       two regex passes before parsing.
    2. Prose before *and* after the JSON object - we isolate the
       outermost ``{...}`` span via ``str.find`` / ``str.rfind``.
    3. The model smuggling its entire response into a single
       ``feedback`` field as an escaped string - we detect by looking
       for a leading ``{``, the ``extracted_steps`` substring, or any
       triple-backticks inside ``feedback`` and replace it with the
       model's own ``error_summary`` (or a generic pointer).

    Raises
    ------
    ValueError
        If no balanced ``{...}`` can be located in ``raw``.
    json.JSONDecodeError
        If the isolated span is not valid JSON.
    """
    # 1. strip ALL markdown artifacts (two passes: ```json and ```)
    text = re.sub(r"```json\s*", "", raw, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # 2. isolate the outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:300]!r}")
    text = text[start : end + 1]

    # 3. parse (json.JSONDecodeError propagates to caller)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(
            f"Top-level JSON value is {type(data).__name__}, expected object."
        )

    # 4. detect the "model nested its JSON inside feedback" failure mode
    feedback = data.get("feedback", "")
    if isinstance(feedback, str):
        stripped = feedback.strip()
        if (
            stripped.startswith("{")
            or "extracted_steps" in feedback
            or "```" in feedback
        ):
            logger.warning(
                "grading: model nested JSON inside 'feedback' field; replacing. "
                "head=%r",
                stripped[:160],
            )
            data["feedback"] = (
                data.get("error_summary") or "See step breakdown."
            )

    return data


# --------------------------------------------------------------------- validation


def _clamp_score(value: Any, max_score: int) -> int:
    """Coerce the model's ``student_score`` into ``[0, max_score]``."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        try:
            score = int(float(value))
        except (TypeError, ValueError):
            return 0
    if score < 0:
        return 0
    if score > max_score:
        return max_score
    return score


def _label_prefix_variants(label: str) -> list[str]:
    """Return common ways a student might write a question label at the
    top of their page.

    We try these in order of *longest first* when stripping so the most
    specific form wins (e.g. we prefer to strip ``"0003."`` over
    ``"0003"`` alone, to avoid leaving a stray ``"."``).
    """
    clean = str(label or "").strip()
    if not clean:
        return []

    variants: list[str] = []
    # Canonical and punctuation variants.
    for suffix in ("", ".", ")", ":", " -", " -"):
        variants.append(f"{clean}{suffix}")
        variants.append(f"Q{clean}{suffix}")
        variants.append(f"({clean}){suffix}".rstrip())
    variants.append(f"Question {clean}")
    variants.append(f"Question {clean}:")
    variants.append(f"Question {clean}.")

    # Also try without leading zeros ("0003" -> "3") since students often
    # write the compact form even when the canonical label is zero-padded.
    if clean.isdigit() and clean.startswith("0"):
        trimmed = clean.lstrip("0") or "0"
        for suffix in ("", ".", ")", ":"):
            variants.append(f"{trimmed}{suffix}")
            variants.append(f"Q{trimmed}{suffix}")
            variants.append(f"({trimmed}){suffix}".rstrip())

    # Dedup, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _strip_leading_label(text: str, variants: list[str]) -> str:
    """Strip the longest matching label variant from ``text``'s left edge.

    A variant is only accepted when it is clearly *standalone*, i.e. the
    character right after it is either:

    - end-of-string, or
    - whitespace (the student wrote "0003. (3x^2...)" with a space), or
    - a punctuation-to-math boundary *when the variant itself ends in
      punctuation* (the student wrote "0003.(3x^2...)" with no space).

    This prevents false-positive strips against genuine math like
    ``"3x^2 + 4"`` (variant ``"3"`` would match the leading coefficient)
    or ``"3.141"`` (variant ``"3."`` would match the integer part).

    Returns the original ``text`` unchanged if no variant matched
    under those constraints.
    """
    if not text:
        return text
    stripped = text.lstrip()

    # Try longest first so "0003." beats "0003".
    for v in sorted(variants, key=len, reverse=True):
        if not v:
            continue
        if not stripped.lower().startswith(v.lower()):
            continue

        after_idx = len(v)
        # Whole string was just the label.
        if after_idx >= len(stripped):
            return ""

        next_char = stripped[after_idx]

        # Safe case: label is followed by whitespace (the normal way
        # students write "0003. <math>" with a space between).
        if next_char.isspace():
            return stripped[after_idx:].lstrip()

        # Edge case: label ends in punctuation AND the next char is not
        # alphanumeric. Covers "0003.(3x^2..." with no space, while
        # still rejecting "3.141" (variant "3." followed by "1" which
        # IS alphanumeric) and "3x^2" (variant "3" doesn't end in
        # punctuation so this branch does not trigger).
        if v[-1] in ".):-" and not next_char.isalnum():
            return stripped[after_idx:].lstrip()

    return text


_MATH_WHITESPACE_RE = re.compile(r"\s+")


def _math_normalize(text: str) -> str:
    """Collapse whitespace and lower-case for tolerant math equality."""
    return _MATH_WHITESPACE_RE.sub("", str(text or "")).lower()


def _repair_question_label_in_step1(
    data: dict[str, Any], question_label: str | None
) -> None:
    """Detect and undo the "label polluted student_wrote" failure mode.

    Context: even with prompt instructions to strip question labels, the
    VLM sometimes writes the label into ``student_wrote`` *and* marks
    ``is_correct: false`` citing the label's presence as the error.
    This helper runs on the parsed dict after JSON extraction and
    *before* ``_validate_score``.

    For the first extracted step, when a known ``question_label`` was
    supplied by the route layer (``/submit-by-image`` knows it):

    1. Strip any leading label variant from ``student_wrote``.
    2. If the stripped transcription now matches ``expected`` under
       whitespace-insensitive comparison, flip ``is_correct`` to True,
       clear the ``error`` field, and fix up top-level
       ``first_error_step`` / ``error_summary`` so the response is
       internally consistent.
    3. If ``student_wrote`` did not start with a label we leave
       everything alone - we never silently upgrade a genuine math error.

    All mutations are logged at WARNING level so we can audit how often
    the model needs bailing out.
    """
    if not question_label:
        return

    steps = data.get("extracted_steps")
    if not isinstance(steps, list) or not steps:
        return

    first = steps[0]
    if not isinstance(first, dict):
        return

    variants = _label_prefix_variants(str(question_label))
    if not variants:
        return

    original = str(first.get("student_wrote") or "")
    stripped = _strip_leading_label(original, variants)
    if stripped == original:
        return  # no leading label present

    logger.warning(
        "grading: repair stripped question label %r from step 1 student_wrote: "
        "%r -> %r",
        question_label,
        original[:80],
        stripped[:80],
    )
    first["student_wrote"] = stripped

    expected = str(first.get("expected") or "")
    if expected and _math_normalize(stripped) == _math_normalize(expected):
        if not first.get("is_correct"):
            logger.warning(
                "grading: repair flipped step 1 is_correct False -> True "
                "(student_wrote matches expected after label strip)"
            )
        first["is_correct"] = True
        first["error"] = None

        # Recompute first_error_step: find the first remaining
        # is_correct=False step, or null if all correct.
        new_first_err: int | None = None
        for s in steps:
            if isinstance(s, dict) and not s.get("is_correct", True):
                try:
                    new_first_err = int(s.get("step_number"))
                except (TypeError, ValueError):
                    new_first_err = None
                break
        data["first_error_step"] = new_first_err

        # Nuke a stale error_summary that was pointing at step 1's label.
        err_summary = str(data.get("error_summary") or "").lower()
        if (
            "label" in err_summary
            or "question number" in err_summary
            or "0003" in err_summary  # defensive against the exact failure mode
            or question_label.lower() in err_summary
        ):
            data["error_summary"] = (
                None if new_first_err is None else data.get("error_summary")
            )


def _normalize_final_answer(text: Any) -> str:
    """Normalise an answer string for tolerant final-answer equality.

    Strips a leading ``=`` (students often write ``"= x+2"``), drops
    common Korean / English answer-prefix tokens (``"답"``, ``"ans"``,
    ``"Answer:"``), whitespace, and lower-cases. Returns the
    comparison key, not a user-facing string.
    """
    s = str(text or "").strip()
    if not s:
        return ""
    # Strip leading "= " that students prepend on a new line.
    while s.startswith("="):
        s = s[1:].lstrip()
    # Drop common answer-word prefixes (case-insensitive, Korean inclusive).
    lower = s.lower()
    for prefix in ("answer:", "ans:", "answer ", "ans ", "=>"):
        if lower.startswith(prefix):
            s = s[len(prefix):].lstrip()
            break
    # Korean "답" prefix (with or without colon/whitespace).
    if s.startswith("\uB2F5"):  # 답
        s = s[1:].lstrip(": ").strip()
    return _math_normalize(s)


def _repair_final_answer_match(
    data: dict[str, Any],
    expected_final_answer: str | None,
    max_score: int,
) -> None:
    """Safety net: if any extracted step matches the expected final answer,
    upgrade the **entire** response to full-marks / all-correct state.

    Rationale
    ---------

    When the student writes the correct final answer on their paper,
    they have reached the correct result. Intermediate-step
    transcription noise (e.g. the VLM reads cursive "3" as "8") is the
    grader's problem, not the student's. A correct final answer is
    strong evidence the student's math was valid; the per-step flags
    the VLM produced against its own (noisy) transcription cannot be
    trusted in that case.

    We therefore normalise the **whole** response so no field
    contradicts another:

    - Top-level: ``student_score -> max_score``; ``is_correct -> True``;
      ``method_correct -> True``.
    - Per-step: every step with ``is_correct=false`` is flipped to
      ``true`` and its ``error`` cleared. The original VLM verdicts
      are logged at WARNING for audit.
    - Derived text: ``first_error_step -> None``; ``error_summary -> None``;
      ``feedback`` replaced with a single "full marks: final answer
      matches" sentence (the original is logged).
    - ``score_breakdown`` is left for ``_validate_score`` to recompute
      from the now-all-correct step list (it will read ``N/N steps
      correct``).

    We check **all** extracted steps (not just the last) because the
    VLM sometimes drops the trailing line or collapses it into the
    previous step; as long as the expected final answer appears
    somewhere in the student's transcribed work, credit it.

    No-op when:

    - ``expected_final_answer`` is falsy (manual ``/submit`` path).
    - No extracted step matches the expected final answer. In that
      case the per-step flags the VLM produced are preserved so the
      caller sees genuine errors.
    """
    if not expected_final_answer:
        return

    expected_norm = _normalize_final_answer(expected_final_answer)
    if not expected_norm:
        return

    steps = data.get("extracted_steps")
    if not isinstance(steps, list) or not steps:
        return

    match_idx: int | None = None
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if _normalize_final_answer(step.get("student_wrote")) == expected_norm:
            match_idx = i
            break

    if match_idx is None:
        return  # student never wrote the expected final answer; trust VLM flags.

    current_score = _clamp_score(data.get("student_score", 0), max_score)
    was_score = current_score
    was_is_correct = bool(data.get("is_correct"))
    was_method_correct = bool(data.get("method_correct"))

    # --- Per-step: flip every flagged step to correct --------------------
    # Rationale: since the final answer is right, any earlier step marked
    # wrong by the VLM is almost certainly a transcription error on the
    # VLM's side (cursive 3 vs 8 is the canonical case). Clearing those
    # flags keeps the response internally consistent with the full-marks
    # top-level verdict. If the math actually was wrong mid-way, the
    # student wouldn't have reached the correct final answer.
    flipped_audit: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if not step.get("is_correct", False):
            flipped_audit.append(
                {
                    "step_number": step.get("step_number"),
                    "student_wrote": str(step.get("student_wrote") or "")[:200],
                    "original_error": step.get("error"),
                }
            )
            step["is_correct"] = True
            step["error"] = None

    # --- Top-level scalars -----------------------------------------------
    data["student_score"] = max_score
    data["is_correct"] = True
    data["method_correct"] = True
    # Flag the response so the UI / frontend can render a "auto-graded
    # via final answer - please review intermediate work" badge. This
    # is the explicit signal that we upgraded the score based on the
    # final-line match rather than the VLM's per-step judgement.
    data["safety_net_engaged"] = True

    # --- Derived text fields --------------------------------------------
    # Everything is now correct -> no first_error_step, no error_summary.
    data["first_error_step"] = None

    original_summary = data.get("error_summary")
    data["error_summary"] = None

    # Preserve the original feedback in logs, replace the user-visible one
    # with a clear single-sentence statement that will not contradict the
    # top-level verdict.
    original_feedback = str(data.get("feedback") or "").strip()
    data["feedback"] = (
        f"Full marks: the student reached the correct final answer "
        f"({expected_final_answer}). Any intermediate-step uncertainty "
        f"flagged during transcription was resolved in the student's "
        f"favour because the final result is correct."
    )

    # --- Audit logging ---------------------------------------------------
    if was_score != max_score or not was_is_correct or not was_method_correct:
        logger.warning(
            "grading: final-answer safety net engaged - student reached the "
            "correct final answer %r (matched in step %s); "
            "score %d -> %d, is_correct %s -> True, method_correct %s -> True",
            expected_final_answer,
            steps[match_idx].get("step_number"),
            was_score,
            max_score,
            was_is_correct,
            was_method_correct,
        )
    if flipped_audit:
        logger.warning(
            "grading: final-answer safety net also cleared %d intermediate "
            "step flag(s) for consistency (likely VLM OCR noise). "
            "original_flagged_steps=%r",
            len(flipped_audit),
            flipped_audit,
        )
    if original_summary or (
        original_feedback and original_feedback != data["feedback"]
    ):
        logger.info(
            "grading: final-answer safety net replaced error_summary=%r "
            "feedback_head=%r",
            original_summary,
            original_feedback[:200] if original_feedback else None,
        )


def _validate_score(data: dict[str, Any], max_score: int) -> dict[str, Any]:
    """Apply score-related sanity checks to ``data`` in place.

    Three guarantees for the caller:

    - ``student_score`` is clamped to ``[0, max_score]``.
    - If the model returned ``student_score=0`` but ``extracted_steps``
      shows any correct steps, we override the score with a
      step-proportional estimate and log a WARNING so the drift is
      auditable.
    - ``first_error_step`` is back-filled from the first incorrect step
      when the model forgot to emit it.
    - ``score_breakdown`` is populated with ``"X/Y steps correct"`` so
      the response model always has a value.

    Returns the mutated ``data`` dict for chained-call convenience.
    """
    steps = data.get("extracted_steps")
    if not isinstance(steps, list):
        steps = []

    total = len(steps)
    correct = sum(
        1 for s in steps if isinstance(s, dict) and s.get("is_correct", False)
    )

    step_score = round((correct / total) * max_score) if total > 0 else 0

    current_score = _clamp_score(data.get("student_score", 0), max_score)

    # Override: model said 0 but its own step list disagrees.
    if current_score == 0 and correct > 0:
        logger.warning(
            "grading: score=0 override correct=%d/%d new_score=%d",
            correct,
            total,
            step_score,
        )
        current_score = step_score

    data["student_score"] = current_score

    # Back-fill first_error_step if the model returned null despite
    # having at least one is_correct=false step.
    if data.get("first_error_step") is None:
        for s in steps:
            if not isinstance(s, dict):
                continue
            if not s.get("is_correct", True):
                try:
                    data["first_error_step"] = int(s.get("step_number"))
                except (TypeError, ValueError):
                    pass
                break

    data["score_breakdown"] = f"{correct}/{total} steps correct"
    return data


# --------------------------------------------------------------------- step sanitisation


def _sanitize_step(raw_step: Any, fallback_index: int) -> dict[str, Any]:
    """Normalise one step dict so Pydantic validation cannot fail on it."""
    if not isinstance(raw_step, dict):
        raw_step = {}

    try:
        step_number = int(raw_step.get("step_number", fallback_index))
    except (TypeError, ValueError):
        step_number = fallback_index

    error_val = raw_step.get("error")
    if error_val is None or (isinstance(error_val, str) and not error_val.strip()):
        error_clean: str | None = None
    else:
        error_clean = str(error_val)[:2000]

    return {
        "step_number": step_number,
        "student_wrote": str(raw_step.get("student_wrote") or "")[:2000],
        "expected": str(raw_step.get("expected") or "")[:2000],
        "is_correct": bool(raw_step.get("is_correct", False)),
        "error": error_clean,
    }


def _sanitize_steps(raw_steps: Any) -> list[dict[str, Any]]:
    """Coerce the model's ``extracted_steps`` field into a clean list."""
    if not isinstance(raw_steps, list):
        return []
    return [_sanitize_step(s, idx + 1) for idx, s in enumerate(raw_steps)]


def _coerce_optional_int(value: Any) -> int | None:
    """Return ``value`` as ``int`` when plausibly numeric, else ``None``."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _coerce_optional_str(value: Any) -> str | None:
    """Return ``value`` as a trimmed string, or ``None`` when empty/missing."""
    if value is None:
        return None
    text = str(value).strip()
    return text[:2000] if text else None


def _normalize_confidence(value: Any) -> str:
    """Map the model's confidence string onto the ``{high,medium,low}`` enum."""
    if not isinstance(value, str):
        return "low"
    v = value.strip().lower()
    if v in {"high", "medium", "low"}:
        return v
    if v in {"hi", "strong"}:
        return "high"
    if v in {"mid", "ok"}:
        return "medium"
    return "low"


# --------------------------------------------------------------------- generation


def _build_messages(
    pil_image: Image.Image,
    question: str,
    answer_key: str,
    max_score: int,
    subject: str,
    *,
    strict_retry: bool,
    question_label: str | None = None,
    other_questions_on_page: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build the Qwen2.5-VL chat-template messages for one grading call.

    The conversation is split into a system message (role definition)
    and a multi-modal user message carrying the PIL image plus the
    step-by-step grading instructions.

    ``question_label`` is the canonical label of the item being graded
    (e.g. ``"0003"``) when known - typically populated by
    ``/submit-by-image`` from the resolved ``AnswerKeyItem.question_no``.
    When supplied, the prompt gains a concrete hint telling the model
    exactly which string to treat as a question label and strip from
    the transcription. When omitted, the generic "strip any leading
    label" guidance still applies.

    ``other_questions_on_page`` is the list of OTHER question labels
    the student also wrote on the same page (populated by
    ``/submit-multi-by-image``). When non-empty, the prompt gains a
    second hint telling the VLM to restrict its transcription to
    ``question_label``'s work only and IGNORE lines labelled with any
    of the other numbers. Prevents cross-question contamination of
    ``extracted_steps`` when one image carries multiple answers.
    """
    # Build the context hint(s). Both notes share the same
    # ``{label_hint}`` substitution point in the template so the
    # prompt ordering is stable whether one or both are present.
    hint_parts: list[str] = []

    if question_label and str(question_label).strip():
        clean = str(question_label).strip()
        hint_parts.append(
            f"Note: this paper is for question number \"{clean}\". "
            f"The student may write \"{clean}\", \"{clean}.\", \"Q{clean}\", "
            f"or similar as a label at the top of the page - that is a "
            f"question identifier, NOT part of the math expression. Strip "
            f"it before comparing to the expected step 1."
        )

    if other_questions_on_page:
        others_clean = [
            str(q).strip()
            for q in other_questions_on_page
            if q is not None and str(q).strip()
        ]
        # De-dup and remove any that equal the target label.
        target = (str(question_label).strip() if question_label else "").lower()
        seen: set[str] = set()
        others_unique: list[str] = []
        for o in others_clean:
            k = o.lower()
            if k == target or k in seen:
                continue
            seen.add(k)
            others_unique.append(o)

        if others_unique:
            others_display = ", ".join(f'"{o}"' for o in others_unique)
            target_display = (
                str(question_label).strip() if question_label else "the target question"
            )
            hint_parts.append(
                f"Note: this page contains work for MULTIPLE questions. "
                f"Only grade the work labelled with question number "
                f"\"{target_display}\". IGNORE any lines labelled with "
                f"{others_display} - those belong to OTHER questions. "
                f"Those other-question lines MUST NOT appear in "
                f"extracted_steps for this grading call, and they MUST "
                f"NOT influence is_correct for \"{target_display}\"'s steps."
            )

    label_hint = "\n".join(hint_parts)
    if label_hint:
        label_hint = label_hint + "\n"

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        question=question,
        answer_key=answer_key,
        max_score=max_score,
        subject=subject,
        label_hint=label_hint,
    )
    if strict_retry:
        user_prompt = user_prompt + _STRICT_RETRY_SUFFIX

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def _run_generation(
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int = _MAX_NEW_TOKENS_PASS,
) -> str:
    """Run one ``model.generate`` pass and return the decoded string.

    Must be called with :data:`_generate_lock` held so two concurrent
    grading requests do not trample each other on the shared GPU.
    Explicitly drops the ``inputs`` tensor and empties the allocator
    cache after decoding so the next request sees a compacted pool.

    ``max_new_tokens`` defaults to the full grading budget; pass a
    tighter value (e.g. 128) for short-form passes like question
    identification.

    Raises
    ------
    RuntimeError
        CUDA OOM or other torch runtime failures propagate unchanged.
    """
    import torch
    from qwen_vl_utils import process_vision_info

    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_device)

    try:
        with torch.inference_mode():
            generated_ids = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Strip the prompt tokens so we decode only the continuation.
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = _processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output[0] if output else ""
    finally:
        # VRAM hygiene: release activations + input tensors so the
        # allocator can compact before the next request.
        try:
            del inputs
        except Exception:
            pass
        try:
            del generated_ids  # noqa: F821 - may be unbound if generate raised
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best-effort
            pass


# --------------------------------------------------------------------- orchestration


def _parse_failed_fallback(
    raw: str, max_score: int, original_bytes_len: int
) -> dict[str, Any]:
    """Build a safe zero-score envelope when both parse attempts failed.

    The route layer still gets a valid :class:`GradingResponse`-shaped
    dict so the client sees structured 200 JSON; the ``graded_by``
    field is suffixed with ``(parse failed)`` so ops/audit can spot
    these rows quickly.
    """
    truncated = (raw or "").strip()[:500] or "Could not parse grading response"
    return {
        "student_score": 0,
        "max_score": max_score,
        "extracted_steps": [],
        "first_error_step": None,
        "error_summary": "Model output could not be parsed",
        "method_correct": False,
        "feedback": truncated,
        "is_correct": False,
        "confidence": "low",
        "score_breakdown": "0/0 steps correct",
        "graded_by": f"{MODEL_ID} (parse failed)",
        "safety_net_engaged": False,
    }


def _grade_sync(
    image_bytes: bytes,
    question: str,
    answer_key: str,
    max_score: int,
    subject: str,
    question_label: str | None = None,
    expected_final_answer: str | None = None,
    other_questions_on_page: list[str] | None = None,
) -> dict[str, Any]:
    """Blocking implementation of :func:`grade_answer`.

    Flow:

    1. Decode + validate the image bytes (raises ``ValueError`` on
       corrupt payloads).
    2. Acquire the GPU lock non-blockingly; fail fast with
       :class:`GradingBusy` if another request is mid-generation.
    3. Run a single ``model.generate`` pass; on JSON parse failure
       retry once with the strict suffix. If both attempts fail,
       return the safe zero-score envelope.
    4. Run ``_validate_score`` to enforce clamping, the score=0
       override, and ``first_error_step`` back-fill.
    5. Emit one structured log line with per-request metrics
       (VRAM, inference time, parse state, step tally, final score).
    """
    if not is_ready():
        raise GradingServiceUnavailable(
            "Grading model is not loaded (see startup logs for details)."
        )

    pil_image = _decode_image(image_bytes)
    pixels_after_resize = _estimate_pixels_after_resize(pil_image)

    if not _generate_lock.acquire(blocking=False):
        raise GradingBusy(
            "A previous grading request is still running on the GPU; "
            "retry shortly."
        )

    t_start = time.perf_counter()
    parse_success = False
    parse_retried = False
    raw = ""

    try:
        # --- Pass 1 -----------------------------------------------------
        messages = _build_messages(
            pil_image, question, answer_key, max_score, subject,
            strict_retry=False,
            question_label=question_label,
            other_questions_on_page=other_questions_on_page,
        )
        raw = _run_generation(messages)

        try:
            data = _parse_json_response(raw)
            parse_success = True
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "grading: first-pass JSON parse failed (%s); retrying with strict suffix. "
                "raw head=%r",
                exc,
                raw[:200],
            )
            parse_retried = True

            # --- Pass 2 (strict retry) ---------------------------------
            messages = _build_messages(
                pil_image, question, answer_key, max_score, subject,
                strict_retry=True,
                question_label=question_label,
                other_questions_on_page=other_questions_on_page,
            )
            raw = _run_generation(messages)
            try:
                data = _parse_json_response(raw)
                parse_success = True
            except (ValueError, json.JSONDecodeError) as exc2:
                logger.error(
                    "grading: retry JSON parse also failed (%s); returning parse-failed "
                    "envelope. raw head=%r",
                    exc2,
                    raw[:200],
                )
                data = None  # sentinel for the fallback below
    finally:
        _generate_lock.release()

    inference_ms = (time.perf_counter() - t_start) * 1000.0

    if data is None:
        result = _parse_failed_fallback(raw, max_score, len(image_bytes))
        logger.info(
            "grading: image_size_bytes=%d image_pixels_after_resize=%d "
            "model_output_len=%d parse_success=%s parse_retried=%s "
            "correct_steps=0 total_steps=0 final_score=0 vram_used_mb=%d "
            "inference_ms=%.1f",
            len(image_bytes),
            pixels_after_resize,
            len(raw),
            parse_success,
            parse_retried,
            _vram_used_mb(),
            inference_ms,
        )
        return result

    # --- Happy path: repair known failure modes, then validate --------
    # Order matters:
    #   1. Strip question label from step 1 (may flip step 1 correct).
    #   2. Final-answer safety net (may flip overall correct + full marks
    #      when the student reached the right answer despite intermediate
    #      transcription noise like cursive 3 vs 8).
    #   3. _validate_score (clamps, score=0 override, first_error_step
    #      back-fill, score_breakdown).
    # Both repair helpers are no-ops when their respective hints weren't
    # supplied by the route (i.e. the manual /submit path runs with
    # both as None and lands directly at _validate_score).
    _repair_question_label_in_step1(data, question_label)
    _repair_final_answer_match(data, expected_final_answer, max_score)

    _validate_score(data, max_score)
    extracted_steps = _sanitize_steps(data.get("extracted_steps"))

    total_steps = len(extracted_steps)
    correct_steps = sum(1 for s in extracted_steps if s.get("is_correct"))
    final_score = _clamp_score(data.get("student_score", 0), max_score)

    result = {
        "student_score": final_score,
        "max_score": max_score,
        "extracted_steps": extracted_steps,
        "first_error_step": _coerce_optional_int(data.get("first_error_step")),
        "error_summary": _coerce_optional_str(data.get("error_summary")),
        "method_correct": bool(data.get("method_correct", False)),
        "feedback": str(data.get("feedback", "") or "")[:4000],
        "is_correct": bool(data.get("is_correct", final_score == max_score)),
        "confidence": _normalize_confidence(data.get("confidence")),
        "score_breakdown": str(
            data.get("score_breakdown") or f"{correct_steps}/{total_steps} steps correct"
        ),
        "graded_by": MODEL_DISPLAY_NAME,
        # True iff the final-answer safety net bumped this response to
        # full marks by matching the last line against the expected
        # final answer. Frontends should show a "teacher-review
        # recommended" badge when this is True - the VLM's per-step
        # flags are not fully trusted in that case.
        "safety_net_engaged": bool(data.get("safety_net_engaged", False)),
    }

    logger.info(
        "grading: image_size_bytes=%d image_pixels_after_resize=%d "
        "model_output_len=%d parse_success=%s parse_retried=%s "
        "correct_steps=%d total_steps=%d final_score=%d vram_used_mb=%d "
        "inference_ms=%.1f",
        len(image_bytes),
        pixels_after_resize,
        len(raw),
        parse_success,
        parse_retried,
        correct_steps,
        total_steps,
        final_score,
        _vram_used_mb(),
        inference_ms,
    )
    return result


# --------------------------------------------------------------------- public async API


async def grade_answer(
    *,
    image_bytes: bytes,
    question: str,
    answer_key: str,
    max_score: int = 10,
    subject: str = "math",
    question_label: str | None = None,
    expected_final_answer: str | None = None,
    other_questions_on_page: list[str] | None = None,
) -> dict[str, Any]:
    """Grade a single handwritten answer image.

    Offloads the blocking HuggingFace ``generate`` call to the default
    asyncio executor so the FastAPI event loop stays responsive.
    Nothing is written to disk; everything is held in memory for the
    duration of the call.

    Parameters
    ----------
    image_bytes:
        Raw bytes of a JPG / PNG / WebP image containing the student's
        handwritten answer. The caller is responsible for size / type
        validation (the route does that before reaching this layer).
    question:
        The original question text shown to the student.
    answer_key:
        The **full worked solution** for the question, step-by-step.
        The grader compares each student step against this reference
        so a single final-answer string is not sufficient.
    max_score:
        Maximum score for this question. ``student_score`` is clamped
        to ``[0, max_score]``.
    subject:
        Free-form subject hint ("math", "physics", ...) used as extra
        grading context.
    question_label:
        Canonical question-number label (e.g. ``"0003"``) when the
        caller already knows it - typically populated by
        ``/submit-by-image`` from the resolved ``AnswerKeyItem.question_no``.
        When supplied, the grading prompt gains an explicit hint telling
        the model exactly which string to strip from the top of the
        student's page before comparing step 1 to the expected
        expression. Prevents the classic failure mode where the VLM
        transcribes the label (e.g. ``"0003."``) as part of the math and
        then flags it as a coefficient mismatch.
    expected_final_answer:
        The canonical final answer for this question when known -
        typically ``AnswerKeyItem.normalized_answer`` (or ``.final_answer``
        as fallback) from the resolver. When supplied, a server-side
        safety net in ``_grade_sync`` inspects the parsed response: if
        the student's transcribed work contains the expected final
        answer (whitespace- and ``=``-insensitive), grading is upgraded
        to full marks regardless of intermediate-step transcription
        noise. Protects against VLM OCR errors on cursive digits
        (3 vs 8, 1 vs 7, ...) in intermediate lines.
    other_questions_on_page:
        List of OTHER question labels present on the same page -
        populated by ``/submit-multi-by-image`` so the grading prompt
        can tell the VLM to grade only this question's work and ignore
        lines labelled with the other numbers. Prevents cross-question
        contamination of ``extracted_steps`` when one image carries
        multiple student answers. Omit (or pass ``None`` / empty list)
        for single-question grading.

    Returns
    -------
    dict
        A dict matching the HTTP response envelope defined in
        :mod:`app.api.routes.grading` (``GradingResponse``).

    Raises
    ------
    GradingServiceUnavailable
        The model failed to load at startup. Route maps to HTTP 503.
    GradingBusy
        Another grading request is still holding the GPU. Route maps
        to HTTP 503 with ``Retry-After: 10``.
    ValueError
        The image bytes could not be decoded. Route maps to HTTP 400.
    RuntimeError
        CUDA OOM or other torch failures mid-generation. Route maps
        OOM-ish messages to HTTP 503.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        _grade_sync,
        image_bytes,
        question,
        answer_key,
        max_score,
        subject,
        question_label,
        expected_final_answer,
        other_questions_on_page,
    )


# --------------------------------------------------------------------- question identification
# A short, low-token Qwen pass that reads the *top* of a handwritten
# page and extracts either the question number the student wrote or
# the problem statement they copied. The caller then uses those hints
# to pick the correct ``AnswerKeyItem`` via either exact SQL lookup
# (on ``(file_id, question_no)``) or semantic similarity in Chroma.
#
# This is deliberately a separate, tiny generation pass (~128 tokens)
# because the full grading pass is slow and bundling "identify + grade"
# into one prompt would (a) bloat the output-token budget and (b)
# force the VLM to simultaneously read AND reason, which empirically
# hurts both tasks.

_IDENTIFY_SYSTEM_PROMPT = (
    "You are reading a student's handwritten math work to identify "
    "which question they are answering. Do NOT grade. Do NOT solve."
)

_IDENTIFY_USER_PROMPT = """Look at the top of this handwritten page. Extract:

1. question_number: if the student wrote a question number at the top
   of the page (e.g. "1.", "Q3", "Question 2", "#5", "2.a") return
   just the number/identifier as a short string. If no question number
   is visible, return null.

2. problem_text: if the student copied the problem statement from the
   exam paper at the top of their page (typically the first 1-3 lines,
   before any working), return it as a single line. If the page only
   contains working/calculation with no restated problem, return null.

3. read_confidence: "high" if the handwriting at the top is clearly
   legible; "medium" if partially legible; "low" if mostly illegible.

OUTPUT RULES:
- Output RAW JSON only
- No prose, no markdown fences, no ```json
- Start your response with { and end with }
- Example: {"question_number": "3", "problem_text": null, "read_confidence": "high"}"""

# Short output budget. The JSON envelope is 60-120 tokens in practice;
# 128 leaves a little headroom without letting the model run on.
_IDENTIFY_MAX_NEW_TOKENS = int(os.getenv("GRADING_IDENTIFY_MAX_NEW_TOKENS", "128"))


def _identify_sync(image_bytes: bytes) -> dict[str, Any]:
    """Blocking implementation of :func:`identify_question`.

    Shares the singleton model + :data:`_generate_lock` with the main
    grading pass. Returns a dict with three keys - ``question_number``,
    ``problem_text``, ``read_confidence`` - any of which may be
    ``None`` / ``"low"`` when the handwriting at the top of the page
    is missing or illegible. On JSON parse failure we fall back to a
    zero-signal response rather than raising, so the caller can still
    ask the student to disambiguate via the 422 path.
    """
    if not is_ready():
        raise GradingServiceUnavailable(
            "Grading model is not loaded (see startup logs for details)."
        )

    pil_image = _decode_image(image_bytes)

    if not _generate_lock.acquire(blocking=False):
        raise GradingBusy(
            "A previous grading request is still running on the GPU; "
            "retry shortly."
        )

    t_start = time.perf_counter()
    try:
        messages = [
            {"role": "system", "content": _IDENTIFY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": _IDENTIFY_USER_PROMPT},
                ],
            },
        ]
        raw = _run_generation(messages, max_new_tokens=_IDENTIFY_MAX_NEW_TOKENS)
    finally:
        _generate_lock.release()

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    try:
        data = _parse_json_response(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "identify: JSON parse failed (%s); returning zero-signal result. "
            "raw head=%r",
            exc,
            raw[:200],
        )
        return {
            "question_number": None,
            "problem_text": None,
            "read_confidence": "low",
        }

    qn = _coerce_optional_str(data.get("question_number"))
    pt = _coerce_optional_str(data.get("problem_text"))
    rc = _normalize_confidence(data.get("read_confidence"))

    logger.info(
        "identify: question_number=%r problem_text_len=%d read_confidence=%s "
        "elapsed_ms=%.1f vram_used_mb=%d",
        qn,
        len(pt or ""),
        rc,
        elapsed_ms,
        _vram_used_mb(),
    )

    return {
        "question_number": qn,
        "problem_text": pt,
        "read_confidence": rc,
    }


async def identify_question(image_bytes: bytes) -> dict[str, Any]:
    """Extract question identifiers from a student's handwritten page.

    Runs a short Qwen2.5-VL pass (~128 tokens) and returns whichever of
    ``question_number`` / ``problem_text`` the model could read from the
    top of the page. Callers feed the result into
    :func:`app.services.question_resolver.resolve_question` to pick the
    matching :class:`AnswerKeyItem`.

    Returns
    -------
    dict
        ``{"question_number": str | None,
           "problem_text":    str | None,
           "read_confidence": "high" | "medium" | "low"}``

    Raises
    ------
    GradingServiceUnavailable
        Model failed to load at startup.
    GradingBusy
        Another grading request is holding the GPU.
    ValueError
        ``image_bytes`` could not be decoded.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _identify_sync, image_bytes)


# --------------------------------------------------------------------- multi-question identification
# Sibling of :func:`identify_question` for the multi-question endpoint
# (``/submit-multi-by-image``). Instead of reading just the top of the
# page and returning ONE label, it scans the whole page and returns a
# list of distinct question labels (up to MAX) the student wrote.
#
# We run this as a single Qwen pass (still cheap - ~256 tokens) rather
# than N independent "is question X here?" calls. The returned list
# then drives the per-question grade loop in the route layer.

_IDENTIFY_ALL_SYSTEM_PROMPT = (
    "You are scanning a handwritten page to list which questions the "
    "student answered. Do NOT grade. Do NOT solve."
)

_IDENTIFY_ALL_USER_PROMPT = """Scan this handwritten page for QUESTION LABELS.

A question label is a number or short identifier at the start of a
distinct section of student work, for example: "1.", "Q3", "0002.",
"Question 5:", "(3)".

Return ONE entry per distinct question the student answered on this
page. Return at most 5 entries. Order entries top-to-bottom as they
appear on the page. Do NOT return the same question twice. Do NOT
include printed headers or answer-key labels - only include labels
the student wrote in their own handwriting.

For each entry, include:
- question_number: the label as written, digits only if it is numeric
  (e.g. "0002", "3", "Q5"). Do NOT include the trailing "." or ")".
- problem_text_preview: up to 60 characters of the first line the
  student wrote for this question - the starting expression they
  copied from the exam. If the student did NOT copy the problem
  statement (started straight with working), set this to null.
- confidence: "high" | "medium" | "low" for how clearly you can see
  this label on the page.

OUTPUT RULES:
- Output RAW JSON only
- No prose, no markdown fences, no ```json
- Start your response with { and end with }
- Shape: {"questions": [{"question_number": "...", "problem_text_preview": "..." or null, "confidence": "..."}, ...]}

Example valid output:
{"questions": [
  {"question_number": "0002", "problem_text_preview": "Simplify (x+2)^2", "confidence": "high"},
  {"question_number": "0003", "problem_text_preview": "(3x^2+2xy-y^2)-(x^2-5xy-4y^2)", "confidence": "high"}
]}"""

# Tighter than the grading pass (1024) but looser than single identify
# (128) because multi-question can emit up to 5 entries.
_IDENTIFY_ALL_MAX_NEW_TOKENS = int(
    os.getenv("GRADING_IDENTIFY_ALL_MAX_NEW_TOKENS", "256")
)

# Hard cap on entries we process downstream even if the model emits
# more. Prevents pathological pages from triggering 10+ grading passes.
_IDENTIFY_ALL_MAX_RESULTS = int(
    os.getenv("GRADING_MAX_QUESTIONS_PER_IMAGE", "5")
)


def _sanitize_identification_entry(raw: Any, fallback_idx: int) -> dict[str, Any] | None:
    """Coerce one model-output entry into ``{question_number, problem_text_preview, confidence}``.

    Returns ``None`` for entries that are clearly bogus (no usable
    question number). Never raises.
    """
    if not isinstance(raw, dict):
        return None

    qn = _coerce_optional_str(raw.get("question_number"))
    if qn is None:
        return None
    # Models sometimes emit the trailing "." / ")" - strip once more.
    qn = qn.strip().rstrip(".").rstrip(")").strip()
    if not qn:
        return None

    preview = _coerce_optional_str(raw.get("problem_text_preview"))
    if preview is not None:
        preview = preview[:200]  # hard cap even if model ignored the 60-char hint

    conf = _normalize_confidence(raw.get("confidence"))

    return {
        "question_number": qn,
        "problem_text_preview": preview,
        "confidence": conf,
    }


def _identify_all_sync(image_bytes: bytes) -> list[dict[str, Any]]:
    """Blocking implementation of :func:`identify_all_questions`.

    Returns a list (possibly empty) of identification entries. Dedups
    by ``question_number`` keeping first occurrence (preserves
    top-to-bottom order). Caps at ``_IDENTIFY_ALL_MAX_RESULTS`` entries.

    Like :func:`_identify_sync` we swallow JSON parse failures rather
    than raising: the route layer can still 422 with "no questions
    detected" and let the client retry manually.
    """
    if not is_ready():
        raise GradingServiceUnavailable(
            "Grading model is not loaded (see startup logs for details)."
        )

    pil_image = _decode_image(image_bytes)

    if not _generate_lock.acquire(blocking=False):
        raise GradingBusy(
            "A previous grading request is still running on the GPU; "
            "retry shortly."
        )

    t_start = time.perf_counter()
    try:
        messages = [
            {"role": "system", "content": _IDENTIFY_ALL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": _IDENTIFY_ALL_USER_PROMPT},
                ],
            },
        ]
        raw = _run_generation(
            messages, max_new_tokens=_IDENTIFY_ALL_MAX_NEW_TOKENS
        )
    finally:
        _generate_lock.release()

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    try:
        data = _parse_json_response(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "identify_all: JSON parse failed (%s); returning empty list. "
            "raw head=%r",
            exc,
            raw[:200],
        )
        return []

    raw_entries = data.get("questions")
    if not isinstance(raw_entries, list):
        logger.warning(
            "identify_all: 'questions' field missing or wrong type; "
            "returning empty list. data=%r",
            dict(list(data.items())[:3]),
        )
        return []

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw_entries):
        sanitized = _sanitize_identification_entry(entry, idx)
        if sanitized is None:
            continue
        key = sanitized["question_number"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(sanitized)
        if len(out) >= _IDENTIFY_ALL_MAX_RESULTS:
            break

    logger.info(
        "identify_all: detected=%d (cap=%d) elapsed_ms=%.1f vram_used_mb=%d "
        "labels=%r",
        len(out),
        _IDENTIFY_ALL_MAX_RESULTS,
        elapsed_ms,
        _vram_used_mb(),
        [e["question_number"] for e in out],
    )
    return out


async def identify_all_questions(image_bytes: bytes) -> list[dict[str, Any]]:
    """List every distinct question the student answered on the page.

    Async wrapper around :func:`_identify_all_sync`. Dispatches onto
    the default executor so the FastAPI event loop stays responsive
    during the ~3-5s Qwen pass.

    Returns
    -------
    list of dict
        Each entry has keys ``question_number`` (non-empty str),
        ``problem_text_preview`` (str | None), and ``confidence``
        (``"high" | "medium" | "low"``). Empty list when the model
        couldn't find any labels OR when JSON parsing failed.

    Raises
    ------
    GradingServiceUnavailable
        Model failed to load at startup.
    GradingBusy
        Another grading request is holding the GPU.
    ValueError
        ``image_bytes`` could not be decoded.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _identify_all_sync, image_bytes)


# --------------------------------------------------------------------- eager load
# Kick the load off at import time so the first HTTP request doesn't
# pay the 30-60s warm-up cost. If CUDA is missing this leaves the
# singletons at ``None`` and :func:`grade_answer` will raise
# :class:`GradingServiceUnavailable` on first use.
_load_model()
