"""Qwen2.5-VL-7B-Instruct-AWQ grading service.

This module owns the vision-language model used to auto-grade a
student's handwritten step-by-step solution. It is tuned for an
RTX 5080 (16 GB VRAM, CUDA 13.1) and designed to stay under a ~14 GB
peak allocation covering both the model weights and the per-request
inference activations.

Architecture
------------

- **Model**: ``Qwen/Qwen2.5-VL-7B-Instruct-AWQ`` in ``float16``. AWQ is
  a 4-bit activation-aware weight quantisation that ships natively in
  recent ``transformers``; no ``bitsandbytes`` runtime is required.
  Weights land in VRAM at roughly ~5 GB.
- **Resolution cap**: the processor is pinned to the range
  ``[256*28*28, 512*28*28]`` pixels (200,704 - 401,408). This single
  knob is the biggest lever on peak VRAM - a 12 MP phone photo would
  otherwise balloon the visual-token activations into the 6 GB range.
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

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
MODEL_DISPLAY_NAME = "Qwen2.5-VL-7B-Instruct-AWQ"

# Image-resolution bounds in pixels. Qwen2.5-VL tiles the image into
# 28x28 patches; clamping both ends of the range keeps the visual
# token sequence predictable and keeps activations in VRAM bounded.
_MIN_PIXELS = 256 * 28 * 28  # 200,704 px  (~448x448 square equivalent)
_MAX_PIXELS = 512 * 28 * 28  # 401,408 px  (~633x633 square equivalent)

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
# Single-braced {question}, {answer_key}, {max_score}, {subject} are
# the real substitution points.
_USER_PROMPT_TEMPLATE = """Grade this handwritten math solution.

Question: {question}
Full correct solution (answer key):
{answer_key}
Maximum score: {max_score}
Subject: {subject}

STEP 1 - READ the handwriting:
Transcribe every line the student wrote, exactly as written.
Do not skip any lines. Do not correct anything yet.

STEP 2 - VERIFY each step:
Compare each transcribed step against the answer key.

CRITICAL VERIFICATION RULES:
- Rule 1: Step 1 must show the EXACT expression from the question.
  Check every coefficient and sign individually.
  (2x^2-xy+3y^2) != (x^2+xy+3y^2) -> step 1 is WRONG.
- Rule 2: Each step must follow mathematically from the previous.
  Check every coefficient, sign, and variable character by character.
- Rule 3: If step N is wrong, steps after it that use the wrong
  result are 'affected by step N error' - note dependency but
  do not penalise the method again if approach is still valid.
- Rule 4: Award marks proportionally per correct step.

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
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception:
            logger.exception(
                "grading: transformers/torch not installed; grading endpoint will 503"
            )
            return

        if not torch.cuda.is_available():
            logger.error(
                "grading: CUDA is not available on this host; "
                "Qwen2.5-VL-AWQ requires a GPU. Grading endpoint will 503."
            )
            return

        _device = "cuda"

        try:
            logger.info(
                "grading: loading %s in float16 on %s (AWQ weights)...",
                MODEL_ID,
                _device,
            )
            _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
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
) -> list[dict[str, Any]]:
    """Build the Qwen2.5-VL chat-template messages for one grading call.

    The conversation is split into a system message (role definition)
    and a multi-modal user message carrying the PIL image plus the
    step-by-step grading instructions.
    """
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        question=question,
        answer_key=answer_key,
        max_score=max_score,
        subject=subject,
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


def _run_generation(messages: list[dict[str, Any]]) -> str:
    """Run one ``model.generate`` pass and return the decoded string.

    Must be called with :data:`_generate_lock` held so two concurrent
    grading requests do not trample each other on the shared GPU.
    Explicitly drops the ``inputs`` tensor and empties the allocator
    cache after decoding so the next request sees a compacted pool.

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
                max_new_tokens=_MAX_NEW_TOKENS_PASS,
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
    }


def _grade_sync(
    image_bytes: bytes,
    question: str,
    answer_key: str,
    max_score: int,
    subject: str,
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

    # --- Happy path: score validation + response envelope -----------
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
    )


# --------------------------------------------------------------------- eager load
# Kick the load off at import time so the first HTTP request doesn't
# pay the 30-60s warm-up cost. If CUDA is missing this leaves the
# singletons at ``None`` and :func:`grade_answer` will raise
# :class:`GradingServiceUnavailable` on first use.
_load_model()
