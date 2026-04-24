"""Hybrid OCR + LLM grading service.

Auto-grades a student's handwritten step-by-step solution using a
two-stage pipeline:

1. **OCR stage (GLM-OCR, ``zai-org/GLM-OCR``)** transcribes the
   handwritten image into plain-text math lines. The OCR engine lives
   in :mod:`app.services.ocr_service` and is loaded lazily; this
   module reuses that singleton via
   :func:`app.services.ocr_service.transcribe_image` so the GLM-OCR
   weights are only loaded once per process even when both answer-key
   ingestion and student-answer grading are active.
2. **LLM stage (``Qwen/Qwen2.5-VL-7B-Instruct``, bnb 4-bit NF4)**
   grades the transcription against a retrieved answer-key. The
   Qwen2.5-VL backbone is used as a pure **text LLM** here - no
   image is passed through ``apply_chat_template`` anymore, so the
   vision encoder and ``qwen_vl_utils.process_vision_info`` step are
   skipped. Qwen performs the RAG generation step: given the
   student's transcribed work plus the retrieved answer-key context,
   it emits the step-by-step grading JSON.

Why split OCR from grading?
---------------------------

The previous architecture used Qwen2.5-VL for both reading AND
reasoning in one shot. That wasted capacity two ways: Qwen's visual
encoder is strong but not OCR-tuned, and forcing a single prompt to
handle "read handwriting correctly" AND "reason about math mistakes"
empirically hurt both. Splitting lets each model do what it's best at:
GLM-OCR is purpose-built for multilingual handwriting transcription,
and Qwen's text side handles math reasoning on clean input.

RAG role
--------

Retrieval is performed upstream by
:func:`app.services.question_resolver.resolve_question`: the caller
embeds the student's detected text/question number with ``bge-m3``,
searches the Chroma collection scoped to the target ``file_id``, and
picks the matching :class:`AnswerKeyItem`. ``grade_answer``'s
``answer_key`` parameter is the retrieved context; an optional
``retrieval_context`` parameter can inject *additional* similar-problem
chunks the caller has already fetched. Qwen then generates against the
combined retrieved context - the "generation" step of RAG.

Hardware notes
--------------

- Tuned for an RTX 5080 (16 GB VRAM, CUDA 12.8). Qwen2.5-VL-7B in
  4-bit NF4 uses ~5-6 GB; GLM-OCR is loaded with ``device_map="auto"``
  so it spills to CPU if VRAM is tight. Set ``GLM_OCR_MODEL_PATH`` to
  a smaller OCR model or run on a bigger GPU if you see OOM.
- ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` is set *before*
  torch is imported so the allocator can grow/shrink segments instead
  of fragmenting across requests.
- ``bitsandbytes`` 0.49+ ships native Windows wheels with Blackwell
  kernels - that's the drop-in substitute for ``autoawq``, which has
  no working wheel on Windows + torch 2.11 / sm_120.

Concurrency
-----------

Only one Qwen generation may run at a time on the GPU. We guard
inference with a non-blocking :class:`threading.Lock`; if a previous
(likely timed-out) request is still holding the GPU, we raise
:class:`GradingBusy` so the caller sees an immediate 503 with a
``Retry-After`` header rather than stacking up executor threads.
GLM-OCR transcription happens BEFORE the lock is acquired so a slow
OCR pass doesn't keep the grading GPU lock held.
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

MODEL_ID: str = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_DISPLAY_NAME: str = "Qwen2.5-VL-7B-Instruct (bnb-4bit) + GLM-OCR"

# Max tokens the Qwen grader may emit in a single pass. Enough headroom
# for ~20 step JSON with prose feedback; override via env for edge cases.
_MAX_NEW_TOKENS_PASS: int = int(os.getenv("GRADING_MAX_NEW_TOKENS", "1024"))

# Tighter budgets for the identification passes - they only return
# a small JSON envelope (question label + optional preview), so we
# do not need the full grading budget.
_IDENTIFY_MAX_NEW_TOKENS: int = int(
    os.getenv("GRADING_IDENTIFY_MAX_NEW_TOKENS", "128")
)
_IDENTIFY_ALL_MAX_NEW_TOKENS: int = int(
    os.getenv("GRADING_IDENTIFY_ALL_MAX_NEW_TOKENS", "256")
)

# Hard cap on detected questions we process downstream even if Qwen
# hallucinates more. Prevents pathological pages from triggering 10+
# grading passes. Must match the cap in the route layer.
_IDENTIFY_ALL_MAX_RESULTS: int = int(
    os.getenv("GRADING_MAX_QUESTIONS_PER_IMAGE", "5")
)

# Generation budget for the handwriting OCR pass used when producing
# the transcription fed to the grading LLM. Real pages with 20+ math
# steps transcribe into ~250-500 tokens; 2048 gives comfortable
# headroom without making the model spin through thousands of empty
# tokens before EOS (each one costs a full forward pass).
_OCR_MAX_NEW_TOKENS: int = int(os.getenv("GRADING_OCR_MAX_NEW_TOKENS", "2048"))

# Tighter OCR budget for the identify passes. They only need enough
# of the page to surface question labels + starting expressions, so
# capping generation much lower here keeps the per-call latency well
# under the grading pass's budget.
_OCR_IDENTIFY_MAX_NEW_TOKENS: int = int(
    os.getenv("GRADING_OCR_IDENTIFY_MAX_NEW_TOKENS", "512")
)

# Prompt sent to GLM-OCR for student handwriting. Plain "Text
# Recognition:" keeps the output verbatim (one line per physical line)
# which is exactly what the grading LLM needs. Override via env if
# you need to specialise for a subject.
_OCR_PROMPT: str = os.getenv("GRADING_OCR_PROMPT", "Text Recognition:")

# Wall-clock budget (in seconds) for a single run_in_executor grading call.
# When exceeded the async wrapper raises GradingTimeout so the route layer
# can return 503 + Retry-After instead of blocking the executor pool.
# Override via GRADING_TIMEOUT_S env var. The route layer adds its own
# (typically larger) asyncio.wait_for budget on top of this.
_GRADING_TIMEOUT_S: float = float(os.getenv("GRADING_TIMEOUT_S", "180"))

# Set GRADING_PREPROCESS_IMAGES=0 to skip opencv preprocessing entirely
# (useful for already-clean digital scans where denoising/deskewing is
# a no-op at best or harmful at worst). Preprocessing now feeds GLM-OCR
# rather than Qwen's visual encoder, so the same denoise/deskew/
# binarise pipeline still improves accuracy. Defaults to enabled.
_GRADING_PREPROCESS: bool = os.getenv("GRADING_PREPROCESS_IMAGES", "1") == "1"

# Optional extra RAG retrieval on top of the answer-key already passed
# in from the resolver. When enabled, the grader embeds the student's
# OCR'd transcription and fetches top-K similar-problem chunks from
# Chroma (scoped to the same file_id). Set to 0 to disable; defaults
# to 0 because the resolver already supplies the best single chunk.
_RAG_EXTRA_TOP_K: int = int(os.getenv("GRADING_RAG_EXTRA_TOP_K", "0"))

# ----- Final-answer safety net ------------------------------------------------
# The safety net in ``_repair_final_answer_match`` upgrades a response to
# full-marks when the expected final answer appears in one of the VLM's
# ``extracted_steps[].student_wrote`` fields. It was designed to paper over
# cursive-3-vs-8 transcription noise, but a too-permissive match can also
# paper over GENUINE errors:
#
#   * Qwen sometimes copies the expected answer out of its own user-prompt
#     (the ``answer_key`` block) into ``student_wrote``, even when the raw
#     GLM-OCR transcription never contained that string. We call that the
#     "VLM leaked the answer key" failure mode.
#   * When the student simplifies in one step, an intermediate line can
#     legitimately equal the final answer. If the student THEN continues
#     and writes a wrong final line, "match-anywhere" still fires and
#     the student gets full marks for work they did wrong.
#
# Three env knobs let you control the behaviour without code changes:
#
#   GRADING_FINAL_ANSWER_SAFETY_NET             (default 1)
#     Master switch. Set to 0 to disable the safety net entirely - useful
#     while you are tuning Qwen's per-step grading accuracy and do not
#     want spurious upgrades masking real errors.
#
#   GRADING_SAFETY_NET_REQUIRE_LAST_STEP        (default 1)
#     Only accept a match that lands on the LAST non-empty extracted step.
#     Set to 0 to restore the legacy "match anywhere in the step list"
#     behaviour, which is more forgiving of VLM line-drops but also more
#     prone to crediting wrong final-line work.
#
#   GRADING_SAFETY_NET_VERIFY_TRANSCRIPTION     (default 1)
#     Cross-check that the expected answer also appears (whitespace-
#     insensitive) in the raw GLM-OCR transcription. This defeats the
#     "Qwen leaked the answer key into student_wrote" failure mode: if
#     OCR never saw the expected string, the match cannot be trusted.
#     Set to 0 to skip the cross-check (legacy behaviour).
_GRADING_SAFETY_NET_ENABLED: bool = (
    os.getenv("GRADING_FINAL_ANSWER_SAFETY_NET", "1") == "1"
)
_GRADING_SAFETY_NET_REQUIRE_LAST_STEP: bool = (
    os.getenv("GRADING_SAFETY_NET_REQUIRE_LAST_STEP", "1") == "1"
)
_GRADING_SAFETY_NET_VERIFY_TRANSCRIPTION: bool = (
    os.getenv("GRADING_SAFETY_NET_VERIFY_TRANSCRIPTION", "1") == "1"
)


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


class GradingTimeout(RuntimeError):
    """Raised when a grading call exceeds ``_GRADING_TIMEOUT_S``.

    The blocking ``model.generate`` call in the executor thread cannot
    be cancelled once started, but we can surface a meaningful 503 to
    the caller instead of letting the request hang indefinitely.
    The route layer should translate this into HTTP 503 + Retry-After.
    """


# --------------------------------------------------------------------- reasoning rule constants
# These are injected into the user prompt (via string concatenation in
# _USER_PROMPT_TEMPLATE) so each rule block can be tuned independently
# without touching the surrounding template. The template calls .format()
# on the concatenated result, so any literal curly braces here must be
# doubled ({{ / }}).
#
# NOTE: these were originally "OCR rules" that asked the VLM to read
# the handwriting. Reading is now GLM-OCR's job, so these blocks have
# been rewritten to operate on the TRANSCRIBED text: how to normalise
# math notation the OCR emitted, and how to line up transcribed lines
# against expected steps.

_SYMBOL_RULES: str = """\
TRANSCRIPTION NORMALISATION:
- The transcription was produced by GLM-OCR and may contain residual \
ambiguities. Before comparing, treat these pairs as interchangeable if \
one reading makes the math follow logically: 3/8, 1/7, 0/6, 4/9, +/t, \
-/em-dash, ×/x-variable, y/q/g, n/u, a/d, z/2.
- Superscripts are written as ``^N`` (x^2, x^3) - compare accordingly.
- Greek letters may appear as words: alpha, beta, pi, sum, delta."""

_STRUCTURE_RULES: str = """\
MATH STRUCTURE CONVENTIONS:
- Fractions: ``(x+1)/(x-2)``
- Exponents: ``x^2``
- Square root: ``sqrt(x+1)``
- Subscripts: ``x_1``
- Absolute value: ``|x+1|``
- Integrals: ``integral(x)dx``
- Mixed numbers: ``2 + 1/2``
- Preserve every parenthesis the transcription contains."""

_LAYOUT_RULES: str = """\
STEP-SPLITTING RULES:
- Each non-empty line of the transcription is ONE candidate step.
- If a line starts with ``=`` alone, still treat it as its own step.
- Skip blank lines and lines that contain only punctuation (``-``, \
``*``, ``_``) - those are stray OCR artefacts from rulings / borders.
- Student work flows strictly top-to-bottom in the transcription, so \
iterate it in order."""

_COUNT_VERIFICATION: str = """\
STEP-COUNT VERIFICATION (do this BEFORE outputting JSON):
1. Count the non-empty lines in the transcription → call this L.
2. Your ``extracted_steps`` list MUST contain exactly L entries \
(one per line), unless two physical lines clearly belong to the \
same math expression (a continuation) - in that case combine them \
into one step.
3. The LAST non-empty line in the transcription MUST be represented \
as the final entry in ``extracted_steps``. Dropping the final line \
is the most common failure mode; guard against it explicitly."""


# --------------------------------------------------------------------- prompt

_SYSTEM_PROMPT: str = (
    "You are a math teacher grading a student's solution. The student's "
    "handwritten work has already been OCR-transcribed for you. Compare "
    "the transcription line-by-line against the answer key and produce "
    "structured JSON feedback."
)

# Double-braces ({{ / }}) are str.format escapes for literal { / }.
# Single-braced {transcription}, {question}, {answer_key}, {max_score},
# {subject}, {label_hint}, {retrieval_block} are the real substitution
# points. ``{label_hint}`` is filled by ``_build_messages`` - either an
# empty string or a line telling the model the exact question-number
# label to strip (we know it when the request came through
# ``/submit-by-image``). ``{retrieval_block}`` is filled with extra RAG
# context chunks when the caller supplies them (empty string otherwise).
#
# The reasoning rule constants (_SYMBOL_RULES, _STRUCTURE_RULES,
# _LAYOUT_RULES, _COUNT_VERIFICATION) are concatenated as separate
# module-level strings so each block can be tuned independently.
_USER_PROMPT_TEMPLATE: str = (
    """\
Grade this student's solution.

Question: {question}
Full correct solution (answer key):
{answer_key}
{retrieval_block}Maximum score: {max_score}
Subject: {subject}
{label_hint}
Student's handwritten work (transcribed by GLM-OCR, one line per \
physical line; do NOT change the order):
<<<TRANSCRIPTION_BEGIN
{transcription}
TRANSCRIPTION_END>>>
"""
    + _SYMBOL_RULES
    + "\n\n"
    + _STRUCTURE_RULES
    + "\n\n"
    + _LAYOUT_RULES
    + "\n\n"
    + _COUNT_VERIFICATION
    + """

STEP 1 - MIRROR the transcription into extracted_steps:
Walk the transcription top-to-bottom. For EACH non-empty line, emit ONE
entry in ``extracted_steps`` with its ``student_wrote`` field set to
that line's math content. Preserve the line's math verbatim - do not
rewrite, do not "correct" what the student wrote, do not reorder.

DO NOT skip any lines. In particular, ALWAYS include the student's
LAST transcription line (the final answer they arrived at) as its own
entry in ``extracted_steps``. Missing the final line is a common
failure.

CRITICAL - Question labels are NOT math. Strip them before writing
``student_wrote``:
Question labels like "1.", "Q3", "<label>.", "Question 5:", "(3)" at
the very start of the transcription are METADATA. They MUST NOT appear
in ``student_wrote``. A label's presence is NEVER grounds for
``is_correct: false`` - strip it, then compare.

Example - this is the exact failure mode to avoid:
  Transcription begins with:  "<label>. (3x^2+2xy-y^2)-(x^2-5xy-4y^2)"

  CORRECT output:
    {{"step_number": 1,
      "student_wrote": "(3x^2+2xy-y^2)-(x^2-5xy-4y^2)",
      "expected":      "(3x^2+2xy-y^2)-(x^2-5xy-4y^2)",
      "is_correct": true, "error": null}}

  WRONG output (do NOT produce this):
    {{"step_number": 1,
      "student_wrote": "<label>. (3x^2+2xy-y^2)-(x^2-5xy-4y^2)",
      "is_correct": false,
      "error": "includes question label"}}

If the student copied the starting expression correctly, step 1
is correct regardless of any label in the transcription.

STEP 2 - VERIFY each step:
The answer key lists the expected steps in order as "Step 1",
"Step 2", ... Match transcription line N against the answer key's
Step N (same number). In the ``expected`` field for each student
step, put the answer key's Step N text. Do NOT put the final
answer there unless this IS the final-answer step.

CRITICAL VERIFICATION RULES:
- Rule 0: Strip question labels before comparing. A leading "Q3.",
  "<label>.", "(3)", "Question 5:" at the top of the transcription is
  a question NUMBER, not a coefficient or part of the math. Never
  flag it as a coefficient mismatch against the expected expression.
- Rule 1: After stripping the label, step 1 must show the EXACT
  expression from the question (this is also the answer key's
  Step 1 "starting expression"). Perform a character-by-character
  comparison on:
    * every coefficient (e.g. "3x^2" vs "2x^2" -> WRONG),
    * every sign before every term (e.g. "+y^2" vs "-y^2" -> WRONG),
    * every exponent (e.g. "x^2" vs "x^3" -> WRONG),
    * every bracket structure.
  A single sign flip or coefficient mismatch in step 1 makes step 1
  wrong, even if subsequent steps "make sense" given the wrong
  starting expression. Do NOT credit step 1 because the student's
  intent is obvious; grade what they actually wrote.
- Rule 2: Each subsequent student step must (a) follow mathematically
  from the previous student step AND (b) match the answer key's step
  with the same number, character by character on coefficients and
  signs.
- Rule 3: If step N is wrong, steps after it that use the wrong
  result are 'affected by step N error' - note dependency but
  do not penalise the method again if approach is still valid.
- Rule 4 (PROPORTIONAL SCORING): Award marks strictly proportionally
  per correct step. If there are N extracted_steps and K of them are
  correct, student_score MUST equal round(K / N * max_score). Do NOT
  award more than this just because the final answer happens to match;
  intermediate-step errors always cost marks. If ``is_correct`` is
  ``false`` for ANY step, ``student_score`` MUST be strictly less than
  ``max_score``. A self-contradictory response (e.g. ``is_correct:
  false`` at top level but ``student_score: max_score``) will be
  rejected.
- Rule 5 (TRANSCRIPTION IS AUTHORITATIVE): The GLM-OCR transcription
  between the <<<TRANSCRIPTION_BEGIN ... TRANSCRIPTION_END>>> markers
  is the SOLE source of truth for what the student wrote. Do NOT
  "correct" the transcription to make the math work. Do NOT
  substitute a character (digit, sign, coefficient, exponent) even
  if an alternate reading would make the step follow logically.
  In particular:
    * If the transcription shows "+y^2" where the question has "-y^2",
      step 1 is WRONG - the student copied the sign incorrectly.
      Do not flip "+" to "-" just because "-" would match the question.
    * If the transcription shows "3y^2" where "5y^2" would make the
      arithmetic close, step N is WRONG. Do not change "3" to "5".
    * If the student's final line happens to equal the expected
      final answer but is inconsistent with their own preceding
      step, the preceding step is STILL wrong - mark it wrong and
      note the inconsistency in the error field.
  Character errors in the transcription ARE student errors for the
  purpose of grading. If OCR truly misread a character the operator
  will see it in the response and can regrade manually; that is
  strictly better than silently crediting wrong work.

FEEDBACK DETAIL REQUIREMENTS (for each wrong step):
- The step's ``error`` field MUST include (1) what the student wrote,
  quoted verbatim, (2) what the answer key expected, quoted verbatim,
  and (3) the specific discrepancy (which term / coefficient / sign /
  operator differs).
  Bad:  "wrong coefficient of xy"
  Good: "student wrote '5xy' but the expected step 2 has '7xy'; the
         coefficient of xy is off by 2 - the student forgot to add the
         '+5xy' that comes from distributing -(x^2-5xy-4y^2)".
- The top-level ``feedback`` field MUST aggregate per-step errors in
  one or two sentences that a student could act on. Include the step
  number, what they wrote, and what they should have written.
  Bad:  "Step 2: wrong coefficient of xy."
  Good: "Step 2 is wrong: you wrote '5x^2+2xy-xy+3y^2+y^2' but the
         expansion should be '5x^2+2xy-xy+3y^2+y^2+4xy' - you dropped
         the '+4xy' term from the third bracket. Step 3 carries the
         same error into '5x^2+xy+4y^2' where the correct result is
         '5x^2+5xy+4y^2'."
- ``error_summary`` is a one-line headline (<= 120 chars) for dashboards
  and should name the single most important thing that went wrong.

OUTPUT RULES (strictly follow):
- Output RAW JSON only
- No markdown fences, no backticks, no ```json
- Start your response with {{ and end with }}
- Do not put JSON inside any field value

{{
  "extracted_steps": [
    {{
      "step_number": 1,
      "student_wrote": "(5x^2+2xy)-(xy-3y^2)+(y^2+4xy)",
      "expected":      "(5x^2+2xy)-(xy-3y^2)+(y^2+4xy)",
      "is_correct": true,
      "error": null
    }},
    {{
      "step_number": 2,
      "student_wrote": "5x^2+2xy-xy+3y^2+y^2",
      "expected":      "5x^2+2xy-xy+3y^2+y^2+4xy",
      "is_correct": false,
      "error": "student wrote '5x^2+2xy-xy+3y^2+y^2' but after distributing -(xy-3y^2) and then +(y^2+4xy) the expansion should be '5x^2+2xy-xy+3y^2+y^2+4xy'; the '+4xy' term from the third bracket was dropped. Discrepancy: missing '+4xy'."
    }},
    {{
      "step_number": 3,
      "student_wrote": "5x^2+xy+4y^2",
      "expected":      "5x^2+5xy+4y^2",
      "is_correct": false,
      "error": "student wrote '5x^2+xy+4y^2' but the correct combination of like terms gives '5x^2+5xy+4y^2'; the coefficient of xy is 1 instead of 5 because the '+4xy' lost in step 2 carries through. Discrepancy: xy coefficient off by 4."
    }}
  ],
  "first_error_step": 2,
  "error_summary": "Dropped '+4xy' when distributing the third bracket in step 2",
  "method_correct": true,
  "student_score": 2,
  "max_score": {max_score},
  "feedback": "Step 1 is correct - you copied the expression accurately. Step 2 is wrong: you wrote '5x^2+2xy-xy+3y^2+y^2' but the full distribution of -(xy-3y^2)+(y^2+4xy) should give '5x^2+2xy-xy+3y^2+y^2+4xy' - you dropped the '+4xy' term from the last bracket. Step 3 carries the same error: you got '5x^2+xy+4y^2' but the correct combined form is '5x^2+5xy+4y^2' (xy coefficient should be 5, not 1).",
  "is_correct": false,
  "confidence": "high"
}}"""
)

# Appended to the user prompt on a retry when the first response was
# not parseable. Intentionally short and blunt.
_STRICT_RETRY_SUFFIX: str = (
    "\n\nIMPORTANT: Your previous response could not be parsed.\n"
    "Output ONLY a valid JSON object. Start with { end with }.\n"
    "No other text."
)

# Required top-level keys in a grading JSON response. Checked after
# json.loads in _parse_json_response when called from _grade_sync.
# Identification responses (identify_question / identify_all_questions)
# do NOT pass this set, so they are not validated against it.
_GRADING_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"extracted_steps", "student_score", "is_correct"}
)


# --------------------------------------------------------------------- module globals
# torch/transformers imports are lazy so that `import app.services.grading_service`
# does not force a multi-second CUDA init (and does not fail on CI
# machines without a GPU).

_model: Any = None
_processor: Any = None
_device: str | None = None
_load_lock: threading.Lock = threading.Lock()
_load_attempted: bool = False
# Human-readable description of the last model-load failure, or None
# when load succeeded (or has not been attempted yet). Exposed via
# load_error() so health-check routes can surface this without tailing
# log files.
_load_error: str | None = None
# Serialises GPU access across request threads dispatched by the
# FastAPI executor pool. Non-blocking acquire in ``_grade_sync`` so we
# can surface 503/GradingBusy instead of queueing up blocked threads.
_generate_lock: threading.Lock = threading.Lock()


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
    stay ``None``, :data:`_load_error` is populated with a human-readable
    description, and :func:`grade_answer` raises
    :class:`GradingServiceUnavailable`.

    Qwen2.5-VL is loaded here for its TEXT capability only; we no longer
    send images through its vision encoder (GLM-OCR handles OCR
    upstream). The processor is still a full ``AutoProcessor`` so the
    chat-template helper continues to work - we just never call
    ``process_vision_info`` or pass ``images=`` to the processor.
    """
    global _model, _processor, _device, _load_attempted, _load_error

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
        except Exception as exc:
            _load_error = f"transformers/torch not installed: {exc}"
            logger.exception(
                "grading: transformers/torch not installed; grading endpoint will 503"
            )
            return

        if not torch.cuda.is_available():
            _load_error = (
                "CUDA is not available on this host; "
                "Qwen2.5-VL bnb-4bit requires a GPU."
            )
            logger.error(
                "grading: CUDA is not available on this host; "
                "Qwen2.5-VL bnb-4bit requires a GPU. Grading endpoint will 503."
            )
            return

        try:
            import bitsandbytes  # noqa: F401
        except Exception as exc:
            _load_error = f"bitsandbytes not installed: {exc}"
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
            _processor = AutoProcessor.from_pretrained(MODEL_ID)
            _model.eval()
            logger.info(
                "grading: Qwen text-grader loaded (max_new_tokens=%d, OCR handled "
                "upstream by GLM-OCR via ocr_service)",
                _MAX_NEW_TOKENS_PASS,
            )
            _log_vram_usage("after-load")
        except Exception as exc:
            if _is_oom(exc):
                _load_error = f"CUDA OOM loading {MODEL_ID}: {exc}"
                logger.error(
                    "grading: CUDA OOM loading %s even in AWQ/fp16 form. "
                    "Grading endpoint will 503. (%s)",
                    MODEL_ID,
                    exc,
                )
            else:
                _load_error = f"Load failed for {MODEL_ID}: {exc}"
                logger.exception(
                    "grading: load failed for %s; grading endpoint will 503",
                    MODEL_ID,
                )
            _model = None
            _processor = None
            _device = None
            try:
                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass


def is_ready() -> bool:
    """Return ``True`` iff the model + processor are loaded and usable."""
    return _model is not None and _processor is not None


def load_error() -> str | None:
    """Return the human-readable model-load failure message, or ``None``.

    Returns ``None`` when the model loaded successfully (or when a load
    has not been attempted yet). Returns a non-empty string on any
    failure path so health-check routes can surface the reason without
    tailing log files.

    Returns
    -------
    str | None
        Load failure description, or ``None`` on success / not-yet-attempted.
    """
    return _load_error


# --------------------------------------------------------------------- image helpers


def _preprocess_image(pil_image: Image.Image) -> Image.Image:
    """Apply opencv-based preprocessing to improve handwriting OCR accuracy.

    Pipeline (when ``GRADING_PREPROCESS_IMAGES=1`` and opencv is installed):

    1. Convert to grayscale.
    2. :func:`cv2.fastNlMeansDenoising` (h=10) to reduce scan/JPEG noise.
    3. Deskew: find the dominant tilt angle of dark pixels via
       :func:`cv2.minAreaRect` and correct with :func:`cv2.warpAffine`
       when ``0.5° < |angle| < 45°``.
    4. :func:`cv2.adaptiveThreshold` (ADAPTIVE_THRESH_GAUSSIAN_C,
       blockSize=31, C=10) to binarise the image.
    5. Convert back to RGB PIL ``Image``.

    If ``opencv-python`` (cv2) or ``numpy`` is not installed, or if
    ``GRADING_PREPROCESS_IMAGES`` is set to ``"0"``, the original image
    is returned unchanged so the function is always safe to call.

    Parameters
    ----------
    pil_image:
        Input image in any PIL mode.

    Returns
    -------
    Image.Image
        Preprocessed RGB image, or the original on any error / missing deps.
    """
    if not _GRADING_PREPROCESS:
        return pil_image

    try:
        import cv2  # type: ignore[import]
        import numpy as np
    except ImportError:
        return pil_image

    try:
        # 1. Grayscale
        gray = np.array(pil_image.convert("L"))

        # 2. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # 3. Deskew using minimum area rectangle of dark pixels
        rows, cols = np.where(denoised < 128)
        if len(rows) >= 4:
            pts = np.column_stack([cols, rows]).astype(np.float32)
            rect = cv2.minAreaRect(pts)
            angle: float = float(rect[-1])
            # Normalize: minAreaRect returns angles in (-90, 0]; map to (-45, 45]
            if angle < -45.0:
                angle += 90.0
            if abs(angle) > 0.5 and abs(angle) < 45.0:
                h_img, w_img = denoised.shape[:2]
                M = cv2.getRotationMatrix2D(
                    (w_img / 2.0, h_img / 2.0), angle, 1.0
                )
                denoised = cv2.warpAffine(
                    denoised,
                    M,
                    (w_img, h_img),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )

        # 4. Adaptive threshold (binarise)
        thresh = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )

        # 5. Back to RGB PIL
        rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    except Exception as exc:  # pragma: no cover - best-effort preprocessing
        logger.warning("grading: image preprocessing failed, using original: %s", exc)
        return pil_image


def _decode_image(image_bytes: bytes) -> Image.Image:
    """Decode raw bytes into a preprocessed PIL ``RGB`` image.

    Responsibilities:

    1. Decode ``image_bytes`` → PIL :class:`~PIL.Image.Image`.
    2. Convert to ``RGB`` mode.
    3. Apply :func:`_preprocess_image` (grayscale → denoise → deskew →
       binarise) to improve downstream OCR accuracy.

    The resulting image is fed to GLM-OCR (not Qwen) - GLM-OCR handles
    its own smart-resize internally, so we don't clamp pixel dimensions
    here anymore.

    Parameters
    ----------
    image_bytes:
        Raw encoded image data (JPEG, PNG, WebP, …).

    Returns
    -------
    Image.Image
        Preprocessed RGB image ready to pass to GLM-OCR.

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
    return _preprocess_image(img)


def _transcribe_handwriting(
    pil_image: Image.Image,
    *,
    prompt: str = _OCR_PROMPT,
    max_new_tokens: int = _OCR_MAX_NEW_TOKENS,
) -> str:
    """Run GLM-OCR on a preprocessed student image and return the transcription.

    Delegates to :func:`app.services.ocr_service.transcribe_image` so the
    GLM-OCR engine singleton is shared with the answer-key ingestion
    path - we never load the OCR weights twice.

    Returns an empty string on any failure (engine missing, OCR
    exception). The caller is responsible for deciding how to react to
    an empty transcription (typically: fail the grading call with a
    low-confidence zero-score envelope).
    """
    try:
        from app.services.ocr_service import transcribe_image
    except Exception:
        logger.exception(
            "grading: could not import ocr_service.transcribe_image; "
            "cannot OCR student handwriting"
        )
        return ""

    try:
        return transcribe_image(
            pil_image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
    except Exception:
        logger.exception("grading: GLM-OCR transcription raised")
        return ""


def _retrieve_grading_context(
    transcription: str,
    file_id: str | None,
    *,
    top_k: int = _RAG_EXTRA_TOP_K,
) -> list[str]:
    """Fetch additional Chroma chunks to feed into the grading prompt.

    The resolver already picks the matching :class:`AnswerKeyItem` for
    the student's target question and the route layer passes that
    item's worked solution as ``answer_key``. This helper adds an
    *optional* second retrieval hop: embed the OCR'd transcription
    with ``bge-m3`` and pull the top-K nearest-neighbour answer-key
    chunks from the same ``file_id``. Useful when the student's work
    looks structurally similar to other solved problems and we want
    the LLM to spot cross-question hints.

    No-op when ``top_k <= 0`` (the default) or when retrieval raises;
    we never let a RAG failure break grading.
    """
    if top_k <= 0 or not transcription.strip():
        return []

    try:
        from app.services.chroma_service import search_documents
        from app.services.embedding_service import get_embedding
    except Exception:
        logger.exception("grading: RAG dependencies unavailable; skipping retrieval")
        return []

    try:
        query = transcription[:2000]  # bge-m3 is happier with bounded inputs
        embedding = get_embedding(query)
        where = {"file_id": str(file_id)} if file_id else None
        result = search_documents(
            query_embedding=embedding,
            top_k=top_k,
            where=where,
        )
        docs_outer = result.get("documents") or []
        docs = docs_outer[0] if docs_outer else []
        return [str(d) for d in docs if d]
    except Exception:
        logger.exception(
            "grading: RAG retrieval failed (top_k=%d, file_id=%s); proceeding without extras",
            top_k,
            file_id,
        )
        return []


# --------------------------------------------------------------------- parsing


def _parse_json_response(
    raw: str,
    required_fields: frozenset[str] | None = None,
) -> dict[str, Any]:
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

    Parameters
    ----------
    raw:
        Raw string output from the model.
    required_fields:
        When provided, every key in this set must be present in the
        parsed dict. If any are missing a ``ValueError`` is raised
        listing the absent keys. Pass ``_GRADING_REQUIRED_FIELDS`` from
        grading call sites; omit (or pass ``None``) for identification
        calls whose response schema differs.

    Returns
    -------
    dict[str, Any]
        Parsed and minimally validated response dictionary.

    Raises
    ------
    ValueError
        If no balanced ``{...}`` can be located in ``raw``, or if
        ``required_fields`` are absent from the parsed dict.
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

    # 4. required-field validation (grading responses only)
    if required_fields:
        missing = required_fields - data.keys()
        if missing:
            raise ValueError(
                f"Model response missing required fields: {sorted(missing)}"
            )

    # 5. detect the "model nested its JSON inside feedback" failure mode
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

    Parameters
    ----------
    label:
        Canonical question label, e.g. ``"0003"`` or ``"5"``.

    Returns
    -------
    list[str]
        Deduplicated list of label variants sorted longest-first.
    """
    clean = str(label or "").strip()
    if not clean:
        return []

    variants: list[str] = []
    # Canonical and punctuation variants (no duplicate suffixes).
    for suffix in ("", ".", ")", ":", " -"):
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

    Parameters
    ----------
    text:
        The ``student_wrote`` string to strip.
    variants:
        Label variants to attempt, produced by :func:`_label_prefix_variants`.

    Returns
    -------
    str
        ``text`` with the leading label removed, or ``text`` unchanged
        when no safe match was found.
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


_MATH_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")


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

    Parameters
    ----------
    data:
        Parsed grading response dict, mutated in place.
    question_label:
        Canonical question label (e.g. ``"0003"``) when known, else ``None``.
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
            or "0003" in err_summary  # legacy: defensive against historical model output
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

    Parameters
    ----------
    text:
        Raw answer string to normalise.

    Returns
    -------
    str
        Comparison key suitable for equality testing.
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
    *,
    transcription: str | None = None,
    question: str | None = None,
    question_label: str | None = None,
) -> None:
    """Safety net: if the extracted final step matches the expected final
    answer AND the answer also appears in the raw OCR transcription, upgrade
    the **entire** response to full-marks / all-correct state.

    Rationale
    ---------

    When the student writes the correct final answer on their paper,
    they have reached the correct result. Intermediate-step
    transcription noise (e.g. GLM-OCR reads cursive "3" as "8") is the
    grader's problem, not the student's. A correct final answer is
    strong evidence the student's math was valid; the per-step flags
    the VLM produced against its own (noisy) transcription cannot be
    trusted in that case.

    Cancelling-errors failure mode
    ------------------------------

    The safety net cannot naively trust *any* final-answer match:
    students sometimes make two (or more) real arithmetic mistakes
    that happen to cancel out, producing the correct final answer via
    genuinely wrong intermediate work. Crediting full marks there is
    pedagogically wrong -- the student did not do the math correctly,
    Qwen's per-step ``is_correct=false`` flags are accurate, and the
    mismatched intermediate steps are real errors rather than OCR
    noise.

    Two additional guards below distinguish "OCR noise cancelling out"
    from "real student errors cancelling out":

    - **Step-1 copy guard**: step 1 is a straight copy from the printed
      question. If the student's step 1 does not match the expected
      starting expression, the student made a real copy error -- not
      OCR noise -- so we must not upgrade.
    - **Transcription-confirmed wrong step guard**: if any step Qwen
      flagged ``is_correct=false`` has a ``student_wrote`` that is
      verbatim present in the raw GLM-OCR transcription AND genuinely
      differs from ``expected`` under normalisation, then the VLM's
      flag is reliable (the student actually wrote that wrong line on
      the page), so the upgrade is suppressed.

    Guards (in order)
    -----------------

    1. ``GRADING_FINAL_ANSWER_SAFETY_NET=0`` - master off switch; no-op.
    2. ``expected_final_answer`` is falsy - no target to match against.
    3. ``GRADING_SAFETY_NET_REQUIRE_LAST_STEP=1`` (default) - we only
       accept a match that lands on the LAST non-empty extracted step.
       This prevents crediting students who wrote the correct
       intermediate form but then finished with a wrong final line.
       Set the env var to 0 to restore the legacy "match anywhere"
       behaviour for cases where the VLM drops the trailing line.
    4. ``GRADING_SAFETY_NET_VERIFY_TRANSCRIPTION=1`` (default) - the
       expected answer must also appear (whitespace-insensitive) in the
       raw GLM-OCR transcription. Defeats the "VLM leaked the answer
       key from its prompt into student_wrote" failure mode: if OCR
       never saw the string, Qwen hallucinated it, and the match
       cannot be trusted.
    5. **Step-1 copy guard** - when ``question`` is supplied and step 1's
       ``student_wrote`` (after label-stripping) does not match the
       starting expression, skip the upgrade. This suppresses the
       cancelling-errors false positive: a mismatched step 1 is a real
       student copy error, not OCR noise.
    6. **Transcription-confirmed wrong step guard** - when
       ``transcription`` is supplied, any step flagged
       ``is_correct=false`` whose ``student_wrote`` is verbatim
       present in the transcription AND differs from its ``expected``
       under normalisation is treated as a confirmed real student
       error. The upgrade is skipped because Qwen's per-step verdicts
       are trustworthy in that case.

    When all guards pass we normalise the **whole** response so no field
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

    Parameters
    ----------
    data:
        Parsed grading response dict, mutated in place.
    expected_final_answer:
        Expected final answer string (e.g. ``"2x^2 + xy + 4y^2"``).
    max_score:
        Maximum score for score clamping.
    transcription:
        Raw GLM-OCR transcription of the student's page. Used for the
        hallucination cross-check AND the transcription-confirmed
        wrong step guard. Pass ``None`` to skip both checks (legacy
        behaviour); the hallucination check is also gated on the
        ``GRADING_SAFETY_NET_VERIFY_TRANSCRIPTION`` env var.
    question:
        Expected starting expression (the original problem text). Used
        by the step-1 copy guard to detect real student copy errors
        that would otherwise be masked by a coincidentally correct
        final answer. Pass ``None`` to skip the guard (manual
        ``/submit`` path).
    question_label:
        Canonical question label (e.g. ``"0003"``) so the step-1 guard
        can strip ``"0003."`` / ``"Q3"`` / ``"(3)"`` prefixes from
        ``student_wrote`` before comparing against ``question``.
    """
    if not _GRADING_SAFETY_NET_ENABLED:
        return

    if not expected_final_answer:
        return

    expected_norm = _normalize_final_answer(expected_final_answer)
    if not expected_norm:
        return

    steps = data.get("extracted_steps")
    if not isinstance(steps, list) or not steps:
        return

    match_idx: int | None = None
    if _GRADING_SAFETY_NET_REQUIRE_LAST_STEP:
        # Walk from the end and find the last step with non-empty
        # ``student_wrote``; only accept a match at that position so we
        # never credit a correct intermediate line while the student's
        # real final line is wrong.
        for i in range(len(steps) - 1, -1, -1):
            step = steps[i]
            if not isinstance(step, dict):
                continue
            if not str(step.get("student_wrote") or "").strip():
                continue
            if _normalize_final_answer(step.get("student_wrote")) == expected_norm:
                match_idx = i
            break  # either way, the last non-empty step is decisive
    else:
        # Legacy "match anywhere" behaviour. Kept for ops who know their
        # VLM tends to drop the trailing line and prefer false positives
        # to false negatives.
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            if _normalize_final_answer(step.get("student_wrote")) == expected_norm:
                match_idx = i
                break

    if match_idx is None:
        return  # student never wrote the expected final answer; trust VLM flags.

    # Anti-hallucination: if the expected answer isn't actually present
    # in what GLM-OCR read off the page, Qwen is almost certainly copying
    # the expected string from the answer_key block in its own prompt.
    # Refuse the upgrade in that case so Qwen's per-step verdicts stand.
    if (
        _GRADING_SAFETY_NET_VERIFY_TRANSCRIPTION
        and transcription is not None
    ):
        transcription_norm = _math_normalize(transcription)
        if expected_norm not in transcription_norm:
            logger.warning(
                "grading: final-answer safety net SKIPPED - expected %r "
                "matched extracted_steps[%d].student_wrote but NOT present "
                "in raw GLM-OCR transcription (len=%d). Likely VLM "
                "hallucination; trusting per-step flags.",
                expected_final_answer,
                match_idx,
                len(transcription_norm),
            )
            return

    # Guard: if step 1 doesn't match the starting expression, the
    # student made a real copy error - do not credit full marks even
    # if the final answer accidentally matches (two errors cancelling
    # out). Step 1 is a straight copy from the printed question, so a
    # mismatch there is never explainable as OCR transcription noise.
    if question:
        student_raw = str(steps[0].get("student_wrote") or "") if isinstance(steps[0], dict) else ""
        if question_label:
            variants = _label_prefix_variants(str(question_label))
            student_clean = _strip_leading_label(student_raw, variants)
        else:
            student_clean = student_raw
        student_norm = _math_normalize(student_clean)
        question_norm = _math_normalize(question)
        if (
            student_norm
            and question_norm
            and student_norm != question_norm
            and student_norm not in question_norm
        ):
            logger.warning(
                "grading: final-answer safety net SKIPPED - step 1 "
                "student_wrote=%r does not match starting expression=%r "
                "(student made a real copy error, not OCR noise; "
                "cancelling-error false positive suppressed).",
                student_clean[:80],
                question[:80],
            )
            return

    # Guard (Qwen-independent): even if ``extracted_steps[0].student_wrote``
    # matches the question, Qwen is known to silently rewrite step 1 so it
    # matches the expected starting expression when the student actually
    # copied it wrong. The raw GLM-OCR transcription is Qwen-independent:
    # if the expected starting expression does NOT appear (normalised) in
    # the transcription, the student wrote something different on the
    # page regardless of what Qwen reported, so the safety net must not
    # fire. This is the bulletproof backstop against Qwen-rewrite
    # cancelling-error cases like Q 0003 ("+y^2" vs "-y^2" sign flip).
    if question and transcription:
        question_norm = _math_normalize(question)
        transcription_norm = _math_normalize(transcription)
        if (
            question_norm
            and transcription_norm
            and question_norm not in transcription_norm
        ):
            logger.warning(
                "grading: final-answer safety net SKIPPED - expected "
                "starting expression %r does NOT appear in raw GLM-OCR "
                "transcription. The student copied the question "
                "incorrectly (real copy error). Qwen may have silently "
                "rewritten step 1 to match the question; the final "
                "answer match is therefore a cancelling-errors artefact "
                "and must not be credited.",
                question[:120],
            )
            return

    # Guard: if any step that Qwen flagged ``is_correct=false`` has a
    # ``student_wrote`` value that exists verbatim in the raw GLM-OCR
    # transcription AND differs from its ``expected`` value under
    # ``_math_normalize``, treat that as a confirmed real student error
    # (not OCR noise) and skip the upgrade. Presence in the
    # transcription confirms the VLM didn't hallucinate the wrong line;
    # the mismatch against ``expected`` confirms it's genuinely wrong
    # math rather than an equivalent alternate form.
    if transcription:
        transcription_norm = _math_normalize(transcription)
        for step in steps:
            if not isinstance(step, dict) or step.get("is_correct", True):
                continue
            wrote_norm = _math_normalize(step.get("student_wrote") or "")
            expected_norm_step = _math_normalize(step.get("expected") or "")
            if (
                wrote_norm
                and expected_norm_step
                and wrote_norm != expected_norm_step
                and wrote_norm in transcription_norm
            ):
                logger.warning(
                    "grading: final-answer safety net SKIPPED - step %s "
                    "student_wrote=%r is confirmed in transcription but "
                    "differs from expected=%r (real student error, not "
                    "OCR noise).",
                    step.get("step_number"),
                    str(step.get("student_wrote") or "")[:80],
                    str(step.get("expected") or "")[:80],
                )
                return

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


def _enforce_step1_starting_expression(
    data: dict[str, Any],
    question: str | None,
    question_label: str | None = None,
    transcription: str | None = None,
) -> None:
    """Force step 1 ``is_correct=false`` when it differs from the question.

    Rationale
    ---------

    The VLM is known to "helpfully" rewrite step 1 so it matches the
    question-text when the student copied the starting expression
    incorrectly. A classic failure mode is a sign flip ("+y^2" vs
    "-y^2") or a coefficient copy error - the student's subsequent
    work flows logically from their own (wrong) starting expression,
    and Qwen credits everything as correct because the alternate
    reading "makes the math work". We already removed the
    prompt-level rule that encouraged this, but a Python-side guard
    here is a deterministic backstop that cannot be negotiated away
    by the LLM.

    The checks (either is sufficient to force step 1 wrong):

    1. **Qwen-reported check**: take the first extracted step's
       ``student_wrote`` field and compare (after label-stripping and
       ``_math_normalize``) against ``question``. If they differ,
       force step 1 wrong. This catches cases where Qwen faithfully
       reports what the student wrote.
    2. **Transcription check** (Qwen-independent): if the expected
       ``question`` does NOT appear (normalised) in the raw GLM-OCR
       ``transcription``, the student wrote something different on
       the page regardless of what Qwen reported in
       ``extracted_steps[0].student_wrote``. Forces step 1 wrong even
       when Qwen silently rewrote the student's copy to match the
       question - the canonical Q 0003 cancelling-errors case.

    No-op when:

    - ``question`` is falsy (manual ``/submit`` path that didn't pass
      the question text).
    - ``extracted_steps`` is missing or empty.
    - Step 1's ``student_wrote`` already matches the question AND the
      expected question appears verbatim in the transcription (or no
      transcription was supplied).
    - Step 1 was already flagged ``is_correct=false`` (nothing to
      override; the VLM agreed with us).

    Parameters
    ----------
    data:
        Parsed grading response dict, mutated in place.
    question:
        Expected starting expression (the original problem text). When
        the question is a sentence like "Simplify (3x^2+2xy-y^2)-..."
        we still compare after normalisation - any leading prose will
        simply never equal ``student_wrote``, in which case we skip
        the override conservatively (we don't want false positives).
    question_label:
        Canonical question label so we can strip "0003." / "Q3" /
        "(3)" prefixes from ``student_wrote`` before comparing.
    transcription:
        Raw GLM-OCR transcription of the student's page. Used for the
        Qwen-independent substring check that defeats the "Qwen
        silently rewrote step 1" failure mode. Pass ``None`` to skip
        the transcription check (legacy behaviour).
    """
    if not question:
        return

    steps = data.get("extracted_steps")
    if not isinstance(steps, list) or not steps:
        return

    first = steps[0]
    if not isinstance(first, dict):
        return

    if not first.get("is_correct", False):
        return  # already flagged wrong - nothing to override

    student_raw = str(first.get("student_wrote") or "")
    if question_label:
        variants = _label_prefix_variants(str(question_label))
        student_clean = _strip_leading_label(student_raw, variants)
    else:
        student_clean = student_raw

    student_norm = _math_normalize(student_clean)
    question_norm = _math_normalize(question)

    if not question_norm:
        return

    # Check 2 (Qwen-independent, runs first): is the expected starting
    # expression actually on the page according to raw GLM-OCR? If not,
    # the student copied it wrong regardless of what Qwen reported for
    # ``student_wrote``. This is the ONLY guard that survives Qwen
    # silently rewriting step 1 to match the question.
    transcription_hit = True
    if transcription:
        transcription_norm = _math_normalize(transcription)
        if transcription_norm and question_norm not in transcription_norm:
            transcription_hit = False
            logger.warning(
                "grading: step 1 transcription check - expected question %r "
                "NOT present in raw GLM-OCR transcription. Qwen may have "
                "rewritten step 1 to match; forcing is_correct=false.",
                question[:120],
            )

    if transcription_hit:
        # Check 1 (Qwen-reported): fall back to comparing Qwen's
        # reported student_wrote against the question. Only meaningful
        # when we haven't already decided to flip via the transcription
        # check.
        if not student_norm:
            return
        # When the question is a bare expression (the common answer-key
        # case), we expect exact equality. When the question is a full
        # sentence ("Simplify ... = ?"), equality will fail naturally -
        # skip so we don't create false positives against sentence prose.
        # Heuristic: if the question contains the student's normalised
        # text as a substring we treat it as "question contains an
        # embedded expression" and require that substring match; else
        # require full equality.
        if student_norm == question_norm:
            return
        if student_norm in question_norm:
            return

    # If the question *contains* the expected expression and the
    # student's differs from the embedded one, we still flag. This
    # catches "Simplify (3x^2+2xy-y^2)-(x^2-5xy-4y^2)" vs the
    # student's "+y^2" copy error.
    logger.warning(
        "grading: step 1 character mismatch - student_wrote=%r does not "
        "match question=%r (transcription_hit=%s); forcing is_correct=false "
        "(was true)",
        student_clean,
        question,
        transcription_hit,
    )
    first["is_correct"] = False
    existing_err = str(first.get("error") or "").strip()
    if transcription_hit:
        new_err = (
            f"student wrote {student_clean!r} but the question starts with "
            f"{question!r}; the student copied the starting expression "
            f"incorrectly (check signs, coefficients, and exponents)"
        )
    else:
        new_err = (
            f"the expected starting expression {question!r} was not "
            f"found in the raw OCR reading of this page; the student "
            f"copied the question incorrectly (check signs, "
            f"coefficients, and exponents)"
        )
    # Preserve any prior error by appending, but put our own first so the
    # root cause reads naturally.
    first["error"] = (
        new_err if not existing_err else f"{new_err}. Prior note: {existing_err}"
    )


# Patterns used by the "Qwen self-contradiction" reconciler below. Both are
# intentionally permissive regexes that match the canonical "everything OK"
# narrative Qwen emits when it *thinks* the student got the question right.
# Keep them anchored to phrase fragments that are NOT ambiguous at the
# sentence level - we do not want to match e.g. "step 3 is correct" (which
# would be a false positive that silently upgrades partially-correct work).
_QWEN_SAYS_ALL_CORRECT_RE = re.compile(
    r"\ball\s+(?:the\s+)?steps?\s+(?:are\s+)?correct\b"
    r"|\ball\s+correct\b"
    r"|\bno\s+errors?\s+(?:were\s+)?(?:detected|found|present)\b"
    r"|\bcorrectly\s+(?:expanded|simplified|solved|computed)\b",
    re.IGNORECASE,
)

_QWEN_EMPTY_SUMMARY_RE = re.compile(
    r"^\s*(?:none|null|no\s+errors?(?:\s+detected|\s+found)?|n/?a)\s*\.?\s*$",
    re.IGNORECASE,
)


def _qwen_says_all_correct(data: dict[str, Any]) -> bool:
    """True when Qwen's top-level narrative clearly claims no step errors.

    Looks at ``feedback`` (the detailed sentence) and ``error_summary``
    (the short one-liner) together. Both must be consistent with
    "all correct" for this to return True - a feedback that says
    "all steps correct" while error_summary specifically names a
    wrong step would NOT match.
    """
    feedback = str(data.get("feedback") or "")
    summary = str(data.get("error_summary") or "")

    feedback_says_correct = bool(_QWEN_SAYS_ALL_CORRECT_RE.search(feedback[:500]))
    summary_says_no_errors = (
        not summary.strip() or bool(_QWEN_EMPTY_SUMMARY_RE.match(summary))
    )
    return feedback_says_correct and summary_says_no_errors


def _reconcile_qwen_self_contradiction(
    data: dict[str, Any],
    *,
    question: str | None,
    expected_final_answer: str | None,
    transcription: str | None,
) -> None:
    """Repair "feedback says all correct but per-step flags disagree".

    Qwen's JSON output is occasionally internally inconsistent: it
    emits one or more ``extracted_steps[].is_correct=false`` flags
    while the top-level ``feedback`` and ``error_summary`` describe
    the work as entirely correct. Empirically the long-form narrative
    text is more reliable than the discrete flags in these cases
    (Qwen's flag generator hallucinates errors that its prose describer
    does not see). We therefore trust the narrative - but only when
    two **Qwen-independent** sanity signals agree with it:

    1. The expected printed ``question`` appears verbatim (normalised)
       in the raw GLM-OCR ``transcription`` - proof that the student
       really did copy the starting expression correctly on the page.
    2. The ``expected_final_answer`` appears verbatim (normalised) in
       the raw GLM-OCR ``transcription`` - proof that the student
       really did reach the right answer on the page.

    When BOTH hold and Qwen's narrative says "all correct", we flip
    every per-step ``is_correct=false`` back to ``true`` so the
    response matches its own narrative. When either signal fails
    we leave the flags alone and let the stricter downstream guards
    (``_enforce_step1_starting_expression``, ``_validate_score``)
    have the final word - this is exactly the case that distinguishes
    Q 0002 (student really correct, safe to trust Qwen's narrative)
    from Q 0003 (real copy error on the page, narrative is itself
    wrong, must not trust it).
    """
    if not _qwen_says_all_correct(data):
        return

    steps = data.get("extracted_steps")
    if not isinstance(steps, list) or not steps:
        return

    wrong_steps = [
        s for s in steps if isinstance(s, dict) and not s.get("is_correct", True)
    ]
    if not wrong_steps:
        return  # already internally consistent; nothing to reconcile

    # Sanity signal 1: the printed question must appear on the page.
    if question and transcription:
        question_norm = _math_normalize(question)
        transcription_norm = _math_normalize(transcription)
        if question_norm and question_norm not in transcription_norm:
            return
    elif not (question and transcription):
        # Without either signal we cannot safely reconcile - defer to
        # the stricter per-step policy.
        return

    # Sanity signal 2: the expected final answer must appear on the page.
    if expected_final_answer and transcription:
        answer_norm = _normalize_final_answer(expected_final_answer)
        transcription_norm = _math_normalize(transcription)
        if answer_norm and answer_norm not in transcription_norm:
            return
    elif not (expected_final_answer and transcription):
        return

    # Both sanity signals pass. Flip all wrong step flags to match
    # Qwen's narrative. Log an audit entry for each flip so we can
    # trace this decision later.
    flipped_audit: list[dict[str, Any]] = []
    for step in wrong_steps:
        flipped_audit.append(
            {
                "step_number": step.get("step_number"),
                "student_wrote": str(step.get("student_wrote") or "")[:120],
                "original_error": step.get("error"),
            }
        )
        step["is_correct"] = True
        step["error"] = None

    data["first_error_step"] = None
    logger.warning(
        "grading: Qwen self-contradiction repaired - top-level feedback "
        "claims all steps correct AND transcription confirms the student "
        "wrote the correct question + final answer on the page, but %d "
        "per-step flag(s) were marked wrong. Flipped those steps to "
        "correct to match Qwen's own narrative. flipped=%r",
        len(flipped_audit),
        flipped_audit,
    )


def _reconcile_top_level_feedback_with_steps(
    data: dict[str, Any],
    expected_final_answer: str | None,
) -> None:
    """Rewrite ``error_summary`` and ``feedback`` to match the step verdict.

    Runs last, after ``_validate_score`` has settled ``is_correct`` and
    ``first_error_step`` from the per-step list. Guarantees the
    user-visible text fields never claim "all correct" while the score
    reflects errors, and vice versa. The score + per-step list are
    always the source of truth at this stage; only the two prose
    fields are adjusted.

    Two cases:

    - **All steps correct**: clear ``error_summary`` and, if the
      feedback narrative does not already read as "all correct",
      replace it with a clean one-sentence statement so the student
      sees a coherent "well done" message rather than whatever
      half-finished text Qwen emitted.
    - **At least one wrong step**: if the top-level narrative still
      claims "all correct" (canonical Qwen self-contradiction), rebuild
      both ``feedback`` and ``error_summary`` from the wrong steps'
      ``error`` fields so the student sees what actually went wrong.
      Narratives that already acknowledge errors (as for Q 0004) are
      left unchanged.
    """
    steps = data.get("extracted_steps") or []
    wrong_steps = [
        s for s in steps if isinstance(s, dict) and not s.get("is_correct", True)
    ]

    feedback = str(data.get("feedback") or "")

    if not wrong_steps:
        # All steps now correct. Make the prose consistent.
        if not _QWEN_SAYS_ALL_CORRECT_RE.search(feedback[:500]):
            original = feedback
            tail = (
                f" ({expected_final_answer})" if expected_final_answer else ""
            )
            data["feedback"] = (
                f"All steps are correct. The student reached the correct "
                f"final answer{tail}."
            )
            logger.info(
                "grading: reconciled feedback for all-correct case "
                "(was=%r now=%r)",
                original[:120],
                str(data["feedback"])[:120],
            )
        data["error_summary"] = None
        data["first_error_step"] = None
        return

    # Wrong steps present. If Qwen's narrative still claims everything
    # is correct, regenerate both text fields from the per-step errors.
    if _qwen_says_all_correct(data):
        parts: list[str] = []
        for ws in wrong_steps:
            sn = ws.get("step_number")
            err = str(ws.get("error") or "").strip()
            if err:
                parts.append(f"Step {sn}: {err}" if sn is not None else err)

        if parts:
            new_feedback = " ".join(parts)[:2000]
            new_summary = parts[0][:400]
        else:
            new_feedback = (
                f"{len(wrong_steps)} step(s) contain errors; please "
                f"review the per-step breakdown."
            )
            new_summary = new_feedback

        logger.warning(
            "grading: reconciled contradictory feedback - Qwen claimed "
            "all correct but %d step(s) are flagged wrong. "
            "Regenerated feedback and error_summary from step-level errors.",
            len(wrong_steps),
        )
        data["feedback"] = new_feedback
        data["error_summary"] = new_summary


def _validate_score(data: dict[str, Any], max_score: int) -> dict[str, Any]:
    """Recompute ``student_score`` deterministically from the step list.

    Policy
    ------

    Qwen is good at judging step-level correctness (``extracted_steps[].
    is_correct``) but historically unreliable at converting that judgement
    into a numeric score: we've seen it output the *count* of correct
    steps as the score (e.g. ``3`` when 3/3 steps are correct and
    ``max_score`` is 5 or 10), output ``max_score`` despite flagging
    intermediate errors, or output ``0`` despite flagging everything
    correct. Instead of reasoning case-by-case about which Qwen mistake
    we're seeing, the Python side simply **recomputes** the score from
    the step list:

        student_score = round(correct_steps / total_steps * max_score)

    with one safety override: if any step is wrong and rounding would
    produce ``max_score``, we cap at ``max_score - 1`` so the numeric
    score can never lie about "all steps correct" when they're not.

    Guarantees
    ----------

    - ``student_score`` is always in ``[0, max_score]``.
    - ``is_correct`` equals ``(total > 0 and wrong == 0)``. The per-step
      list is authoritative; any stale top-level flag Qwen produced is
      overwritten.
    - When all steps are correct, ``student_score == max_score``.
    - When any step is wrong, ``student_score < max_score``.
    - ``first_error_step`` is back-filled from the first
      ``is_correct=false`` step when the model forgot to emit it; or
      cleared to ``None`` when every step is correct.
    - ``score_breakdown`` is populated with ``"X/Y steps correct"``.

    Parameters
    ----------
    data:
        Parsed grading response dict, mutated in place.
    max_score:
        Maximum score used for scaling.

    Returns
    -------
    dict[str, Any]
        The mutated ``data`` dict (for chained-call convenience).
    """
    steps = data.get("extracted_steps")
    if not isinstance(steps, list):
        steps = []

    total = len(steps)
    correct = sum(
        1 for s in steps if isinstance(s, dict) and s.get("is_correct", False)
    )
    wrong = total - correct

    # --- Reconcile top-level ``is_correct`` with the per-step verdict ----
    steps_all_correct = (total > 0 and wrong == 0)
    was_is_correct = bool(data.get("is_correct", False))
    data["is_correct"] = steps_all_correct
    if was_is_correct and not steps_all_correct:
        logger.warning(
            "grading: top-level is_correct=true but %d/%d steps wrong; "
            "forcing is_correct=false",
            wrong,
            total,
        )
    elif not was_is_correct and steps_all_correct:
        logger.info(
            "grading: top-level is_correct=false but all %d steps correct; "
            "forcing is_correct=true",
            total,
        )

    # --- Deterministic score computation ---------------------------------
    reported_score = _clamp_score(data.get("student_score", 0), max_score)

    if total == 0:
        # No steps at all (degenerate response). Fall back to whatever
        # Qwen reported, clamped, but mark the response as wrong so a
        # human reviewer sees it - "no steps extracted" is always a
        # red flag, not a scoring event.
        computed_score = reported_score
        data["is_correct"] = False
        logger.warning(
            "grading: extracted_steps empty; keeping clamped reported_score=%d",
            reported_score,
        )
    else:
        ratio = correct / total
        computed_score = round(ratio * max_score)
        # When any step is wrong, the student cannot have full marks
        # regardless of rounding. This handles e.g. 9/10 correct with
        # max_score=5 -> round(4.5)=5, which would contradict the
        # flagged wrong step.
        if not steps_all_correct and computed_score >= max_score:
            computed_score = max(0, max_score - 1)

    data["student_score"] = computed_score

    if computed_score != reported_score:
        logger.info(
            "grading: student_score recomputed %d -> %d (steps %d/%d correct, "
            "max_score=%d, is_correct=%s)",
            reported_score,
            computed_score,
            correct,
            total,
            max_score,
            data["is_correct"],
        )

    # --- Back-fill first_error_step --------------------------------------
    if wrong == 0:
        data["first_error_step"] = None
    elif data.get("first_error_step") is None:
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
    """Normalise one step dict so Pydantic validation cannot fail on it.

    Parameters
    ----------
    raw_step:
        Raw step dict from the model output.
    fallback_index:
        1-based index used when ``step_number`` is missing or unparseable.

    Returns
    -------
    dict[str, Any]
        Clean step dict with all required fields present and typed correctly.
    """
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
    """Coerce the model's ``extracted_steps`` field into a clean list.

    Parameters
    ----------
    raw_steps:
        Raw ``extracted_steps`` value from the parsed response.

    Returns
    -------
    list[dict[str, Any]]
        List of sanitised step dicts; empty list for non-list inputs.
    """
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
    """Map the model's confidence string onto the ``{high,medium,low}`` enum.

    Parameters
    ----------
    value:
        Raw confidence value from model output.

    Returns
    -------
    str
        One of ``"high"``, ``"medium"``, ``"low"``.
    """
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
    transcription: str,
    question: str,
    answer_key: str,
    max_score: int,
    subject: str,
    *,
    strict_retry: bool,
    question_label: str | None = None,
    other_questions_on_page: list[str] | None = None,
    retrieval_context: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build the Qwen chat-template messages for one grading call.

    Text-only: the student's handwriting has already been transcribed
    by GLM-OCR upstream, so this builds a plain text prompt with the
    transcription inlined between sentinel markers.

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

    Parameters
    ----------
    transcription:
        GLM-OCR output of the student's handwritten page, one line per
        physical line. Whitespace-only lines are tolerated but the
        grader will be told to ignore them.
    question:
        Original question text.
    answer_key:
        Full worked solution (step-by-step).
    max_score:
        Maximum score for this question.
    subject:
        Subject hint string.
    strict_retry:
        When ``True`` append the strict-retry suffix to the user prompt.
    question_label:
        Canonical question label when known, else ``None``.
    other_questions_on_page:
        Other question labels visible on the same page, else ``None``.
    retrieval_context:
        Optional extra RAG chunks (similar problems pulled from Chroma)
        to inline as reference material. Empty or ``None`` disables the
        block entirely.

    Returns
    -------
    list[dict[str, Any]]
        Chat-template messages list ready for ``apply_chat_template``.
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

    # Build the optional retrieval block. Kept blank when no extra
    # chunks were supplied so the prompt stays tight on the common path.
    retrieval_block = ""
    if retrieval_context:
        cleaned_chunks = [c.strip() for c in retrieval_context if c and c.strip()]
        if cleaned_chunks:
            joined = "\n---\n".join(cleaned_chunks)
            retrieval_block = (
                "Additional reference material retrieved from the answer-key "
                "collection (similar problems - use for context only, do NOT "
                "grade against these):\n"
                f"<<<RAG_CONTEXT_BEGIN\n{joined}\nRAG_CONTEXT_END>>>\n"
            )

    # Guard against degenerate transcription (empty string etc.) so the
    # prompt still looks coherent and the grader doesn't see the sentinel
    # markers wrapping nothing.
    body_transcription = (transcription or "").strip() or (
        "(GLM-OCR returned no text - the page may be blank, illegible, "
        "or the OCR engine failed. Emit a single extracted_steps entry "
        'explaining that the transcription was empty.)'
    )

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        transcription=body_transcription,
        question=question,
        answer_key=answer_key,
        max_score=max_score,
        subject=subject,
        label_hint=label_hint,
        retrieval_block=retrieval_block,
    )
    if strict_retry:
        user_prompt = user_prompt + _STRICT_RETRY_SUFFIX

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _run_generation(
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int = _MAX_NEW_TOKENS_PASS,
) -> str:
    """Run one Qwen ``model.generate`` pass on TEXT-only messages.

    Must be called with :data:`_generate_lock` held so two concurrent
    grading requests do not trample each other on the shared GPU.
    Explicitly drops the ``inputs`` tensor and empties the allocator
    cache after decoding so the next request sees a compacted pool.

    ``max_new_tokens`` defaults to the full grading budget; pass a
    tighter value (e.g. 128) for short-form passes like question
    identification.

    This function intentionally does NOT call
    ``qwen_vl_utils.process_vision_info`` or pass ``images=`` /
    ``videos=`` into the processor - image understanding is now done
    upstream by GLM-OCR in :mod:`app.services.ocr_service` and the
    transcription is embedded inside the prompt text.

    Parameters
    ----------
    messages:
        Chat-template message list. Every message's ``content`` must
        be a plain string (no image/video content blocks); the caller
        is responsible for inlining any transcription into the text.
    max_new_tokens:
        Token budget for this generation pass.

    Returns
    -------
    str
        Decoded model output (continuation only, prompt tokens stripped).

    Raises
    ------
    GradingServiceUnavailable
        If ``_device`` is ``None`` (model not loaded or load failed).
    RuntimeError
        CUDA OOM or other torch runtime failures propagate unchanged.
    """
    if _device is None:
        raise GradingServiceUnavailable(
            "Grading model device is not initialised (load failed or CUDA unavailable)."
        )

    import torch

    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    ).to(_device)

    # Bind to None BEFORE generate so the finally-block del is always safe,
    # even if generate() raises before assigning generated_ids.
    generated_ids = None
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
        # VRAM hygiene: release activations + input tensors in a strict
        # order so the allocator can compact before the next request.
        # Each del has its own try/except so one failure cannot skip
        # the others.
        try:
            del inputs
        except Exception:
            pass
        try:
            del generated_ids
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best-effort
            pass


# --------------------------------------------------------------------- orchestration


def _parse_failed_fallback(raw: str, max_score: int) -> dict[str, Any]:
    """Build a safe zero-score envelope when both parse attempts failed.

    The route layer still gets a valid :class:`GradingResponse`-shaped
    dict so the client sees structured 200 JSON; the ``graded_by``
    field is suffixed with ``(parse failed)`` so ops/audit can spot
    these rows quickly.

    Parameters
    ----------
    raw:
        Raw model output that could not be parsed.
    max_score:
        Maximum score (echoed in the response).

    Returns
    -------
    dict[str, Any]
        Zero-score grading response envelope.
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
    file_id: str | None = None,
    retrieval_context: list[str] | None = None,
) -> dict[str, Any]:
    """Blocking implementation of :func:`grade_answer`.

    Flow:

    1. Decode + preprocess the image bytes (``ValueError`` on corrupt
       payloads).
    2. OCR the student's handwriting with GLM-OCR to produce a
       plain-text transcription. This runs BEFORE the GPU lock is
       acquired: the OCR model holds its own weights and a slow OCR
       pass shouldn't keep the Qwen grading lock held.
    3. Optionally expand the RAG context by searching Chroma with the
       transcription (no-op when ``_RAG_EXTRA_TOP_K == 0`` or the
       caller passed ``retrieval_context`` explicitly).
    4. Acquire the GPU lock non-blockingly; fail fast with
       :class:`GradingBusy` if another request is mid-generation.
    5. Run a single Qwen ``model.generate`` pass on the text-only
       prompt. On JSON parse failure retry once with the strict
       suffix. If both attempts fail, return the safe zero-score
       envelope.
    6. Run ``_validate_score`` to enforce clamping, the score=0
       override (majority-steps guard), and ``first_error_step``
       back-fill.
    7. Emit one structured log line with per-request metrics
       (image_size, transcription_len, output_len, parse, retry,
       step tally, score, confidence, safety_net, VRAM, latency).

    Parameters
    ----------
    image_bytes:
        Raw image bytes.
    question:
        Question text.
    answer_key:
        Full worked solution.
    max_score:
        Maximum score.
    subject:
        Subject hint.
    question_label:
        Canonical question label when known.
    expected_final_answer:
        Expected final answer for the safety-net comparison.
    other_questions_on_page:
        Other question labels on the same page.
    file_id:
        When provided, scope the optional secondary RAG retrieval
        to the answer-key file the student is being graded against.
    retrieval_context:
        Pre-computed extra RAG chunks to inline into the prompt.
        When ``None`` and ``_RAG_EXTRA_TOP_K > 0`` we run our own
        retrieval pass keyed off the transcription.

    Returns
    -------
    dict[str, Any]
        Grading response dict matching the ``GradingResponse`` schema.

    Raises
    ------
    GradingServiceUnavailable
        Model not loaded.
    GradingBusy
        GPU lock held by another request.
    ValueError
        Image cannot be decoded.
    RuntimeError
        CUDA OOM or other torch failure.
    """
    if not is_ready():
        raise GradingServiceUnavailable(
            "Grading model is not loaded (see startup logs for details)."
        )

    pil_image = _decode_image(image_bytes)

    # --- OCR stage (GLM-OCR) ----------------------------------------------
    # Runs BEFORE the Qwen lock is taken so a slow OCR pass cannot
    # starve the grading queue.
    t_ocr_start = time.perf_counter()
    transcription = _transcribe_handwriting(pil_image)
    ocr_ms = (time.perf_counter() - t_ocr_start) * 1000.0
    logger.info(
        "grading: GLM-OCR transcription done chars=%d ms=%.1f",
        len(transcription),
        ocr_ms,
    )
    # Dump the transcription itself so accuracy issues can be diagnosed
    # without re-running OCR. 600 chars is enough for a typical 3-step
    # polynomial and keeps the log line manageable. Toggle off via
    # ``GRADING_LOG_TRANSCRIPTION=0`` if this gets too noisy.
    if os.getenv("GRADING_LOG_TRANSCRIPTION", "1") == "1":
        preview = transcription if len(transcription) <= 600 else transcription[:600] + "..."
        logger.info(
            "grading: GLM-OCR transcription text (question_label=%r):\n%s",
            question_label,
            preview,
        )

    # --- RAG expansion (optional) -----------------------------------------
    # Uses the OCR transcription as the retrieval key. Falls back to
    # whatever the caller passed in; skipped silently on any error.
    effective_retrieval = retrieval_context
    if effective_retrieval is None and _RAG_EXTRA_TOP_K > 0:
        effective_retrieval = _retrieve_grading_context(
            transcription,
            file_id=file_id,
            top_k=_RAG_EXTRA_TOP_K,
        )
        logger.debug(
            "grading: RAG retrieval top_k=%d fetched=%d",
            _RAG_EXTRA_TOP_K,
            len(effective_retrieval or []),
        )

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
            transcription, question, answer_key, max_score, subject,
            strict_retry=False,
            question_label=question_label,
            other_questions_on_page=other_questions_on_page,
            retrieval_context=effective_retrieval,
        )
        raw = _run_generation(messages)

        try:
            data = _parse_json_response(raw, required_fields=_GRADING_REQUIRED_FIELDS)
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
                transcription, question, answer_key, max_score, subject,
                strict_retry=True,
                question_label=question_label,
                other_questions_on_page=other_questions_on_page,
                retrieval_context=effective_retrieval,
            )
            raw = _run_generation(messages)
            try:
                data = _parse_json_response(
                    raw, required_fields=_GRADING_REQUIRED_FIELDS
                )
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
        result = _parse_failed_fallback(raw, max_score)
        logger.info(
            "grading: image_size_bytes=%d ocr_chars=%d output_len=%d parse_success=%s "
            "retried=%s rag_chunks=%d correct=0/%d score=0/%d confidence=low "
            "safety_net=False vram_mb=%d ocr_ms=%.1f llm_ms=%.1f",
            len(image_bytes),
            len(transcription),
            len(raw),
            parse_success,
            parse_retried,
            len(effective_retrieval or []),
            0,
            max_score,
            _vram_used_mb(),
            ocr_ms,
            inference_ms,
        )
        return result

    # --- Happy path: repair known failure modes, then validate --------
    # Order matters:
    #   1. Strip question label from step 1 (may flip step 1 correct).
    #   2. Final-answer safety net (may upgrade overall correct + full
    #      marks when the student reached the right answer despite
    #      intermediate OCR noise; now protected by the step-1 copy,
    #      transcription-presence, and transcription-confirmed-wrong-step
    #      guards so cancelling-errors cases like Q 0003 are NOT upgraded).
    #   3. Reconcile Qwen self-contradiction: if Qwen's narrative says
    #      "all correct" but per-step flags disagree AND the raw OCR
    #      transcription confirms the student wrote the correct question
    #      + final answer, trust the narrative and flip the contradictory
    #      flags. Fixes Q 0002-style "score 3/5 but feedback says perfect"
    #      output.
    #   4. Enforce step 1 copy from the question (may force step 1 wrong
    #      when the student copied the starting expression incorrectly;
    #      uses transcription to defeat Qwen-rewrites step-1 edits).
    #   5. _validate_score (clamps, score recomputation, first_error_step
    #      back-fill, score_breakdown).
    #   6. Reconcile top-level feedback + error_summary with the final
    #      step verdict so the user never sees "all correct" when the
    #      score shows errors (or vice versa).
    # All repair helpers are no-ops when their respective hints weren't
    # supplied by the route (the manual /submit path runs with most as
    # None and lands directly at _validate_score).
    _repair_question_label_in_step1(data, question_label)
    _repair_final_answer_match(
        data,
        expected_final_answer,
        max_score,
        transcription=transcription,
        question=question,
        question_label=question_label,
    )
    _reconcile_qwen_self_contradiction(
        data,
        question=question,
        expected_final_answer=expected_final_answer,
        transcription=transcription,
    )
    # Deterministic backstop: if Qwen credited step 1 but its
    # ``student_wrote`` does NOT match the question's starting
    # expression character-for-character (or the expected question
    # isn't present in the raw OCR transcription at all), force step 1
    # wrong. Handles the "student copied +y^2 instead of -y^2 but Qwen
    # helpfully rewrote it to match" failure mode.
    _enforce_step1_starting_expression(
        data, question, question_label, transcription=transcription
    )

    _validate_score(data, max_score)
    _reconcile_top_level_feedback_with_steps(data, expected_final_answer)
    extracted_steps = _sanitize_steps(data.get("extracted_steps"))

    total_steps = len(extracted_steps)
    correct_steps = sum(1 for s in extracted_steps if s.get("is_correct"))
    final_score = _clamp_score(data.get("student_score", 0), max_score)
    confidence_val = _normalize_confidence(data.get("confidence"))
    safety_net_val = bool(data.get("safety_net_engaged", False))

    result = {
        "student_score": final_score,
        "max_score": max_score,
        "extracted_steps": extracted_steps,
        "first_error_step": _coerce_optional_int(data.get("first_error_step")),
        "error_summary": _coerce_optional_str(data.get("error_summary")),
        "method_correct": bool(data.get("method_correct", False)),
        "feedback": str(data.get("feedback", "") or "")[:4000],
        "is_correct": bool(data.get("is_correct", final_score == max_score)),
        "confidence": confidence_val,
        "score_breakdown": str(
            data.get("score_breakdown") or f"{correct_steps}/{total_steps} steps correct"
        ),
        "graded_by": MODEL_DISPLAY_NAME,
        # True iff the final-answer safety net bumped this response to
        # full marks by matching the last line against the expected
        # final answer. Frontends should show a "teacher-review
        # recommended" badge when this is True - the VLM's per-step
        # flags are not fully trusted in that case.
        "safety_net_engaged": safety_net_val,
    }

    logger.info(
        "grading: image_size_bytes=%d ocr_chars=%d output_len=%d parse_success=%s "
        "retried=%s rag_chunks=%d correct=%d/%d score=%d/%d confidence=%s "
        "safety_net=%s vram_mb=%d ocr_ms=%.1f llm_ms=%.1f",
        len(image_bytes),
        len(transcription),
        len(raw),
        parse_success,
        parse_retried,
        len(effective_retrieval or []),
        correct_steps,
        total_steps,
        final_score,
        max_score,
        confidence_val,
        safety_net_val,
        _vram_used_mb(),
        ocr_ms,
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
    file_id: str | None = None,
    retrieval_context: list[str] | None = None,
) -> dict[str, Any]:
    """Grade a single handwritten answer image.

    Offloads the blocking HuggingFace ``generate`` call and the
    GLM-OCR transcription to the default asyncio executor so the
    FastAPI event loop stays responsive. Nothing is written to disk;
    everything is held in memory for the duration of the call.

    Pipeline (per request):

    1. GLM-OCR transcribes the handwritten image to plain text.
    2. (Optional) Chroma + bge-m3 fetches extra similar-problem
       chunks as reference material.
    3. Qwen2.5-VL (text-only) grades the transcription against the
       answer key and produces the JSON envelope.

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
        can tell the LLM to grade only this question's work and ignore
        lines labelled with the other numbers. Prevents cross-question
        contamination of ``extracted_steps`` when one image carries
        multiple student answers. Omit (or pass ``None`` / empty list)
        for single-question grading.
    file_id:
        UUID of the :class:`AnswerKeyFile` we are grading against.
        Used to scope the optional secondary RAG retrieval so it only
        returns chunks from the same answer-key PDF. Safe to omit.
    retrieval_context:
        Pre-fetched extra RAG chunks the caller wants inlined as
        reference material. When ``None`` and
        ``GRADING_RAG_EXTRA_TOP_K > 0``, the grader performs its own
        bge-m3 + Chroma retrieval hop keyed off the OCR'd transcription.

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
    GradingTimeout
        The grading call exceeded ``_GRADING_TIMEOUT_S``. Route maps
        to HTTP 503 with ``Retry-After: 30``.
    ValueError
        The image bytes could not be decoded. Route maps to HTTP 400.
    RuntimeError
        CUDA OOM or other torch failures mid-generation. Route maps
        OOM-ish messages to HTTP 503.
    """
    loop = asyncio.get_running_loop()

    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
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
                file_id,
                retrieval_context,
            ),
            timeout=_GRADING_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        raise GradingTimeout(
            f"Grading timed out after {_GRADING_TIMEOUT_S:.0f}s; retry shortly."
        ) from exc


# --------------------------------------------------------------------- question identification
# A short, low-token Qwen pass that looks at the GLM-OCR transcription
# of a handwritten page and extracts either the question number the
# student wrote or the problem statement they copied. The caller then
# uses those hints to pick the correct ``AnswerKeyItem`` via either
# exact SQL lookup (on ``(file_id, question_no)``) or semantic
# similarity in Chroma.
#
# This is deliberately a separate, tiny generation pass (~128 tokens):
# it runs against text only (the OCR output), so it's much cheaper
# than the full grading pass and gives us a reliable identification
# signal without asking the grading LLM to multiplex "identify + grade".

_IDENTIFY_SYSTEM_PROMPT: str = (
    "You are reading a GLM-OCR transcription of a student's handwritten "
    "math work to identify which question they are answering. Do NOT grade. "
    "Do NOT solve. Output only the requested JSON envelope."
)

_IDENTIFY_USER_PROMPT_TEMPLATE: str = """Below is the OCR transcription of the TOP portion of a student's handwritten
page. Using only what is in the transcription, extract:

1. question_number: if the student wrote a question number at the top
   of the page (e.g. "1.", "Q3", "Question 2", "#5", "2.a") return
   just the number/identifier as a short string. If no question number
   is visible, return null.

2. problem_text: if the student copied the problem statement from the
   exam paper at the top of their page (typically the first 1-3 lines,
   before any working), return it as a single line. If the page only
   contains working/calculation with no restated problem, return null.

3. read_confidence: "high" if the transcription's opening lines are
   clean (no ``?``, no garbled characters); "medium" if partially
   clean; "low" if mostly garbled or empty.

TRANSCRIPTION:
<<<OCR_BEGIN
{transcription}
OCR_END>>>

OUTPUT RULES:
- Output RAW JSON only
- No prose, no markdown fences, no ```json
- Start your response with {{ and end with }}
- Example: {{"question_number": "3", "problem_text": null, "read_confidence": "high"}}"""


def _identify_sync(image_bytes: bytes) -> dict[str, Any]:
    """Blocking implementation of :func:`identify_question`.

    Runs GLM-OCR over the image first, then asks Qwen (text-only) to
    pull the question number / problem preview / confidence out of the
    transcription. Shares the singleton Qwen model +
    :data:`_generate_lock` with the main grading pass. Returns a dict
    with three keys - ``question_number``, ``problem_text``,
    ``read_confidence`` - any of which may be ``None`` / ``"low"`` when
    the transcription is missing or illegible. On JSON parse failure we
    fall back to a zero-signal response rather than raising, so the
    caller can still ask the student to disambiguate via the 422 path.

    Parameters
    ----------
    image_bytes:
        Raw image bytes.

    Returns
    -------
    dict[str, Any]
        Identification result with keys ``question_number``,
        ``problem_text``, ``read_confidence``.

    Raises
    ------
    GradingServiceUnavailable
        Model not loaded.
    GradingBusy
        GPU lock held by another request.
    ValueError
        Image cannot be decoded.
    """
    if not is_ready():
        raise GradingServiceUnavailable(
            "Grading model is not loaded (see startup logs for details)."
        )

    pil_image = _decode_image(image_bytes)

    # GLM-OCR transcribes the page first (outside the Qwen GPU lock).
    # Identify only needs enough to read the question label + any copied
    # problem statement, so we cap the budget lower than the grading
    # transcription to keep this pass fast.
    t_ocr_start = time.perf_counter()
    transcription = _transcribe_handwriting(
        pil_image, max_new_tokens=_OCR_IDENTIFY_MAX_NEW_TOKENS
    )
    ocr_ms = (time.perf_counter() - t_ocr_start) * 1000.0

    if not transcription.strip():
        logger.info(
            "identify: GLM-OCR returned empty transcription ocr_ms=%.1f; "
            "returning zero-signal result",
            ocr_ms,
        )
        return {
            "question_number": None,
            "problem_text": None,
            "read_confidence": "low",
        }

    # Only the top ~15 lines inform the identification; passing the
    # whole page costs tokens and can confuse the LLM with mid-page
    # math that mentions other numbers.
    top_lines = [
        line for line in transcription.splitlines() if line.strip()
    ][:15]
    top_transcription = "\n".join(top_lines)

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
                "content": _IDENTIFY_USER_PROMPT_TEMPLATE.format(
                    transcription=top_transcription,
                ),
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
        "ocr_ms=%.1f llm_ms=%.1f vram_mb=%d",
        qn,
        len(pt or ""),
        rc,
        ocr_ms,
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

    Runs GLM-OCR over the page to produce a plain-text transcription,
    then uses a short Qwen text pass (~128 tokens) to pull the
    question number / problem preview / read confidence out of the
    top of the transcription. Callers feed the result into
    :func:`app.services.question_resolver.resolve_question` to pick
    the matching :class:`AnswerKeyItem`.

    Parameters
    ----------
    image_bytes:
        Raw image bytes.

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
    GradingTimeout
        The identification call exceeded ``_GRADING_TIMEOUT_S``.
    ValueError
        ``image_bytes`` could not be decoded.
    """
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _identify_sync, image_bytes),
            timeout=_GRADING_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        raise GradingTimeout(
            f"Question identification timed out after {_GRADING_TIMEOUT_S:.0f}s; "
            "retry shortly."
        ) from exc


# --------------------------------------------------------------------- multi-question identification
# Sibling of :func:`identify_question` for the multi-question endpoint
# (``/submit-multi-by-image``). Instead of reading just the top of the
# page and returning ONE label, it scans the full GLM-OCR transcription
# and returns a list of distinct question labels (up to MAX) the
# student wrote.
#
# We run this as a single Qwen text pass (still cheap - ~256 tokens)
# rather than N independent "is question X here?" calls. The returned
# list then drives the per-question grade loop in the route layer.

_IDENTIFY_ALL_SYSTEM_PROMPT: str = (
    "You are scanning a GLM-OCR transcription of a student's handwritten "
    "page to list which questions they answered. Do NOT grade. Do NOT solve. "
    "Output only the requested JSON envelope."
)

_IDENTIFY_ALL_USER_PROMPT_TEMPLATE: str = """Below is the OCR transcription of an entire handwritten page. Scan it top to
bottom and list every QUESTION LABEL the student wrote.

A question label is a number or short identifier at the start of a
distinct section of student work, for example: "1.", "Q3", "0002.",
"Question 5:", "(3)".

IMPORTANT - be thorough:
- Scan the ENTIRE transcription top to bottom, including the middle
  and lower sections. Do NOT stop after the first label you find.
- If you see two or more labelled sections, you MUST list all of
  them. Missing a label is a worse error than listing an extra one.
- Treat each distinct number the student wrote as its own entry,
  even if the on-page expressions happen to look similar to examples
  you have seen in instructions.

Return ONE entry per distinct question the student answered on this
page. Return at most 5 entries. Order entries top-to-bottom as they
appear in the transcription. Do NOT return the same question twice.
Do NOT include printed headers or answer-key labels - only include
labels the student wrote in their own handwriting (i.e. that appear
in the transcription).

For each entry, include:
- question_number: the label as written, digits only if it is numeric
  (e.g. "0002", "3", "Q5"). Do NOT include the trailing "." or ")".
- problem_text_preview: up to 60 characters of the first line the
  student wrote for this question - the starting expression they
  copied from the exam. If the student did NOT copy the problem
  statement (started straight with working), set this to null.
- confidence: "high" | "medium" | "low" for how clearly the label
  reads in the transcription (low if the surrounding text is garbled).

TRANSCRIPTION:
<<<OCR_BEGIN
{transcription}
OCR_END>>>

OUTPUT RULES:
- Output RAW JSON only
- No prose, no markdown fences, no ```json
- Start your response with {{ and end with }}
- Shape: {{"questions": [{{"question_number": "...", "problem_text_preview": "..." or null, "confidence": "..."}}, ...]}}

Example valid output (illustrative only - the labels and text below
are placeholders; use what you ACTUALLY see in the transcription):
{{"questions": [
  {{"question_number": "<label_a>", "problem_text_preview": "<first line of a>", "confidence": "high"}},
  {{"question_number": "<label_b>", "problem_text_preview": null, "confidence": "medium"}}
]}}"""


def _sanitize_identification_entry(raw: Any) -> dict[str, Any] | None:
    """Coerce one model-output entry into ``{question_number, problem_text_preview, confidence}``.

    Returns ``None`` for entries that are clearly bogus (no usable
    question number). Never raises.

    Parameters
    ----------
    raw:
        Raw entry dict from the model output.

    Returns
    -------
    dict[str, Any] | None
        Sanitised entry, or ``None`` when the entry has no usable number.
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

    Runs GLM-OCR over the whole page, then asks Qwen (text-only) to
    extract up to ``_IDENTIFY_ALL_MAX_RESULTS`` distinct question
    labels. Dedups by ``question_number`` keeping first occurrence
    (preserves top-to-bottom order).

    Like :func:`_identify_sync` we swallow JSON parse failures rather
    than raising: the route layer can still 422 with "no questions
    detected" and let the client retry manually.

    Parameters
    ----------
    image_bytes:
        Raw image bytes.

    Returns
    -------
    list[dict[str, Any]]
        Identified questions list (possibly empty).

    Raises
    ------
    GradingServiceUnavailable
        Model not loaded.
    GradingBusy
        GPU lock held by another request.
    ValueError
        Image cannot be decoded.
    """
    if not is_ready():
        raise GradingServiceUnavailable(
            "Grading model is not loaded (see startup logs for details)."
        )

    pil_image = _decode_image(image_bytes)

    # identify_all needs the WHOLE page (labels scattered top-to-bottom),
    # so we give it twice the per-identify budget but still keep it well
    # below the grading transcription budget.
    t_ocr_start = time.perf_counter()
    transcription = _transcribe_handwriting(
        pil_image, max_new_tokens=_OCR_IDENTIFY_MAX_NEW_TOKENS * 2
    )
    ocr_ms = (time.perf_counter() - t_ocr_start) * 1000.0

    if not transcription.strip():
        logger.info(
            "identify_all: GLM-OCR returned empty transcription ocr_ms=%.1f; "
            "returning empty list",
            ocr_ms,
        )
        return []

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
                "content": _IDENTIFY_ALL_USER_PROMPT_TEMPLATE.format(
                    transcription=transcription,
                ),
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
    for entry in raw_entries:
        sanitized = _sanitize_identification_entry(entry)
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
        "identify_all: detected=%d (cap=%d) ocr_ms=%.1f llm_ms=%.1f vram_mb=%d "
        "labels=%r",
        len(out),
        _IDENTIFY_ALL_MAX_RESULTS,
        ocr_ms,
        elapsed_ms,
        _vram_used_mb(),
        [e["question_number"] for e in out],
    )
    return out


async def identify_all_questions(image_bytes: bytes) -> list[dict[str, Any]]:
    """List every distinct question the student answered on the page.

    Async wrapper around :func:`_identify_all_sync`. Dispatches onto
    the default executor so the FastAPI event loop stays responsive
    during the GLM-OCR pass + the short Qwen text pass.

    Parameters
    ----------
    image_bytes:
        Raw image bytes.

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
    GradingTimeout
        The identification call exceeded ``_GRADING_TIMEOUT_S``.
    ValueError
        ``image_bytes`` could not be decoded.
    """
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _identify_all_sync, image_bytes),
            timeout=_GRADING_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        raise GradingTimeout(
            f"Multi-question identification timed out after {_GRADING_TIMEOUT_S:.0f}s; "
            "retry shortly."
        ) from exc


# --------------------------------------------------------------------- eager load
# Kick the load off at import time so the first HTTP request doesn't
# pay the 30-60s warm-up cost. If CUDA is missing this leaves the
# singletons at ``None`` and :func:`grade_answer` will raise
# :class:`GradingServiceUnavailable` on first use.
_load_model()
