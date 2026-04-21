"""Unit tests for app.services.grading_service.

All tests mock _model, _processor, and _device so no real GPU is
required. Tests are fully runnable with ``pytest`` on any machine.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(width: int = 10, height: int = 10) -> Image.Image:
    """Return a small solid-colour RGB PIL image for testing."""
    return Image.new("RGB", (width, height), color=(200, 200, 200))


def _valid_grading_response(
    steps: list[dict[str, Any]] | None = None,
    student_score: int = 10,
    is_correct: bool = True,
) -> dict[str, Any]:
    """Return a minimal valid grading response dict."""
    if steps is None:
        steps = [
            {
                "step_number": 1,
                "student_wrote": "x + 1",
                "expected": "x + 1",
                "is_correct": True,
                "error": None,
            }
        ]
    return {
        "extracted_steps": steps,
        "student_score": student_score,
        "is_correct": is_correct,
        "first_error_step": None,
        "error_summary": None,
        "method_correct": True,
        "feedback": "Good work.",
        "confidence": "high",
        "max_score": 10,
    }


# ---------------------------------------------------------------------------
# Imports under test (after PIL is available so the module loads)
# ---------------------------------------------------------------------------

import app.services.grading_service as gs  # noqa: E402


# ---------------------------------------------------------------------------
# 3a – _preprocess_image
# ---------------------------------------------------------------------------


class TestPreprocessImage:
    """_preprocess_image should always return a PIL Image."""

    def test_returns_pil_image_without_cv2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When cv2 is not importable the original image is returned unchanged."""
        import builtins
        real_import = builtins.__import__

        def _block_cv2(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in ("cv2", "numpy"):
                raise ImportError(f"Mocked missing: {name}")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_cv2):
            img = _make_image()
            result = gs._preprocess_image(img)
        assert isinstance(result, Image.Image)

    def test_returns_pil_image_with_cv2(self) -> None:
        """When cv2 IS importable the result is still a PIL Image (or original on error)."""
        pytest.importorskip("cv2")
        img = _make_image(64, 64)
        result = gs._preprocess_image(img)
        assert isinstance(result, Image.Image)

    def test_disabled_by_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GRADING_PREPROCESS_IMAGES=0 bypasses all processing."""
        monkeypatch.setattr(gs, "_GRADING_PREPROCESS", False)
        img = _make_image()
        result = gs._preprocess_image(img)
        assert result is img  # exact same object returned


# ---------------------------------------------------------------------------
# 2f – _parse_json_response required fields
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    """_parse_json_response raises ValueError when required fields are absent."""

    def test_valid_grading_response_passes(self) -> None:
        raw = json.dumps(
            {
                "extracted_steps": [],
                "student_score": 0,
                "is_correct": False,
            }
        )
        data = gs._parse_json_response(raw, required_fields=gs._GRADING_REQUIRED_FIELDS)
        assert data["student_score"] == 0

    def test_missing_all_required_fields_raises(self) -> None:
        raw = json.dumps({"feedback": "nice"})
        with pytest.raises(ValueError, match="missing required fields"):
            gs._parse_json_response(raw, required_fields=gs._GRADING_REQUIRED_FIELDS)

    def test_missing_one_required_field_raises(self) -> None:
        raw = json.dumps({"extracted_steps": [], "student_score": 5})
        with pytest.raises(ValueError, match="is_correct"):
            gs._parse_json_response(raw, required_fields=gs._GRADING_REQUIRED_FIELDS)

    def test_no_required_fields_parameter_skips_check(self) -> None:
        """Identification responses omit the grading keys; no error without required_fields."""
        raw = json.dumps({"question_number": "3", "problem_text": None})
        data = gs._parse_json_response(raw)  # no required_fields
        assert data["question_number"] == "3"

    def test_no_json_object_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="No JSON object"):
            gs._parse_json_response("This is not JSON at all.")

    def test_strips_markdown_fences(self) -> None:
        raw = "```json\n{\"extracted_steps\": [], \"student_score\": 0, \"is_correct\": false}\n```"
        data = gs._parse_json_response(raw, required_fields=gs._GRADING_REQUIRED_FIELDS)
        assert data["student_score"] == 0


# ---------------------------------------------------------------------------
# 2d – _validate_score step_ratio guard
# ---------------------------------------------------------------------------


class TestValidateScore:
    """_validate_score must only override score=0 when step_ratio >= 0.5."""

    def _make_steps(self, total: int, correct: int) -> list[dict[str, Any]]:
        steps = []
        for i in range(total):
            steps.append(
                {
                    "step_number": i + 1,
                    "student_wrote": "x",
                    "expected": "x",
                    "is_correct": i < correct,
                    "error": None,
                }
            )
        return steps

    def test_does_not_override_when_step_ratio_below_0_5(self) -> None:
        """1 correct out of 4 steps (ratio=0.25) → score stays 0."""
        data: dict[str, Any] = {
            "student_score": 0,
            "extracted_steps": self._make_steps(total=4, correct=1),
        }
        gs._validate_score(data, max_score=10)
        assert data["student_score"] == 0, (
            "Score must NOT be overridden when fewer than half of steps are correct"
        )

    def test_does_not_override_when_exactly_one_correct_of_many(self) -> None:
        """1 correct out of 10 steps (ratio=0.1) → score stays 0."""
        data: dict[str, Any] = {
            "student_score": 0,
            "extracted_steps": self._make_steps(total=10, correct=1),
        }
        gs._validate_score(data, max_score=10)
        assert data["student_score"] == 0

    def test_overrides_when_step_ratio_exactly_0_5(self) -> None:
        """2 correct out of 4 steps (ratio=0.5) → score overridden."""
        data: dict[str, Any] = {
            "student_score": 0,
            "extracted_steps": self._make_steps(total=4, correct=2),
        }
        gs._validate_score(data, max_score=10)
        assert data["student_score"] > 0, (
            "Score MUST be overridden when step_ratio >= 0.5"
        )

    def test_overrides_when_step_ratio_above_0_5(self) -> None:
        """3 correct out of 4 steps (ratio=0.75) → score overridden."""
        data: dict[str, Any] = {
            "student_score": 0,
            "extracted_steps": self._make_steps(total=4, correct=3),
        }
        gs._validate_score(data, max_score=10)
        assert data["student_score"] > 0

    def test_does_not_override_positive_score(self) -> None:
        """Non-zero score from the model is not changed by the override guard."""
        data: dict[str, Any] = {
            "student_score": 5,
            "extracted_steps": self._make_steps(total=4, correct=4),
        }
        gs._validate_score(data, max_score=10)
        assert data["student_score"] == 5

    def test_score_breakdown_populated(self) -> None:
        data: dict[str, Any] = {
            "student_score": 8,
            "extracted_steps": self._make_steps(total=3, correct=2),
        }
        gs._validate_score(data, max_score=10)
        assert data["score_breakdown"] == "2/3 steps correct"


# ---------------------------------------------------------------------------
# 2g – _label_prefix_variants no duplicate suffixes
# ---------------------------------------------------------------------------


class TestLabelPrefixVariants:
    """_label_prefix_variants must not produce duplicate entries."""

    def test_no_duplicate_variants(self) -> None:
        for label in ("0003", "1", "Q5", "123"):
            variants = gs._label_prefix_variants(label)
            assert len(variants) == len(set(variants)), (
                f"Duplicates found in variants for label={label!r}: {variants}"
            )

    def test_no_duplicate_space_dash_suffix(self) -> None:
        """Specifically verify the old ' -' duplicate is gone."""
        variants = gs._label_prefix_variants("5")
        space_dash_count = sum(1 for v in variants if v.endswith(" -"))
        assert space_dash_count <= len(
            ["5 -", "Q5 -", "(5) -"]
        ), "Each base prefix should appear at most once with ' -' suffix"
        # More precisely: dedup means each variant appears exactly once
        assert len(variants) == len(set(variants))

    def test_empty_label_returns_empty(self) -> None:
        assert gs._label_prefix_variants("") == []

    def test_zero_padded_label_includes_trimmed(self) -> None:
        variants = gs._label_prefix_variants("0003")
        # Should include both "0003" and "3" forms
        assert "0003" in variants
        assert "3" in variants


# ---------------------------------------------------------------------------
# _strip_leading_label – genuine math coefficients must not be stripped
# ---------------------------------------------------------------------------


class TestStripLeadingLabel:
    """_strip_leading_label must not strip genuine math like '3x^2 + 4'."""

    def test_coefficient_not_stripped(self) -> None:
        """Variant '3' must not strip the '3' from '3x^2 + 4'."""
        text = "3x^2 + 4"
        variants = gs._label_prefix_variants("3")
        result = gs._strip_leading_label(text, variants)
        assert result == text, (
            f"Expected '{text}' to be unchanged but got '{result}'"
        )

    def test_label_with_space_is_stripped(self) -> None:
        """'3. (3x^2+2xy)' has a clear label boundary (space after '.')."""
        text = "3. 3x^2+2xy"
        variants = gs._label_prefix_variants("3")
        result = gs._strip_leading_label(text, variants)
        assert result == "3x^2+2xy"

    def test_zero_padded_label_stripped(self) -> None:
        text = "0003. (3x^2+2xy-y^2)"
        variants = gs._label_prefix_variants("0003")
        result = gs._strip_leading_label(text, variants)
        assert "0003" not in result
        assert "3x^2" in result

    def test_decimal_not_stripped(self) -> None:
        """'3.14' — variant '3.' should NOT match because '1' is alphanumeric."""
        text = "3.14"
        variants = gs._label_prefix_variants("3")
        result = gs._strip_leading_label(text, variants)
        assert result == text


# ---------------------------------------------------------------------------
# 2i – grade_answer raises GradingTimeout on asyncio timeout
# ---------------------------------------------------------------------------


class TestGradeAnswerTimeout:
    """grade_answer must raise GradingTimeout when run_in_executor times out."""

    @pytest.mark.asyncio
    async def test_raises_grading_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Simulate _GRADING_TIMEOUT_S expiring by patching asyncio.wait_for.

        We patch asyncio.wait_for in the grading_service module namespace so
        the inner wait_for call raises asyncio.TimeoutError, which grade_answer
        must convert to GradingTimeout. Using a module-level patch avoids the
        executor thread running and producing an unrelated ValueError from the
        fake image bytes.
        """
        monkeypatch.setattr(gs, "_model", MagicMock())
        monkeypatch.setattr(gs, "_processor", MagicMock())
        monkeypatch.setattr(gs, "_device", "cuda")

        async def _fake_wait_for(fut: Any, timeout: float) -> None:
            # Cancel the future/coroutine if possible, then raise TimeoutError
            if hasattr(fut, "cancel"):
                fut.cancel()
            raise asyncio.TimeoutError

        with patch.object(asyncio, "wait_for", side_effect=_fake_wait_for):
            with pytest.raises(gs.GradingTimeout):
                await gs.grade_answer(
                    image_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 100,
                    question="Solve x+1=3",
                    answer_key="Step 1: x=2",
                    max_score=10,
                )


# ---------------------------------------------------------------------------
# 2e – load_error() exposes failure reason
# ---------------------------------------------------------------------------


class TestLoadError:
    """load_error() must return None on success and a string on failure."""

    def test_returns_none_when_no_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gs, "_load_error", None)
        assert gs.load_error() is None

    def test_returns_string_when_error_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gs, "_load_error", "CUDA not available")
        result = gs.load_error()
        assert isinstance(result, str)
        assert "CUDA" in result


# ---------------------------------------------------------------------------
# 2c – _run_generation raises GradingServiceUnavailable when _device is None
# ---------------------------------------------------------------------------


class TestRunGenerationDeviceGuard:
    """_run_generation must raise GradingServiceUnavailable when _device is None."""

    def test_raises_when_device_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gs, "_device", None)
        with pytest.raises(gs.GradingServiceUnavailable, match="not initialised"):
            gs._run_generation([])


# ---------------------------------------------------------------------------
# Prompt template sanity checks
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    """Sanity checks on the prompt template and rule constants."""

    def test_template_does_not_contain_literal_0003_example(self) -> None:
        """'0003' must not appear as a hardcoded example in the template.

        It was replaced with '<label>' in fix 2h to avoid biasing the model
        toward that specific question number.
        """
        assert "0003" not in gs._USER_PROMPT_TEMPLATE, (
            "Literal '0003' found in _USER_PROMPT_TEMPLATE; "
            "it should have been replaced with '<label>'"
        )

    def test_template_contains_label_placeholder(self) -> None:
        assert "<label>" in gs._USER_PROMPT_TEMPLATE

    def test_rule_constants_are_non_empty(self) -> None:
        for name, const in [
            ("_SYMBOL_RULES", gs._SYMBOL_RULES),
            ("_STRUCTURE_RULES", gs._STRUCTURE_RULES),
            ("_LAYOUT_RULES", gs._LAYOUT_RULES),
            ("_COUNT_VERIFICATION", gs._COUNT_VERIFICATION),
        ]:
            assert isinstance(const, str) and const.strip(), (
                f"{name} must be a non-empty string"
            )

    def test_rule_constants_injected_in_template(self) -> None:
        """Each rule block must appear verbatim in the template."""
        for rule in [
            gs._SYMBOL_RULES,
            gs._STRUCTURE_RULES,
            gs._LAYOUT_RULES,
            gs._COUNT_VERIFICATION,
        ]:
            # Take the first 40 chars as a unique-enough fingerprint
            fragment = rule[:40]
            assert fragment in gs._USER_PROMPT_TEMPLATE, (
                f"Rule fragment {fragment!r} not found in _USER_PROMPT_TEMPLATE"
            )

    def test_template_format_raises_on_bad_key(self) -> None:
        """The template must only have the five expected format slots."""
        with pytest.raises(KeyError):
            gs._USER_PROMPT_TEMPLATE.format(
                question="q", answer_key="a", max_score=10, subject="math"
                # missing label_hint → KeyError
            )

    def test_template_format_succeeds_with_all_keys(self) -> None:
        rendered = gs._USER_PROMPT_TEMPLATE.format(
            question="Solve x",
            answer_key="x=1",
            max_score=10,
            subject="math",
            label_hint="",
        )
        assert "Solve x" in rendered
        assert "x=1" in rendered
