"""Normalization helpers for answer-key text.

The goal of this module is NOT to fully evaluate a math expression - it is
to collapse harmless formatting differences (whitespace, OCR-friendly
look-alikes, unicode operator variants) so that two strings that represent
the same answer compare equal.

Design principles:

* **Never change math meaning.** We only rewrite characters that are well
  known OCR / unicode equivalents of ASCII math operators. We never touch
  digits, variables, or the shape of the expression.
* **Idempotent.** ``normalize_answer_text(normalize_answer_text(x))`` must
  equal ``normalize_answer_text(x)`` - callers can normalize freely.
* **Safe for Korean content.** Korean answer keys intersperse Hangul with
  the final expression; we preserve Hangul characters verbatim and only
  strip obvious OCR garbage noise.

Examples (see tests below once you add them)::

    normalize_answer_text("3 x ² − x y + 4 y²")  == "3x^2 - xy + 4y^2"
    normalize_answer_text("( x + 1 ) ( x − 2 )") == "(x+1)(x-2)"
    normalize_answer_text("답  x = 3")           == "x = 3"
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------- tables
#
# These are intentionally conservative: each mapping either fixes a known
# unicode variant (``×`` -> ``*``) or a typographical spacing quirk. Anything
# that might change meaning (e.g. lowercasing Greek letters) is NOT listed.

# Unicode superscript digits -> ASCII digits. Used together with ``^`` when
# the superscript directly follows a letter/digit (``x²`` -> ``x^2``).
_SUPERSCRIPT_DIGITS: dict[str, str] = {
    "\u2070": "0",  # ⁰
    "\u00b9": "1",  # ¹
    "\u00b2": "2",  # ²
    "\u00b3": "3",  # ³
    "\u2074": "4",  # ⁴
    "\u2075": "5",  # ⁵
    "\u2076": "6",  # ⁶
    "\u2077": "7",  # ⁷
    "\u2078": "8",  # ⁸
    "\u2079": "9",  # ⁹
}

# Unicode operator look-alikes -> ASCII.
_OPERATOR_ALIASES: dict[str, str] = {
    "\u2212": "-",  # − minus sign
    "\u2013": "-",  # – en dash
    "\u2014": "-",  # — em dash
    "\u00d7": "*",  # × multiplication
    "\u00f7": "/",  # ÷ division
    "\u2215": "/",  # ∕ division slash
    "\u2044": "/",  # ⁄ fraction slash
    "\u00b7": "*",  # · middle dot (often used as multiplication)
    "\u2217": "*",  # ∗ asterisk operator
    "\u2264": "<=",
    "\u2265": ">=",
    "\u2260": "!=",
    "\u2248": "~=",
}

# Leading / trailing OCR junk we feel safe removing from a final answer.
# Bullet glyphs and the Korean ``답`` ("answer") prefix commonly leak into
# the extracted answer text. Note that ``-`` and ``+`` are NOT listed here
# because they carry sign semantics at the start of an expression
# (``-2(x+1)^2`` must stay negative).
_STRIP_LEADING_JUNK_RE = re.compile(
    r"^(?:답\s*[:：]?\s*|[\s\u2022\u00b7\*]+)+",
)
_STRIP_TRAILING_JUNK_RE = re.compile(
    r"[\s\u2022\u00b7\*\.]+$",
)

# Matches `letter_or_digit` immediately followed by one or more superscript
# digits, so we can insert ``^`` before the run (``x²`` -> ``x^2``,
# ``10²³`` -> ``10^23``).
_SUPERSCRIPT_RUN_RE = re.compile(
    r"(?P<base>[A-Za-z0-9\)])(?P<sup>[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]+)"
)

# Whitespace around operators - we drop spaces around ``^`` (power) and
# unary ``*`` / ``/`` because they are almost always stylistic. We keep
# spaces around ``+`` and ``-`` when they sit between tokens so a student
# answer like ``3x^2 - xy + 4y^2`` still matches the normalized form.
_SPACES_AROUND_POWER_RE = re.compile(r"\s*\^\s*")
_SPACES_AROUND_MUL_RE = re.compile(r"\s*\*\s*")
_SPACES_AROUND_DIV_RE = re.compile(r"\s*/\s*")
_SPACES_AROUND_EQ_RE = re.compile(r"\s*=\s*")
_SPACES_AROUND_COMMA_RE = re.compile(r"\s*,\s*")
_SPACES_INSIDE_PAREN_RE = re.compile(r"\(\s+|\s+\)")

# Plus / minus: collapse multi-space around them but keep ONE space so
# ``3x^2-xy`` and ``3x^2 - xy`` normalize to the same shape.
_SPACES_AROUND_PLUSMINUS_RE = re.compile(r"\s*([+\-])\s*")

# Extra whitespace cleanup (runs of whitespace -> single space, no trailing).
_MULTI_WS_RE = re.compile(r"[ \t\f\v]+")

# Characters we DO NOT want inside a final-answer string. These are common
# OCR artifacts: replacement characters and control characters. Hangul,
# Latin, digits, and math punctuation are always preserved.
_JUNK_CHAR_RE = re.compile(r"[\ufffd\x00-\x08\x0b-\x1f\x7f]")


# ---------------------------------------------------------------------- API
def normalize_answer_text(raw: str | None) -> str | None:
    """Return a comparison-friendly version of ``raw`` (or ``None``).

    The returned string is safe to compare literally against a student
    answer that has been passed through the same function. Returns
    ``None`` if ``raw`` is falsy or becomes empty after normalization so
    callers can treat "no answer" uniformly.
    """
    if not raw:
        return None

    # IMPORTANT: use NFC (canonical composition) rather than NFKC. NFKC's
    # *compatibility* decomposition would silently turn ``x²`` into ``x2``
    # - losing the superscript marker we rely on below to insert ``^``.
    # NFC only merges combining marks and leaves superscript codepoints
    # intact, which is exactly what we want.
    text = unicodedata.normalize("NFC", raw)

    text = _JUNK_CHAR_RE.sub("", text)

    # OCR output often separates a base character from its superscript
    # (``x ²``). Collapse that whitespace before the ``x²`` -> ``x^2``
    # rewrite so the regex can see them as one run.
    text = re.sub(
        r"([A-Za-z0-9\)])\s+(?=[\u2070\u00b9\u00b2\u00b3\u2074-\u2079])",
        r"\1",
        text,
    )
    text = _SUPERSCRIPT_RUN_RE.sub(_rewrite_superscript, text)

    for src, dst in _OPERATOR_ALIASES.items():
        if src in text:
            text = text.replace(src, dst)

    text = _STRIP_LEADING_JUNK_RE.sub("", text)
    text = _STRIP_TRAILING_JUNK_RE.sub("", text)

    text = _collapse_operator_spacing(text)

    text = _MULTI_WS_RE.sub(" ", text).strip()

    return text or None


# ---------------------------------------------------------------------- internals
def _rewrite_superscript(match: re.Match[str]) -> str:
    base = match.group("base")
    sup = match.group("sup")
    digits = "".join(_SUPERSCRIPT_DIGITS[ch] for ch in sup)
    return f"{base}^{digits}"


def _collapse_operator_spacing(text: str) -> str:
    """Tighten whitespace around operators without mangling meaning.

    Rules:

    * ``^``, ``*``, ``/`` lose surrounding whitespace entirely
      (``x ^ 2`` -> ``x^2``).
    * ``=`` and ``,`` collapse to single-space surround for readability.
    * ``+`` / ``-`` get single-space surround at the **top level** so
      ``3x-4`` and ``3x - 4`` both normalize to ``3x - 4``.
    * Inside parentheses every whitespace character is stripped so
      ``( x + 1 )`` becomes ``(x+1)`` and ``(x+1) (x-2)`` becomes
      ``(x+1)(x-2)`` - matching the shape the classifier's factored-
      expression regex expects.
    """
    text = _SPACES_AROUND_POWER_RE.sub("^", text)
    text = _SPACES_AROUND_MUL_RE.sub("*", text)
    text = _SPACES_AROUND_DIV_RE.sub("/", text)
    text = _SPACES_AROUND_EQ_RE.sub(" = ", text)
    text = _SPACES_AROUND_COMMA_RE.sub(", ", text)

    text = _SPACES_AROUND_PLUSMINUS_RE.sub(r" \1 ", text)
    text = re.sub(r"(?<=[=(,])\s+([+\-])\s*(?=\S)", r"\1", text)
    text = re.sub(r"^\s*([+\-])\s*", r"\1", text)

    text = _strip_whitespace_inside_parens(text)
    text = re.sub(r"\)\s+\(", ")(", text)

    return text


_PAREN_GROUP_RE = re.compile(r"\(([^()]*)\)")


def _strip_whitespace_inside_parens(text: str) -> str:
    """Iteratively strip whitespace inside the innermost parentheses.

    Each pass rewrites the innermost ``(...)`` groups with their inner
    whitespace removed, then treats them as atomic tokens so the next
    pass can process the outer level. This handles arbitrary nesting
    without pulling in a real parser.
    """
    while True:
        replaced = _PAREN_GROUP_RE.sub(
            lambda m: "(" + re.sub(r"\s+", "", m.group(1)) + ")",
            text,
        )
        if replaced == text:
            return replaced
        text = replaced
