#!/usr/bin/env python3
"""
Language filtering and validation utilities.

Exports:
  - strip_math_and_symbols(text, ...)
  - validate_language(response, reasoning_language, ...)

Goals:
  • Robustly strip LaTeX/math/code and non-linguistic symbols while preserving
    natural-language text across target languages.
  • Validate that a model's response is predominantly in the requested language,
    using a mix of script heuristics (for non-Latin scripts) and Lingua detection.

Notes:
  • For English targets, validation is skipped and returns (1, 1.0, 0, [], []).
  • For non-Latin targets, we apply script-ratio overrides (e.g., Kana for Japanese)
    to reduce confusion between Chinese and Japanese.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

# Public API
__all__ = ["strip_math_and_symbols", "validate_language"]

# =========================
# Regexes (precompiled)
# =========================

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]*`")

# LaTeX math
_MATH_DOLLAR_DBL = re.compile(r"(?<!\\)\$\$(?:\\.|[^$\\])+\$\$", re.DOTALL)
_MATH_DOLLAR_SGL = re.compile(r"(?<!\\)\$(?:\\.|[^$\\])+\$", re.DOTALL)
_MATH_PARENS = re.compile(r"\\\((?:.|\n)*?\\\)")
_MATH_BRACKETS = re.compile(r"\\\[(?:.|\n)*?\\\]")

_ENV_NAMES = [
    "equation", "align", "alignat", "gather", "multline", "flalign",
    "eqnarray", "split", "cases", "matrix", "pmatrix", "bmatrix",
    "vmatrix", "Vmatrix", "smallmatrix", "aligned", "alignedat", "array",
]
_ENV_BLOCK = re.compile(
    r"\\begin\{(" + "|".join(map(re.escape, _ENV_NAMES + [n + "*" for n in _ENV_NAMES])) + r")\}.*?\\end\{\1\}",
    re.DOTALL,
)

_LATEX_TAGLIKE = re.compile(r"\\(?:tag|label|ref|eqref|cite|footnote)\*?\{.*?\}", re.DOTALL)
_LATEX_COMMENTS = re.compile(r"(^|(?<!\\)\s)%(?!\S).*?$", re.MULTILINE)

# Remaining LaTeX commands: optional [..] and one optional {..}
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+(?:\s*\[[^\]]*\])?(?:\s*\{[^{}]*\})?")

# Zero-width/invisible characters
_ZERO_WIDTH = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
_NBSP = "\u00A0"

# Unwrap selected text macros iteratively (handles shallow nesting)
_TEXT_MACROS = (
    "textbf", "textit", "emph", "underline", "texttt", "textrm", "textsf",
    "textsc", "textup", "textmd", "textnormal", "text", "mbox",
)
_UNWRAP_TEXT_MACROS = re.compile(
    r"\\(?:%s)\s*\{((?:[^{}]|\{[^{}]*\})*)\}" % "|".join(map(re.escape, _TEXT_MACROS))
)

# Remove core TeX punctuation/symbols (ASCII core punctuation intentionally excluded)
_TEX_JUNK = re.compile(r"[{}^_~&%$#]")

# =========================
# Unicode helpers
# =========================

def _is_math_alphanumeric(c: str) -> bool:
    cp = ord(c)
    if 0x1D400 <= cp <= 0x1D7FF:  # Mathematical Alphanumeric Symbols
        return True
    if 0x2100 <= cp <= 0x214F:    # Letterlike Symbols
        return True
    return False

def _is_greek(c: str) -> bool:
    cp = ord(c)
    return (0x0370 <= cp <= 0x03FF) or (0x1F00 <= cp <= 0x1FFF)

def _allowed_punct(extra: Iterable[str]) -> set:
    """
    Whitelist of punctuation to preserve (ASCII core punctuation is excluded).
    Pass additional characters via `extra`.
    """
    base = {
        # Typographic quotes/dashes/ellipsis/middots (non-ASCII)
        "’", "‘", "“", "”", "–", "—", "…", "·",
        # CJK punctuation
        "。", "、", "，", "；", "：", "？", "！",
        "「", "」", "『", "』", "〈", "〉", "《", "》", "【", "】", "（", "）", "・",
        # Thai
        "ฯ", "ๆ",
        # Indic (e.g., Devanagari)
        "।", "॥",
    }
    base.update(extra)
    return base

def _keep_char(c: str, *, keep_digits: bool, allowed: set, drop_greek: bool) -> bool:
    # Whitespace is preserved; normalized later.
    if c in ("\n", "\t", " "):
        return True

    cat = unicodedata.category(c)  # Ll, Lu, Lo, Nd, Po, Sm, Sc, Cf, etc.

    # Letters from any script are OK, except math alphabets and (optionally) Greek.
    if cat.startswith("L"):
        if _is_math_alphanumeric(c):
            return False
        if drop_greek and _is_greek(c):
            return False
        return True

    # Digits (optional)
    if keep_digits and cat == "Nd":
        return True

    # Punctuation (only whitelisted)
    if cat.startswith("P") and c in allowed:
        return True

    # Everything else -> drop
    return False

def strip_math_and_symbols(
    text: str,
    *,
    keep_digits: bool = False,
    extra_punct: Iterable[str] = (),
    drop_greek: bool = True,
) -> str:
    """
    Remove LaTeX/math/code and non-linguistic symbols, preserving natural-language text.

    Parameters
    ----------
    text : str
        Input string potentially containing LaTeX/markdown and math.
    keep_digits : bool, default False
        If True, preserve decimal digits.
    extra_punct : Iterable[str], default ()
        Additional punctuation characters to keep.
    drop_greek : bool, default True
        If True, drop Greek letters (helps avoid math noise).

    Returns
    -------
    str
        Cleaned text with paragraphs preserved.
    """
    if not text:
        return ""

    # Normalize; also normalize NBSP and remove zero-widths early
    s = unicodedata.normalize("NFKC", text).replace(_NBSP, " ")
    s = _ZERO_WIDTH.sub("", s)

    # 1) Remove code & LaTeX comments
    s = _CODE_FENCE.sub(" ", s)
    s = _INLINE_CODE.sub(" ", s)
    s = _LATEX_COMMENTS.sub(" ", s)

    # 2) Remove LaTeX math blocks (iterate to exhaustion)
    for _ in range(4):
        s_old = s
        s = _ENV_BLOCK.sub(" ", s)
        s = _MATH_DOLLAR_DBL.sub(" ", s)
        s = _MATH_DOLLAR_SGL.sub(" ", s)
        s = _MATH_PARENS.sub(" ", s)
        s = _MATH_BRACKETS.sub(" ", s)
        s = _LATEX_TAGLIKE.sub(" ", s)
        if s == s_old:
            break

    # 3) Unwrap text macros (\textbf{...} -> ...)
    for _ in range(6):  # catch shallow nesting
        s_new = _UNWRAP_TEXT_MACROS.sub(r"\1", s)
        if s_new == s:
            break
        s = s_new

    # 4) Drop remaining LaTeX commands and TeX junk
    s = _LATEX_COMMAND.sub(" ", s)
    s = _TEX_JUNK.sub(" ", s)

    # 5) Whitespace normalize (keep paragraph breaks)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    # 6) Keep only letters + whitelisted punctuation + (optional) digits
    allowed = _allowed_punct(extra_punct)
    out: List[str] = []
    for ch in s:
        if ch == "\n":
            out.append(ch)
            continue
        if _keep_char(ch, keep_digits=keep_digits, allowed=allowed, drop_greek=drop_greek):
            out.append(ch)
        elif not ch.isspace():  # keep word boundaries
            out.append(" ")

    s = "".join(out)
    # Final whitespace cleanup
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


# =========================
# Language validation (Lingua)
# =========================

from lingua import Language as LinguaLanguage, LanguageDetectorBuilder  # noqa: E402

# Map UI language names → lingua enums
_LINGUA_MAP: Dict[str, LinguaLanguage] = {
    "English":   LinguaLanguage.ENGLISH,
    "French":    LinguaLanguage.FRENCH,
    "Chinese":   LinguaLanguage.CHINESE,
    "Japanese":  LinguaLanguage.JAPANESE,
    "Thai":      LinguaLanguage.THAI,
    "Afrikaans": LinguaLanguage.AFRIKAANS,
    "Latvian":   LinguaLanguage.LATVIAN,
    "Marathi":   LinguaLanguage.MARATHI,
    "Telugu":    LinguaLanguage.TELUGU,
    "Swahili":   LinguaLanguage.SWAHILI,
}

def _to_lingua_enum(language_name: str) -> LinguaLanguage:
    if language_name not in _LINGUA_MAP:
        raise ValueError(f"Language '{language_name}' not supported by Lingua mapping.")
    return _LINGUA_MAP[language_name]

# Cache detectors keyed by the set of languages requested
@lru_cache(maxsize=32)
def _build_detector(langs: Tuple[LinguaLanguage, ...]):
    return LanguageDetectorBuilder.from_languages(*langs).build()


# =========================
# Non-Latin heuristics
# =========================

_MATH_WORDS = re.compile(
    r"\b(?:arcsin|arccos|cosh|sinh|tanh|sin|cos|tan|cot|sec|csc|asin|acos|atan|atan2|log|ln|sqrt|gcd|lcm|mod|deg|rad)\b",
    re.IGNORECASE,
)
_LATIN_SHORT = re.compile(r"\b[a-zA-Z]{1,3}\b")  # standalone 1–3 letter tokens (variables)

NON_LATIN_TARGETS = {"Chinese", "Japanese", "Thai", "Marathi", "Telugu"}
_MIN_KANA_RATIO = 0.05  # 5% kana to count as Japanese in script override

def _scrub_latin_math_tokens(s: str) -> str:
    s = _MATH_WORDS.sub(" ", s)
    s = _LATIN_SHORT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def _script_ratio(s: str, lang: str) -> float:
    """Fraction of letters in the target script for selected non-Latin langs."""
    ranges = {
        "Chinese":  [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],               # CJK Unified + Extension A
        "Japanese": [(0x4E00, 0x9FFF), (0x3040, 0x309F), (0x30A0, 0x30FF)],  # Kanji + Hiragana + Katakana
        "Thai":     [(0x0E00, 0x0E7F)],
        "Marathi":  [(0x0900, 0x097F)],  # Devanagari
        "Telugu":   [(0x0C00, 0x0C7F)],
    }
    rs = ranges.get(lang)
    if not rs:
        return 0.0
    tgt = latin = other = 0
    for ch in s:
        if not unicodedata.category(ch).startswith("L"):
            continue
        cp = ord(ch)
        if any(a <= cp <= b for a, b in rs):
            tgt += 1
        elif "LATIN" in unicodedata.name(ch, ""):
            latin += 1
        else:
            other += 1
    total = tgt + latin + other
    return (tgt / total) if total else 0.0

def _kana_counts(s: str) -> Tuple[int, int]:
    """(#hiragana, #katakana) letters in s."""
    hira = kata = 0
    for ch in s:
        if not unicodedata.category(ch).startswith("L"):
            continue
        cp = ord(ch)
        if 0x3040 <= cp <= 0x309F:
            hira += 1
        elif 0x30A0 <= cp <= 0x30FF:
            kata += 1
    return hira, kata

def _kana_ratio(s: str) -> float:
    hira, kata = _kana_counts(s)
    kana = hira + kata
    letters = sum(1 for ch in s if unicodedata.category(ch).startswith("L"))
    return (kana / letters) if letters else 0.0


# =========================
# Chunking
# =========================

def _chunk_chars(s: str, size: int) -> List[str]:
    """
    Split string into non-overlapping chunks of `size` characters.
    Counts Unicode code points (may split grapheme clusters, which is OK here).
    """
    return [s[i:i + size] for i in range(0, len(s), size) if s[i:i + size]]


# =========================
# Public: validate_language
# =========================

def validate_language(
    response: str,
    reasoning_language: str,
    confidence_threshold: float,
    compliance_threshold: float,
    script_ratio_threshold: float = 0.75,
    chars_per_chunk: int = 128,
) -> Tuple[int, float, int, List[str], List[str]]:
    """
    Validate that `response` is predominantly in `reasoning_language`.

    Procedure (char-based):
      1) strip_math_and_symbols on the entire response.
      2) For non-Latin targets, scrub common Latin math tokens/short vars.
      3) Normalize whitespace to single spaces.
      4) Split into fixed-size character chunks (drop short and duplicate chunks).
      5) For non-Latin targets, apply script-ratio override:
         • Japanese requires ≥5% Kana to avoid misclassifying pure Han text.
         • Chinese rejects Kana-heavy text.
      6) Otherwise, use Lingua over {EN, Target, (ZH if Target≠ZH)}.
         Drop chunks if max confidence < `confidence_threshold`.
      7) compliance = target_chunks / total_chunks.
      8) Return (compliant ∈ {0,1}, compliance_score, kept_chunks, valid_chunks, invalid_chunks).

    Returns
    -------
    compliant : int
        1 if compliance_score ≥ compliance_threshold, else 0.
    compliance_score : float
        Fraction of chunks classified as the target language.
    kept_chunks : int
        Number of chunks considered (after filtering).
    valid_chunks : List[str]
        Chunks counted as target language.
    invalid_chunks : List[str]
        Chunks counted as non-target language.
    """
    if reasoning_language == "English":
        # English is permissive; skip detection for speed.
        return 1, 1.0, 0, [], []

    target_enum = _to_lingua_enum(reasoning_language)

    # Build language set for Lingua
    langs = {LinguaLanguage.ENGLISH, target_enum}
    if target_enum != LinguaLanguage.CHINESE:
        langs.add(LinguaLanguage.CHINESE)
    langs_tuple = tuple(sorted(langs, key=lambda x: x.name))
    detector = _build_detector(langs_tuple)

    if not response or not response.strip():
        return 0, 0.0, 0, [], []

    # Global cleaning
    cleaned_all = strip_math_and_symbols(response)
    if reasoning_language in NON_LATIN_TARGETS:
        cleaned_all = _scrub_latin_math_tokens(cleaned_all)
    cleaned_all = re.sub(r"\s+", " ", cleaned_all).strip()

    # Chunk and deduplicate
    chunks = _chunk_chars(cleaned_all, max(1, int(chars_per_chunk)))
    seen = set()
    total_chunks = 0
    target_chunks = 0
    valid_chunks: List[str] = []
    invalid_chunks: List[str] = []

    for chunk in chunks:
        if len(chunk) < chars_per_chunk:
            continue
        if chunk in seen:
            continue
        seen.add(chunk)

        # Script override for non-Latin targets
        if reasoning_language in NON_LATIN_TARGETS:
            ratio = _script_ratio(chunk, reasoning_language)
            if reasoning_language == "Japanese":
                if ratio >= script_ratio_threshold and _kana_ratio(chunk) >= _MIN_KANA_RATIO:
                    total_chunks += 1
                    target_chunks += 1
                    valid_chunks.append(chunk)
                    continue
            elif reasoning_language == "Chinese":
                if ratio >= script_ratio_threshold and _kana_ratio(chunk) < _MIN_KANA_RATIO:
                    total_chunks += 1
                    target_chunks += 1
                    valid_chunks.append(chunk)
                    continue
            else:
                if ratio >= script_ratio_threshold:
                    total_chunks += 1
                    target_chunks += 1
                    valid_chunks.append(chunk)
                    continue

        # Lingua decision (EN / ZH / Target)
        confs = detector.compute_language_confidence_values(chunk)
        scores = {c.language: c.value for c in confs}

        tgt_score = scores.get(target_enum, 0.0)
        en_score = scores.get(LinguaLanguage.ENGLISH, 0.0)
        zh_score = scores.get(LinguaLanguage.CHINESE, 0.0) if LinguaLanguage.CHINESE in langs else 0.0

        # Drop low-confidence chunks
        if max(tgt_score, en_score, zh_score) < confidence_threshold:
            continue

        total_chunks += 1
        if tgt_score > max(en_score, zh_score):  # strict win
            target_chunks += 1
            valid_chunks.append(chunk)
        else:
            invalid_chunks.append(chunk)

    compliance_score = (target_chunks / total_chunks) if total_chunks else 0.0
    compliant = 1 if compliance_score >= compliance_threshold else 0
    return compliant, compliance_score, total_chunks, valid_chunks, invalid_chunks