"""Stage 0 — input normalisation and de-obfuscation.

Attackers rarely send ``ignore previous instructions`` in clear text. This
module reverses the cheap evasion tricks *before* any detector runs and exposes
every recovered representation as a :class:`~app.core.base.TextView`:

* invisible / zero-width characters, bidi overrides, stray control characters
* Unicode tag characters and variation selectors (ASCII / emoji smuggling)
* mixed-script homoglyphs (Cyrillic or Greek letters inside Latin words)
* Base64, hex and ROT13 payloads, decoded recursively up to a bounded depth
* leetspeak and letter-spacing (``i-g-n-o-r-e``)

Detectors then scan *all* views, so a payload hidden in Base64 is judged by the
same signatures and the same classifier as plain text.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.config import NormalizerSettings
from app.core.base import DetectionStage, PipelineContext, StageOutcome, TextView
from app.models.schemas import Finding, Severity

CATEGORY = "obfuscation"

# --------------------------------------------------------------------------- hidden characters

_ZERO_WIDTH: frozenset[int] = frozenset(
    {
        0x00AD, 0x034F, 0x061C, 0x115F, 0x1160, 0x17B4, 0x17B5, 0x180E,
        0x200B, 0x200C, 0x200D, 0x200E, 0x200F,
        0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0x3164, 0xFEFF, 0xFFA0,
    }
)  # fmt: skip
_BIDI_CONTROLS: frozenset[int] = frozenset(
    {0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069}
)
_ZWJ = 0x200D
_EMOJI_SELECTOR = 0xFE0F

# --------------------------------------------------------------------------- homoglyphs

_CONFUSABLES: dict[str, str] = {
    # Cyrillic -> Latin
    "а": "a", "в": "b", "с": "c", "ԁ": "d", "е": "e", "һ": "h", "і": "i", "ј": "j",
    "к": "k", "м": "m", "н": "h", "о": "o", "р": "p", "ԛ": "q", "ѕ": "s", "т": "t",
    "у": "y", "ѵ": "v", "ԝ": "w", "х": "x",
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "Ј": "J", "К": "K",
    "М": "M", "О": "O", "Р": "P", "Ѕ": "S", "Т": "T", "У": "Y", "Х": "X",
    # Greek -> Latin
    "α": "a", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t",
    "υ": "u", "χ": "x",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Latin look-alikes outside ASCII
    "ı": "i", "ɑ": "a", "ɡ": "g", "ǀ": "l",
}  # fmt: skip

_LETTER_RUN = re.compile(r"[^\W\d_]+")

# --------------------------------------------------------------------------- whitespace / spacing

_HORIZONTAL_WS = re.compile(r"[^\S\n]+")
_NEWLINE_PADDING = re.compile(r" *\n *")
_NEWLINE_RUN = re.compile(r"\n{3,}")
_SPACED_LETTERS = re.compile(r"(?<![A-Za-z])(?:[A-Za-z][ ._\-*|]){2,}[A-Za-z](?![A-Za-z])")
_SPACING_SEPARATORS = re.compile(r"[ ._\-*|]")

# --------------------------------------------------------------------------- leetspeak

_LEET_TOKEN = re.compile(r"[A-Za-z0-9@$]{3,}")
_LEET_CHARS = frozenset("0134578@$")
_LEET_BASE = {"0": "o", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"}
_LEET_AS_I = str.maketrans({**_LEET_BASE, "1": "i"})
_LEET_AS_L = str.maketrans({**_LEET_BASE, "1": "l"})

# --------------------------------------------------------------------------- encodings

_BASE64_CANDIDATE = re.compile(
    r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/_-]{%d,}={0,2}(?![A-Za-z0-9+/=_-])"
)
_HEX_CONTIGUOUS = re.compile(r"(?<![0-9A-Za-z])(?:0x)?((?:[0-9A-Fa-f]{2}){8,})(?![0-9A-Za-z])")
_HEX_SEPARATED = re.compile(
    r"(?<![0-9A-Za-z])((?:[0-9A-Fa-f]{2}[ :,]){7,}[0-9A-Fa-f]{2})(?![0-9A-Za-z])"
)
_HEX_ESCAPED = re.compile(r"((?:\\x[0-9A-Fa-f]{2}){6,})")
_NON_HEX = re.compile(r"[^0-9A-Fa-f]")

_ROT13_VOCABULARY: frozenset[str] = frozenset(
    [
        "the",
        "and",
        "you",
        "your",
        "all",
        "are",
        "now",
        "not",
        "must",
        "will",
        "should",
        "that",
        "this",
        "with",
        "from",
        "have",
        "what",
        "ignore",
        "disregard",
        "forget",
        "previous",
        "prior",
        "above",
        "instructions",
        "instruction",
        "rules",
        "system",
        "prompt",
        "reveal",
        "show",
        "tell",
        "print",
        "output",
        "repeat",
        "everything",
        "secret",
        "password",
        "hidden",
        "confidential",
        "admin",
        "developer",
        "mode",
        "override",
        "bypass",
        "restrictions",
        "follow",
        "new",
    ]
)
_ROT13_MIN_HITS = 3
_WORD = re.compile(r"[a-z]{3,}")


@dataclass(slots=True)
class _HiddenStats:
    invisible: int = 0
    bidi: int = 0
    tag_chars: int = 0
    variation_selectors: int = 0
    control: int = 0
    tag_payload: str = ""
    selector_payload: str = ""


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Everything the normaliser learned about a text."""

    canonical: str
    views: tuple[TextView, ...]
    findings: tuple[Finding, ...]
    decoded_sources: frozenset[str]
    stats: dict[str, int]


@dataclass(slots=True)
class _Collector:
    """Accumulates views while enforcing the configured upper bound."""

    max_views: int
    views: list[TextView] = field(default_factory=list)
    decoded_sources: set[str] = field(default_factory=set)
    encodings: Counter[str] = field(default_factory=Counter)
    max_depth: int = 0
    _seen: set[str] = field(default_factory=set)

    @property
    def full(self) -> bool:
        return len(self.views) >= self.max_views

    def add(self, view: TextView) -> bool:
        if self.full or not view.text or view.text in self._seen:
            return False
        self._seen.add(view.text)
        self.views.append(view)
        return True


class TextNormalizer:
    """Pure text canonicaliser and de-obfuscator (no pipeline state, reusable for outputs)."""

    def __init__(self, settings: NormalizerSettings | None = None) -> None:
        self._settings = settings or NormalizerSettings()
        self._base64 = re.compile(_BASE64_CANDIDATE.pattern % self._settings.min_encoded_length)

    # ------------------------------------------------------------------ public API

    @staticmethod
    def strip_invisible(text: str) -> str:
        """Remove invisible code points without otherwise changing the text."""
        return _strip_hidden(text)[0]

    def normalize(self, text: str) -> NormalizationResult:
        stripped, hidden = _strip_hidden(text)
        folded, homoglyph_tokens = _fold_mixed_script(unicodedata.normalize("NFKC", stripped))
        pre_whitespace = folded.replace("\r\n", "\n").replace("\r", "\n")
        canonical = _collapse_whitespace(pre_whitespace)

        out = _Collector(max_views=self._settings.max_views)
        out.add(TextView(name="canonical", text=canonical))

        for name, payload in (
            ("unicode-tags", hidden.tag_payload),
            ("variation-selectors", hidden.selector_payload),
        ):
            if out.add(TextView(name=name, text=_collapse_whitespace(payload), obfuscated=True)):
                self._decode_payloads(payload, depth=0, chain=name, out=out)

        self._decode_payloads(canonical, depth=0, chain="", out=out)

        despaced = _collapse_whitespace(_SPACED_LETTERS.sub(_join_spaced, pre_whitespace))
        out.add(TextView(name="despaced", text=despaced, obfuscated=True))
        for name, table in (("leet", _LEET_AS_I), ("leet-l", _LEET_AS_L)):
            out.add(TextView(name=name, text=_deleet(canonical, table), obfuscated=True))

        skeleton = "".join(_CONFUSABLES.get(ch, ch) for ch in canonical)
        out.add(TextView(name="skeleton", text=skeleton, obfuscated=True, speculative=True))

        rot13_detected = self._add_rot13(canonical, out)

        stats = {
            "invisible_chars": hidden.invisible,
            "bidi_controls": hidden.bidi,
            "tag_chars": hidden.tag_chars,
            "variation_selectors": hidden.variation_selectors,
            "control_chars": hidden.control,
            "homoglyph_tokens": homoglyph_tokens,
            "base64_payloads": out.encodings["base64"],
            "hex_payloads": out.encodings["hex"],
            "rot13_detected": int(rot13_detected),
            "decode_depth": out.max_depth,
            "views": len(out.views),
        }
        return NormalizationResult(
            canonical=canonical,
            views=tuple(out.views),
            findings=tuple(_build_findings(stats, hidden)),
            decoded_sources=frozenset(out.decoded_sources),
            stats=stats,
        )

    # ------------------------------------------------------------------ encodings

    def _decode_payloads(self, text: str, *, depth: int, chain: str, out: _Collector) -> None:
        """Decode Base64 / hex payloads found in ``text``, recursing into the results."""
        if depth >= self._settings.max_decode_depth:
            return
        for encoding, source, decoded in self._find_encoded(text):
            if out.full:
                return
            name = f"{chain}>{encoding}" if chain else encoding
            clean = _collapse_whitespace(_strip_hidden(decoded)[0])
            clean = clean[: self._settings.max_decoded_chars]
            if not out.add(TextView(name=name, text=clean, obfuscated=True)):
                continue
            out.decoded_sources.add(source)
            out.encodings[encoding] += 1
            out.max_depth = max(out.max_depth, depth + 1)
            self._decode_payloads(clean, depth=depth + 1, chain=name, out=out)

    def _find_encoded(self, text: str) -> Iterator[tuple[str, str, str]]:
        """Yield ``(encoding, source_substring, decoded_text)`` for every decodable payload."""
        hex_spans: list[tuple[int, int]] = []
        for pattern in (_HEX_CONTIGUOUS, _HEX_SEPARATED, _HEX_ESCAPED):
            for match in pattern.finditer(text):
                decoded = _try_hex(match.group(1))
                if decoded is not None:
                    hex_spans.append(match.span())
                    yield "hex", match.group(0), decoded
        for match in self._base64.finditer(text):
            if any(start <= match.start() < end for start, end in hex_spans):
                continue
            decoded = _try_base64(match.group(0))
            if decoded is not None:
                yield "base64", match.group(0), decoded

    def _add_rot13(self, canonical: str, out: _Collector) -> bool:
        """Add the ROT13 view; return ``True`` when the text really looks ROT13-encoded.

        ROT13 has no syntactic fingerprint, so the view is always offered to the
        signature engine (a benign text never matches an attack signature after
        rotation). It only counts as *detected* obfuscation, and is only shown
        to the classifier, when the rotated text contains real English words.
        """
        if sum(ch.isalpha() for ch in canonical) < 12:
            return False
        rotated = codecs.decode(canonical, "rot_13")
        hits = sum(1 for word in _WORD.findall(rotated.lower()) if word in _ROT13_VOCABULARY)
        detected = hits >= _ROT13_MIN_HITS
        out.add(TextView(name="rot13", text=rotated, obfuscated=True, speculative=not detected))
        return detected


class NormalizerStage(DetectionStage):
    """Pipeline stage that canonicalises the prompt and publishes the decoded views."""

    name = "normalizer"

    def __init__(self, normalizer: TextNormalizer, *, weight: float, enabled: bool = True) -> None:
        super().__init__(weight=weight, enabled=enabled)
        self._normalizer = normalizer

    def _analyze(self, ctx: PipelineContext) -> StageOutcome:
        result = self._normalizer.normalize(ctx.raw_text)
        ctx.canonical = result.canonical
        ctx.views = list(result.views)
        ctx.decoded_sources = set(result.decoded_sources)
        return StageOutcome(findings=list(result.findings), details=dict(result.stats))


# --------------------------------------------------------------------------- helpers


def _selector_byte(codepoint: int) -> int | None:
    """Map a variation selector to the byte it encodes in 'emoji smuggling' payloads."""
    if 0xFE00 <= codepoint <= 0xFE0F:
        return codepoint - 0xFE00
    if 0xE0100 <= codepoint <= 0xE01EF:
        return codepoint - 0xE0100 + 16
    return None


def _strip_hidden(text: str) -> tuple[str, _HiddenStats]:
    """Remove invisible code points, recovering any payload smuggled inside them."""
    stats = _HiddenStats()
    kept: list[str] = []
    tag_chars: list[str] = []
    selector_bytes = bytearray()
    emoji_context = False

    for ch in text:
        cp = ord(ch)
        if 0xE0000 <= cp <= 0xE007F:
            stats.tag_chars += 1
            if 0xE0020 <= cp <= 0xE007E:
                tag_chars.append(chr(cp - 0xE0000))
            continue
        if cp in _BIDI_CONTROLS:
            stats.bidi += 1
            continue
        selector = _selector_byte(cp)
        if selector is not None:
            if cp == _EMOJI_SELECTOR and emoji_context:
                kept.append(ch)  # ordinary emoji presentation selector
                continue
            stats.variation_selectors += 1
            selector_bytes.append(selector)
            continue
        if cp == _ZWJ and emoji_context:
            kept.append(ch)  # zero-width joiner inside an emoji sequence
            continue
        category = unicodedata.category(ch)
        if cp in _ZERO_WIDTH or category == "Cf":
            stats.invisible += 1
            continue
        if category == "Cc" and ch not in "\n\r\t":
            stats.control += 1
            continue
        kept.append(ch)
        emoji_context = cp >= 0x1F000 or category in ("So", "Sk")

    if len(tag_chars) >= 3:
        stats.tag_payload = "".join(tag_chars)
    if len(selector_bytes) >= 4:
        stats.selector_payload = _decode_text(bytes(selector_bytes)) or ""
    return "".join(kept), stats


def _fold_mixed_script(text: str) -> tuple[str, int]:
    """Replace look-alike letters inside words that also contain Latin letters.

    Only *mixed-script* tokens are folded and counted: a fully Cyrillic or Greek
    word is legitimate text, while ``іgnоrе`` (Cyrillic і/о/е among Latin g/n/r)
    has no innocent explanation.
    """
    folded_tokens = 0

    def fold(match: re.Match[str]) -> str:
        nonlocal folded_tokens
        token = match.group(0)
        if not any(ch in _CONFUSABLES for ch in token):
            return token
        if not any("a" <= ch.lower() <= "z" for ch in token):
            return token
        folded_tokens += 1
        return "".join(_CONFUSABLES.get(ch, ch) for ch in token)

    return _LETTER_RUN.sub(fold, text), folded_tokens


def _collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HORIZONTAL_WS.sub(" ", text)
    text = _NEWLINE_PADDING.sub("\n", text)
    return _NEWLINE_RUN.sub("\n\n", text).strip()


def _join_spaced(match: re.Match[str]) -> str:
    return _SPACING_SEPARATORS.sub("", match.group(0))


def _deleet(text: str, table: dict[int, str]) -> str:
    """Translate leetspeak inside tokens that mix letters with digit/symbol substitutes."""

    def translate(match: re.Match[str]) -> str:
        token = match.group(0)
        letters = sum(ch.isalpha() for ch in token)
        if letters < 2 or letters == len(token):
            return token
        if not any(ch in _LEET_CHARS for ch in token):
            return token
        return token.translate(table)

    return _LEET_TOKEN.sub(translate, text)


def _looks_like_text(text: str) -> bool:
    """Heuristic: is a decoded byte string human-readable text rather than noise?"""
    if len(text) < 6:
        return False
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\t")
    if printable / len(text) < 0.95:
        return False
    alpha_or_space = sum(1 for ch in text if ch.isalpha() or ch.isspace())
    return alpha_or_space / len(text) >= 0.6


def _decode_text(raw: bytes) -> str | None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text if _looks_like_text(text) else None


def _try_base64(token: str) -> str | None:
    body = token.rstrip("=")
    if len(body) % 4 == 1:
        return None
    standard = body.replace("-", "+").replace("_", "/")
    padded = standard + "=" * (-len(standard) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None
    return _decode_text(raw)


def _try_hex(token: str) -> str | None:
    digits = _NON_HEX.sub("", token.replace("\\x", ""))
    if len(digits) % 2:
        return None
    try:
        raw = bytes.fromhex(digits)
    except ValueError:
        return None
    return _decode_text(raw)


def _finding(rule_id: str, severity: Severity, score: float, description: str) -> Finding:
    return Finding(
        rule_id=rule_id,
        category=CATEGORY,
        severity=severity,
        description=description,
        score=score,
        view="raw",
    )


def _build_findings(stats: dict[str, int], hidden: _HiddenStats) -> list[Finding]:
    findings: list[Finding] = []
    invisible = stats["invisible_chars"] + stats["control_chars"]
    if invisible:
        findings.append(
            _finding(
                "OBF-INVISIBLE",
                Severity.LOW if invisible < 5 else Severity.MEDIUM,
                30.0 if invisible < 5 else 50.0,
                f"{invisible} invisible or control character(s) removed",
            )
        )
    if stats["bidi_controls"]:
        findings.append(
            _finding(
                "OBF-BIDI",
                Severity.HIGH,
                60.0,
                f"{stats['bidi_controls']} bidirectional override character(s) removed",
            )
        )
    if hidden.tag_payload:
        findings.append(
            _finding(
                "OBF-UNICODE-TAGS",
                Severity.CRITICAL,
                90.0,
                "Hidden text smuggled in Unicode tag characters (ASCII smuggling)",
            )
        )
    elif stats["tag_chars"]:
        findings.append(
            _finding("OBF-UNICODE-TAGS", Severity.MEDIUM, 45.0, "Unicode tag characters removed")
        )
    if hidden.selector_payload:
        findings.append(
            _finding(
                "OBF-VARIATION-SELECTORS",
                Severity.CRITICAL,
                90.0,
                "Hidden text smuggled in Unicode variation selectors",
            )
        )
    elif stats["variation_selectors"] >= 4:
        findings.append(
            _finding(
                "OBF-VARIATION-SELECTORS",
                Severity.MEDIUM,
                45.0,
                f"{stats['variation_selectors']} stray variation selectors removed",
            )
        )
    if stats["homoglyph_tokens"]:
        findings.append(
            _finding(
                "OBF-HOMOGLYPH",
                Severity.MEDIUM,
                45.0,
                f"{stats['homoglyph_tokens']} mixed-script word(s) with look-alike letters",
            )
        )
    if stats["base64_payloads"]:
        findings.append(
            _finding("OBF-BASE64", Severity.LOW, 25.0, "Base64 payload decoded to readable text")
        )
    if stats["hex_payloads"]:
        findings.append(
            _finding("OBF-HEX", Severity.LOW, 25.0, "Hex payload decoded to readable text")
        )
    if stats["decode_depth"] >= 2:
        findings.append(
            _finding(
                "OBF-NESTED-ENCODING",
                Severity.MEDIUM,
                55.0,
                f"Payload wrapped in {stats['decode_depth']} encoding layers",
            )
        )
    if stats["rot13_detected"]:
        findings.append(
            _finding("OBF-ROT13", Severity.MEDIUM, 40.0, "Text contains a ROT13-encoded message")
        )
    return findings
