"""Step 2: turn raw OCR tokens into numbers, fixing common OCR digit errors."""
from dataclasses import dataclass
from typing import Optional, Union

Number = Union[int, float]

# OCR lookalike -> digit
DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5"})


@dataclass
class NormalizedToken:
    raw: str
    value: Optional[Number]   # None if it could not be parsed
    is_percent: bool
    corrections: int          # how many characters had to be fixed


def _to_number(s: str) -> Optional[Number]:
    s = s.replace(",", "")                 # 1,200 / 1,20,000 (Indian grouping)
    if s.count(".") > 1:                   # "1.200.50" -> keep only the last dot
        head, _, tail = s.rpartition(".")
        s = head.replace(".", "") + "." + tail
    try:
        value = float(s)
    except ValueError:
        return None
    return int(value) if value.is_integer() else round(value, 2)


def normalize_token(raw: str) -> NormalizedToken:
    is_percent = raw.endswith("%")
    body = raw.rstrip("%")
    fixed = body.translate(DIGIT_FIXES)
    corrections = sum(1 for a, b in zip(body, fixed) if a != b)
    return NormalizedToken(raw=raw, value=_to_number(fixed), is_percent=is_percent, corrections=corrections)


def token_confidence(tok: NormalizedToken) -> float:
    """1.0 for a clean token, minus 0.15 for every character we had to guess."""
    if tok.value is None:
        return 0.0
    return max(0.3, 1.0 - 0.15 * tok.corrections)
