"""Step 1: get text out of the input and pull raw numeric tokens.

- Typed text is used as-is.
- Images go through light preprocessing + Tesseract. Tesseract's per-word
  confidences are averaged to give the step's confidence score.
"""
import io
import re
from dataclasses import dataclass
from typing import List, Optional

import pytesseract
from PIL import Image, ImageChops, ImageFilter, ImageOps

from . import config


@dataclass
class ExtractedText:
    text: str
    confidence: float
    source: str  # "text" or "image"


class OCRError(Exception):
    pass


# ---------- Text acquisition ----------

def from_text(text: str) -> ExtractedText:
    return ExtractedText(text=text.strip(), confidence=config.TEXT_INPUT_CONFIDENCE, source="text")


TARGET_WIDTH = 1600     # Tesseract reads best when text is roughly 30px tall
ADAPTIVE_RADIUS = 15    # neighbourhood size for adaptive thresholding
ADAPTIVE_OFFSET = 25    # how much darker than its surroundings a pixel must be to count as ink


def _upscale(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("L")
    if img.width < TARGET_WIDTH:
        scale = TARGET_WIDTH / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return ImageOps.autocontrast(img)


def _preprocess_simple(img: Image.Image) -> Image.Image:
    """Grayscale + median filter. Best for clean scans and speckle noise."""
    return img.filter(ImageFilter.MedianFilter(size=3))


def _preprocess_adaptive(img: Image.Image) -> Image.Image:
    """Adaptive threshold: each pixel is compared with the average brightness of
    its neighbourhood instead of one global cut-off, so shadows, tinted paper
    and uneven lighting in phone photos don't wash out digits."""
    background = img.filter(ImageFilter.BoxBlur(ADAPTIVE_RADIUS))
    darkness = ImageChops.subtract(background, img)
    return darkness.point(lambda p: 0 if p > ADAPTIVE_OFFSET else 255)


# (preprocessing, tesseract page-segmentation mode). psm 4 = single column of
# variable-size text (bills with columns), psm 6 = one uniform block (receipts).
OCR_CANDIDATES = [
    (_preprocess_simple, 6),
    (_preprocess_adaptive, 4),
    (_preprocess_adaptive, 6),
]


def _run_tesseract(img: Image.Image, psm: int) -> dict:
    return pytesseract.image_to_data(img, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)


def from_image(data: bytes) -> ExtractedText:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise OCRError(f"could not decode image: {exc}") from exc

    base = _upscale(img)
    try:
        # Run each candidate and keep the best read, scored by TOTAL word
        # confidence: an attempt that drops half the words can have a higher
        # average confidence but has read less of the document.
        best, best_score = None, -1.0
        for preprocess, psm in OCR_CANDIDATES:
            d = _run_tesseract(preprocess(base), psm)
            score = sum(float(c) for c, w in zip(d["conf"], d["text"]) if w.strip() and float(c) >= 0)
            if score > best_score:
                best, best_score = d, score
        data_dict = best
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRError("tesseract is not installed on the server") from exc

    # Rebuild text line by line, and average word confidences
    lines: dict = {}
    confs: List[float] = []
    for i, word in enumerate(data_dict["text"]):
        word = word.strip()
        conf = float(data_dict["conf"][i])
        if not word or conf < 0:
            continue
        key = (data_dict["block_num"][i], data_dict["par_num"][i], data_dict["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)

    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    confidence = round(sum(confs) / len(confs) / 100, 2) if confs else 0.0
    return ExtractedText(text=text, confidence=confidence, source="image")


# ---------- Token extraction ----------

# Characters Tesseract commonly confuses with digits
OCR_DIGIT_LOOKALIKES = "OoIlS|"

# A numeric-looking token: must contain at least one real digit, may contain
# lookalike chars (l200, 1O00), thousands separators, decimals, and a trailing %.
# Must not be glued to letters on either side (so "T0tal" is NOT a token).
TOKEN_RE = re.compile(
    rf"(?<![A-Za-z0-9])"
    rf"[\d{OCR_DIGIT_LOOKALIKES}]*\d[\d{OCR_DIGIT_LOOKALIKES},.]*%?"
    rf"(?![A-Za-z0-9])"
)

# Currency symbols only count when they sit right next to a number ("$100",
# "Rs 100"). A stray symbol in a smudged stamp or footer is ignored.
CURRENCY_PATTERNS = [
    ("INR", re.compile(r"\b(INR|Rupees?)\b|\bRs\.?\s?[\dlIO]|₹\s?[\dlIO]", re.I)),
    ("USD", re.compile(r"\b(USD|dollars?)\b|\$\s?\d", re.I)),
    ("EUR", re.compile(r"\bEUR\b|€\s?\d", re.I)),
    ("GBP", re.compile(r"\bGBP\b|£\s?\d", re.I)),
]


@dataclass
class RawToken:
    token: str
    start: int
    end: int


# Things that look numeric but are never amounts. They are blanked out (same
# length, so character offsets stay valid) before token search.
NON_AMOUNT_PATTERNS = [
    re.compile(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b"),        # dates 12/09/2025
    re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\s*(am|pm)?\b", re.I),  # times 10:30
    re.compile(r"(?<!\d)(\+91[\s-]?)?[6-9]\d{9}(?!\d)"),          # IN phone numbers
]


def mask_non_amounts(text: str) -> str:
    for pattern in NON_AMOUNT_PATTERNS:
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return text


# "$,500.00" is never a real dollar amount: a "$" directly before a thousands
# comma is OCR misreading a 5 ("5,500.00"). Same length, so offsets stay valid.
DOLLAR_AS_FIVE = re.compile(r"\$(?=,\d{3})")


def find_tokens(text: str) -> List[RawToken]:
    tokens = []
    cleaned = DOLLAR_AS_FIVE.sub("5", mask_non_amounts(text))
    for m in TOKEN_RE.finditer(cleaned):
        tok = m.group(0).rstrip(",.")  # trailing punctuation from "1200." or "1,200,"
        if tok:
            tokens.append(RawToken(tok, m.start(), m.start() + len(tok)))
    return tokens


def detect_currency(text: str) -> Optional[str]:
    for code, pattern in CURRENCY_PATTERNS:
        if pattern.search(text):
            return code
    return None
