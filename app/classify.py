"""Step 3: label each amount using the words right before it.

Rule-based first (deterministic, explainable). The LLM (see llm.py) is only
used to fill gaps and to cross-check, never to invent amounts.
"""
import difflib
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .normalize import NormalizedToken
from .ocr import RawToken

# type -> keywords. Multi-word phrases are allowed.
KEYWORDS = {
    "subtotal": ["sub total", "subtotal", "sub-total"],
    "total_bill": ["grand total", "net total", "total amount", "bill amount", "net amount", "gross amount",
                   "amount payable", "net payable", "payable", "total"],
    "paid": ["amount paid", "paid", "received", "advance", "payment"],
    "due": ["balance due", "amount due", "due", "balance", "outstanding", "pending"],
    "discount": ["discount", "disc", "concession"],
    "tax": ["cgst", "sgst", "igst", "gst", "tax", "vat"],
    # Numbers next to these are identifiers / quantities, not money -> dropped
    "_ignore": ["invoice no", "bill no", "receipt no", "invoice", "gstin", "phone", "mobile",
                "uhid", "reg no", "patient id", "qty", "quantity", "age", "pin", "room no"],
}

# Single words we allow fuzzy matching against (handles "Pald", "Tota1", "Dve")
FUZZY_VOCAB = sorted({w for kws in KEYWORDS.values() for kw in kws for w in kw.split() if len(w) >= 3})
FUZZY_CUTOFF = 0.75

# OCR damage inside label words: digits that should be letters
LABEL_FIXES = str.maketrans({"0": "o", "1": "l", "5": "s", "|": "l", "@": "a", "$": "s"})

SEGMENT_SPLIT = re.compile(r"[|\n;]")


@dataclass
class Classified:
    type: str
    value: float
    source: str        # provenance snippet, e.g. "Total: INR 1200"
    label: str         # cleaned label text that was matched against
    confidence: float
    method: str        # "exact" | "fuzzy" | "none" | "llm"


def _segment_bounds(text: str, pos: int) -> Tuple[int, int]:
    """Start/end of the line or '|'-separated cell containing pos."""
    start = 0
    for m in SEGMENT_SPLIT.finditer(text, 0, pos):
        start = m.end()
    m = SEGMENT_SPLIT.search(text, pos)
    end = m.start() if m else len(text)
    return start, end


def _clean_label(label: str) -> Tuple[str, bool]:
    """Lowercase, undo digit-for-letter OCR errors, fuzzy-fix words.
    Returns (clean_label, used_fuzzy)."""
    words = re.findall(r"[a-z\-]+", label.lower().translate(LABEL_FIXES))
    fixed, used_fuzzy = [], False
    for w in words:
        if len(w) >= 3 and w not in FUZZY_VOCAB:
            # same first letter required: stops "count" -> "discount"
            match = [m for m in difflib.get_close_matches(w, FUZZY_VOCAB, n=3, cutoff=FUZZY_CUTOFF)
                     if m[0] == w[0]][:1]
            if match:
                w, used_fuzzy = match[0], True
        fixed.append(w)
    return " ".join(fixed), used_fuzzy


def match_type(label: str) -> Optional[str]:
    """Pick the keyword that appears CLOSEST to the number (latest end position).
    'Sub Total' -> subtotal (longer phrase wins ties); 'Total Paid' -> paid."""
    best: Optional[Tuple[int, int, str]] = None  # (end_pos, kw_len, type)
    for amount_type, keywords in KEYWORDS.items():
        for kw in keywords:
            for m in re.finditer(rf"\b{re.escape(kw)}\b", label):
                cand = (m.end(), len(kw), amount_type)
                if best is None or cand[:2] > best[:2]:
                    best = cand
    return best[2] if best else None


def classify(
    text: str,
    raw_tokens: List[RawToken],
    normalized: List[NormalizedToken],
    input_source: str,
) -> List[Classified]:
    results: List[Classified] = []
    prev_end_in_segment = {}

    # If most sizeable amounts are written with decimals ("400.00"), whole
    # numbers without decimals are probably not money (addresses, PIN codes,
    # stray digits). Only applied when the document clearly uses that style.
    sizeable = [r.token for r, n in zip(raw_tokens, normalized)
                if n.value is not None and not n.is_percent and n.value >= 100]
    with_decimals = sum(1 for t in sizeable if "." in t)
    decimal_style = len(sizeable) >= 3 and with_decimals / len(sizeable) >= 0.6

    for raw, norm in zip(raw_tokens, normalized):
        seg_start, seg_end = _segment_bounds(text, raw.start)
        # label = text between the previous amount in this segment (or the
        # segment start) and this token
        label_start = max(seg_start, prev_end_in_segment.get(seg_start, seg_start))

        if norm.value is None or norm.is_percent:
            prev_end_in_segment[seg_start] = raw.end
            continue  # percentages (e.g. "Discount: 10%") are rates, not amounts

        # OCR garbage like "{ 00": all zeros with no decimal point is not an amount
        if set(raw.token) <= {"0", "O", "o"} and "." not in raw.token:
            continue

        raw_label = text[label_start:raw.start]
        label, used_fuzzy = _clean_label(raw_label)
        amount_type = match_type(label)

        if amount_type == "_ignore":
            prev_end_in_segment[seg_start] = raw.end
            continue
        # Small whole numbers with no money keyword are serial numbers or
        # quantities (Sl.No / Qty columns), not amounts. They are skipped
        # WITHOUT ending the label, so "ECG 1 300.00" keeps "ECG" as 300's label.
        if amount_type is None and isinstance(norm.value, int) and norm.value < 100 \
                and "." not in raw.token:
            continue
        # In a decimal-style document, a whole number is only kept if it has a
        # money keyword and is at least 10 (drops "#184", "560067", "Total: 7").
        if decimal_style and "." not in raw.token and (amount_type is None or norm.value < 10):
            continue
        prev_end_in_segment[seg_start] = raw.end
        if amount_type is None:
            amount_type, conf, method = "other", 0.4, "none"
        elif used_fuzzy:
            conf, method = 0.75, "fuzzy"
        else:
            conf, method = 0.92, "exact"

        snippet = text[label_start:raw.end].strip(" :-\t")
        results.append(Classified(
            type=amount_type,
            value=norm.value,
            source=f"{input_source}: '{snippet}'",
            label=label,
            confidence=conf,
            method=method,
        ))
    return results


def consistency_check(items: List[Classified]) -> Optional[bool]:
    """Arithmetic sanity check: total - paid == due (when all three exist).
    Returns None if it can't be checked."""
    def first(t):
        return next((c.value for c in items if c.type == t), None)

    total, paid, due = first("total_bill"), first("paid"), first("due")
    if None in (total, paid, due):
        return None
    return abs((total - paid) - due) < 0.01
