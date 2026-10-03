"""Chains the four steps together and applies guardrails between them.

    input -> [1] OCR/tokens -> [2] normalize -> [3] classify (+LLM) -> [4] final

Any step can stop the chain with a GuardrailResult.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from . import classify as clf
from . import config, llm, normalize, ocr
from .schemas import (
    ClassificationResult, FinalResult, GuardrailResult, NormalizationResult, OCRResult,
)


class GuardrailTriggered(Exception):
    def __init__(self, reason: str, status: str = "no_amounts_found"):
        self.result = GuardrailResult(status=status, reason=reason)
        super().__init__(reason)


@dataclass
class PipelineOutput:
    step1: OCRResult
    step2: Optional[NormalizationResult] = None
    step3: Optional[ClassificationResult] = None
    step4: Optional[FinalResult] = None
    debug: dict = field(default_factory=dict)


def _clamp(x: float) -> float:
    return round(max(0.0, min(1.0, x)), 2)


def run(extracted: ocr.ExtractedText, use_llm: Optional[bool] = None, stop_after: int = 4) -> PipelineOutput:
    """Run steps 1..stop_after. Guardrails of later steps don't fire if we stop early."""
    text = extracted.text
    use_llm = config.llm_enabled() if use_llm is None else (use_llm and config.llm_enabled())

    # ---------- Step 1: text + raw tokens ----------
    if not text.strip():
        raise GuardrailTriggered("no readable text in document")
    if extracted.source == "image" and extracted.confidence < config.MIN_OCR_CONFIDENCE:
        raise GuardrailTriggered("document too noisy")

    raw_tokens = ocr.find_tokens(text)
    if not raw_tokens:
        raise GuardrailTriggered("no numeric values found in document")

    step1 = OCRResult(
        raw_tokens=[t.token for t in raw_tokens],
        currency_hint=ocr.detect_currency(text),
        confidence=_clamp(extracted.confidence),
    )

    if stop_after == 1:
        return PipelineOutput(step1)

    # ---------- Step 2: normalize ----------
    norm_tokens = [normalize.normalize_token(t.token) for t in raw_tokens]
    amount_tokens = [n for n in norm_tokens if n.value is not None and not n.is_percent]
    if not amount_tokens:
        raise GuardrailTriggered("no monetary amounts found (only percentages or unreadable numbers)")

    step2 = NormalizationResult(
        normalized_amounts=[n.value for n in amount_tokens],
        normalization_confidence=_clamp(
            sum(normalize.token_confidence(n) for n in amount_tokens) / len(amount_tokens) * 0.98
        ),
    )

    if stop_after == 2:
        return PipelineOutput(step1, step2)

    # ---------- Step 3: classify by context ----------
    source_label = "text" if extracted.source == "text" else "ocr"
    items = clf.classify(text, raw_tokens, norm_tokens, source_label)
    if not items:
        raise GuardrailTriggered("numbers found, but none look like monetary amounts")

    if use_llm:
        items = llm.refine_with_llm(text, items)

    consistent = clf.consistency_check(items)
    conf = sum(c.confidence for c in items) / len(items)
    if consistent is True:
        conf += 0.05
    elif consistent is False:
        conf -= 0.10
    # labels can't be more reliable than the text they were read from
    conf = min(conf, (extracted.confidence + 1) / 2)

    step3 = ClassificationResult(
        amounts=[{"type": c.type, "value": c.value} for c in items],
        confidence=_clamp(conf),
    )

    # ---------- Step 4: final output with provenance ----------
    step4 = FinalResult(
        currency=step1.currency_hint,
        amounts=[{"type": c.type, "value": c.value, "source": c.source} for c in items],
        status="ok",
    )

    debug = {
        "extracted_text": text,
        "input_source": extracted.source,
        "llm_used": use_llm,
        "arithmetic_check": {None: "not_applicable", True: "passed", False: "failed"}[consistent],
        "per_amount": [
            {"value": c.value, "type": c.type, "method": c.method,
             "label": c.label, "confidence": round(c.confidence, 2)}
            for c in items
        ],
    }
    return PipelineOutput(step1, step2, step3, step4, debug)
