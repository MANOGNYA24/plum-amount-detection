"""Optional LLM step (Google Gemini via REST).

Design rule: the LLM never produces numbers. It only receives the list of
amounts we already extracted (by index) and returns a label for each index.
Its output is schema-validated; anything invalid is discarded and the
rule-based label is kept. This makes hallucinated amounts impossible.
"""
import json
import logging
from typing import Dict, List, Optional

import httpx

from . import config
from .classify import Classified

log = logging.getLogger(__name__)

VALID_TYPES = {"total_bill", "paid", "due", "discount", "tax", "subtotal", "other"}

PROMPT = """You label monetary amounts found in a medical bill or receipt.
The text came from OCR and may contain errors (e.g. "Pald" = "Paid", "T0tal" = "Total").

Full document text:
<<<
{text}
>>>

Amounts already extracted (index: value | surrounding text):
{items}

For each index, choose exactly one type from:
total_bill, paid, due, discount, tax, subtotal, other

Rules:
- Use only the surrounding text as evidence. Use "other" if unsure.
- Do not add, remove or change any amounts.

Respond with JSON only, in this shape:
{{"labels": [{{"index": 0, "type": "total_bill"}}]}}"""


def _call_gemini(prompt: str) -> Optional[str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    try:
        resp = httpx.post(
            url, json=body, timeout=config.LLM_TIMEOUT_SECONDS,
            headers={"x-goog-api-key": config.GEMINI_API_KEY},
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:  # network error, quota, bad response shape...
        log.warning("LLM call failed, falling back to rules: %s", exc)
        return None


def parse_labels(raw: Optional[str], n_items: int) -> Dict[int, str]:
    """Validate LLM output. Keeps only well-formed {index, type} pairs."""
    if not raw:
        return {}
    try:
        data = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
        labels = data["labels"]
    except (json.JSONDecodeError, KeyError, TypeError):
        log.warning("LLM returned unparseable output: %r", raw[:200])
        return {}

    valid: Dict[int, str] = {}
    for entry in labels if isinstance(labels, list) else []:
        if not isinstance(entry, dict):
            continue
        idx, typ = entry.get("index"), entry.get("type")
        if isinstance(idx, int) and 0 <= idx < n_items and typ in VALID_TYPES:
            valid[idx] = typ
    return valid


def refine_with_llm(text: str, items: List[Classified], call=_call_gemini) -> List[Classified]:
    """Merge LLM labels with rule labels.

    - Rule had no match ('other') or a fuzzy match -> LLM label wins.
    - Rule had an exact keyword match -> rule wins; agreement raises confidence,
      disagreement lowers it (that's the cross-validation).
    """
    if not items:
        return items

    listing = "\n".join(f"{i}: {c.value} | {c.source.split(': ', 1)[1]}" for i, c in enumerate(items))
    labels = parse_labels(call(PROMPT.format(text=text, items=listing)), len(items))
    if not labels:
        return items

    for i, c in enumerate(items):
        llm_type = labels.get(i)
        if llm_type is None:
            continue
        if c.method in ("none", "fuzzy"):
            if llm_type == c.type:
                c.confidence = min(0.95, c.confidence + 0.1)
            elif llm_type != "other":
                c.type, c.confidence, c.method = llm_type, 0.8, "llm"
        else:  # exact rule match
            c.confidence = min(0.98, c.confidence + 0.04) if llm_type == c.type else c.confidence - 0.15
    return items
