"""Response schemas. These mirror the JSON shapes in the assignment exactly."""
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field

# int when the amount is whole (1200), float otherwise (1200.50)
Number = Union[int, float]

AmountType = Literal[
    "total_bill", "paid", "due", "discount", "tax", "subtotal", "other"
]


class TextInput(BaseModel):
    text: str = Field(..., min_length=1, description="Raw bill/receipt text")


# ---- Step 1 ----
class OCRResult(BaseModel):
    raw_tokens: List[str]
    currency_hint: Optional[str]
    confidence: float


# ---- Step 2 ----
class NormalizationResult(BaseModel):
    normalized_amounts: List[Number]
    normalization_confidence: float


# ---- Step 3 ----
class ClassifiedAmount(BaseModel):
    type: AmountType
    value: Number


class ClassificationResult(BaseModel):
    amounts: List[ClassifiedAmount]
    confidence: float


# ---- Step 4 ----
class FinalAmount(BaseModel):
    type: AmountType
    value: Number
    source: str


class FinalResult(BaseModel):
    currency: Optional[str]
    amounts: List[FinalAmount]
    status: Literal["ok"] = "ok"


# ---- Guardrail / errors ----
class GuardrailResult(BaseModel):
    status: str
    reason: str
