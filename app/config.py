"""Central configuration, read from environment variables."""
import os

# LLM (optional). If no key is set, the pipeline runs fully rule-based.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "15"))

# Upload guardrails
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))  # 5 MB
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "10000"))
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp", "image/tiff"}

# Confidence thresholds
MIN_OCR_CONFIDENCE = float(os.getenv("MIN_OCR_CONFIDENCE", "0.30"))
TEXT_INPUT_CONFIDENCE = 0.95  # typed text has no OCR noise


def llm_enabled() -> bool:
    return bool(GEMINI_API_KEY)
