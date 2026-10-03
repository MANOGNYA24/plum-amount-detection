"""FastAPI entrypoint.

Every POST endpoint accepts EITHER:
  - JSON:       {"text": "Total: INR 1200 | Paid: 1000"}
  - multipart:  file=@bill.jpg   (or a `text` form field)
"""
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from . import config, ocr, pipeline

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("amount-detector")

app = FastAPI(
    title="AI-Powered Amount Detection in Medical Documents",
    version="1.0.0",
    description="OCR -> numeric normalization -> context classification, with guardrails.",
)


# ---------------- error handling ----------------

def error_body(reason: str) -> dict:
    return {"status": "error", "reason": reason}


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=error_body(str(exc.detail)))


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content=error_body(f"invalid request: {exc.errors()}"))


@app.exception_handler(Exception)
async def unhandled_error(_: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content=error_body("internal server error"))


# ---------------- input handling ----------------

async def read_input(request: Request) -> ocr.ExtractedText:
    ctype = request.headers.get("content-type", "")

    if ctype.startswith("application/json"):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "body is not valid JSON")
        text = body.get("text") if isinstance(body, dict) else None
        return _validate_text(text)

    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is not None and hasattr(upload, "read"):
            if upload.content_type not in config.ALLOWED_IMAGE_TYPES:
                raise HTTPException(415, f"unsupported file type '{upload.content_type}'. "
                                         f"Use one of: {sorted(config.ALLOWED_IMAGE_TYPES)}")
            data = await upload.read()
            if not data:
                raise HTTPException(400, "uploaded file is empty")
            if len(data) > config.MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"file too large (max {config.MAX_UPLOAD_BYTES // 1024 // 1024} MB)")
            try:
                return ocr.from_image(data)
            except ocr.OCRError as exc:
                raise HTTPException(422, str(exc))
        return _validate_text(form.get("text"))

    if ctype.startswith("text/plain"):
        return _validate_text((await request.body()).decode("utf-8", errors="replace"))

    raise HTTPException(415, "send JSON {'text': ...}, multipart with 'file' or 'text', or text/plain")


def _validate_text(text) -> ocr.ExtractedText:
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "provide a non-empty 'text' field or an image 'file'")
    if len(text) > config.MAX_TEXT_CHARS:
        raise HTTPException(413, f"text too long (max {config.MAX_TEXT_CHARS} characters)")
    return ocr.from_text(text)


async def run_pipeline(request: Request, use_llm: Optional[bool], stop_after: int = 4):
    extracted = await read_input(request)
    try:
        return pipeline.run(extracted, use_llm=use_llm, stop_after=stop_after), None
    except pipeline.GuardrailTriggered as g:
        log.info("guardrail: %s", g.result.reason)
        return None, JSONResponse(status_code=200, content=g.result.model_dump())


LLM_PARAM = Query(None, description="Override: use the LLM step (needs GEMINI_API_KEY)")

# Declared only so Swagger's "Try it out" renders input fields. The actual
# parsing still happens in read_input(), which also accepts raw JSON
# ({"text": "..."}) or a text/plain body -- these two are simply the
# multipart/form-data alternative, both optional so JSON keeps working.
TEXT_FORM = Form(None, description='Bill text. Alternatively send JSON {"text": "..."} or a text/plain body.')
FILE_FORM = File(None, description="Bill image (png/jpeg/webp/bmp/tiff, max 5 MB). Ignored if 'text' is also set and no file is chosen.")


# ---------------- endpoints ----------------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "llm_enabled": config.llm_enabled()}


@app.post("/api/v1/extract", summary="Step 1: OCR / raw numeric tokens")
async def extract(request: Request, text: Optional[str] = TEXT_FORM, file: Optional[UploadFile] = FILE_FORM):
    out, guard = await run_pipeline(request, use_llm=False, stop_after=1)
    return guard or out.step1


@app.post("/api/v1/normalize", summary="Step 2: fix OCR digit errors, map to numbers")
async def normalize(request: Request, text: Optional[str] = TEXT_FORM, file: Optional[UploadFile] = FILE_FORM):
    out, guard = await run_pipeline(request, use_llm=False, stop_after=2)
    return guard or out.step2


@app.post("/api/v1/classify", summary="Step 3: label amounts by context")
async def classify(
    request: Request,
    use_llm: Optional[bool] = LLM_PARAM,
    text: Optional[str] = TEXT_FORM,
    file: Optional[UploadFile] = FILE_FORM,
):
    out, guard = await run_pipeline(request, use_llm)
    return guard or out.step3


@app.post("/api/v1/detect", summary="Step 4: full pipeline, final output with provenance")
async def detect(
    request: Request,
    use_llm: Optional[bool] = LLM_PARAM,
    debug: bool = Query(False, description="Include every intermediate step in the response"),
    text: Optional[str] = TEXT_FORM,
    file: Optional[UploadFile] = FILE_FORM,
):
    out, guard = await run_pipeline(request, use_llm)
    if guard:
        return guard
    if not debug:
        return out.step4
    return {
        **out.step4.model_dump(),
        "steps": {
            "step1_ocr": out.step1, "step2_normalization": out.step2,
            "step3_classification": out.step3,
        },
        "debug": out.debug,
    }
