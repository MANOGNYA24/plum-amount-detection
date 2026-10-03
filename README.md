# Amount Detection in Medical Documents

FastAPI service that extracts financial amounts from medical bills and receipts, given as typed text or images (scanned or photographed, possibly noisy). It corrects OCR digit errors, labels each amount by context (total, paid, due, and so on), and returns structured JSON with provenance: the exact text snippet each amount came from.

Problem Statement 4: OCR, Numeric Normalization, Context Classification.

Live demo: https://cosmos-pointing-earpiece.ngrok-free.dev (frontend) and https://cosmos-pointing-earpiece.ngrok-free.dev/docs (API docs)

## Architecture

```
 Input: text/JSON or image, via POST /api/v1/detect

 [1] OCR / token extraction     (app/ocr.py)
     Tesseract + preprocessing, regex tokens, currency hint
     -> guardrail: no_amounts_found (no text / too noisy / no numbers)

 [2] Normalization               (app/normalize.py)
     l->1, O->0, S->5, "1,20,000" -> 120000
     -> guardrail: only % / unreadable

 [3] Context classification      (app/classify.py)
     keyword rules + fuzzy match ("Pald"), optional LLM refine (app/llm.py)
     arithmetic check: total - paid = due
     -> guardrail: no monetary amounts

 [4] Final JSON with provenance
```

| File | Responsibility |
|---|---|
| `app/main.py` | API routes, input parsing (JSON / multipart / text), error handling |
| `app/pipeline.py` | Chains the 4 steps, applies guardrails, computes confidences |
| `app/ocr.py` | Image preprocessing, Tesseract OCR, token regex, currency detection |
| `app/normalize.py` | OCR digit correction and number parsing |
| `app/classify.py` | Context labeling (rules + fuzzy), arithmetic consistency check |
| `app/llm.py` | Optional Gemini step: labels ambiguous amounts, cross-validates rules |
| `app/schemas.py` | Pydantic response models that mirror the assignment's JSON schemas |
| `app/config.py` | Env-based settings and thresholds |
| `app/static/index.html` | Frontend that visualizes the pipeline (see Frontend below) |

### Design decisions

Rules run first; the LLM is optional and only used where rules are weak, such as amounts with no keyword (e.g. "Consultation 500") or fuzzy, OCR-damaged labels.

The LLM cannot invent amounts. It receives the amounts already extracted, by index, and returns only a label per index. Output is schema-validated: unknown indices, invalid types, or malformed JSON are discarded and the rule result is kept.

When a rule matches an exact keyword, the rule wins. If the LLM agrees, confidence goes up; if it disagrees, confidence goes down. The arithmetic check (`total - paid == due`) adjusts confidence the same way.

Each amount carries the exact source snippet as it appeared, including OCR errors (e.g. `ocr: 'T0tal: Rs l200'`), for auditability.

Dates, times, phone numbers, and numbers labeled invoice no / UHID / qty are filtered out so they are not reported as money.

If `GEMINI_API_KEY` is not set, or the LLM call fails, the pipeline runs fully rule-based.

### Confidence scores
| Field | Meaning |
|---|---|
| `step1.confidence` | Mean Tesseract word confidence (images), or 0.95 for typed text |
| `normalization_confidence` | Lower when more characters had to be corrected (-0.15 per fix) |
| `step3.confidence` | Mean label confidence (exact 0.92, fuzzy 0.75, none 0.40, LLM 0.80), adjusted by the arithmetic check, capped by OCR quality |

## Setup

### Local
```bash
# 1. Install Tesseract
#    macOS:   brew install tesseract
#    Ubuntu:  sudo apt install tesseract-ocr
#    Windows: https://github.com/UB-Mannheim/tesseract/wiki (add to PATH)

# 2. Python deps
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. (Optional) enable the LLM step
export GEMINI_API_KEY=your_key                          # free: https://aistudio.google.com/apikey

# 4. Run
uvicorn app.main:app --reload --port 8000
```
Interactive docs: `http://localhost:8000/docs`
Frontend: `http://localhost:8000/`

### Expose publicly with ngrok
```bash
ngrok http 8000
```

### Docker / cloud (Render, Railway, etc.)
```bash
docker build -t amount-detector .
docker run -p 8000:8000 -e GEMINI_API_KEY=$GEMINI_API_KEY amount-detector
```
On Render: New, Web Service, connect the repo. It detects the Dockerfile automatically.

### Tests
```bash
pytest -q        # unit, API, image OCR, guardrails, mocked LLM validation
```

## API

Every POST endpoint accepts either:
- JSON: `{"text": "..."}`
- multipart: `file=@bill.jpg` (png / jpeg / webp / bmp / tiff, max 5 MB), or a `text` form field
- `text/plain` body

Every POST endpoint also declares optional `text` and `file` parameters so Swagger's "Try it out" (`/docs`) renders real input fields. This is additive only; JSON and `text/plain` bodies work exactly as before.

| Endpoint | Returns |
|---|---|
| `GET /health` | `{"status":"ok","llm_enabled":false}` |
| `POST /api/v1/extract` | Step 1: raw tokens, currency hint, OCR confidence |
| `POST /api/v1/normalize` | Step 2: corrected numeric amounts |
| `POST /api/v1/classify` | Step 3: typed amounts. `?use_llm=false` forces rules only |
| `POST /api/v1/detect` | Step 4: final output. `?debug=true` adds all intermediate steps |

Amount types: `total_bill`, `paid`, `due`, `discount`, `tax`, `subtotal`, `other`.

### Examples

Step 1: extract
```bash
curl -X POST localhost:8000/api/v1/extract -H "Content-Type: application/json" \
  -d '{"text":"Total: INR 1200 | Paid: 1000 | Due: 200 | Discount: 10%"}'
```
```json
{"raw_tokens":["1200","1000","200","10%"],"currency_hint":"INR","confidence":0.95}
```

Step 2: normalize (noisy OCR text)
```bash
curl -X POST localhost:8000/api/v1/normalize -H "Content-Type: application/json" \
  -d '{"text":"T0tal: Rs l200 | Pald: 1000 | Due: 200"}'
```
```json
{"normalized_amounts":[1200,1000,200],"normalization_confidence":0.93}
```

Step 3: classify
```bash
curl -X POST localhost:8000/api/v1/classify -H "Content-Type: application/json" \
  -d '{"text":"Total: INR 1200 | Paid: 1000 | Due: 200"}'
```
```json
{"amounts":[{"type":"total_bill","value":1200},{"type":"paid","value":1000},{"type":"due","value":200}],"confidence":0.97}
```

Step 4: final output (image upload)
```bash
curl -X POST localhost:8000/api/v1/detect -F "file=@samples/bill_noisy.jpg"
```
```json
{
  "currency": "INR",
  "amounts": [
    {"type": "total_bill", "value": 1200, "source": "ocr: 'TOtal: Rs 1200'"},
    {"type": "paid",       "value": 1000, "source": "ocr: 'Pald: 1000'"},
    {"type": "due",        "value": 200,  "source": "ocr: 'Due: 200'"}
  ],
  "status": "ok"
}
```

### Guardrails

```bash
curl -X POST localhost:8000/api/v1/detect -F "file=@samples/no_amounts.jpg"
```
```json
{"status":"no_amounts_found","reason":"no numeric values found in document"}
```

Other guardrail reasons: `document too noisy` (OCR confidence < 0.30), `no readable text in document`, `no monetary amounts found (only percentages or unreadable numbers)`.

Input errors return a proper HTTP status with `{"status":"error","reason":"..."}`: 400 (empty/missing input), 413 (too large), 415 (unsupported file type), 422 (undecodable image).

### All sample requests
- `bash samples/requests.sh` (set `BASE=https://xxx.ngrok-free.app` to hit the deployed URL)
- Postman: import `samples/postman_collection.json` and set the `base_url` variable
- Regenerate sample images: `python samples/make_samples.py`

## Frontend

A single self-contained page at `app/static/index.html`, served at `GET /` (plain HTML/CSS/JS, no build step). It calls the same `POST /api/v1/detect?debug=true` endpoint and renders the response as four step cards, in pipeline order:

1. Text extraction (OCR): raw tokens, with OCR-lookalike tokens (e.g. `l200`) highlighted.
2. Normalization: raw to corrected number mapping (e.g. `l200 -> 1200`) and the normalization confidence.
3. Classification: each amount's label, match method (exact / fuzzy / llm), confidence, and the `total - paid = due` arithmetic check.
4. Final output: currency and the amounts table with each value's source snippet.

Confidence scores are shown as color-coded progress bars. A guardrail response (`status: "no_amounts_found"`) shows a yellow banner; an error response (`status: "error"`) shows a red banner with the reason. A collapsible "Raw JSON" section always shows the exact API response.

Input is a textarea plus three sample buttons (clean text, noisy OCR text, no amounts), or an image upload with preview.

## Limitations and next steps
- Line items without keywords ("Medicines 700") are labeled `other` unless the LLM is enabled.
- A label on a different line than its number ("Gross Amount" on one line, the value on the next) is not matched, since classification only looks at text on the same line or `|`-separated cell. The amount is still extracted, labeled `other`.
- OCR tries up to 3 preprocessing/PSM passes per image and keeps the one with the highest total word confidence. This takes roughly 5-8 seconds for a phone photo, versus under a second for typed text.
- Single-page images only. PDF support would use `pdf2image` before OCR.
- Tesseract struggles with heavily crumpled or low-light photos. Deskewing, or a vision-LLM OCR fallback, would help.
- Percentages (e.g. `Discount: 10%`) appear in step 1 tokens but are excluded from amounts, because they are rates, not money.
