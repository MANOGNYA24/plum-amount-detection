from pathlib import Path

from fastapi.testclient import TestClient

from app import classify, llm, normalize, ocr, pipeline
from app.main import app

client = TestClient(app)
SAMPLES = Path(__file__).parent.parent / "samples"
SPEC_TEXT = "Total: INR 1200 | Paid: 1000 | Due: 200 | Discount: 10%"
NOISY_TEXT = "T0tal: Rs l200 | Pald: 1000 | Due: 200"


# ---------- unit: step 1 ----------

def test_tokens_from_spec_example():
    assert [t.token for t in ocr.find_tokens(SPEC_TEXT)] == ["1200", "1000", "200", "10%"]


def test_ocr_lookalike_token_kept_but_label_word_ignored():
    # "l200" is a token, the 0 inside "T0tal" is not
    assert [t.token for t in ocr.find_tokens(NOISY_TEXT)] == ["l200", "1000", "200"]


def test_dates_times_phones_are_not_amounts():
    text = "Date: 12/09/2025 10:30 AM Ph: 9876543210 Total 500"
    assert [t.token for t in ocr.find_tokens(text)] == ["500"]


def test_currency_detection():
    assert ocr.detect_currency("Rs 100") == "INR"
    assert ocr.detect_currency("₹100") == "INR"
    assert ocr.detect_currency("$100") == "USD"
    assert ocr.detect_currency("100") is None
    assert ocr.detect_currency("RE $ only") is None           # stray symbol, no number next to it
    assert ocr.detect_currency("T0tal: Rs l200") == "INR"      # "Rs" + OCR lookalike digit ("l200")


# ---------- unit: step 2 ----------

def test_digit_fixes():
    assert normalize.normalize_token("l200").value == 1200
    assert normalize.normalize_token("1O0O").value == 1000
    assert normalize.normalize_token("1,20,000").value == 120000
    assert normalize.normalize_token("1,250.50").value == 1250.5
    assert normalize.normalize_token("10%").is_percent


def test_dollar_as_five():
    # "$" directly before a thousands comma is a misread "5" ("5,500.00"),
    # not a dollar sign glued to "500.00".
    tokens = ocr.find_tokens("$,500.00")
    assert [t.token for t in tokens] == ["5,500.00"]
    assert normalize.normalize_token(tokens[0].token).value == 5500


# ---------- unit: step 3 ----------

def test_keyword_closest_to_number_wins():
    assert classify.match_type("sub total") == "subtotal"
    assert classify.match_type("total paid") == "paid"
    assert classify.match_type("balance due") == "due"


def test_gross_amount_is_total_bill():
    out = pipeline.run(ocr.from_text("Gross Amount: 5,500.00"), use_llm=False)
    assert [(a.type, a.value) for a in out.step3.amounts] == [("total_bill", 5500)]


def test_arithmetic_check():
    out = pipeline.run(ocr.from_text("Total 1200 | Paid 1000 | Due 300"), use_llm=False)
    assert out.debug["arithmetic_check"] == "failed"


def test_fuzzy_match_requires_same_first_letter():
    # "count" is close enough to "discount" by edit distance, but shouldn't
    # match it -- a real quantity column ("Complete Blood Count 1") must not
    # be mislabeled as a discount.
    out = pipeline.run(ocr.from_text("Complete Blood Count 1 | 400.00"), use_llm=False)
    assert all(a.type != "discount" for a in out.step3.amounts)


def test_quantity_before_amount_is_dropped_but_stays_in_source():
    # "2 ECG 1" are a quantity and a serial number, not amounts; "300.00" is
    # the only real amount and should still carry them in its provenance.
    out = pipeline.run(ocr.from_text("2 ECG 1 300.00"), use_llm=False)
    assert [(a.type, a.value) for a in out.step3.amounts] == [("other", 300)]
    assert "ECG" in out.step4.amounts[0].source


def test_zero_padding_is_not_mistaken_for_a_total():
    out = pipeline.run(ocr.from_text("Total : { 00 | Paid 100"), use_llm=False)
    assert not any(a.type == "total_bill" for a in out.step3.amounts)
    assert [(a.type, a.value) for a in out.step3.amounts] == [("paid", 100)]


def test_decimal_style_drops_whole_number_without_keyword():
    # Most sizeable amounts here are written with decimals ("400.00"), so a
    # bare whole number with no money keyword nearby ("#184") is a reference
    # number, not an amount, and should be dropped.
    text = "#184 Medicines 400.00 | Labs 300.00 | Total 1,500.00"
    out = pipeline.run(ocr.from_text(text), use_llm=False)
    values = [a.value for a in out.step3.amounts]
    assert 184 not in values
    assert set(values) == {400, 300, 1500}


# ---------- LLM merge + validation (mocked, no network) ----------

def _items(text):
    raw = ocr.find_tokens(text)
    return classify.classify(text, raw, [normalize.normalize_token(t.token) for t in raw], "text")


def test_llm_fills_unknown_label():
    items = _items("Consultation 500 | Total 500")
    assert items[0].type == "other"
    fake = lambda _: '{"labels":[{"index":0,"type":"subtotal"},{"index":1,"type":"total_bill"}]}'
    out = llm.refine_with_llm("...", items, call=fake)
    assert out[0].type == "subtotal" and out[0].method == "llm"


def test_llm_cannot_override_exact_rule_or_inject_garbage():
    items = _items("Total 1200 | Paid 1000")
    fake = lambda _: ('{"labels":[{"index":0,"type":"paid"},{"index":7,"type":"due"},'
                      '{"index":1,"type":"made_up_type"}]}')
    out = llm.refine_with_llm("...", items, call=fake)
    assert [c.type for c in out] == ["total_bill", "paid"]   # rules kept
    assert len(out) == 2                                      # no injected items
    assert out[0].confidence < 0.92                           # disagreement penalised


def test_llm_bad_json_falls_back():
    items = _items("Total 1200")
    out = llm.refine_with_llm("...", items, call=lambda _: "not json at all")
    assert out[0].type == "total_bill"


# ---------- API ----------

def test_api_step_endpoints_match_spec():
    r1 = client.post("/api/v1/extract", json={"text": SPEC_TEXT}).json()
    assert r1["raw_tokens"] == ["1200", "1000", "200", "10%"] and r1["currency_hint"] == "INR"

    r2 = client.post("/api/v1/normalize", json={"text": SPEC_TEXT}).json()
    assert r2["normalized_amounts"] == [1200, 1000, 200]

    r3 = client.post("/api/v1/classify", json={"text": SPEC_TEXT}).json()
    assert r3["amounts"] == [{"type": "total_bill", "value": 1200},
                             {"type": "paid", "value": 1000},
                             {"type": "due", "value": 200}]


def test_api_final_output_matches_spec():
    r = client.post("/api/v1/detect", json={"text": SPEC_TEXT})
    assert r.status_code == 200
    assert r.json() == {
        "currency": "INR",
        "amounts": [
            {"type": "total_bill", "value": 1200, "source": "text: 'Total: INR 1200'"},
            {"type": "paid", "value": 1000, "source": "text: 'Paid: 1000'"},
            {"type": "due", "value": 200, "source": "text: 'Due: 200'"},
        ],
        "status": "ok",
    }


def test_api_noisy_text():
    body = client.post("/api/v1/detect", json={"text": NOISY_TEXT}).json()
    assert [(a["type"], a["value"]) for a in body["amounts"]] == \
        [("total_bill", 1200), ("paid", 1000), ("due", 200)]


def test_api_image_upload():
    with open(SAMPLES / "bill_noisy.jpg", "rb") as f:
        body = client.post("/api/v1/detect", files={"file": ("b.jpg", f, "image/jpeg")}).json()
    assert body["status"] == "ok" and body["currency"] == "INR"
    assert {(a["type"], a["value"]) for a in body["amounts"]} >= \
        {("total_bill", 1200), ("paid", 1000), ("due", 200)}


def test_api_guardrail_no_amounts():
    body = client.post("/api/v1/detect", json={"text": "Thank you for visiting"}).json()
    assert body["status"] == "no_amounts_found"


def test_api_bad_inputs():
    assert client.post("/api/v1/detect", json={}).status_code == 400
    assert client.post("/api/v1/detect", json={"text": "   "}).status_code == 400
    r = client.post("/api/v1/detect", files={"file": ("a.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 415 and r.json()["status"] == "error"
    r = client.post("/api/v1/detect", files={"file": ("a.png", b"not an image", "image/png")})
    assert r.status_code == 422


def test_debug_mode_shows_all_steps():
    body = client.post("/api/v1/detect?debug=true", json={"text": SPEC_TEXT}).json()
    assert set(body["steps"]) == {"step1_ocr", "step2_normalization", "step3_classification"}
    assert body["debug"]["arithmetic_check"] == "passed"


# ---------- frontend ----------

def test_root_serves_html_frontend():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<html" in r.text.lower()
