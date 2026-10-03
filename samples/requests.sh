#!/usr/bin/env bash
# Sample requests. Usage: BASE=https://your-url.ngrok-free.app bash samples/requests.sh
BASE=${BASE:-http://localhost:8000}
DIR="$(cd "$(dirname "$0")" && pwd)"
run() { echo -e "\n### $1"; shift; curl -s "$@"; echo; }

run "Health"                    "$BASE/health"
run "Step 1 - extract (text)"   -X POST "$BASE/api/v1/extract"   -H "Content-Type: application/json" -d '{"text":"Total: INR 1200 | Paid: 1000 | Due: 200 | Discount: 10%"}'
run "Step 2 - normalize (noisy OCR text)" -X POST "$BASE/api/v1/normalize" -H "Content-Type: application/json" -d '{"text":"T0tal: Rs l200 | Pald: 1000 | Due: 200"}'
run "Step 3 - classify"         -X POST "$BASE/api/v1/classify"  -H "Content-Type: application/json" -d '{"text":"Total: INR 1200 | Paid: 1000 | Due: 200"}'
run "Step 4 - detect (text)"    -X POST "$BASE/api/v1/detect"    -H "Content-Type: application/json" -d '{"text":"Total: INR 1200 | Paid: 1000 | Due: 200 | Discount: 10%"}'
run "Step 4 - detect (noisy image)" -X POST "$BASE/api/v1/detect" -F "file=@$DIR/bill_noisy.jpg"
run "Step 4 - detect (clean image, debug)" -X POST "$BASE/api/v1/detect?debug=true" -F "file=@$DIR/bill_clean.jpg"
run "Guardrail - no amounts"    -X POST "$BASE/api/v1/detect"    -F "file=@$DIR/no_amounts.jpg"
run "Error - empty text"        -X POST "$BASE/api/v1/detect"    -H "Content-Type: application/json" -d '{"text":""}'
