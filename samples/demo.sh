#!/usr/bin/env bash
# Walkthrough of every endpoint. Press Enter to run each step.
BASE=${BASE:-https://cosmos-pointing-earpiece.ngrok-free.dev}
cd "$(dirname "$0")/.."

step() {
  clear
  echo "# $1"
  echo "\$ $2"
  echo
  eval "$2"
  echo
  read -r -p "Press Enter to continue..."
}

step "Step 1: extract (text)" \
  "curl -s -X POST $BASE/api/v1/extract -H 'Content-Type: application/json' -d '{\"text\":\"Total: INR 1200 | Paid: 1000 | Due: 200 | Discount: 10%\"}' | python3 -m json.tool"
step "Step 1: extract (image, OCR)" \
  "curl -s -X POST $BASE/api/v1/extract -F 'file=@samples/bill_noisy.jpg' | python3 -m json.tool"
step "Step 2: normalize (noisy OCR text)" \
  "curl -s -X POST $BASE/api/v1/normalize -H 'Content-Type: application/json' -d '{\"text\":\"T0tal: Rs l200 | Pald: 1000 | Due: 200\"}' | python3 -m json.tool"
step "Step 3: classify" \
  "curl -s -X POST $BASE/api/v1/classify -H 'Content-Type: application/json' -d '{\"text\":\"Total: INR 1200 | Paid: 1000 | Due: 200\"}' | python3 -m json.tool"
step "Step 4: detect (text)" \
  "curl -s -X POST $BASE/api/v1/detect -H 'Content-Type: application/json' -d '{\"text\":\"Total: INR 1200 | Paid: 1000 | Due: 200 | Discount: 10%\"}' | python3 -m json.tool"
step "Step 4: detect (image)" \
  "curl -s -X POST $BASE/api/v1/detect -F 'file=@samples/bill_noisy.jpg' | python3 -m json.tool"
step "Guardrail: no amounts" \
  "curl -s -X POST $BASE/api/v1/detect -F 'file=@samples/no_amounts.jpg' | python3 -m json.tool"
step "Error handling: empty input, unsupported file type" \
  "curl -s -w '\nHTTP %{http_code}\n' -X POST $BASE/api/v1/detect -H 'Content-Type: application/json' -d '{\"text\":\"\"}'; curl -s -w '\nHTTP %{http_code}\n' -X POST $BASE/api/v1/detect -F 'file=@README.md;type=text/plain'"
step "Debug mode: full chain and arithmetic check" \
  "curl -s -X POST '$BASE/api/v1/detect?debug=true' -H 'Content-Type: application/json' -d '{\"text\":\"Total: INR 1200 | Paid: 1000 | Due: 200\"}' | python3 -m json.tool"

clear
echo "Done."
