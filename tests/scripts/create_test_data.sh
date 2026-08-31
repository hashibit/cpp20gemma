#!/usr/bin/env bash
# Regenerate all test fixtures and golden data (deterministic, fixed seeds).
set -euo pipefail
cd "$(dirname "$0")/../.."

mkdir -p weights tests/golden
python3 py/make_test_weights.py --size tiny \
    --out-weights weights/tiny_weights.bin \
    --out-tok weights/tiny_tokenizer.bin
python3 py/generate_golden.py
python3 py/e2e_reference.py
echo "test data ready (weights/ + tests/golden/)"
