#!/usr/bin/env bash
# Run the C++ unit tests against the golden data.
set -euo pipefail
cd "$(dirname "$0")/../.."

./build/unit_tests tests/golden weights/tiny_weights.bin weights/tiny_tokenizer.bin
