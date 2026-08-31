#!/usr/bin/env bash
# Run the end-to-end integration test (64-token greedy decode vs Python).
set -euo pipefail
cd "$(dirname "$0")/../.."

./build/integration weights/tiny_weights.bin weights/tiny_tokenizer.bin \
    tests/golden/e2e_tokens.bin tests/golden/e2e_last_logits.bin

# CLI regression: prompt + n_dec must never exceed the cache cap. The engine
# clamps n_dec to the remaining capacity and warns (KVCache::write has no
# bounds check, so exceeding it would be a silent out-of-bounds write).
# Prompt "The meaning of life" = 7 tokens; cap 32 -> 25 positions left.
out=$(./build/gemma --weights_path weights/tiny_weights.bin \
    --tok_path weights/tiny_tokenizer.bin \
    --prompt "The meaning of life" --n_dec 64 --max_cache_len 32 \
    --terminate_on_eos 0 --seed 42 2>&1)
echo "$out" | grep -q "clamping n_dec from 64 to 25" \
    || { echo "FAIL: expected n_dec clamp warning:"; echo "$out"; exit 1; }
echo "$out" | grep -q "generated 25 tokens" \
    || { echo "FAIL: expected exactly 25 generated tokens:"; echo "$out"; exit 1; }
