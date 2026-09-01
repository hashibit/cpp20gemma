# cpp20gemma

A single-threaded, zero-dependency C++20 inference engine implementing a Gemma-style (RoPE + GQA + GeGLU) language model:

- **INT8 weight quantization** (symmetric per-tensor scale, 75% memory savings); quantization happens exactly once, in the Python conversion script
- **Hand-written SIMD kernels**: NEON on arm64 (int8→int16→int32→f32 widening + FMLA, the arm64 counterpart of the article's AVX2 approach), AVX2 on x86-64, plus a scalar fallback with no SIMD
- **Complete model architecture**: learnable position bias, RMSNorm, RoPE (GPT-NeoX interleaved pairs), GQA (4 Q heads sharing 1 KV head), GeGLU, KV cache (caches the already-rotated k), Min-P sampling
- **Zero dependencies**: one Makefile, `make` is all you need; no BLAS/MKL, no protobuf
- **Per-layer unit tests**: every layer's C++ output is compared against a NumPy reference implementation; end-to-end 64-step greedy decoding matches the Python reference exactly, token for token

## Build

Requirements: a C++20-capable compiler (Apple clang 14+ / GCC 10+), Python 3 + numpy (only needed for tests and conversion scripts).

```bash
make gemma        # main inference binary: build/gemma
make unittests    # unit tests: build/unit_tests
make integration  # integration test: build/integration
make test         # generate test data and run all tests
```

On arm64 no special flags are needed (NEON is part of the arm64 baseline); on x86-64 `-march=native` is added automatically.

## Quick start

```bash
# 1. Generate a test model (deterministic random weights, no HuggingFace login needed)
python3 py/make_test_weights.py --size tiny \
    --out-weights weights/tiny_weights.bin \
    --out-tok weights/tiny_tokenizer.bin

# 2. Inference (Min-P sampling)
./build/gemma --weights_path weights/tiny_weights.bin \
    --tok_path weights/tiny_tokenizer.bin \
    --model_size tiny --n_dec 64 --minp 0.1 --temp 0.7 --seed 42 \
    --prompt "The meaning of life"

# 3. Greedy decoding (omit --minp for greedy)
./build/gemma --weights_path weights/tiny_weights.bin \
    --tok_path weights/tiny_tokenizer.bin \
    --n_dec 64 --prompt "What is a transformer?"
```

> The test model's weights are random numbers, so its output is gibberish — it validates the whole inference pipeline (quantize → load → forward → sample → decode), not model quality.

### CLI options

| Option | Default | Description |
|---|---|---|
| `--weights_path` | required | Weights file in GMW1 format |
| `--tok_path` | `weights_path + ".tok"` | Tokenizer in GMT1 format |
| `--model_size` | — | Preset name check: `270m` / `1B` / `tiny` |
| `--prompt` | "What is a transformer?" | Input prompt |
| `--n_dec` | 250 | Maximum number of tokens to generate |
| `--temp` | 0.7 | Temperature; ≤0 selects greedy |
| `--minp` | unset = greedy | Min-P sampling threshold (0–1; lower = more conservative) |
| `--max_cache_len` | 8192 | KV cache allocation cap (in positions) |
| `--seed` | clock | Sampling RNG seed |
| `--terminate_on_eos` | 1 | Stop on EOS |
| `--chat_format` | 0 | Wrap the prompt in the Gemma chat template (`<start_of_turn>` format) |
| `--dump` | — | Print the model config and tensor layout, then exit |

## Real Gemma weights (best effort)

Real Gemma 3n is a gated model (requires HF login + license acceptance), so this path could not be verified locally. Conversion scripts are provided, and shapes are strictly validated:

```bash
pip install safetensors sentencepiece
huggingface-cli login
hf download google/gemma-3n-E2B-it tokenizer.model --local-dir weights
hf download google/gemma-3n-E2B-it model.safetensors --local-dir weights

python3 py/convert.py --weights-in weights/model.safetensors \
    --weights-out weights/gemma_i8.bin --model-size 270m
python3 py/convert_tokenizer.py --tok-in weights/tokenizer.model \
    --tok-out weights/gemma_i8.bin.tok
```

Note: this engine implements the **architecture described in the articles** (a simplified version). Real Gemma 3n has q_norm/k_norm, sliding-window attention, and other structures the conversion script will report explicitly rather than silently mis-load. When `embed_positions` is missing, zeros are written automatically (RoPE already provides the position information).

## Architecture & implementation

```
hidden = embed_tokens[tok] + embed_positions[pos]      # learnable position bias
each layer:  h = h + attn(rmsnorm(h));  h = h + mlp(rmsnorm(h))
logits = lm_head(rmsnorm(h))
```

- **Quantization**: `scale = max(|w|, 1e-9)/127`, `w8 = clip(round(w/scale), -128, 127)`. Dequantization trick: `(x·w8) × scale` — the scale is multiplied in only once, after the dot product
- **GQA**: the KV projection computes only `n_kv_heads × head_dim` rows; KV cache memory is 1/4 of MHA
- **RoPE**: cos/sin are materialized lazily per position (no full 128K precompute); RoPE is applied to k **before** it is written to the cache
- **Min-P**: a dynamic threshold of `max_p × min_p`, better suited to small models than Top-K/Top-P; `--seed` fixes a splitmix64 RNG so tests are reproducible
- **Weight layout**: all matrices are stored row-major as `[out][in]`, so each output row is a contiguous stretch of memory — a cache-friendly GEMV
- **Loading**: one `aligned_alloc(64)` plus one `read` for the whole file; the file size is fully determined by header fields (shape validation comes for free)

## Binary formats

### Weights: GMW1 v1 (little-endian)

A 64-byte fixed header: `magic "GMW1" | version=1 | n_layers | dim | n_heads | n_kv_heads | mlp_dim | vocab_size | max_seq_len | head_dim | f32 rope_base | reserved(20, must be all 0)`

Tensor section in fixed order (each tensor = `N bytes int8 + 4-byte f32 scale`):

1. `embed_tokens [vocab][dim]`, `embed_positions [max_seq_len][dim]`
2. Per layer: `q_proj [dim][dim]`, `k_proj/v_proj [n_kv_heads×head_dim][dim]`, `o_proj [dim][dim]`, `gate/up_proj [mlp_dim][dim]`, `down_proj [dim][mlp_dim]`, `gamma_attn/ffn [dim]`
3. `final_norm_gamma [dim]`, `lm_head [vocab][dim]`

### Tokenizer: GMT1 v1 (little-endian)

A 48-byte header: `magic "GMT1" | version | vocab_size | bos_id | eos_id | unk_id | pad_id | reserved`

Each entry: `u32 len | u32 type (0 normal / 1 byte-fallback / 2 control) | f32 score | len bytes UTF-8`

Encoding: greedy longest-prefix matching + per-byte fallback (`<0xXX>` byte tokens); control tokens are skipped during decoding.

## Tests

```bash
make test
```

- **13 unit tests**: loader (including corrupt-file rejection), NEON/f32 GEMV, dequantized rows, RMSNorm, RoPE, SiLU/GeGLU, softmax (including large-number stability), per-layer forward pass (embedding/norm/attention/MLP/final/logits all compared against NumPy references), raw attention scores + the rope-before-cache convention for k, KV cache, sampler (greedy + Min-P sampled ids exactly equal to Python), tokenizer round-trip
- **Integration test**: 64-step greedy decoding matches the Python reference **exactly, id for id**; final logits tolerance 2e-3 (observed max deviation ~3e-5); on near-ties, the logit difference between both sides is printed for diagnosis
- Tolerance policy: the reference uses **dequantized weights** (same source as the engine), so the only difference is fp32 accumulation order; matmul/attention/logits use `atol 2e-3 / rtol 1e-3`, and mlp (the largest-magnitude intermediate) is relaxed to `rtol 5e-3`

## Repository layout

```
src/          C++ engine: config / weights / kernel / ops / kv_cache / tokenizer / sampler / model / main
py/           Python: test-weight generation, NumPy reference implementation, golden generation, safetensors/sentencepiece conversion
tests/        Unit tests, integration test, data generation and run scripts
```

## Known limitations

- Single-threaded (a deliberate design choice from the articles: no lock contention, better cache hit rates)
- Implements the articles' simplified architecture, not bit-exact real Gemma 3n (q_norm/k_norm etc. are unsupported)
- Prefill (prompt processing) is not batched into GEMM, so long prompts are slow on CPU; decoding is the optimization target
- A full 128K-context KV allocation is impractical (≈10 GB); the default is `--max_cache_len 8192`
