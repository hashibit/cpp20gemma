# Learn to Build an Inference Engine from Scratch

This is a tutorial series written for complete beginners, built around this repository's C++20 inference engine.
Every article follows the same structure: **why you need it → the core concept from zero → a worked example by hand → our real code → pitfalls we hit**.

The best way to learn: read an article with the corresponding source file open, run `make test` to check your understanding, then change the code and watch the tests fail.

## Reading path (in dependency order)

| # | Article | Source | In one sentence |
|---|---|---|---|
| 00 | [Overview: what an inference engine does](00-Overview.md) | `src/main.cpp` | The full-pipeline map; see the big picture first |
| 01 | [Tokenizer: how text becomes numbers](01-Tokenizer.md) | `src/tokenizer.cpp` | The model only knows numbers |
| 02 | [Binary format design: what a weights file looks like](02-Binary-Format.md) | `src/weights.cpp` `py/common.py` | Fixed header + fixed order = zero-cost validation |
| 03 | [Embeddings and positional encoding](03-Embeddings-and-Positional-Encoding.md) | `src/model.cpp` | Vector lookup + position bias |
| 04 | [RMSNorm: keeping the numbers from exploding](04-RMSNorm.md) | `src/ops.cpp` | Normalization in a one-line formula |
| 05 | [RoPE: rotary position embedding](05-RoPE.md) | `src/ops.cpp` | Encoding position into vectors via rotation |
| 06 | [Attention: how the model "reads" context](06-Attention.md) | `src/model.cpp` | Q/K/V, softmax, causality |
| 07 | [KV cache and GQA: the key to fast decoding](07-KV-Cache-and-GQA.md) | `src/kv_cache.h` | Never recompute history |
| 08 | [MLP and GeGLU: processing each word independently](08-MLP-and-GeGLU.md) | `src/model.cpp` | Attention exchanges information; the MLP processes it |
| 09 | [INT8 quantization: compressing 1 GB into 250 MB](09-INT8-Quantization.md) | `py/common.py` `src/kernel.cpp` | How the memory savings work |
| 10 | [SIMD and matrix multiplication: why it's fast](10-SIMD-and-Matrix-Multiplication.md) | `src/kernel.cpp` | One instruction computes 8 numbers |
| 11 | [Sampling: how the model "picks words"](11-Sampling.md) | `src/sampler.cpp` | Greedy, temperature, Min-P |
| 12 | [Testing strategy: how to prove the math is right](12-Testing-Strategy.md) | `tests/` `py/generate_golden.py` | NumPy reference + tolerances + real bug retrospectives |

## Suggested hands-on exercises (in order)

1. After 00–02: run `./build/gemma --weights_path weights/tiny_weights.bin --dump` and match every field against the articles
2. After 03–08: run `make test`, then deliberately break something (e.g. change `kvh = h * nkv / nh` inside `attention_block` to `0`) and watch how the tests fail and from which layer the error surfaces
3. After 09–11: change the temperature/sampling parameters in `py/make_test_weights.py` and observe the output; drop `--minp` and compare against the greedy output
4. After 12: write a golden test for a new operator (e.g. GELU instead of SiLU) — this is the real test of whether you've mastered it

## Prerequisites

Only this: one programming language (any), and knowing what functions and arrays are. Floating point, SIMD, and matrices are all taught from zero in the articles.

## What this engine is (honest disclaimer)

This engine implements a **simplified teaching architecture**, not bit-exact real Gemma 3n.
As teaching material it is complete enough: quantization, SIMD, GQA, RoPE, KV cache, sampling, per-layer tests — every part an inference engine needs, in roughly 1500 lines of C++, which is a size you can actually read through.
