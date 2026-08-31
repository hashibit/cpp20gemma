#!/usr/bin/env python3
"""Generate golden data for the C++ unit tests (all deterministic).

Runs the NumPy reference model over a fixed 8-token prompt with recording on,
and dumps per-layer intermediate outputs plus standalone op test vectors into
tests/golden/ as raw little-endian f32/u32 files. Shapes are hardcoded in the
C++ test (tiny config: 2 layers, dim 256, vocab 1024).
"""

import argparse
import os
import struct

import numpy as np

from common import quantize
from reference_model import (RefModel, SplitMix64, min_p_sample, silu, softmax)

PROMPT_TOKS = [13, 7, 42, 99, 5, 21, 8, 3]
RNG_SEED = 20260818


def write(path, arr):
    np.asarray(arr, dtype=np.float32).astype(np.float32, copy=False).tofile(path)


def write_u32(path, arr):
    np.asarray(arr, dtype=np.uint32).tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/tiny_weights.bin")
    ap.add_argument("--out", default="tests/golden")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = RefModel(args.weights)
    c = model.cfg
    d, v = c["dim"], c["vocab_size"]
    np_tok = len(PROMPT_TOKS)

    emb = np.zeros((np_tok, d), np.float32)
    norm_a = np.zeros((c["n_layers"], np_tok, d), np.float32)
    attn = np.zeros_like(norm_a)
    norm_f = np.zeros_like(norm_a)
    mlp = np.zeros_like(norm_a)
    fin = np.zeros((np_tok, d), np.float32)
    logits_all = np.zeros((np_tok, v), np.float32)
    scores_l1 = k_raw_l1 = None

    for pi, tok in enumerate(PROMPT_TOKS):
        logits, rec = model.forward_token(tok, pi, record=True)
        emb[pi] = rec["embed"]
        for l in range(c["n_layers"]):
            norm_a[l, pi] = rec["norm_a"][l]
            attn[l, pi] = rec["attn_out"][l]
            norm_f[l, pi] = rec["norm_f"][l]
            mlp[l, pi] = rec["mlp_out"][l]
        fin[pi] = rec["final_norm"]
        logits_all[pi] = logits
        if pi == np_tok - 1:
            scores_l1 = rec["scores_L1"].copy()
            k_raw_l1 = rec["k_raw_L1"].copy()

    # k_cached_L1 = rope-applied k as written into the cache at the last pos.
    k_cached_l1 = model.kv[1][0][np_tok - 1].copy()

    write(f"{args.out}/embedding_out.bin", emb)
    write(f"{args.out}/norm_a.bin", norm_a)
    write(f"{args.out}/attn_out.bin", attn)
    write(f"{args.out}/norm_f.bin", norm_f)
    write(f"{args.out}/mlp_out.bin", mlp)
    write(f"{args.out}/final_norm_out.bin", fin)
    write(f"{args.out}/logits.bin", logits_all)
    write(f"{args.out}/scores_L1.bin", scores_l1)
    write(f"{args.out}/k_raw_L1.bin", k_raw_l1)
    write(f"{args.out}/k_cached_L1.bin", k_cached_l1)

    rng = np.random.default_rng(RNG_SEED)

    # --- standalone op goldens -------------------------------------------
    rope_in = (np.arange(64, dtype=np.float32) * 0.13 - 3.7)
    write(f"{args.out}/rope_in.bin", rope_in)
    write(f"{args.out}/rope_out.bin", model.rope_apply(rope_in, 3))

    # dequant_row: embed_tokens row 13, dequantized exactly like the kernel.
    write(f"{args.out}/dequant_row.bin", model.t["embed_tokens"][13])

    # int8 GEMV: [16x32] int8 weights + scale + [32] x + [16] y.
    W = rng.standard_normal((16, 32)).astype(np.float32)
    x = rng.standard_normal((32,)).astype(np.float32)
    Wq, Ws = quantize(W)
    y = (Wq.astype(np.float32) @ x) * Ws
    with open(f"{args.out}/mm_i8.bin", "wb") as f:
        f.write(Wq.tobytes())
        f.write(struct.pack("<f", Ws))
        f.write(x.tobytes())
        f.write(y.astype(np.float32).tobytes())

    # f32 GEMV, same shape.
    W = rng.standard_normal((16, 32)).astype(np.float32)
    x = rng.standard_normal((32,)).astype(np.float32)
    with open(f"{args.out}/mm_f32.bin", "wb") as f:
        f.write(W.tobytes())
        f.write(x.tobytes())
        f.write((W @ x).astype(np.float32).tobytes())

    silu_in = rng.standard_normal((64,)).astype(np.float32)
    write(f"{args.out}/silu_in.bin", silu_in)
    write(f"{args.out}/silu_out.bin", silu(silu_in))

    gate_in = rng.standard_normal((64,)).astype(np.float32)
    up_in = rng.standard_normal((64,)).astype(np.float32)
    write(f"{args.out}/geglu_gate_in.bin", gate_in)
    write(f"{args.out}/geglu_up_in.bin", up_in)
    write(f"{args.out}/geglu_out.bin", silu(gate_in) * up_in)

    softmax_in = rng.standard_normal((64,)).astype(np.float32) * 2.0
    write(f"{args.out}/softmax_in.bin", softmax_in)
    write(f"{args.out}/softmax_out.bin", softmax(softmax_in))
    softmax_big = softmax_in.copy()
    softmax_big[::3] += 1000.0
    softmax_big[1::3] -= 1000.0
    write(f"{args.out}/softmax_big_in.bin", softmax_big)
    write(f"{args.out}/softmax_big_out.bin", softmax(softmax_big))

    # --- sampler goldens ---------------------------------------------------
    slogits = logits_all[-1].astype(np.float32)
    write(f"{args.out}/sampler_logits.bin", slogits)
    t = np.float32(0.7)
    write(f"{args.out}/sampler_probs.bin", softmax(slogits / t))
    write_u32(f"{args.out}/sampler_greedy_id.bin", [int(np.argmax(slogits))])
    rng2 = SplitMix64(42)
    draws = [min_p_sample(slogits, 0.7, 0.1, rng2) for _ in range(8)]
    write_u32(f"{args.out}/sampler_draws.bin", draws)

    print(f"golden data written to {args.out}")


if __name__ == "__main__":
    main()
