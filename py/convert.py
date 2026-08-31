#!/usr/bin/env python3
"""Convert HuggingFace safetensors weights to the GMW1 INT8 format (best effort).

Requires the `safetensors` package:  pip install safetensors
Real Gemma 3n checkpoints are gated on HuggingFace — this script cannot be
verified locally without access; it validates shapes and fails loudly on
anything the engine's architecture doesn't support.

Usage:
    python3 py/convert.py --weights-in weights/model.safetensors \
        --weights-out weights/gemma_i8.bin --model-size 270m \
        [--rope-base 10000] [--max-seq-len N]
"""

import argparse
import sys

import numpy as np

from common import CONFIGS, write_weights, tensor_shapes

SUPPORTED = {
    # HF name pattern -> our tensor name pattern; None = optional
    "embed_tokens": "model.embed_tokens.weight",
    "embed_positions": None,  # real Gemma has no learned positions -> zeros
    "q_proj": "model.layers.{l}.self_attn.q_proj.weight",
    "k_proj": "model.layers.{l}.self_attn.k_proj.weight",
    "v_proj": "model.layers.{l}.self_attn.v_proj.weight",
    "o_proj": "model.layers.{l}.self_attn.o_proj.weight",
    "gate_proj": "model.layers.{l}.mlp.gate_proj.weight",
    "up_proj": "model.layers.{l}.mlp.up_proj.weight",
    "down_proj": "model.layers.{l}.mlp.down_proj.weight",
    "gamma_attn": "model.layers.{l}.input_layernorm.weight",
    "gamma_ffn": "model.layers.{l}.post_attention_layernorm.weight",
    "final_norm": "model.norm.weight",
    "lm_head": None,  # tied to embed_tokens unless model.lm_head.weight exists
}

# Tensors this engine's architecture does not support — fail loudly.
UNSUPPORTED = (
    "q_norm", "k_norm", "qkv_norm", "mamba", "sliding_window",
    "rotary_emb", "vision",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-in", required=True)
    ap.add_argument("--weights-out", required=True)
    ap.add_argument("--model-size", default="270m", choices=sorted(CONFIGS))
    ap.add_argument("--rope-base", type=float, default=None)
    ap.add_argument("--max-seq-len", type=int, default=None)
    args = ap.parse_args()

    try:
        from safetensors import safe_open
    except ImportError:
        sys.exit("convert.py needs the 'safetensors' package: pip install safetensors")

    cfg = dict(CONFIGS[args.model_size])
    if args.max_seq_len is not None:
        cfg["max_seq_len"] = args.max_seq_len
    if args.rope_base is not None:
        cfg["rope_base"] = args.rope_base

    with safe_open(args.weights_in, framework="pt") as f:
        names = set(f.keys())
        for u in UNSUPPORTED:
            if any(u in n for n in names):
                sys.exit(f"unsupported tensor family '{u}' in checkpoint — the engine "
                         "implements the teaching architecture, not real Gemma 3n")

        tensors = {}
        shapes = tensor_shapes(cfg)
        got_vocab = got_pos = None

        # embed_tokens defines vocab + dim
        emb = f.get_tensor(SUPPORTED["embed_tokens"]).numpy()
        expect = shapes["embed_tokens"]
        if emb.shape != expect:
            sys.exit(f"embed_tokens shape {tuple(emb.shape)} != expected {expect}")
        tensors["embed_tokens"] = emb
        got_vocab, _ = emb.shape

        for l in range(cfg["n_layers"]):
            for kind, pat in SUPPORTED.items():
                if kind in ("embed_tokens", "embed_positions", "final_norm", "lm_head"):
                    continue
                name = pat.format(l=l)
                if name not in names:
                    sys.exit(f"missing tensor {name}")
                t = f.get_tensor(name).numpy()
                if t.shape != shapes[f"{kind}_{l}"]:
                    sys.exit(f"{name} shape {tuple(t.shape)} != expected {shapes[f'{kind}_{l}']}")
                tensors[f"{kind}_{l}"] = t

        fn = SUPPORTED["final_norm"]
        if fn not in names:
            sys.exit(f"missing tensor {fn}")
        tensors["final_norm_gamma"] = f.get_tensor(fn).numpy()

        # learned position bias: real Gemma checkpoints don't have one (RoPE
        # only). The engine adds embed_positions[pos], so zeros keep the
        # forward pass identical to the checkpoint's math.
        if "model.embed_positions.weight" in names:
            tensors["embed_positions"] = f.get_tensor("model.embed_positions.weight").numpy()
            got_pos = tensors["embed_positions"].shape[0]
            if got_pos != cfg["max_seq_len"]:
                cfg["max_seq_len"] = got_pos
                print(f"note: embed_positions has {got_pos} rows; max_seq_len updated")
        else:
            print("note: no embed_positions in checkpoint; writing zeros (RoPE provides positions)")
            tensors["embed_positions"] = np.zeros(
                (cfg["max_seq_len"], cfg["dim"]), np.float32)

        # lm_head: tied to embed_tokens unless present
        if "model.lm_head.weight" in names:
            tensors["lm_head"] = f.get_tensor("model.lm_head.weight").numpy()
        else:
            tensors["lm_head"] = tensors["embed_tokens"]

    write_weights(args.weights_out, cfg, tensors)
    print(f"wrote {args.weights_out} (vocab={got_vocab}, layers={cfg['n_layers']}, "
          f"dim={cfg['dim']}, max_seq_len={cfg['max_seq_len']})")


if __name__ == "__main__":
    main()
