"""Shared configs, INT8 quantization, and binary writers.

Weight layout note: every matrix is stored [out][in] row-major so each output
element is a dot over one contiguous row (cache-friendly GEMV). In PyTorch /
HuggingFace terms that is simply the weight matrix itself in row-major order.
"""

import os
import struct

import numpy as np

EPS = 1e-6  # RMSNorm epsilon (matches Gemma)

# Named configs. "270m"/"1B" are the larger-scale fixtures;
# vocab_size/max_seq_len are fixture defaults (a real conversion overrides
# them from the actual safetensors shapes).
CONFIGS = {
    "tiny": dict(n_layers=2, dim=256, n_heads=4, n_kv_heads=1, mlp_dim=1024,
                 vocab_size=1024, max_seq_len=256, head_dim=64, rope_base=10000.0),
    "270m": dict(n_layers=26, dim=1536, n_heads=4, n_kv_heads=1, mlp_dim=6144,
                 vocab_size=4096, max_seq_len=4096, head_dim=384, rope_base=10000.0),
    "1B": dict(n_layers=28, dim=2048, n_heads=4, n_kv_heads=1, mlp_dim=8192,
               vocab_size=4096, max_seq_len=4096, head_dim=512, rope_base=10000.0),
}

def tensor_order(cfg):
    n = cfg["n_layers"]
    order = ["embed_tokens", "embed_positions"]
    for l in range(n):
        order += [f"q_proj_{l}", f"k_proj_{l}", f"v_proj_{l}", f"o_proj_{l}",
                  f"gate_proj_{l}", f"up_proj_{l}", f"down_proj_{l}",
                  f"gamma_attn_{l}", f"gamma_ffn_{l}"]
    order += ["final_norm_gamma", "lm_head"]
    return order


def tensor_shapes(cfg):
    d, m, v, s = cfg["dim"], cfg["mlp_dim"], cfg["vocab_size"], cfg["max_seq_len"]
    kv = cfg["n_kv_heads"] * cfg["head_dim"]
    shapes = {
        "embed_tokens": (v, d),
        "embed_positions": (s, d),
        "final_norm_gamma": (d,),
        "lm_head": (v, d),
    }
    for l in range(cfg["n_layers"]):
        shapes[f"q_proj_{l}"] = (d, d)
        shapes[f"k_proj_{l}"] = (kv, d)
        shapes[f"v_proj_{l}"] = (kv, d)
        shapes[f"o_proj_{l}"] = (d, d)
        shapes[f"gate_proj_{l}"] = (m, d)
        shapes[f"up_proj_{l}"] = (m, d)
        shapes[f"down_proj_{l}"] = (d, m)
        shapes[f"gamma_attn_{l}"] = (d,)
        shapes[f"gamma_ffn_{l}"] = (d,)
    return shapes


def quantize(w):
    """Symmetric per-tensor INT8 quantization. The zero-tensor guard is
    mandatory: without it scale = 0 and every dequantized value becomes NaN."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    scale = max(float(np.max(np.abs(w))), 1e-9) / 127.0
    q = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return q, np.float32(scale)


def write_weights(path, cfg, tensors):
    """tensors: dict name -> fp32 ndarray. Every tensor is [N int8][f32 scale]."""
    order = tensor_order(cfg)
    header = struct.pack(
        "<4sI" + "I" * 8 + "f" + "20s",
        b"GMW1", 1,
        cfg["n_layers"], cfg["dim"], cfg["n_heads"], cfg["n_kv_heads"],
        cfg["mlp_dim"], cfg["vocab_size"], cfg["max_seq_len"], cfg["head_dim"],
        np.float32(cfg["rope_base"]), b"\x00" * 20)
    assert len(header) == 64
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        for name in order:
            q, s = quantize(tensors[name])
            f.write(q.tobytes())
            f.write(struct.pack("<f", s))


def write_tokenizer(path, entries, ids):
    """entries: list of (piece: bytes, type: int, score: float).
    ids: dict with bos/eos/unk/pad. Types: 0 normal, 1 byte, 2 control."""
    header = struct.pack(
        "<4sI" + "I" * 5 + "20s",
        b"GMT1", 1, len(entries),
        ids["bos"], ids["eos"], ids["unk"], ids["pad"], b"\x00" * 20)
    assert len(header) == 48
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        for piece, typ, score in entries:
            f.write(struct.pack("<IIf", len(piece), typ, score))
            f.write(piece)
