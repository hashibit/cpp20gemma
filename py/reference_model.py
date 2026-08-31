"""NumPy reference implementation — the single source of truth for all golden
data. Every computation mirrors the C++ math:

  * weights are loaded INT8 and dequantized (fp32) exactly like the engine
  * RMSNorm: sequential-mean-style fp32 sum of squares (np.mean float32)
  * RoPE: GPT-NeoX interleaved pairs, rope-applied k goes into the KV cache
  * attention: causal by loop bounds, softmax with max subtraction
  * MLP: GeGLU — silu(gate) * up, then down projection
  * sampling: splitmix64 + min-p mirrored bit-for-bit (float32 softmax,
    double cumulative walk)

Only fp32 accumulation *order* differs from C++ (NumPy matmul vs sequential
FMA) — the tolerances absorb that.
"""

import struct

import numpy as np

from common import EPS, tensor_order, tensor_shapes


# ---------------------------------------------------------------------------
# Binary readers
# ---------------------------------------------------------------------------
def read_weights(path):
    """Load a GMW1 file; dequantize everything into a dict of fp32 tensors."""
    with open(path, "rb") as f:
        data = f.read()
    magic, ver, n_layers, dim, n_heads, n_kv, mlp, vocab, max_seq, head_dim = \
        struct.unpack_from("<4sI8I", data, 0)
    rope_base, = struct.unpack_from("<f", data, 40)
    assert magic == b"GMW1" and ver == 1
    cfg = dict(n_layers=n_layers, dim=dim, n_heads=n_heads, n_kv_heads=n_kv,
               mlp_dim=mlp, vocab_size=vocab, max_seq_len=max_seq,
               head_dim=head_dim, rope_base=float(rope_base))
    shapes = tensor_shapes(cfg)
    tensors = {}
    off = 64
    for name in tensor_order(cfg):
        sh = shapes[name]
        is_vec = len(sh) == 1
        if is_vec:
            sh = (sh[0], 1)  # gammas are vectors; stored as rows
        r, c = sh
        n = int(r) * int(c)
        q = np.frombuffer(data, np.int8, n, off).reshape(r, c)
        scale = np.frombuffer(data, np.float32, 1, off + n)[0]
        tensors[name] = (q.reshape(-1) if is_vec else q).astype(np.float32) * np.float32(scale)
        off += n + 4
    assert off == len(data)
    return cfg, tensors


def read_tokenizer(path):
    """Load a GMT1 file. Returns (entries, ids) where entries is a list of
    (piece: bytes, type: int) and ids is a dict bos/eos/unk/pad."""
    with open(path, "rb") as f:
        data = f.read()
    magic, ver, vocab, bos, eos, unk, pad = struct.unpack_from("<4sI5I", data, 0)
    assert magic == b"GMT1" and ver == 1
    entries = []
    off = 48
    for _ in range(vocab):
        ln, typ, score = struct.unpack_from("<IIf", data, off)
        off += 12
        piece = data[off:off + ln]
        off += ln
        entries.append((piece, typ))
    return entries, dict(bos=bos, eos=eos, unk=unk, pad=pad)


# ---------------------------------------------------------------------------
# Ops (mirror src/ops.cpp and src/sampler.cpp)
# ---------------------------------------------------------------------------
def rmsnorm(x, gamma):
    x = np.asarray(x, np.float32)
    ss = np.mean(x * x, dtype=np.float32)
    return (x / np.sqrt(ss + np.float32(EPS), dtype=np.float32) * gamma).astype(np.float32)


def silu(x):
    x = np.asarray(x, np.float32)
    e = np.exp(x)
    return (x * e / (np.float32(1.0) + e)).astype(np.float32)


def softmax(x):
    x = np.asarray(x, np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return (e / np.sum(e, dtype=np.float32)).astype(np.float32)


class SplitMix64:
    """Bit-for-bit mirror of the C++ RNG."""

    MASK = 0xFFFFFFFFFFFFFFFF

    def __init__(self, seed):
        self.s = seed & self.MASK

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & self.MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return (z ^ (z >> 31)) & self.MASK

    def uniform(self):
        return float(self.next() >> 11) * (1.0 / 9007199254740992.0)


def min_p_sample(logits, temp, min_p, rng):
    """Mirror of Sampler::min_p: float32 softmax, min-p filter, double walk."""
    logits = np.asarray(logits, np.float32)
    t = np.float32(temp)
    x = logits / t
    mx = np.max(x)
    p = softmax(x - mx)
    thr = np.float32(min_p) * np.float32(np.max(p))
    cands = np.nonzero(p >= thr)[0]
    if len(cands) == 0:
        cands = np.array([int(np.argmax(logits))])
    u = rng.uniform()
    cum = 0.0
    for c in cands:
        cum += float(p[c])
        if u < cum:
            return int(c)
    return int(cands[-1])


# ---------------------------------------------------------------------------
# Tokenizer mirror (greedy longest prefix + byte fallback)
# ---------------------------------------------------------------------------
class TokenizerRef:
    def __init__(self, path):
        self.entries, self.ids = read_tokenizer(path)
        self.piece_to_id = {piece: i for i, (piece, _) in enumerate(self.entries)}
        if len(self.piece_to_id) != len(self.entries):
            raise ValueError("tokenizer has duplicate pieces; encode() would be ambiguous")
        self.byte_token = {}
        for i, (piece, typ) in enumerate(self.entries):
            if typ == 1:
                self.byte_token[piece[0]] = i
        self.max_piece_len = max((len(p) for p, _ in self.entries), default=0)

    def encode(self, text):
        if isinstance(text, str):
            text = text.encode("utf-8")
        ids = []
        i = 0
        while i < len(text):
            L = min(self.max_piece_len, len(text) - i)
            matched = False
            while L > 0:
                piece = text[i:i + L]
                if piece in self.piece_to_id:
                    ids.append(self.piece_to_id[piece])
                    i += L
                    matched = True
                    break
                L -= 1
            if not matched:
                ids.append(self.byte_token.get(text[i], self.ids["unk"]))
                i += 1
        return ids

    def decode_id(self, tid):
        piece, typ = self.entries[tid]
        return b"" if typ == 2 else piece

    def decode(self, ids):
        return b"".join(self.decode_id(t) for t in ids)


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------
class RefModel:
    def __init__(self, path):
        self.cfg, self.t = read_weights(path)
        c = self.cfg
        self.dim = c["dim"]
        self.hd = c["head_dim"]
        self.nh = c["n_heads"]
        self.nkv = c["n_kv_heads"]
        self.n_layers = c["n_layers"]
        # KV cache: [n_layers] of (k, v) arrays [max_seq_len][nkv*hd]
        self.kv = [
            (np.zeros((c["max_seq_len"], c["n_kv_heads"] * c["head_dim"]), np.float32),
             np.zeros((c["max_seq_len"], c["n_kv_heads"] * c["head_dim"]), np.float32))
            for _ in range(c["n_layers"])
        ]
        # RoPE tables: cos/sin [max_seq_len][hd/2]
        nhalf = self.hd // 2
        freqs = (1.0 / (c["rope_base"] ** (2 * np.arange(nhalf) / self.hd))).astype(np.float32)
        ang = (np.arange(c["max_seq_len"], dtype=np.float32)[:, None] * freqs[None, :])
        self.rope_cos = np.cos(ang).astype(np.float32)
        self.rope_sin = np.sin(ang).astype(np.float32)

    def rope_apply(self, x, pos):
        """GPT-NeoX interleaved rotation. x: [..., hd]."""
        x = np.asarray(x, np.float32).copy()
        c = self.rope_cos[pos]
        s = self.rope_sin[pos]
        # copy(): these are strided views of x — the first assignment would
        # overwrite the inputs of the second (the classic in-place RoPE bug)
        x0, x1 = x[..., 0::2].copy(), x[..., 1::2].copy()
        x[..., 0::2] = x0 * c - x1 * s
        x[..., 1::2] = x0 * s + x1 * c
        return x

    def forward_token(self, token, pos, record=False):
        """One forward step; fills the KV cache at `pos` and returns logits."""
        c = self.cfg
        h = (self.t["embed_tokens"][token] + self.t["embed_positions"][pos]).astype(np.float32)
        rec = {"embed": h.copy()} if record else None

        for l in range(self.n_layers):
            gamma_a = self.t[f"gamma_attn_{l}"]
            hn = rmsnorm(h, gamma_a)
            if record:
                rec.setdefault("norm_a", []).append(hn.copy())

            q = self.t[f"q_proj_{l}"] @ hn
            k = self.t[f"k_proj_{l}"] @ hn
            v = self.t[f"v_proj_{l}"] @ hn
            if record and l == 1:
                rec["k_raw_L1"] = k.copy()
            q = self.rope_apply(q.reshape(self.nh, self.hd), pos).reshape(-1)
            k = self.rope_apply(k.reshape(self.nkv, self.hd), pos).reshape(-1)

            k_arr, v_arr = self.kv[l]
            k_arr[pos] = k
            v_arr[pos] = v

            out = np.zeros((self.nh, self.hd), np.float32)
            if record and l == 1:
                rec["scores_L1"] = np.zeros(pos + 1, np.float32)
            for hq in range(self.nh):
                kh = hq * self.nkv // self.nh
                kk = k_arr[:pos + 1, kh * self.hd:(kh + 1) * self.hd]
                s = (kk @ q[hq * self.hd:(hq + 1) * self.hd] / np.sqrt(self.hd)).astype(np.float32)
                if record and l == 1 and hq == 0:
                    rec["scores_L1"][:] = s
                p = softmax(s)
                out[hq] = p @ v_arr[:pos + 1, kh * self.hd:(kh + 1) * self.hd]

            attn = self.t[f"o_proj_{l}"] @ out.reshape(-1)
            if record:
                rec.setdefault("attn_out", []).append(attn.copy())
            h = h + attn

            hn = rmsnorm(h, self.t[f"gamma_ffn_{l}"])
            if record:
                rec.setdefault("norm_f", []).append(hn.copy())
            g = silu(self.t[f"gate_proj_{l}"] @ hn) * (self.t[f"up_proj_{l}"] @ hn)
            mo = self.t[f"down_proj_{l}"] @ g
            if record:
                rec.setdefault("mlp_out", []).append(mo.copy())
            h = h + mo

        hn = rmsnorm(h, self.t["final_norm_gamma"])
        logits = self.t["lm_head"] @ hn
        if record:
            rec["final_norm"] = hn.copy()
        return logits.astype(np.float32), rec
