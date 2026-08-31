#!/usr/bin/env python3
"""Generate deterministic random weights + a tiny tokenizer in our binary
formats. No HuggingFace access needed — this is the test fixture and the
perf-smoke model.

Usage:
    python3 py/make_test_weights.py --size tiny \
        --out-weights weights/tiny_weights.bin \
        --out-tok weights/tiny_tokenizer.bin
"""

import argparse
import struct

import numpy as np

from common import CONFIGS, write_weights, write_tokenizer

SEED = 20260818

# Common English words (including the e2e prompt words so it encodes as word
# pieces rather than bytes). The rest of the vocab is filled with "w<i>".
WORDS = [
    "The", "the", "a", "an", "of", "to", "in", "is", "it", "for", "on", "with",
    "as", "at", "by", "from", "that", "this", "these", "those", "and", "or",
    "but", "not", "no", "yes", "be", "are", "was", "were", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "should", "may", "might", "must", "if", "then", "than", "so", "such", "only",
    "also", "very", "just", "more", "most", "some", "any", "all", "each", "every",
    "other", "one", "two", "first", "last", "new", "old", "good", "great", "big",
    "small", "high", "low", "long", "short", "right", "left", "time", "year",
    "day", "way", "life", "world", "people", "man", "woman", "child", "thing",
    "things", "place", "work", "hand", "part", "case", "point", "fact", "word",
    "question", "answer", "problem", "system", "model", "data", "language",
    "meaning", "transformer", "attention", "token", "vector", "matrix", "layer",
    "network", "neural", "learning", "machine", "computer", "program", "code",
    "function", "value", "result", "example", "difference", "between", "same",
    "like", "know", "think", "make", "take", "give", "use", "find", "need",
    "want", "get", "go", "come", "see", "look", "say", "tell", "ask", "help",
    "try", "start", "stop", "run", "write", "read", "call", "name", "number",
    "story", "history", "science", "math", "art", "music", "book", "house",
    "city", "country", "water", "air", "fire", "earth", "sun", "moon", "star",
    "sky", "sea", "river", "tree", "flower", "animal", "bird", "fish", "food",
    "bread", "water", "money", "school", "teacher", "student", "game", "play",
    "love", "hate", "happy", "sad", "angry", "afraid", "strong", "weak", "fast",
    "slow", "easy", "hard", "soft", "warm", "cold", "hot", "clean", "dirty",
    "light", "dark", "quiet", "loud", "early", "late", "always", "never",
    "often", "sometimes", "again", "still", "here", "there", "where", "what",
    "who", "why", "how", "when", "which", "because", "while", "after", "before",
    "during", "through", "about", "into", "over", "under", "above", "below",
    "around", "near", "far", "next", "back", "front", "side", "end", "begin",
    "middle", "top", "bottom", "inside", "outside", "together", "apart",
    "yes", "hello", "world", "today", "tomorrow", "yesterday", "morning",
    "night", "week", "month", "hour", "minute", "second", "moment", "space",
    "mind", "body", "head", "hand", "eye", "ear", "face", "voice", "sound",
    "color", "red", "blue", "green", "yellow", "black", "white", " ", ",", ".",
    "?", "!", "\n", "'s", "ly", "ing", "ed", "er", "es", "'",
]


def make_weights(cfg):
    rng = np.random.default_rng(SEED)
    t = {}
    t["embed_tokens"] = rng.standard_normal(
        (cfg["vocab_size"], cfg["dim"])).astype(np.float32)
    t["embed_positions"] = rng.standard_normal(
        (cfg["max_seq_len"], cfg["dim"])).astype(np.float32)
    kv = cfg["n_kv_heads"] * cfg["head_dim"]
    for l in range(cfg["n_layers"]):
        d, m = cfg["dim"], cfg["mlp_dim"]
        t[f"q_proj_{l}"] = rng.standard_normal((d, d)).astype(np.float32)
        t[f"k_proj_{l}"] = rng.standard_normal((kv, d)).astype(np.float32)
        t[f"v_proj_{l}"] = rng.standard_normal((kv, d)).astype(np.float32)
        t[f"o_proj_{l}"] = rng.standard_normal((d, d)).astype(np.float32)
        t[f"gate_proj_{l}"] = rng.standard_normal((m, d)).astype(np.float32)
        t[f"up_proj_{l}"] = rng.standard_normal((m, d)).astype(np.float32)
        t[f"down_proj_{l}"] = rng.standard_normal((d, m)).astype(np.float32)
        t[f"gamma_attn_{l}"] = rng.standard_normal((d,)).astype(np.float32)
        t[f"gamma_ffn_{l}"] = rng.standard_normal((d,)).astype(np.float32)
    t["final_norm_gamma"] = rng.standard_normal((cfg["dim"],)).astype(np.float32)
    t["lm_head"] = rng.standard_normal(
        (cfg["vocab_size"], cfg["dim"])).astype(np.float32)
    return t


def make_tokenizer_entries(vocab_size):
    """Control tokens + word pieces + 256 byte-fallback tokens."""
    entries = []
    ids = {}
    for name, typ in (("pad", 2), ("eos", 2), ("bos", 2), ("unk", 2)):
        entries.append((f"<{name}>".encode(), typ, 0.0))
        ids[name] = len(entries) - 1
    for name in ("start_of_turn", "end_of_turn"):
        entries.append((f"<{name}>".encode(), 2, 0.0))
    n_words = vocab_size - len(entries) - 256
    assert n_words > 0, "vocab too small for words + 256 byte tokens"
    # Dedupe: duplicate pieces would make C++ (map keeps first id) and Python
    # (dict keeps last id) disagree on encode(). Also drop single-byte pieces —
    # every byte already has a byte-fallback token, so a 1-byte word piece is a
    # guaranteed duplicate.
    seen = set()
    words = []
    for w in WORDS:
        b = w.encode()
        if len(b) > 1 and b not in seen:
            seen.add(b)
            words.append(b)
    i = 0
    while len(words) < n_words:
        b = f"w{i}".encode()
        if b not in seen:
            seen.add(b)
            words.append(b)
        i += 1
    words = words[:n_words]
    for w in words:
        entries.append((w, 0, 0.0))
    for b in range(256):
        entries.append((bytes([b]), 1, 0.0))
    assert len(entries) == vocab_size
    return entries, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny", choices=sorted(CONFIGS))
    ap.add_argument("--vocab", type=int, default=None,
                    help="override vocab_size (fixture default is config value)")
    ap.add_argument("--out-weights", required=True)
    ap.add_argument("--out-tok", required=True)
    args = ap.parse_args()

    cfg = dict(CONFIGS[args.size])
    if args.vocab is not None:
        cfg["vocab_size"] = args.vocab

    write_weights(args.out_weights, cfg, make_weights(cfg))
    entries, ids = make_tokenizer_entries(cfg["vocab_size"])
    write_tokenizer(args.out_tok, entries, ids)
    print(f"wrote {args.out_weights} ({args.size} config) and {args.out_tok}")


if __name__ == "__main__":
    main()
