#!/usr/bin/env python3
"""End-to-end reference: greedy-decode the tiny model for 64 tokens and dump
the token sequence + final logits. The C++ integration test must reproduce the
token ids exactly (both sides encode the prompt with the same longest-match
tokenizer and both decode greedily with no EOS termination)."""

import argparse
import os

import numpy as np

from reference_model import RefModel, TokenizerRef

PROMPT = "The meaning of life"
N_DEC = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/tiny_weights.bin")
    ap.add_argument("--tok", default="weights/tiny_tokenizer.bin")
    ap.add_argument("--out", default="tests/golden")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = RefModel(args.weights)
    tok = TokenizerRef(args.tok)

    prompt_toks = tok.encode(PROMPT)
    logits = None
    for pi, t in enumerate(prompt_toks):
        logits, _ = model.forward_token(t, pi)

    gen = []
    for step in range(N_DEC):
        nxt = int(np.argmax(logits))
        gen.append(nxt)
        logits, _ = model.forward_token(nxt, len(prompt_toks) + step)

    np.asarray(gen, np.uint32).tofile(f"{args.out}/e2e_tokens.bin")
    np.asarray(logits, np.float32).tofile(f"{args.out}/e2e_last_logits.bin")
    print(f"e2e reference: prompt '{PROMPT}' -> {len(prompt_toks)} tokens, "
          f"{N_DEC} decoded, written to {args.out}")


if __name__ == "__main__":
    main()
