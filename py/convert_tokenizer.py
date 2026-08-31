#!/usr/bin/env python3
"""Convert a SentencePiece tokenizer.model to the GMT1 binary format (best effort).

Requires the `sentencepiece` package:  pip install sentencepiece

Usage:
    python3 py/convert_tokenizer.py --tok-in weights/tokenizer.model \
        --tok-out weights/tokenizer_gemma3.bin
"""

import argparse
import sys

from common import write_tokenizer

# SentencePiece piece types -> our token types
_PIECE_TO_TYPE = {1: 0, 2: 2, 3: 2, 4: 0, 5: 1, 6: 2}  # NORMAL/UNK/CONTROL/USER/BYTE/UNUSED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-in", required=True)
    ap.add_argument("--tok-out", required=True)
    args = ap.parse_args()

    try:
        import sentencepiece as spm
    except ImportError:
        sys.exit("convert_tokenizer.py needs 'sentencepiece': pip install sentencepiece")

    sp = spm.SentencePieceProcessor(model_file=args.tok_in)
    vocab = sp.get_piece_size()
    entries = []
    for i in range(vocab):
        piece = sp.id_to_piece(i)
        typ = _PIECE_TO_TYPE.get(sp.get_piece_type(i), 0)
        if typ == 1 and piece.startswith("<0x") and piece.endswith(">"):
            byte = int(piece[3:5], 16)
            piece = bytes([byte])  # byte-fallback pieces are raw bytes
        else:
            piece = piece.encode("utf-8")
        score = sp.get_score(i)
        entries.append((piece, typ, score))

    ids = {
        "bos": int(sp.bos_id()),
        "eos": int(sp.eos_id()),
        "unk": int(sp.unk_id()),
        "pad": int(sp.pad_id()),
    }
    write_tokenizer(args.tok_out, entries, ids)
    print(f"wrote {args.tok_out} ({vocab} tokens, bos={ids['bos']} eos={ids['eos']} "
          f"unk={ids['unk']} pad={ids['pad']})")


if __name__ == "__main__":
    main()
