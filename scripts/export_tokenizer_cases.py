#!/usr/bin/env python3
"""Pin GPT-2 tokenization edge cases for the JavaScript tokenizer.

    python scripts/export_tokenizer_cases.py     # -> web/tokenizer_cases.json

The browser demo needs to turn text into ids, which nothing in the engine does
-- deliberately, since Phase 1: every entry point takes ids as data. So
web/tokenizer.js is new, hand-written, and the component of Phase 5 most
likely to disagree with the reference. This writes the cases it is held to.

HF's tokenizer produces the expected values and is used for nothing else,
which is the role HF's GPT2LMHeadModel has for the model: it validates the
hand-written thing, it is never the thing. It runs here, once, and the result
is committed, so web/test_tokenizer.mjs needs neither Python nor HF.

Two kinds of case:

  encode   text -> ids. Hand-picked hard inputs -- whitespace of every Unicode
           kind, contractions, digits, CJK, emoji, combining marks, control
           characters -- plus a seeded random set drawn from a pool that mixes
           those categories, so the pre-tokenizer regex meets boundaries
           nobody thought to write down.
  decode   ids -> text, including id sequences that cut a multi-byte
           character in half. Byte-level BPE can do that, generation does it
           routinely, and the text a user sees depends on what happens.

The committed corpus ids are a third check, run by the test directly rather
than pinned here: decode then re-encode 24,576 real tokens.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER = ROOT / "weights" / "gpt2-124m"
DEFAULT_OUT = ROOT / "web" / "tokenizer_cases.json"

HAND = [
    "", " ", "  ", "\n", "\n\n", "\r\n", "\t", " \t\n ", "a", " a", "a ", "a  ",
    "Hello world", "Hello  world", "Hello world ", " Hello world",
    "The capital of France is",
    "don't won't I'm you're we've they'll she'd it's",
    "DON'T I'M YOU'RE",                      # the contractions are case-sensitive
    "'s 's''s rock'n'roll",
    "1234567890 3.14159 1,000,000 -42 1e10",
    "x=1;y=2\nif x<y:\n    print(x)\n",
    "def fibonacci(n):\n\treturn n if n < 2 else fibonacci(n-1)+fibonacci(n-2)",
    "naïve café résumé Zürich",
    "cafe\u0301 n\u0303 a\u0308",               # combining marks, not precomposed
    "日本語のテキスト、中文文本，한국어 텍스트",
    "Здравствуй, мир! Γειά σου Κόσμε! مرحبا بالعالم שלום עולם",
    "emoji 😀 👍🏽 👨‍👩‍👧‍👦 🇫🇷 ❤️",
    "math ∑∫√∞ ≠ ≤ ≥ → ⇒ ∀∃",
    "tab\there, nbsp\u00a0here, thin\u2009here, ideographic\u3000here",
    "nel\u0085here, bom\ufeffhere, zwsp\u200bhere, lsep\u2028here",
    "trailing spaces   \n   leading",
    "many\n\n\n\nnewlines   \n\n  ",
    "ALL CAPS and MiXeD CaSe",
    "http://example.com/a?b=c&d=e#f user@example.com",
    # The one special token. HF recognises it in text and so does
    # web/tokenizer.js; these pin what happens to the whitespace around it.
    "<|endoftext|>", "a<|endoftext|>b", " <|endoftext|> ", "x\n<|endoftext|>\ny",
    "<|endoftext|><|endoftext|>", "<|endoftext|", "<|endoftext|>>",
    "\u0000\u0001\u001f\u007f control",
    "Ⅻ ½ ² ٣ ๔",                              # numbers that are not ASCII digits
    " = Robert Boulter = \n Robert Boulter is an English film , television",
]

POOL = (
    list("abcxyzABCXYZ0123456789") * 3
    + list(" " * 12 + "\n" * 4 + "\t\r")
    + list(".,;:!?'\"-_()[]{}<>/\\@#$%^&*+=|~`")
    + ["'s", "'t", "'re", "'ll", "'d", "'m", "'ve"]
    + list("éüñçß") + list("日本語中文") + list("Ωπ") + list("ا")
    + ["😀", "👍🏽", "\u0301", "\u00a0", "\u3000", "\u2009", "\u0085", "\ufeff"]
    + list("½²٣")
)


def fuzz(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(POOL) for _ in range(rng.randint(1, 24)))
            for _ in range(n)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fuzz", type=int, default=400)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    from transformers import AutoTokenizer  # validation only, see above
    import transformers
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer))

    encode = [{"text": t, "ids": tok(t).input_ids} for t in HAND + fuzz(args.fuzz, args.seed)]

    # Decode cases: whole sequences, then every single id of a few multi-byte
    # texts on its own, which is how streaming generation meets them.
    def dec(ids):
        return tok.decode(ids, clean_up_tokenization_spaces=False)
    decode = []
    for t in ["emoji 😀 👍🏽 👨‍👩‍👧‍👦 🇫🇷 ❤️", "日本語のテキスト", "naïve café"]:
        ids = tok(t).input_ids
        decode.append({"ids": ids, "text": dec(ids)})
        decode += [{"ids": [i], "text": dec([i])} for i in ids]
    decode.append({"ids": [50256], "text": dec([50256])})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema": 1,
        "generated_by": "scripts/export_tokenizer_cases.py",
        "reference": f"transformers {transformers.__version__} "
                     f"{type(tok).__name__}, {args.tokenizer.name}",
        "fuzz": {"n": args.fuzz, "seed": args.seed},
        "encode": encode,
        "decode": decode,
    }, ensure_ascii=True, indent=1) + "\n")
    print(f"wrote {args.out.relative_to(ROOT)}: {len(encode)} encode, "
          f"{len(decode)} decode cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
