#!/usr/bin/env python3
"""Pin the WikiText-2 slice that Phase 4 measures perplexity on.

    python scripts/export_eval_corpus.py        # -> eval/corpus.json

CLAUDE.md's quantized tolerance policy names "perplexity on a fixed WikiText-2
slice". Fixed is the operative word: a perplexity number means nothing unless
the text behind it is the same text every time, so this runs once and commits
the token ids, and nothing afterwards needs the network or the tokenizer.

Rows come from the HuggingFace datasets-server, which serves them as JSON --
no `datasets` package, no parquet reader, no new dependency for something done
once. The slice is a deterministic prefix of the test split: offset 0, a fixed
row count, joined in order.

Two files, because the engine parses no JSON -- that is the point of
engine/include/gpt2/json.h being a writer only:

    eval/corpus.tsv    the ids, in the format engine/src/runs.cpp already reads
    eval/corpus.json   provenance for the Python side: where the text came
                       from, checksums, counts, a preview

The ids live in exactly one of them. The checksum in the JSON is over the same
bytes the engine hashes, so an eval run and this file can be checked against
each other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "eval" / "corpus.json"
DEFAULT_TSV = ROOT / "eval" / "corpus.tsv"
NAME = "wikitext2"
DEFAULT_WEIGHTS = ROOT / "weights" / "gpt2-124m"

SERVER = "https://datasets-server.huggingface.co/rows"
DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"
SPLIT = "test"
PAGE = 100  # the server's maximum rows per request


def fetch_rows(rows: int) -> list[str]:
    out: list[str] = []
    for offset in range(0, rows, PAGE):
        length = min(PAGE, rows - offset)
        query = urllib.parse.urlencode({
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        })
        # The server returns a transient 502 often enough to matter, and a
        # half-fetched corpus is worse than a slow one.
        payload = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(f"{SERVER}?{query}",
                                            timeout=60) as resp:
                    payload = json.load(resp)
                break
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 4:
                    raise SystemExit(f"rows at offset {offset}: {exc}")
                delay = 2 ** attempt
                print(f"    {exc} -- retrying in {delay}s", flush=True)
                time.sleep(delay)
        got = [r["row"]["text"] for r in payload["rows"]]
        if len(got) != length:
            raise SystemExit(
                f"asked for {length} rows at offset {offset}, got {len(got)}")
        out.extend(got)
        print(f"  rows {offset}..{offset + length - 1}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=400,
                        help="rows of the test split to take, from offset 0")
    parser.add_argument("--tokens", type=int, default=8192,
                        help="how many ids to keep; eval uses a prefix of these")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="directory holding the tokenizer files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    args = parser.parse_args()

    print(f"fetching {args.rows} rows of {DATASET}/{CONFIG}/{SPLIT}")
    rows = fetch_rows(args.rows)

    # WikiText rows already carry their own newlines, so they concatenate
    # directly; joining with anything would insert text that is not in the
    # corpus.
    text = "".join(rows)
    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"  {len(text)} chars, sha256 {text_sha[:16]}...")

    from transformers import AutoTokenizer  # tokenization only
    tok = AutoTokenizer.from_pretrained(str(args.weights))
    ids = tok(text).input_ids
    print(f"  {len(ids)} tokens")
    if len(ids) < args.tokens:
        raise SystemExit(
            f"slice tokenizes to {len(ids)} ids, fewer than the requested "
            f"{args.tokens}; raise --rows")
    ids = ids[:args.tokens]

    # Over the raw little-endian int32 ids, which is what the engine hashes
    # from the other side.
    ids_sha = hashlib.sha256(
        b"".join(int(i).to_bytes(4, "little") for i in ids)).hexdigest()

    corpus = {
        "schema": 1,
        "name": NAME,
        "source": {
            "server": SERVER,
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "rows": args.rows,
            "offset": 0,
        },
        "text_sha256": text_sha,
        "text_chars": len(text),
        "n_tokens": len(ids),
        "ids_sha256": ids_sha,
        # The first characters, so a human can see at a glance that this is
        # WikiText and not something else that tokenized to the right length.
        "preview": text[:200],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(corpus, indent=2) + "\n")

    # The engine reads this one. Same three-column format as engine/runs.tsv;
    # the middle field is a description rather than the text, because nothing
    # in the engine computes on it and 134 kB of escaped prose on one line
    # would help no one.
    args.tsv.write_text(
        "\n".join([
            "# generated by scripts/export_eval_corpus.py",
            "# name\tdescription\tinput_ids",
            f"{NAME}\t{DATASET} {CONFIG} {SPLIT} rows 0..{args.rows - 1}\t"
            + ",".join(str(i) for i in ids),
        ]) + "\n")

    print(f"wrote {args.out.relative_to(ROOT)}  (provenance)")
    print(f"wrote {args.tsv.relative_to(ROOT)}  ({len(ids)} ids)")
    print(f"  ids sha256 {ids_sha[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
