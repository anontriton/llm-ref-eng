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


def fetch_rows(rows: int, split: str = SPLIT) -> list[str]:
    out: list[str] = []
    for offset in range(0, rows, PAGE):
        length = min(PAGE, rows - offset)
        query = urllib.parse.urlencode({
            "dataset": DATASET,
            "config": CONFIG,
            "split": split,
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
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing corpus. Required, because "
                             "the committed one is pinned and results depend "
                             "on it")
    parser.add_argument("--split", default=SPLIT,
                        help="which split to draw from. Calibration data must "
                             "not come from the split the eval measures on, or "
                             "the quantizer is tuned on its own test set")
    args = parser.parse_args()

    # The corpus is an input that committed results were measured against, not
    # a cache to be refreshed. Re-fetching can return different rows, and every
    # perplexity in eval/results/ would quietly stop meaning what it says.
    if args.tsv.exists() and not args.force:
        raise SystemExit(
            f"{args.tsv} already exists and is pinned; results in eval/results/ "
            f"were measured on it. Pass --force only if you intend to invalidate "
            f"them.")

    print(f"fetching {args.rows} rows of {DATASET}/{CONFIG}/{args.split}")
    rows = fetch_rows(args.rows, args.split)

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
        "name": args.name,
        "source": {
            "server": SERVER,
            "dataset": DATASET,
            "config": CONFIG,
            "split": args.split,
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
            f"{args.name}\t{DATASET} {CONFIG} {args.split} rows 0..{args.rows - 1}\t"
            + ",".join(str(i) for i in ids),
        ]) + "\n")

    def shown(p: Path) -> Path:
        try:
            return p.resolve().relative_to(ROOT)
        except ValueError:
            return p

    print(f"wrote {shown(args.out)}  (provenance)")
    print(f"wrote {shown(args.tsv)}  ({len(ids)} ids)")
    print(f"  ids sha256 {ids_sha[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
