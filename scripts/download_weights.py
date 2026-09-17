#!/usr/bin/env python3
"""Download the GPT-2 124M checkpoint and verify it against a pinned lockfile.

Fetches raw files over HTTPS rather than going through huggingface_hub, so the
exact bytes we validate against are the exact bytes we pinned. Nothing here
imports transformers.

    python scripts/download_weights.py            # download + verify
    python scripts/download_weights.py --pin      # download + record hashes

`--pin` rewrites scripts/weights.lock.json from whatever was just downloaded.
Run it once, commit the lockfile, and every later run is a verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = "openai-community/gpt2"
BASE = f"https://huggingface.co/{REPO}/resolve/main"

# Weights, plus the tokenizer files. The tokenizer is the one other thing we
# are allowed to take from upstream rather than write ourselves.
FILES = [
    "config.json",
    "model.safetensors",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
]

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "weights" / "gpt2-124m"
LOCKFILE = Path(__file__).resolve().parent / "weights.lock.json"

CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, dest: Path) -> None:
    url = f"{BASE}/{name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    # Only draw a progress bar on a terminal; in a log it is just noise.
    live = sys.stderr.isatty()
    print(f"  fetching {url}")
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        seen = 0
        with tmp.open("wb") as f:
            while chunk := response.read(CHUNK):
                f.write(chunk)
                seen += len(chunk)
                if live and total:
                    pct = 100 * seen / total
                    print(f"\r  {name}: {seen >> 20} / {total >> 20} MiB "
                          f"({pct:.0f}%)", end="", file=sys.stderr, flush=True)
    if live and total:
        print(file=sys.stderr)
    tmp.rename(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", action="store_true",
                        help="rewrite the lockfile from the downloaded files")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file is already present")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    lock = {}
    if LOCKFILE.exists() and not args.pin:
        lock = json.loads(LOCKFILE.read_text())["files"]

    computed = {}
    failures = []

    for name in FILES:
        path = DEST / name
        if args.force or not path.exists():
            download(name, path)
        else:
            print(f"  {name}: already present")

        digest = sha256(path)
        size = path.stat().st_size
        computed[name] = {"sha256": digest, "bytes": size}

        if name in lock:
            if lock[name]["sha256"] != digest:
                failures.append(
                    f"{name}: expected {lock[name]['sha256'][:16]}..., "
                    f"got {digest[:16]}..."
                )
            else:
                print(f"  {name}: sha256 ok ({size >> 10} KiB)")
        elif not args.pin:
            failures.append(f"{name}: not in {LOCKFILE.name}; run --pin once")

    if args.pin:
        LOCKFILE.write_text(json.dumps(
            {"repo": REPO, "base_url": BASE, "files": computed}, indent=2
        ) + "\n")
        print(f"\npinned {len(computed)} files -> {LOCKFILE}")
        return 0

    if failures:
        print("\nCHECKSUM FAILURES:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(f"\nall {len(FILES)} files verified against {LOCKFILE.name}")
    print(f"weights at {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
