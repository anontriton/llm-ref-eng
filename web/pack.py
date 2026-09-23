#!/usr/bin/env python3
"""Assemble the demo as a static site: web/build/site/.

    python3 web/pack.py

What `web/serve.sh` serves locally and what GitHub Pages publishes are the same
directory, built by this, so the thing tested is the thing shipped:

    index.html, worker.js,          the page and its modules, copied from web/
    engine.js, sampling.js,
    tokenizer.js
    wasm/gpt2_web.{mjs,wasm}        the engine, from web/build.sh wasm_simd128
    model/vocab.json, merges.txt    the tokenizer's files
    model/weights.json              the weights' manifest: policy, size, sha256,
                                    and the parts, in order
    model/weights-NNN.bin           the weight file cut into parts

The weights are split because GitHub refuses any file over 100 MB, and the
file is 129 MB. The page fetches the parts in order into one buffer.

The weight file must be byte-identical to the one pinned in
web/weights.lock.json -- the file eval/results/ measured -- or this refuses to
build the site. A demo that shipped some other file would be showing a model
nobody validated. The page checks the same sha256 once it has downloaded it.

Standard library only: nothing to install beyond Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
LOCK = WEB / "weights.lock.json"
ENGINE = WEB / "build" / "engine-wasm_simd128" / "tools"
TOKENIZER = ROOT / "weights" / "gpt2-124m"
DEFAULT_OUT = WEB / "build" / "site"

PAGE_FILES = ["index.html", "worker.js", "engine.js", "sampling.js", "tokenizer.js"]
PART_BYTES = 32 << 20  # 32 MiB: far under GitHub's limit, few enough requests


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fail(msg: str) -> int:
    print(f"pack: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    lock = json.loads(LOCK.read_text())
    weights = ROOT / lock["file"]
    needed = [weights, ENGINE / "gpt2_web.mjs", ENGINE / "gpt2_web.wasm",
              TOKENIZER / "vocab.json", TOKENIZER / "merges.txt"]
    for path in needed:
        if not path.is_file():
            return fail(f"missing {path.relative_to(ROOT)} -- run ./demo.sh, or "
                        "web/build.sh wasm_simd128 and scripts/quantize_weights.py "
                        "--wte-outliers 8")

    size = weights.stat().st_size
    digest = sha256_file(weights)
    if size != lock["bytes"] or digest != lock["sha256"]:
        return fail(f"{lock['file']} is not the pinned file\n"
                    f"  built   {size} bytes, sha256 {digest}\n"
                    f"  pinned  {lock['bytes']} bytes, sha256 {lock['sha256']}\n"
                    f"  (web/weights.lock.json; measured by {lock['eval_result']})")

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "wasm").mkdir(parents=True)
    (out / "model").mkdir()

    for name in PAGE_FILES:
        shutil.copy2(WEB / name, out / name)
    for name in ["gpt2_web.mjs", "gpt2_web.wasm"]:
        shutil.copy2(ENGINE / name, out / "wasm" / name)
    for name in ["vocab.json", "merges.txt"]:
        shutil.copy2(TOKENIZER / name, out / "model" / name)
    # GitHub Pages runs Jekyll unless told not to, which skips some files.
    (out / ".nojekyll").write_text("")

    parts = []
    with weights.open("rb") as f:
        for i in range(0, size, PART_BYTES):
            data = f.read(PART_BYTES)
            name = f"weights-{i // PART_BYTES:03d}.bin"
            (out / "model" / name).write_bytes(data)
            parts.append({"file": name, "bytes": len(data)})
    (out / "model" / "weights.json").write_text(json.dumps({
        "policy": lock["policy"],
        "bytes": size,
        "sha256": digest,
        "eval_result": lock["eval_result"],
        "parts": parts,
    }, indent=2) + "\n")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out.relative_to(ROOT)}: {total / 1e6:.1f} MB, weights in "
          f"{len(parts)} parts, sha256 {digest[:16]}... matches the pin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
