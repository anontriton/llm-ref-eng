#!/usr/bin/env bash
# Serve the demo locally.
#
#     web/serve.sh [port]        # then open http://localhost:8000/web/demo/
#
# From the repo root, because the page reaches three places: web/ for the page
# and the JS, web/build/ for the wasm engine, weights/ for the 129 MB weight
# file and the tokenizer's vocab.json and merges.txt. Build and quantize first:
#
#     web/build.sh wasm_simd128
#     .venv/bin/python scripts/quantize_weights.py --wte-outliers 8
#
# Bound to localhost: this serves the whole repository.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
port="${1:-8000}"

for f in web/build/engine-wasm_simd128/tools/gpt2_web.mjs \
         weights/gpt2-124m-int8-wte-o8.bin \
         weights/gpt2-124m/vocab.json weights/gpt2-124m/merges.txt; do
  [[ -f "$root/$f" ]] || { echo "missing $f -- see the header of $0" >&2; exit 1; }
done

echo "http://localhost:$port/web/demo/"
exec python3 -m http.server "$port" --bind 127.0.0.1 --directory "$root"
