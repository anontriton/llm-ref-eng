#!/usr/bin/env bash
# Build the engine to WebAssembly: the same engine/ sources, configured with
# emcmake. Every tool and test comes out as a .js + .wasm pair run by Node.
#
#     web/build.sh [scalar|wasm_simd128] [fast] [mt]
#                                  # -> web/build/engine-<backend>[-fast][-mt]/
#
# Then, from the repo root, the Phase 2 proof unchanged:
#
#     node web/build/engine-scalar/tools/gpt2_dump.js --out web/build/dumps
#     .venv/bin/python oracle/compare.py web/build/dumps/manifest.json
#     .venv/bin/python oracle/check_greedy.py \
#         --engine web/build/engine-scalar/tools/gpt2_generate.js
#     ctest --test-dir web/build/engine-scalar
set -euo pipefail

backend="${1:-scalar}"
shift || true
# Options after the backend, in any order:
#   fast  the fast numerics tier -- relaxed-SIMD madd, judged by tolerance and
#         eval rather than bit-identity
#   mt    pthreads, for a cross-origin-isolated page (or Node)
numerics=exact
threads=OFF
suffix=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    fast) numerics=fast; suffix="$suffix-fast" ;;
    mt) threads=ON; suffix="$suffix-mt" ;;
    *) echo "web/build.sh: unknown option $1 (expected fast or mt)" >&2; exit 2 ;;
  esac
  shift
done
root="$(cd "$(dirname "$0")/.." && pwd)"
build="$root/web/build/engine-$backend$suffix"

# Arch's emscripten package leaves the compiler drivers off PATH.
if ! command -v emcmake >/dev/null; then
  export PATH="$PATH:/usr/lib/emscripten"
fi

emcmake cmake -S "$root/engine" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGPT2_BACKEND="$backend" \
  -DGPT2_NUMERICS="$numerics" -DGPT2_WASM_THREADS="$threads"
cmake --build "$build" -j
