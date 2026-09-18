# engine/ -- C++ inference engine

C++17, no external dependencies in the core.

- `include/gpt2/`  public headers (tensor, model config, ops, backend)
- `src/`           implementation, one translation unit per concern
- `tests/`         unit tests for kernels and weight loading

The engine dumps its own activations in the oracle layout so `oracle/compare.py`
can diff them against the PyTorch reference. The engine does NOT read `.npy` --
it only writes it. Comparison lives in Python.

SIMD sits behind a backend abstraction (`include/gpt2/backend/`): a scalar
backend for correctness, AVX2 for Phase 3, wasm_simd128 for Phase 5. Kernel
logic never contains raw intrinsics.

## Build and prove it

From the repo root:

    .venv/bin/python scripts/convert_weights.py --verify   # -> weights/gpt2-124m.bin
    .venv/bin/python scripts/export_runs.py                # -> engine/runs.tsv
    cmake -S engine -B engine/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
    cmake --build engine/build -j
    ctest --test-dir engine/build                          # unit tests

    engine/build/tools/gpt2_dump                           # -> engine/dumps/
    .venv/bin/python oracle/compare.py engine/dumps/manifest.json
    .venv/bin/python oracle/check_greedy.py                # 50 greedy steps, ~3 min

`compare.py` is the authority. `ctest` checks that each kernel computes the
right formula; `compare.py` checks the whole engine against the reference.

## Layout

- `backend/backend.h` + `src/backend_scalar.cpp` -- the SIMD seam. Every
  reduction accumulates across a fixed `kAccumLanes` (8) and reduces them
  pairwise, so an AVX2 or wasm_simd128 backend can reproduce scalar results
  bit for bit.
- `ops` -- layernorm, linear (pairwise-summed along the inner dimension),
  tied lm_head, gelu_new, softmax. No intrinsics.
- `model` -- the forward pass, a line-by-line port of `reference/model.py`,
  with a tap for every oracle tensor.
- `weights` -- reads the flat file `scripts/convert_weights.py` writes.
- `dump`, `npy`, `sha256`, `json`, `runs` -- the oracle contract: write-only
  `.npy`, per-tensor checksums, a manifest `compare.py` accepts.
- `tools/gpt2_dump`, `tools/gpt2_generate` -- the two executables.

Phase: 2 complete. No KV cache yet: `gpt2_generate` re-runs the full prefix
every step, which is Phase 3's first target.
