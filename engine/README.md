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
    .venv/bin/python oracle/check_greedy.py --kv-cache     # 50 greedy steps

## Backends

    cmake -S engine -B engine/build       -DGPT2_BACKEND=scalar   # default
    cmake -S engine -B engine/build-avx2  -DGPT2_BACKEND=avx2

Exactly one `src/backend_*.cpp` compiles -- they define the same symbols -- and
only that translation unit gets ISA flags, so the kernels stay baseline and
reach the vector units through `gpt2::backend::*`. Keep a build directory per
backend; the two are not interchangeable and the CMake cache remembers which
is which.

The AVX2 backend refuses FMA on purpose, built with `-mavx2 -ffp-contract=off`
and no `-mfma`. Fusing rounds once where scalar rounds twice, which would put
the backends in disagreement -- and wasm_simd128 in Phase 5 has no FMA to
offer. All 615 oracle tensors are bit-identical between the two backends, and
that property is the point of the layer.

## Tools

    gpt2_dump      activations in the oracle layout; --kv-cache dumps the last
                   token through the cache instead, and compare.py slices the
                   reference to the matching row
    gpt2_generate  greedy decode; --kv-cache selects the cached path
    gpt2_bench     prefill and decode timings -> bench/run.py
    gpt2_eval      per-position nll, top-1 and sampled logits -> eval/metrics.py

All four take `--threads N` (default 1) and read the weight file's header to
decide fp32 or int8 -- never a flag. `--help` on any of them is current.

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
- `kv_cache` -- keys and values retained across decode steps. `Model::forward`
  has two spellings and one implementation: the stateless call is the
  empty-cache case of `forward_impl(new_ids, start_pos, cache)`. Two forward
  passes that must agree forever is the arrangement that rots.
- `threading` -- a pool and a `parallel_for`. Work items own disjoint tiles of
  the output and nothing reduces across a tile boundary, so thread count is not
  a numerical parameter: 615 tensors are bit-identical at 1, 3 and 8 threads.
  Serial by default; at `count() == 1` there is no pool and no locks.
- `tools/` -- four executables, above.

## Things that will bite

- **Blocking in `ops::linear` tiles rows, not columns.** `W` is row-major along
  `n_out`, so a column strip reads it with a stride; four of five shapes got
  *slower* that way. `linear_tied` is the transpose and wants the opposite.
  Both leave the reduction along `n_in` alone -- same leaf size, same merge
  tree -- which is why every Phase 3 step came out bit-identical. Tiling `n_in`
  would reshape the tree and change the answer.
- **Decode takes the unblocked path** (`rows == 1`), deliberately: a single row
  reuses nothing and still pays the stride. It cost 1.7x before it was gated.
- **int8 scales apply once after the reduction**, not per element -- one
  multiply and one rounding instead of `n_in` of them. `Matrix` carries
  whichever representation the file holds; above the kernels nothing branches.

Phase: 4 complete. fp32 and int8 both ship; int4 was measured and rejected, see
`reference/gptq.py`. Next is a wasm_simd128 backend beside the other two.
