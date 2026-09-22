# web/ -- Emscripten build and demo

Compiles the same `engine/` sources to WebAssembly. No engine fork: if the web
build needs a change, it goes into the engine behind the backend abstraction.

Nothing here is written yet. `demo/` is an empty placeholder and there is no
`build.sh`; the plan below is the whole of it.

Phase: 5, not started. Requires the Emscripten SDK (`emcc`), **not installed**.

Oracle validation still applies: the wasm build is compared against the same
PyTorch dumps, run in Node. A backend that cannot reproduce the oracle is not
a backend, it is a different model.

## What Phases 2-4 already decided for this

Most of the awkward choices were made early, on purpose, so that this phase is
a port rather than a redesign.

- **`backend::kAccumLanes` is 8.** One AVX2 register of floats, and *two*
  wasm_simd128 registers, which hold four each. The wasm backend spends an
  extra instruction per lane group and keeps the arithmetic identical. This was
  fixed in Phase 2, before either vector backend existed.
- **No FMA anywhere.** wasm_simd128 has no fused multiply-add, so AVX2 refuses
  to use one -- built without `-mfma` and with `-ffp-contract=off`, verified by
  checking the object has zero `vfmadd`. That is a cost paid in Phase 3
  specifically to keep this phase honest: the wasm backend can be bit-identical
  to scalar and AVX2 rather than merely close.
- **The build seam exists.** `-DGPT2_BACKEND=wasm_simd128` selects
  `src/backend_wasm_simd128.cpp`; exactly one backend compiles and only that
  file gets ISA flags. Adding a third is adding one file and one branch in
  `engine/CMakeLists.txt`.
- **`max()` is scalar in every backend.** `_mm256_max_ps` disagrees with `>` on
  NaN and signed zero, and softmax calls it once per row. Do the same here
  rather than reaching for `f32x4.max`.

## What is genuinely open

**Size is the problem, not speed.** The weight file is the download:

    fp32                        497.8 MB
    int8, wte fp32              243.3 MB   <- what ships today
    int8, wte int8              127.7 MB   <- rejected: 17% of argmaxes change

Of the 243 MB, **154 MB is `wte`**, kept in fp32 because Phase 4 measured
twice that quantizing it costs far more than it saves (`scripts/quantize_weights.py`
documents the numbers). int4 would have taken the *layers* from 85 MB to 48 MB
and left `wte` untouched, so it would have moved the total to about 202 MB --
which is part of why rejecting it cost less than it sounds.

Nobody downloads 243 MB for a demo. Options not yet evaluated, roughly in
order of how much they change:

1. Quantize `wte` for lm_head only and keep fp32 for the embedding lookup.
   The lookup reads *one row* per token, 3 KB; lm_head streams the whole
   table. Phase 4 never separated the two uses, so it is unknown which one the
   damage came from. If it is the embedding side, this recovers 115 MB for
   free. This is the cheapest experiment and the one to run first --
   `reference/quant_sim.py` can test it without touching C++.
2. A smaller context or a trimmed vocabulary for the demo specifically.
3. Streaming the weights and starting generation before the tail arrives.
4. Accepting a long first load with a service worker cache.

**Threads.** `gpt2::threads` is `std::thread` plus a condition variable, which
Emscripten supports only with pthreads enabled, which needs
`SharedArrayBuffer`, which needs COOP/COEP headers on whatever serves the demo.
Single-threaded is the safe default; the pool already runs with no threads and
no locks at `count() == 1`, so this degrades cleanly rather than needing a
second code path.

**The tokenizer.** The engine does not tokenize -- deliberately, since Phase 1 --
and every existing entry point takes token ids from a TSV. A browser demo needs
real tokenization in JavaScript, which is the one genuinely new component in
this phase and the one most likely to disagree with the reference. Pin it
against `eval/corpus.tsv`: the same text tokenized by the Python side is
already committed, ids and checksum.

**Validation in Node.** `gpt2_dump` writes `.npy` through `engine/src/npy.cpp`;
under Emscripten that needs a filesystem mount (`NODEFS`) or the dumps have to
come back over a binding. Either is fine -- `oracle/compare.py` only cares that
the files land somewhere with a manifest beside them.

## Suggested order

1. Install `emcc`, build the *scalar* backend to wasm, dump, and run
   `oracle/compare.py` on it. That is the whole Phase 2 proof, unchanged, and
   it separates "does it compile and run" from "is the SIMD right".
2. Add `src/backend_wasm_simd128.cpp` against `backend.h`, and hold it to the
   same standard AVX2 met: 615 tensors bit-identical to scalar, not merely
   within tolerance.
3. Benchmark with `bench/run.py`, which already records the backend the binary
   reports rather than a flag.
4. Then the demo page, and the size question above.
