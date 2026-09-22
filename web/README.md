# web/ -- Emscripten build and demo

Compiles the same `engine/` sources to WebAssembly. No engine fork: if the web
build needs a change, it goes into the engine behind the backend abstraction.

Phase: 5, steps 1-3 of 4 done -- the scalar and wasm_simd128 backends both
build to wasm and pass the oracle under Node, the two are bit-identical to each
other, and both are benchmarked beside native. `demo/` is still an empty
placeholder.

    web/build.sh                     # -> web/build/engine-scalar/
    web/build.sh wasm_simd128        # -> web/build/engine-wasm_simd128/
    node web/build/engine-scalar/tools/gpt2_dump.js --out web/build/dumps
    .venv/bin/python oracle/compare.py web/build/dumps/manifest.json
    .venv/bin/python oracle/check_greedy.py \
        --engine web/build/engine-scalar/tools/gpt2_generate.js
    ctest --test-dir web/build/engine-scalar

`build.sh` finds `emcc` itself; Arch keeps the drivers in `/usr/lib/emscripten`,
off PATH. See CLAUDE.md's Environment section.

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
    int8, wte fp32              243.3 MB   <- Phase 4's file, native
    int8, wte int8              127.7 MB   <- rejected: 17% of argmaxes change
    int8, wte int8 + 8 cols     129.3 MB   <- Phase 5's file, for the browser

The last line is the answer, and the rest of this section is how it was
reached; see "Weights for the browser" below for the result.

Of the 243 MB, **154 MB is `wte`**, kept in fp32 because Phase 4 measured
twice that quantizing it costs far more than it saves (`scripts/quantize_weights.py`
documents the numbers). int4 would have taken the *layers* from 85 MB to 48 MB
and left `wte` untouched, so it would have moved the total to about 202 MB --
which is part of why rejecting it cost less than it sounds.

Nobody downloads 243 MB for a demo. Options not yet evaluated, roughly in
order of how much they change:

1. **Measured -- `reference/wte_sim.py`.** The first version of this item
   proposed int8 for lm_head and fp32 for the lookup, and could not have saved
   anything: any token can be looked up, so the fp32 table would still ship.
   Separating the two uses was still the right diagnosis, and it came out the
   other way round. The embedding takes int8 for free; **all** the damage is
   lm_head, because ln_f's output has a few huge hidden dims (dim 496 averages
   |x| = 201, median 0.35) that multiply the int8 error in those columns of
   wte. Two formats pass every `eval/metrics.py` gate in simulation:

       int8 wte + 8 outlier columns fp32   129.3 MB   top-1 97.41%   decisive 0.024%
       fp16 wte                            166.1 MB   top-1 97.53%   decisive 0.024%

   against 243.3 MB shipping today. Neither is in the engine yet; the
   simulation predicted int8 to five decimals last time, but `eval/metrics.py`
   on the engine is the authority, not this.
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

**Validation in Node** is settled: the build links with `-sNODERAWFS=1`, so the
tools see the real filesystem and every path argument means what it does
natively. The flags and why are in `engine/CMakeLists.txt`. That is a Node-only
arrangement; the browser build will load weights differently.

## Step 1 result: scalar under Node

    oracle/compare.py         615/615 pass, worst block.11.mlp.out at 58.1%
    oracle/compare.py (kv)    615/615 pass
    check_greedy.py           50/50, with and without --kv-cache
    ctest                     6/6 pass, test_threading disabled (no pthreads)
    gpt2_dump, all 5 runs     21 s single-threaded, Node 22

**It is not bit-identical to the native engine, and cannot be while the
kernels call libm.** The two agree exactly through embeddings, layernorm, QKV,
attention and softmax, and part at `block.0.mlp.act.out` -- GELU, the first
`std::tanh`. Native links glibc; Emscripten links musl. Measured over every
float in [2^-20, 10]:

    glibc tanhf   correctly rounded everywhere
    musl  tanhf   23.3% of inputs off, by up to 2 ulp

`expf`, softmax's one call, is correctly rounded 99.96% of the time in both,
but not on the same inputs; the oracle prompts happen not to reach the
difference. Hence 58.1% of budget where native reports 52.0%: still well
inside tolerance, but a different number.

What that means for step 2: the wasm_simd128 backend is held bit-identical to
the *wasm* scalar build, which shares its libm -- not to native. The backend
seam was never where the transcendentals live (they are in `ops.cpp`), so this
does not touch the SIMD work. If cross-platform bit-identity is ever wanted,
it means the engine carrying its own `tanh` and `exp` built from `+ - * /`,
re-validated against the oracle, since glibc's `tanhf` is the one being
matched today.

## Step 2 result: wasm_simd128

`src/backend_wasm_simd128.cpp` is the AVX2 backend at half the width: the
eight accumulator lanes are two v128, `lo` and `hi`, and the reduction tree is
the scalar one step for step. Only that file gets `-msimd128`. The linked
module has no relaxed-simd, no fused multiply-add and no `f32x4.max`, which is
checked with `wasm-dis`, not assumed.

    vs wasm scalar, sha256    615/615 identical; 615/615 via --kv-cache
    vs wasm scalar, compare   0.0% of budget, abs 0.000e+00
    int8, gpt2_eval           nll, top1, 64 rows of logits byte-identical
    oracle/compare.py         615/615 pass, 58.1% -- the scalar build's figure
    check_greedy.py           50/50, with and without --kv-cache
    ctest                     6/6 pass, test_threading disabled

The fp32 oracle never calls `dot_i8` or `axpy_i8`, so the int8 kernels are
proved by `gpt2_eval` on `gpt2-124m-int8.bin` in both builds instead, two
512-token windows. Speed, single-threaded under Node, as a sanity check rather
than a benchmark:

    gpt2_dump, all 5 runs     scalar 21 s   wasm_simd128  8.5 s   2.5x
    gpt2_eval int8, 2 windows scalar 221 s  wasm_simd128  59 s    3.7x

## Step 3 result: benchmarked beside native

`bench/run.py --tool .../gpt2_bench.js` runs a wasm build under Node and records
it with `config.target: wasm32` and the Node version. Eight results at 2a5a496,
all on one machine in one sitting: T=128 prompt, 128 generated, KV cache, one
thread, on AC under the performance profile.

    fp32                  tok/s   prefill ms   decode ms/tok
    native scalar         12.26      4424         47.40
    native avx2           23.96      1676         28.87
    wasm   scalar          7.66      7726         70.75
    wasm   wasm_simd128   17.70      2550         36.86

    int8                  tok/s   prefill ms   decode ms/tok
    native scalar          9.78      6099         55.03
    native avx2           32.30      1582         18.75
    wasm   scalar          6.60      9236         79.95
    wasm   wasm_simd128   21.77      2449         27.02

- **wasm_simd128 is 74% of native AVX2 end to end** in fp32, 67% in int8.
  Decode is 1.28x behind, prefill 1.52x. Half the vector width is the obvious
  suspect for prefill, which is compute-bound; decode is closer because it is
  bound by memory traffic, which the width does not change.
- **SIMD is 2.3x over wasm scalar**, 3.0x on prefill -- a larger gain than AVX2
  gets over native scalar on the same backend seam.
- **int8 pays off in wasm too**, 1.23x end to end, from decode: 36.9 to 27.0
  ms/token, half the weight bytes per step. Scalar int8 is *slower* than scalar
  fp32 on both targets; the widening only earns its keep once it is vectorized.
- **Weight loading is 2.7x slower** under Node (212 vs 80 ms), which is
  NODERAWFS reading through JS. It is outside every metric above and says
  nothing about the browser, which will not read a file at all.
- Native avx2 fp32 reproduces Phase 3's single-thread figure, 23.96 against
  23.94, and every fp32 run decodes exactly Phase 3's 128 ids, every int8 run
  exactly Phase 4's. The harness is measuring the same programs.

Single-threaded on both sides, which flatters nothing: native reaches 42.8
tok/s at 8 threads and the wasm build has none yet (see Threads above).

These replace a first attempt that is not in `results/`: measured on battery
under the low-power profile, 2.4x slow across the board. `bench/run.py` now
records the power state and names such a run `...-lowpower.json`.

## Weights for the browser: int8-wte-o8

    .venv/bin/python scripts/quantize_weights.py --wte-outliers 8
                                    # -> weights/gpt2-124m-int8-wte-o8.bin

`wte` in int8 like the layers, except the 8 columns where ln_f's output is
largest -- picked on `eval/calib.tsv` by the same function
`reference/wte_sim.py` used -- which are zeroed in the int8 table and stored
whole beside it. lm_head adds an 8-wide fp32 dot per logit; the embedding
lookup overwrites 8 values. The file format gained an int32 dtype for the
column indices, and the loader checks every property the kernel relies on,
including that the int8 table really is zero in those columns.

Judged by `eval/metrics.py` on the engine, against the fp32 engine:

                          int8 (243.3 MB)   int8-wte-o8 (129.3 MB)   simulated
    perplexity ratio          x0.99844          x0.99805             x0.99805
    top-1 agreement            97.480%           97.383%              97.407%
    mean KL                   0.001168          0.001345             0.001327
    decisive disagreement      0/256             0/256                    --

The simulation called the perplexity ratio to five decimals again. The policy
name is `int8-wte-o8`, derived by the loader from what the file holds, so
nothing downstream can report it as Phase 4's `int8`.

Backends: native scalar and AVX2 are byte-identical on it, as are wasm scalar
and wasm_simd128; native and wasm differ by at most 9.2e-5 in a logit, the
libm difference from step 1, with top-1 identical at every position. fp32 is
untouched -- 615/615 oracle tensors identical to before the change.

## Suggested order

1. **Done.** Build the *scalar* backend to wasm, dump, and run
   `oracle/compare.py` on it. That is the whole Phase 2 proof, unchanged, and
   it separates "does it compile and run" from "is the SIMD right".
2. **Done.** Add `src/backend_wasm_simd128.cpp` against `backend.h`, and hold it to the
   same standard AVX2 met: 615 tensors bit-identical to scalar, not merely
   within tolerance.
3. **Done.** Benchmark with `bench/run.py`, which already records the backend the binary
   reports rather than a flag.
4. Then the demo page, and the size question above.
