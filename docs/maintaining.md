# Maintaining

How to pick this repository up, prove it still works, and change the engine
without breaking what the earlier phases proved. The rules themselves are in
[CLAUDE.md](../CLAUDE.md); this is the working procedure around them.

## Picking this up

Almost nothing runnable is in the repo. Weights, activation dumps, build
directories and eval dumps are all gitignored and have to be regenerated; what
*is* committed is everything needed to regenerate them identically.

| Committed | Regenerate |
|---|---|
| `oracle/manifest.json` -- shapes and a sha256 per tensor | `oracle/activations/*.npy` (`reference/dump.py`) |
| `scripts/weights.lock.json` -- checkpoint checksum | `weights/` (download, convert, quantize) |
| `bench/prompts.tsv`, `eval/corpus.tsv`, `eval/calib.tsv` | -- pinned inputs, never regenerate casually |
| `web/tokenizer_cases.json` -- HF's ids for the tokenizer's edge cases | (`scripts/export_tokenizer_cases.py`) |
| `bench/results/*.json`, `eval/results/*.json` | -- the record; commit-tagged |
| -- | `engine/runs.tsv`, `engine/build*/`, `engine/dumps*/`, `eval/runs/`, `web/build/` |

Two consequences worth knowing before you start:

- **Regenerating the oracle is itself a test.** The manifest is committed with
  a checksum per tensor, and `compare.py` verifies dumps against it. If
  `reference/dump.py` produces different bytes on your machine, that is a real
  finding, not a setup problem.
- **`eval/metrics.py` needs the fp32 *dump*, not the committed summary.**
  `eval/results/*.json` records what was measured; the logits it was measured
  from live in `eval/runs/fp32/`, which is gitignored. Regenerate the fp32
  baseline before judging any quantized run against it.

## Before you commit a change to the engine

The non-negotiable in CLAUDE.md is that every optimization commit re-runs
oracle validation. In practice that is:

    ctest --test-dir engine/build && ctest --test-dir engine/build-avx2
    engine/build-avx2/tools/gpt2_dump --threads 8
    .venv/bin/python oracle/compare.py engine/dumps/manifest.json
    engine/build-avx2/tools/gpt2_dump --kv-cache --threads 8 --out engine/dumps_kv
    .venv/bin/python oracle/compare.py engine/dumps_kv/manifest.json
    .venv/bin/python oracle/check_greedy.py --kv-cache \
        --engine engine/build-avx2/tools/gpt2_generate
    .venv/bin/python oracle/test_compare.py

If the change touches anything the wasm build compiles -- which is all of
`engine/src` -- run the same proof there. The wasm_simd128 build must stay
bit-identical to the wasm scalar build, not merely within tolerance:

    web/build.sh scalar && web/build.sh wasm_simd128
    ctest --test-dir web/build/engine-scalar && ctest --test-dir web/build/engine-wasm_simd128
    node web/build/engine-wasm_simd128/tools/gpt2_dump.js --out web/build/dumps
    .venv/bin/python oracle/compare.py web/build/dumps/manifest.json
    .venv/bin/python oracle/check_greedy.py --kv-cache \
        --engine web/build/engine-wasm_simd128/tools/gpt2_generate.js
    node web/test_web.mjs
    node web/test_tokenizer.mjs

Watch the budget percentage `compare.py` prints, not just the pass. It has been
52.0% natively since Phase 2 and stayed there through four optimizations, and
58.1% under wasm (the libm difference in [findings](findings.md), 7); a change
that moves either has changed the arithmetic even if it still passes.

For anything claiming to be faster, benchmark on a **clean tree**, **on AC
power** -- `run.py` records dirty and throttled runs as such and they do not
count -- and compare against a result with the same `config` and `prompt`.
Commit anything else a run writes (an eval result, say) before benchmarking,
or the tree is dirty. For anything quantized,
`eval/metrics.py` is the authority and `compare.py` will refuse the run.

## Reproduce everything

From the repo root, with the `.venv` described in CLAUDE.md:

    .venv/bin/python scripts/download_weights.py           # checksum-verified
    .venv/bin/python reference/validate_hf.py              # Phase 0
    .venv/bin/python reference/dump.py                     # Phase 1: build the oracle
    .venv/bin/python oracle/test_compare.py

    .venv/bin/python scripts/convert_weights.py --verify   # Phase 2
    .venv/bin/python scripts/export_runs.py
    cmake -S engine -B engine/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
    cmake --build engine/build -j
    ctest --test-dir engine/build
    engine/build/tools/gpt2_dump
    .venv/bin/python oracle/compare.py engine/dumps/manifest.json
    .venv/bin/python oracle/check_greedy.py --kv-cache     # seconds; ~3 min without

    # Phase 3: the AVX2 build, and the benchmark
    cmake -S engine -B engine/build-avx2 -G Ninja -DGPT2_BACKEND=avx2
    cmake --build engine/build-avx2 -j
    .venv/bin/python bench/run.py --kv-cache --threads 8 \
        --tool engine/build-avx2/tools/gpt2_bench

    # Phase 4: quantize, then judge against the fp32 engine.
    # eval/corpus.tsv and eval/calib.tsv are committed and pinned -- do NOT
    # re-run scripts/export_eval_corpus.py to "set up". Re-fetching could
    # return different rows, which would silently invalidate every result in
    # eval/results/ that was measured on the old text.
    .venv/bin/python eval/run.py --out eval/runs/fp32
    .venv/bin/python eval/validate_ppl.py eval/runs/fp32   # harness vs PyTorch
    .venv/bin/python scripts/quantize_weights.py
    .venv/bin/python eval/run.py --out eval/runs/int8 \
        --weights weights/gpt2-124m-int8.bin --reference eval/runs/fp32

    # Phase 5: the wasm build, the browser's weights, the tokenizer, the demo.
    export PATH="$PATH:/usr/lib/emscripten"                # Arch keeps emcc here
    web/build.sh scalar && web/build.sh wasm_simd128
    node web/build/engine-wasm_simd128/tools/gpt2_dump.js --out web/build/dumps
    .venv/bin/python oracle/compare.py web/build/dumps/manifest.json
    .venv/bin/python scripts/quantize_weights.py --wte-outliers 8
    .venv/bin/python eval/run.py --out eval/runs/int8-wte-o8 \
        --weights weights/gpt2-124m-int8-wte-o8.bin --reference eval/runs/fp32
    node web/test_tokenizer.mjs
    node web/test_web.mjs
    web/serve.sh                                           # localhost:8000/web/demo/
