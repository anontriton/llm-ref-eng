# bench/ -- benchmark harness

Built BEFORE optimization starts, so Phase 3 has a baseline to move.

## Running

    cmake --build engine/build
    .venv/bin/python bench/run.py

`engine/tools/gpt2_bench` measures; `bench/run.py` wraps it with provenance and
writes the result. Timing lives in C++ so that process startup and weight
loading stay outside the measurement. Recording lives in Python, the same split
the oracle uses: the C++ side produces, the Python side records and judges.

`bench/prompts.tsv` holds the benchmark prompt as token ids, in the format
`engine/src/runs.cpp` already reads -- the engine does not tokenize, so ids
travel as data. Regenerate it with `scripts/export_bench_prompt.py`; it is
committed, so a result stays reproducible.

Useful flags: `--generate N`, `--prefill-repeat N`, `--no-write`, and
`--backend/--threads/--kv-cache`, which declare how the binary was built so the
config is recorded honestly. There is no build-level backend switch yet, so
`--backend` is an assertion rather than a detection; that changes when the AVX2
backend lands.

## What is measured

- **prefill_ms** -- one forward pass over the whole prompt, best of
  `--prefill-repeat`. The minimum rather than the mean: every source of noise
  adds time and none removes it, so the fastest iteration is the closest to the
  machine's real capability. Compute-bound on the matmuls; this is what blocked
  GEMM, AVX2, and threads go after.
- **decode_ms_per_token** -- the mean cost of extending the sequence by one
  token. Until the KV cache lands, every step re-runs the entire prefix through
  the full forward pass, so this is enormous and grows with each step. That is
  the honest baseline the cache gets measured against.
- **tokens_per_sec** -- end-to-end: `generate` tokens over the total time a
  caller waits, prefill included. The two numbers above break it down.

## Result format

Every run emits one JSON file into `results/`, committed to the repo:

    {
      "schema": 1,
      "commit": "<git SHA>",
      "dirty": false,
      "timestamp": "<ISO 8601>",
      "config":  { "build": "RelWithDebInfo", "backend": "scalar|avx2|wasm_simd128",
                   "threads": 1, "dtype": "fp32", "kv_cache": false },
      "prompt":  { "name": "bench128", "tokens": 128, "generate": 128 },
      "metrics": { "tokens_per_sec": 0.0, "prefill_ms": 0.0,
                   "decode_ms_per_token": 0.0 },
      "detail":  { "prefill_repeat": 3, "decode_total_ms": 0.0,
                   "weights_load_ms": 0.0 },
      "output":  { "generated_ids": [ ... ] },
      "machine": { "cpu": "...", "cores": 8, "platform": "...",
                   "compiler": "..." }
    }

Results are commit-tagged so a speedup claim can always be traced back to the
change that produced it. A benchmark run on a dirty tree is recorded as dirty,
is named `...-dirty.json`, and does not count as a result.

`output.generated_ids` is not a metric. It is a tripwire: an optimization that
changes what the engine decodes has changed the program, and the timings above
are then measuring something other than the thing they are being compared to.
It catches that without waiting for a full oracle run -- it does not replace
one.

Comparisons are only valid between results that agree on `commit`-adjacent
context: `config`, `prompt.tokens`, and `prompt.generate`. Changing
`--generate` changes `decode_ms_per_token`, because without a KV cache the
per-step cost grows with sequence length.

Phase: end of 2 (baseline), then every commit in 3+.
