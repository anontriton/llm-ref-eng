# bench/ -- benchmark harness

Built BEFORE optimization starts, so Phase 3 has a baseline to move.

Every run emits one JSON file into `results/`, committed to the repo:

    {
      "commit": "<git SHA>",
      "dirty": false,
      "timestamp": "<ISO 8601>",
      "config":  { "build": "...", "backend": "scalar|avx2|wasm_simd128",
                   "threads": 1, "dtype": "fp32", "kv_cache": false },
      "prompt":  { "tokens": 128, "generate": 128 },
      "metrics": { "tokens_per_sec": 0.0, "prefill_ms": 0.0,
                   "decode_ms_per_token": 0.0 },
      "machine": { "cpu": "...", "cores": 8 }
    }

Results are commit-tagged so a speedup claim can always be traced back to the
change that produced it. A benchmark run on a dirty tree is recorded as dirty
and does not count as a result.

Phase: end of 2 (baseline), then every commit in 3+.
