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
  token, over `generate - 1` steps. The first token is not one of them: it
  falls out of prefill, whose last row is already the distribution over what
  follows the prompt. Without `--kv-cache` each step re-runs the entire prefix
  through the full forward pass, so this is enormous and grows with each step;
  with it, each step extends the sequence by one position.
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
      "detail":  { "prefill_repeat": 3, "decode_steps": 127,
                   "decode_total_ms": 0.0, "weights_load_ms": 0.0 },
      "output":  { "generated_ids": [ ... ] },
      "machine": { "cpu": "...", "cores": 8, "platform": "...",
                   "compiler": "...", "runtime": "native|node vX",
                   "power": { "ac": true, "platform_profile": "performance",
                              "governor": "...",
                              "energy_performance_preference": "..." } }
    }

Results are commit-tagged so a speedup claim can always be traced back to the
change that produced it. A benchmark run on a dirty tree is recorded as dirty,
is named `...-dirty.json`, and does not count as a result.

The same holds for a throttled machine: on battery, or under a `low-power`
platform profile, a run is named `...-lowpower.json` and does not count. This
is not hypothetical. The first Phase 5 matrix was measured on battery at
`low-power`, with the clock at 1.27 of 4.2 GHz, and came out 2.4x slower in
every metric including weight loading -- while decoding exactly the Phase 3
tokens, so nothing but the clock had changed. Nothing in the JSON said so; now
`machine.power` does. Earlier results predate the field.

`output.generated_ids` is not a metric. It is a tripwire: an optimization that
changes what the engine decodes has changed the program, and the timings above
are then measuring something other than the thing they are being compared to.
It catches that without waiting for a full oracle run -- it does not replace
one.

It has already earned its place. The first cached benchmark decoded a sequence
that matched the uncached one for 66 tokens and then diverged: the decode loop
was re-feeding the last prompt token, which prefill had already put in the
cache, so the cached path was continuing from a prompt with a duplicated final
token. Every timing in that run was valid; the program being timed was not the
one it was being compared to. Nothing else in the harness would have noticed.

Under a quantized policy the tripwire means something different, and weaker.
int8 decodes 80 of 128 ids the same as fp32, and that is correct behaviour
rather than breakage -- the engine is a different function now, not a broken
one. `config.dtype` says which reading applies: for fp32 runs a difference
still means something changed that should not have; for quantized runs the
authority is `eval/metrics.py`, and the ids are only a record of what was
generated.

Comparisons are only valid between results that agree on `commit`-adjacent
context: `config`, `prompt.tokens`, and `prompt.generate`. Changing
`--generate` changes `decode_ms_per_token`, because without a KV cache the
per-step cost grows with sequence length.

## History

Results from f4e17d7 divided decode time by `generate`, counting one step that
repeated prefill's work. Later ones divide by `generate - 1` and record the
count in `detail.decode_steps`; the presence of that field is what tells the
two apart. The f4e17d7 baseline stays as the record of what was measured, but
the comparable scalar, no-KV baseline is the one at 237311a.

Phase: end of 2 (baseline), then every commit in 3+.
