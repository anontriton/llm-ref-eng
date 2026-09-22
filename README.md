# GPT-2 124M Inference Engine

A from-scratch GPT-2 124M inference engine, built in three stages -- a
hand-written PyTorch reference, a C++ engine, and a WebAssembly build -- where
the C++ engine is continuously validated layer-by-layer against the PyTorch
reference.

See [CLAUDE.md](CLAUDE.md) for project invariants, tolerance policy, and phase
order. Each directory has its own README describing what lives there.

    reference/   hand-written PyTorch GPT-2 + activation dumping
    engine/      C++ engine (src/, include/gpt2/, tests/, tools/)
    oracle/      dumped reference activations + manifest + compare.py
    bench/       benchmark harness + committed, commit-tagged results
    eval/        quantized-phase harness: pinned corpus, metrics, results
    web/         Emscripten build + demo page
    scripts/     weight download, format conversion

## Status

| Phase | State | Proof |
|---|---|---|
| 0 foundations | done | `reference/validate_hf.py` |
| 1 oracle | done | `oracle/test_compare.py` |
| 2 correct C++ | done | `oracle/compare.py`, `oracle/check_greedy.py`, `ctest` |
| 3 perf | done | `bench/results/`, oracle re-run per commit |
| 4 quantization | done | `eval/metrics.py`, `eval/results/` -- int8 ships, int4 rejected |
| 5 WASM | -- | |
| 6 docs | -- | |

## Picking this up

Almost nothing runnable is in the repo. Weights, activation dumps, build
directories and eval dumps are all gitignored and have to be regenerated; what
*is* committed is everything needed to regenerate them identically.

| Committed | Regenerate |
|---|---|
| `oracle/manifest.json` -- shapes and a sha256 per tensor | `oracle/activations/*.npy` (`reference/dump.py`) |
| `scripts/weights.lock.json` -- checkpoint checksum | `weights/` (download, convert, quantize) |
| `bench/prompts.tsv`, `eval/corpus.tsv`, `eval/calib.tsv` | -- pinned inputs, never regenerate casually |
| `bench/results/*.json`, `eval/results/*.json` | -- the record; commit-tagged |
| -- | `engine/runs.tsv`, `engine/build*/`, `engine/dumps*/`, `eval/runs/` |

Two consequences worth knowing before you start:

- **Regenerating the oracle is itself a test.** The manifest is committed with
  a checksum per tensor, and `compare.py` verifies dumps against it. If
  `reference/dump.py` produces different bytes on your machine, that is a real
  finding, not a setup problem.
- **`eval/metrics.py` needs the fp32 *dump*, not the committed summary.**
  `eval/results/*.json` records what was measured; the logits it was measured
  from live in `eval/runs/fp32/`, which is gitignored. Regenerate the fp32
  baseline before judging any quantized run against it.

### Before you commit a change to the engine

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

Watch the budget percentage `compare.py` prints, not just the pass. It has been
52.0% since Phase 2 and stayed there through four optimizations; a change that
moves it has changed the arithmetic even if it still passes.

For anything claiming to be faster, benchmark on a **clean tree** -- `run.py`
records dirty runs as dirty and they do not count -- and compare against a
result with the same `config` and `prompt`. For anything quantized,
`eval/metrics.py` is the authority and `compare.py` will refuse the run.

## Reproduce

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

## Results

### Phase 0 -- reference vs Hugging Face

The hand-written reference agrees with HF's `GPT2LMHeadModel` to a max absolute
logit error of 7.6e-5 across 4 prompts (max relative 8.4e-7), with 100% top-1
agreement and 50/50 identical greedy tokens.

### Phase 1 -- the oracle

615 checksummed tensors: 123 activations per forward pass across 5 fixed
prompts (T = 1, 5, 7, 17, 90). `compare.py` reports the first divergence in
forward order, and its self-test suite proves it rejects above-tolerance error,
NaN, shape drift, stale dumps, wrong weights, wrong prompts, and quantized
policies.

### Phase 2 -- C++ engine vs the oracle

| Check | Result |
|---|---|
| Layer-wise, `oracle/compare.py` | 615/615 tensors pass; worst uses 52% of its tolerance budget |
| Greedy decode, `oracle/check_greedy.py` | 50/50 tokens identical |
| Top-1 at every dumped position | 120/120 identical |
| Determinism | two independent dumps bit-identical, 615/615 |
| Unit tests, `ctest` | 5/5 suites |
| Comparison self-tests | 15/15 |

Scalar backend, single thread: all 5 oracle runs dump in about 16 s.

### Phase 3 -- performance

Four optimizations in the order CLAUDE.md fixes, each oracle-validated, each
leaving a committed benchmark. At T = 128 prompt / 128 generated:

| Step | prefill | decode/token | tokens/sec |
|---|---|---|---|
| baseline (scalar, no cache) | 5784 ms | 8781.5 ms | 0.114 |
| + KV cache | 5825 ms | 48.5 ms | 10.68 |
| + blocked GEMM | 4561 ms | 47.5 ms | 12.09 |
| + AVX2 | 1573 ms | 29.7 ms | 23.94 |
| + threads (8) | **493 ms** | **19.8 ms** | **42.55** |

373x end to end; 443x on decode; 12x on prefill. Every step is bit-identical to
the one before it -- verified by diffing engine dumps against engine dumps, not
argued from tolerance -- so the engine still sits at 52.0% of its fp32 budget
against PyTorch, the figure it has carried since Phase 2.

### Phase 4 -- int8

Weight-only, symmetric, per output channel. `wte` stays fp32; the twelve
layers' projections quantize. 497.8 MB to 243.3 MB.

| Metric | Result | Limit |
|---|---|---|
| Perplexity | 36.2798 vs 36.3366 (x0.99844) | <= x1.02 |
| Top-1 agreement | 97.480% (3985/4088) | >= 97% |
| Mean KL | 0.001168 nats | <= 0.01 |
| p99 KL | 0.006155 nats | <= 0.05 |
| Decisive disagreement | 0.000% (0 of 256) | <= 0.5% |

Speed, avx2, 8 threads: prefill 484 to 450 ms, decode 19.73 to 17.15 ms/token,
1.14x end to end.

int4 was implemented (`reference/gptq.py`) and rejected. GPTQ cuts the damage
four-fold against round-to-nearest, 42.3685 to 37.8621 perplexity, and still
misses every criterion -- decisive disagreement 1.562% against a limit of 0.5%,
where int8 scored 0.000%. It lives in the reference and not in the engine.

## Findings

These came out of the phases named and changed the project, so they are
recorded here rather than only in commit history.

### 1. The original tolerance rule could not be met in fp32

The fp32 policy began as two independent limits: max absolute error < 1e-4
**and** relative error < 1e-3. Against it, the finished engine -- after the
summation fixes in finding 3 -- still failed 87 of 615 tensors (the first, naive
version failed over 100), while matching the reference's top-1 token at every
position.

The failures tracked magnitude, not correctness. GPT-2's residual stream has a
few outlier dimensions (dimension 447 among them) that reach ~2650. At that
magnitude one fp32 ulp is 2.4e-4, so a 1e-4 absolute limit is *less than one
ulp*: passing would require bit-exact agreement with PyTorch's GEMM summation
order, which is an implementation detail, not a correctness property. The
clearest symptom: `block.2.mlp.out` failed at 9 ulps while `block.3.attn.out`
passed at 103 ulps, purely because the first is ~800x larger.

| \|x\| | one fp32 ulp | 1e-4 absolute limit |
|---|---|---|
| 1 | 1.2e-7 | reachable |
| 512 | 6.1e-5 | reachable |
| 1024 | 1.2e-4 | below one ulp |
| 2650 | 2.4e-4 | below one ulp |

The six relative-criterion failures had the mirror-image cause: values of
0.010-0.037, just above the 1e-2 relative floor, produced by summing 3072
products of magnitude ~2000 that nearly cancel.

**Resolution.** The two numbers now combine in the numpy.allclose form,
per element: `|engine - reference| <= 1e-4 + 1e-3 * |reference|`. The absolute
term governs near zero, the relative term at large magnitude. Three new
self-tests pin it: relative error under 1e-3 passes even when absolute error is
large; relative error over 1e-3 is still caught at large magnitude; and a
masked (exactly zero) attention probability that leaks 5e-4 is still caught.

Candidates considered and rejected, measured on the same dump:

| Rule | Tensors failing |
|---|---|
| abs < 1e-4 AND rel < 1e-3 (original) | 87 |
| raise abs to 5e-3, keep AND | 6 |
| error <= N ulps of the reference value | 313+ even at N = 1024 |
| **1e-4 + 1e-3·\|ref\| (adopted)** | **0**, worst at 52% of budget |
| 1e-4 + 1e-4·\|ref\| | 0, worst at 89% of budget |

Ulps of the reference value is the wrong unit: a near-zero result of a large
cancellation has an enormous ulp count relative to itself. The tighter
rtol = 1e-4 also passes, but with too little headroom to survive a Phase 3
reordering.

### 2. The engine is not less accurate than the reference

Before loosening anything, the question was whether the engine was simply
sloppy. Each matmul was recomputed in float64 from identical inputs and both
implementations were measured against that ground truth:

| Tensor | PyTorch error | Engine error |
|---|---|---|
| capital / block.2.mlp.out (\|x\| ~2317) | 3.6e-4 | 3.6e-4 (1.00x) |
| capital / block.11.mlp.out | 1.3e-5 | 6.3e-6 (0.50x) |
| newline / block.2.mlp.out | 5.7e-4 | 1.6e-4 (0.29x) |
| newline / block.11.mlp.out | 5.7e-5 | 3.2e-6 (0.06x) |

Given the same inputs, the engine's matmul is as accurate as PyTorch's or more
so. The residual gap between them is drift accumulated through 12 layers by two
legitimate but different summation orders, typically 3-30 ulps.

An earlier version of this comparison fed the engine its *own* upstream
activations, which made PyTorch look 10x better; isolating the kernel from
upstream drift is what the table above corrects.

### 3. Summation order is the whole game

The first engine summed each dot product in one sequential fp32 chain -- up to
3072 terms, losing roughly n·eps. Two changes cut the worst error from 6.1e-3 to
2.5e-3, and made a full dump faster (23 s to 16 s) through better cache use:

- `linear` accumulates **pairwise** along the inner dimension: 64-row leaves
  merged as a balanced binary tree, O(log n · eps) instead of O(n · eps).
- Every `dot` and `sum` accumulates across **exactly 8 lanes**
  (`backend::kAccumLanes`) and reduces them pairwise. The count is fixed in the
  backend contract rather than chosen per backend, because 8 is one AVX2
  register and two wasm_simd128 registers -- so the Phase 3 and Phase 5
  backends can reproduce scalar results bit for bit, and a backend swap cannot
  move the numbers.

Accumulating in double would also have passed, and was rejected for that
reason: no SIMD backend would reproduce it, so Phase 3 would regress on day one.

### 4. Smaller things worth knowing

- **`layer_norm_eps` is stored as f64 in the weight file.** Narrowing 1e-5 to
  f32 and widening it back gives 1.0000000116860974e-05, which `compare.py`'s
  exact config check reads as a provenance failure. Kernels still use it as f32,
  matching what PyTorch does with a Python float against an fp32 tensor.
- **tanh vs erf GELU peak 4.7e-4 apart, near |x| ~ 2.7** -- mid-range, not in
  the tails, where both converge. That is ~5x the absolute tolerance, and a unit
  test now pins the gap instead of a comment asserting it.
- **The engine does not tokenize.** The oracle already pins exact `input_ids`;
  a second tokenizer in C++ would produce disagreements that look like engine
  bugs. `scripts/export_runs.py` carries the ids across as data.
- **The weight file carries its source checkpoint's sha256**, so the engine's
  dump manifest can prove which weights it ran on without ever reading the
  original safetensors.

### 5. One tensor carried all of int8's damage

Per-channel int8 on the twelve layers' projections costs nothing measurable.
Quantizing `wte` as well is what hurts, and only one of the four metrics
noticed:

| Config | Perplexity | Top-1 | Mean KL | Size |
|---|---|---|---|---|
| int8, `wte` fp32 | x0.99844 | 97.48% | 0.00117 | 243.3 MB |
| int8, `wte` int8 | x1.01536 | 83.02% | 0.04139 | 127.7 MB |

Seventeen percent of argmaxes change and the distribution moves 35x further,
for 1.9x more compression -- and **perplexity would have waved it through** at
+1.54%. That is the argument for measuring more than one thing. `wte` is the
input embedding and the tied lm_head at once, so its error enters at the bottom
of the stack and again at the top.

Per-tensor scales were disqualified before any of this: 42.30 perplexity
against 36.34, because one scale for a whole matrix is set by that matrix's
worst outlier.

### 6. int8 collected less speed than Phase 3 left available

1.14x end to end, against 2.06x less weight traffic. Prefill barely moves
because blocked GEMM already made it compute-bound -- once weights stream once
instead of once per row, making them smaller stops mattering, and int8 adds a
widening instruction per lane. Decode is where it helps, and the ceiling says
why it does not help more:

| Config | traffic/token | decode | implied bandwidth |
|---|---|---|---|
| fp32 | 494.1 MB | 19.73 ms | 25.0 GB/s |
| int8, `wte` fp32 | 239.3 MB | 17.15 ms | 14.0 GB/s |
| int8, `wte` int8 | 123.5 MB | 12.66 ms | 9.8 GB/s |

fp32 decode runs at about what this machine's memory will do. Every step down
lands further below the limit, so saved bytes stop converting into saved time.
Keeping `wte` in fp32 leaves it as 64% of what decode still streams.

## Known limits going into Phase 5

- `Model::forward` computes logits for every position; prefill needs only the
  last row, and lm_head is 31% of prefill's arithmetic. Narrowing it changes
  the contract the oracle dumps are written against.
- The generated-ids tripwire in `bench/results/` means something weaker under a
  quantized policy: differing ids are expected, and `eval/metrics.py` is the
  authority instead.
- KL and decisive disagreement are computed on a strided sample, not every
  position. Full logits for the slice would be 800 MB.
- The tolerance rule lives in `compare.py`; the manifest records the numbers
  but not how they combine. Stamping the rule into the manifest would need the
  oracle regenerated.
- Watch the budget percentage `compare.py` prints. An optimization that moves
  it sharply has changed the arithmetic, even if it still passes.
