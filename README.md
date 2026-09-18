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
    web/         Emscripten build + demo page
    scripts/     weight download, format conversion

## Status

| Phase | State | Proof |
|---|---|---|
| 0 foundations | done | `reference/validate_hf.py` |
| 1 oracle | done | `oracle/test_compare.py` |
| 2 correct C++ | done | `oracle/compare.py`, `oracle/check_greedy.py`, `ctest` |
| 3 perf | next | KV cache, then blocked GEMM, then AVX2, then threads |
| 4 quantization | -- | |
| 5 WASM | -- | |
| 6 docs | -- | |

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
    .venv/bin/python oracle/check_greedy.py                # ~3 min, no KV cache yet

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

## Findings

These came out of Phase 2 and changed the project, so they are recorded here
rather than only in commit history.

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

## Known limits going into Phase 3

- No KV cache: `gpt2_generate` re-runs the full prefix every step, so the
  50-step greedy check takes about 3 minutes. It is also the baseline the cache
  will be validated against.
- The tolerance rule lives in `compare.py`; the manifest records the numbers
  but not how they combine. Stamping the rule into the manifest would need the
  oracle regenerated.
- Watch the budget percentage `compare.py` prints. An optimization that moves
  it sharply has changed the arithmetic, even if it still passes.
