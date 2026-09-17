# GPT-2 Inference Engine

A GPT-2 124M inference engine built from scratch in three stages: a hand-written
PyTorch reference, a C++ engine, and a WebAssembly build. The defining feature is
that the C++ engine is continuously validated layer-by-layer against the PyTorch
reference.

## Non-negotiables
- The PyTorch reference is hand-written. HF's GPT2LMHeadModel is used ONLY
  to validate the reference, never as the reference itself.
- The C++ engine must match the PyTorch oracle layer-by-layer before ANY
  optimization work begins.
- Every optimization commit re-runs oracle validation. No exceptions.
- SIMD goes behind an abstraction layer (AVX2 backend now, wasm_simd128 later).
  No raw intrinsics in kernel logic.
- C++ activations are dumped to .npy and compared in Python. Do not write an
  npy reader in C++.

## Architecture (GPT-2 124M)
12 layers, 12 heads, d_model 768, d_ff 3072, ctx 1024, vocab 50257

## Weight-loading gotchas
- HF Conv1D stores weights TRANSPOSED vs nn.Linear: x @ W + b
- attn.c_attn is fused QKV: 768 -> 2304, split into three 768 chunks
- GELU is the tanh approximation (gelu_new), NOT erf
- Positional embeddings: learned absolute (wpe), added to wte output
- lm_head is tied to wte (no separate output projection in checkpoint)
- LayerNorm eps = 1e-5
- Attention scale = 1/sqrt(64) (head dim, not d_model)

## Tolerance policy (fp32 phases)
- Per-layer max absolute error < 1e-4
- Per-layer relative error < 1e-3 above a magnitude floor
- Top-1 token identical for 50 consecutive greedy steps

## Tolerance policy (quantized phases)
Layer-wise matching is void. Use instead:
- Perplexity on a fixed WikiText-2 slice
- Top-1 agreement rate vs the fp32 engine
- KL divergence of logits vs fp32

## Phase order
0 foundations -> 1 oracle -> 2 correct C++ -> 3 perf (KV cache, then blocked
GEMM, then AVX2, then threads) -> 4 quantization (int8, then int4) ->
5 WASM -> 6 docs

Work strictly in phase order. Do not begin a phase before the previous one's
exit criteria are met. See "Phase status" below for where we actually are.

## Benchmarking
Emit JSON per run: tokens/sec, prefill ms, per-token decode ms, commit SHA.
Commit to bench/results/. Built BEFORE optimization starts.

---

## Repo layout
- `reference/` - hand-written PyTorch GPT-2 + activation dumping
- `engine/` - C++ engine (`src/`, `include/gpt2/`, `tests/`)
- `oracle/` - dumped reference activations + manifest + `compare.py`
- `bench/` - benchmark harness + `results/` (committed JSON, commit-tagged)
- `web/` - Emscripten build + demo page
- `scripts/` - weight download, format conversion

Generated artifacts (weights, `.npy` activation dumps, build dirs) are
gitignored. `bench/results/*.json` is the one generated thing that IS committed.

## Oracle contract
The oracle is a directory of `.npy` tensors plus `oracle/manifest.json`
describing them. Both the reference dumper and the C++ engine write dumps in the
same layout; `oracle/compare.py` is the only thing that reads them and is the
sole authority on pass/fail.

Tensor naming is stable and hierarchical, e.g.:
`embed.out`, `block.{i}.ln_1.out`, `block.{i}.attn.qkv`,
`block.{i}.attn.scores`, `block.{i}.attn.out`, `block.{i}.mlp.fc.out`,
`block.{i}.out`, `ln_f.out`, `logits`

The manifest pins what a run means so comparisons are apples-to-apples: model id
and weight checksum, prompt token ids, dtype, shapes, and the tolerance policy in
force. A dump whose manifest disagrees with the run being compared is a failure,
not a warning.

## Environment
Missing on this machine as of setup; install before the phase that needs it:
- `torch`, `numpy`, `transformers` (Phase 0/1) - `transformers` is for
  validation and tokenization ONLY
- `cmake` (Phase 2)
- Emscripten SDK / `emcc` (Phase 5)

Present: g++ 16.2.1, make, ninja, 8 cores, AVX2 + FMA + AVX512F.
Target the AVX2 backend regardless of AVX512 availability - WASM SIMD is 128-bit
and the abstraction layer is designed against that width.

## Conventions
- Python: keep the reference readable over clever; it is documentation as much
  as it is code.
- C++: C++17, no external dependencies in the engine core. Headers in
  `include/gpt2/`, one translation unit per concern in `src/`.
- Determinism beats speed until Phase 3. No fast-math, no reassociation flags.
- Each phase leaves behind a way to prove it worked: a test, a comparison run,
  or a committed benchmark JSON.

## Phase status
- [ ] 0 foundations
- [ ] 1 oracle
- [ ] 2 correct C++
- [ ] 3 perf
- [ ] 4 quantization
- [ ] 5 WASM
- [ ] 6 docs

Update this checklist when a phase's exit criteria are met, not when its code is
merely written.
