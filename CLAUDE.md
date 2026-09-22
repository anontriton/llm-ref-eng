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
- Per element: |engine - reference| <= 1e-4 + 1e-3 * |reference|
  (the numpy.allclose form: the 1e-4 term governs near zero, the 1e-3 term at
  large magnitude)
- Top-1 token identical for 50 consecutive greedy steps

The two numbers combine; they are not independent limits. GPT-2's residual
stream has outlier dimensions near 2650, where one fp32 ulp is 2.4e-4, so a
bare 1e-4 absolute limit would demand bit-exact agreement with PyTorch's GEMM
summation order -- not a correctness property. Revised in Phase 2 on evidence;
see the Phase 2 note below.

## Tolerance policy (quantized phases)
Layer-wise matching is void. `oracle/compare.py` refuses a non-fp32 policy
outright; `eval/metrics.py` is the authority instead. Use:
- Perplexity on a fixed WikiText-2 slice
- Top-1 agreement rate vs the fp32 engine
- KL divergence of logits vs fp32
- Decisive disagreement: added in Phase 4, because a raw agreement rate scores
  a coin-flip between two equally likely tokens the same as overwriting a
  confident answer. A flip counts only when fp32 preferred its own pick by more
  than 0.05 probability.

## Phase order
0 foundations -> 1 oracle -> 2 correct C++ -> 3 perf (KV cache, then blocked
GEMM, then AVX2, then threads) -> 4 quantization (int8, then int4) ->
5 WASM -> 6 docs

Phase 4 shipped int8 and declined int4 on measurement; see the exit criteria
below. A phase's contents are a plan, not a promise that every item survives
contact with its own acceptance criteria.

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
- `eval/` - quantized-phase harness: pinned corpus, `metrics.py`, `results/`
- `web/` - Emscripten build + demo page
- `scripts/` - weight download, format conversion, quantization, pinned inputs
- `docs/` - findings, and the procedure for maintaining the repository

Generated artifacts (weights, `.npy` activation dumps, build dirs) are
gitignored. The committed exceptions are `bench/results/*.json` and
`eval/results/*.json`, both commit-tagged, plus the pinned inputs they depend
on: `bench/prompts.tsv`, `eval/corpus.tsv`, `eval/calib.tsv`, and the
tokenizer's fixture `web/tokenizer_cases.json`.

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
Python deps live in `.venv/` (gitignored). Python 3.14.7, torch 2.14.0+cpu,
numpy 2.5.3, transformers 5.17.0. Recreate with:

    python3 -m venv .venv
    .venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
    .venv/bin/python -m pip install numpy transformers

Run Python entry points from the repo root, e.g. `.venv/bin/python
reference/validate_hf.py`.

Emscripten 6.0.9-git (4e42238) for Phase 5, from Arch's `extra/emscripten`.
It does **not** put `emcc` on PATH -- only binaryen's `wasm-*` tools land in
`/usr/bin`, and the compiler drivers stay in `/usr/lib/emscripten`:

    export PATH="$PATH:/usr/lib/emscripten"     # or call emcc by full path

`emcc --version` runs a sanity check on first use, so a slow first invocation
is normal rather than a symptom. Node 22.23.2 is present, which Emscripten
depends on and which Phase 5 also needs to run the wasm build against the
oracle.

Present: g++ 16.2.1, cmake, make, ninja, 4 cores / 8 threads, AVX2 + FMA +
AVX512F.
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
- [x] 0 foundations
- [x] 1 oracle
- [x] 2 correct C++
- [x] 3 perf
- [x] 4 quantization
- [x] 5 WASM
- [x] 6 docs

Update this checklist when a phase's exit criteria are met, not when its code is
merely written.

Phase 0 exit criteria, met: weights downloaded and checksum-pinned; the
hand-written reference loads them; `reference/validate_hf.py` agrees with HF on
logits within tolerance across 4 prompts and reproduces 50 identical greedy
tokens.

Phase 1 exit criteria, met: `reference/dump.py` writes 615 checksummed tensors
across 5 runs plus `oracle/manifest.json`; `oracle/compare.py` enforces
provenance and tolerance and reports the first divergence in forward order;
`oracle/test_compare.py` passes 12/12, proving the comparison rejects
above-tolerance error, NaN, shape drift, stale dumps, wrong weights, wrong
prompts, and quantized policies -- and reports the earliest divergence rather
than the largest.

Phase 2 exit criteria, met: the C++ engine dumps all 615 tensors across the 5
oracle runs and `oracle/compare.py` passes every one, worst tensor at 52% of
its tolerance budget; `oracle/check_greedy.py` reproduces 50/50 greedy tokens;
5 ctest suites pass; `oracle/test_compare.py` passes 15/15. The tolerance rule
was revised from independent abs-AND-rel limits to the combined form above,
after measurement showed the engine's matmul is as accurate or more accurate
than PyTorch's against a float64 ground truth given identical inputs -- the old
rule was failing fp32 resolution, not the engine. Every dot product and sum
accumulates across a fixed 8 lanes (`backend::kAccumLanes`) so the Phase 3
AVX2 and Phase 5 wasm_simd128 backends can reproduce scalar results exactly.

Phase 3 exit criteria, met: KV cache, blocked GEMM, AVX2 backend, and threads,
in that order, each validated against the oracle and each leaving a committed
benchmark JSON. At T=128/128, 8 threads, avx2:

    tokens/sec         0.114 ->  42.554     373x
    prefill ms          5784 ->     493      12x
    decode ms/token     8781 ->    19.8     443x

Every step is bit-identical to the one before it -- verified by comparing
engine dumps against engine dumps, not argued from tolerance -- so the engine
still sits at 52.0% of budget against the PyTorch oracle, the same figure it
reported at the end of Phase 2. `-DGPT2_BACKEND=avx2` selects the backend; only
that translation unit gets ISA flags. The AVX2 backend refuses FMA on purpose,
since fusing rounds once where the scalar backend rounds twice and
wasm_simd128 has no FMA to offer in Phase 5. Thread count is not a numerical
parameter: 615 tensors bit-identical at 1, 3 and 8 threads, and test_threading
pins that property directly.

Two notes for Phase 4. The `bench/results/` entry from f4e17d7 predates a fix
to how decode steps were counted and is not comparable to later runs; results
carrying `detail.decode_steps` use the current accounting. And `Model::forward`
computes logits for every position, which prefill does not need -- only the
last row feeds generation, and lm_head is 31% of prefill's arithmetic. It was
left alone because it changes the forward contract the oracle dumps depend on.

Phase 4 exit criteria, met: the eval harness exists and is validated against
something other than itself, int8 ships, int4 was implemented and rejected.

int8 is weight-only, symmetric, per output channel, on the twelve layers'
projections. Against the fp32 engine on the pinned WikiText-2 slice:
perplexity x0.99844, top-1 97.480%, mean KL 0.001168, p99 KL 0.006155,
decisive disagreement 0 of 256. Weight file 497.8 MB -> 243.3 MB, 1.14x end to
end. `wte` stays fp32: quantizing it as well compresses to 127.7 MB and still
passes on perplexity, at +1.54%, while changing 17% of argmaxes and moving the
distribution 35x further. Perplexity alone would have shipped that, which is
why there is more than one metric.

int4 is implemented in `reference/gptq.py` and is not in the engine. GPTQ cuts
the damage four-fold against round-to-nearest -- 42.3685 to 37.8621 perplexity
at group 64 -- and still misses every criterion: ratio x1.04198, top-1 86.106%,
mean KL 0.041376, decisive disagreement 1.562%. That last number is the reason.
It was added during int8 to stop a raw agreement rate from counting a coin-flip
as damage, and there it cleared the quantizer at 0.000%; at int4 it says the
engine overwrites answers the fp32 model was sure about. Activation ordering
improves three of the metrics and makes that one worse. Four bits does not fit
a 124M model, and no threshold should be moved to pretend otherwise.

The thresholds in `eval/metrics.py` were revised once, on evidence, the way the
fp32 tolerance was in Phase 2 -- top-1 from a guessed 98% to a measured 97%,
after the disagreements were shown to sit where fp32's own top-1/top-2 margin
is 0.00093 against 0.10553 elsewhere. The int4 rejection above is what that
revision bought the right to say.

Phase 5 exit criteria, met: the engine builds to wasm and passes the oracle
under Node, the wasm_simd128 backend is bit-identical to wasm scalar, both are
benchmarked beside native, and a browser demo runs the engine and shows
exactly its output. `web/build.sh` builds it; `web/README.md` has the detail.

The wasm build is not bit-identical to native, and cannot be while the kernels
call libm: GELU's tanh comes from musl under Emscripten, off by up to 2 ulp on
23% of inputs where glibc's is correctly rounded. Everything before
`block.0.mlp.act.out` agrees exactly; the oracle passes at 58.1% of budget
against native's 52.0%. So wasm_simd128 is held to the wasm scalar build --
615/615 tensors identical, int8 outputs byte-identical -- the standard AVX2
met against native scalar. Single-threaded, T=128/128:

    tokens/sec    native avx2 23.96    wasm_simd128 17.70    74%

The download was the problem, not speed. `reference/wte_sim.py` found int8
wte's damage is entirely lm_head, and entirely a few hidden dims where ln_f's
output is two hundred times the median. `int8-wte-o8` zeroes those 8 columns
in the int8 table and keeps them in fp32: 243.3 MB -> 129.3 MB, perplexity
x0.99805, top-1 97.383%, decisive disagreement 0/256 on the engine -- the
simulation predicted the perplexity to five decimals -- and decode 1.15x
faster, since lm_head stops streaming fp32.

The tokenizer is `web/tokenizer.js`, hand-written and held to HF's ids on a
committed fixture, to every committed prompt, and to a decode/re-encode round
trip over 24,576 corpus tokens; its test was shown to fail on six planted
bugs. The demo (`web/serve.sh`) streams the weights into wasm memory, runs in
a worker, and at temperature 0 puts on screen exactly what `gpt2_generate`'s
ids decode to, checked in headless Chrome. Decode is 21 ms/token in the
browser. Threads are not in the wasm build: they need SharedArrayBuffer and
cross-origin isolation, and single-threaded was enough to meet the rest.

`bench/run.py` records the machine's power state since this phase, after a
matrix measured on battery came out 2.4x slow in every metric and nothing in
the JSON said so.

Phase 6 exit criteria, met: every document is current, every documented
command runs, and the README is written for someone arriving cold. The README
says what the project is, how correctness is proven, and what it measured;
`docs/findings.md` holds the findings, ten now, and the known limits;
`docs/maintaining.md` holds the procedure for changing the engine without
breaking what it proves. `./demo.sh` is the one-command path to the browser
demo from a fresh clone, prerequisites checked before anything is downloaded.

Proven rather than proofread: in a clean clone, every command block in the
README and `docs/maintaining.md` -- about 60 commands, the full reproduce
included -- ran without a failure, and the regenerated oracle matched the
committed manifest's 615 checksums, the three weight files byte for byte; and
`./demo.sh` took a clone with nothing installed to a served page. Every
relative link and anchor in the docs resolves.

