# Findings

What building this engine taught, in the order it was learned. Each one changed
the project -- a rule, a design, or a number -- so it is written down here
rather than left in commit history. The project's summary is the
[README](../README.md); the rules these findings produced are in
[CLAUDE.md](../CLAUDE.md).


## 1. The original tolerance rule could not be met in fp32

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

## 2. The engine is not less accurate than the reference

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

## 3. Summation order is the whole game

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

## 4. Smaller things worth knowing

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

## 5. One tensor carried all of int8's damage

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

Phase 5 found where that damage actually comes from, and took most of the
compression back without it -- see finding 8.

Per-tensor scales were disqualified before any of this: 42.30 perplexity
against 36.34, because one scale for a whole matrix is set by that matrix's
worst outlier.

## 6. int8 collected less speed than Phase 3 left available

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

## 7. wasm cannot be bit-identical to native while the kernels call libm

The Emscripten build of the scalar backend passes the oracle, but at 58.1% of
budget where the native engine has reported 52.0% since Phase 2. The two agree
bit for bit through embeddings, layernorm, QKV, attention and softmax, and part
at `block.0.mlp.act.out` -- GELU, the first `std::tanh`. Native links glibc;
Emscripten links musl. Measured over every float in [2^-20, 10]:

| libm | `tanhf` |
|---|---|
| glibc | correctly rounded everywhere |
| musl | off on 23.3% of inputs, by up to 2 ulp |

`expf` is correctly rounded 99.96% of the time in both, but not on the same
inputs. The backend seam never held the transcendentals -- they are in
`ops.cpp` -- so this is a property of the platform, not of any backend, and it
set the bar for the wasm_simd128 backend: bit-identical to the *wasm* scalar
build, which shares its libm. It is, 615 of 615. Cross-platform identity would
mean the engine carrying its own `tanh` and `exp`.

## 8. int8's damage in wte was 8 columns of lm_head

Finding 5 left `wte` in fp32, 154 of the 243 MB the browser would download.
`wte` does two jobs, and `reference/wte_sim.py` quantized each alone:

| Variant | Top-1 | Mean KL | Decisive |
|---|---|---|---|
| int8 `wte`, lm_head only | 83.12% | 0.04185 | 4.16% |
| int8 `wte`, embedding only | 97.58% | 0.00113 | 0.02% |

The embedding absorbs int8 for free; all the damage is the head. The cause is
its input: `ln_f`'s output has a few enormous hidden dimensions -- dim 496
averages |x| = 201 against a median of 0.35 -- so an int8 error in those
columns of `wte` is multiplied two-hundredfold, and those same columns set
every row's scale. Finer groups along `d_model` only reach 89% top-1, and bf16
reaches 93.7%. Zeroing the 8 largest columns in the int8 table and keeping them
in fp32 fixes it: `int8-wte-o8`, 129.3 MB, on the engine x0.99805 perplexity,
97.383% top-1, 0/256 decisive -- the simulation predicted the perplexity to five
decimal places -- and decode 1.15x faster, since lm_head stops streaming fp32.

The plan this came from was wrong in an instructive way: it proposed int8 for
lm_head and fp32 for the lookup, which could not have saved a byte -- any token
can be looked up, so the fp32 table would still ship. The diagnosis it wanted
was the right one; the conclusion was backwards.

## 9. A benchmark on battery measured the clock

The first Phase 5 benchmark matrix came out 2.4x slower than Phase 3 in every
metric, weight loading included, while decoding exactly Phase 3's tokens. The
laptop was on battery under the low-power profile, clock at 1.27 of 4.2 GHz.
Nothing in the JSON said so. `bench/run.py` now records AC state, platform
profile and governor, and names a throttled run `...-lowpower.json` so it does
not count -- the rule a dirty tree already had. The rerun on AC reproduced
Phase 3's single-thread AVX2 figure to 0.1%.

## 10. The tokenizer's strongest test needed no committed text

The JavaScript tokenizer has no oracle behind it, and the plan was to pin it
against the corpus text -- which turned out not to be committed; only its ids
and a checksum are. Byte-level BPE is lossless, though, so `decode(ids)` *is*
the text the ids came from, and encoding it again must reproduce HF's
tokenization token for token. Over 24,576 corpus tokens it does, in 60 ms.

Two behaviours were decided by HF's output rather than by guessing: GPT-2's
`\s` is Unicode White_Space, which JavaScript's `\s` is not (it adds U+FEFF
and drops U+0085); and `<|endoftext|>` typed as text becomes the special token,
with the whitespace around it kept. The test was then shown to fail on six
planted bugs before it was trusted.

## 11. The first live deploy failed on a network the tests never had

Every check passed in CI, and the live page failed on first load with a bare
"network error". Replaying the loader's requests on the live site showed one
32 MiB weight part cut off mid-download while the CDN was still cold; `curl`
fetched the same part cleanly five times afterwards. The local server and CI's
test server had never dropped a connection -- or compressed a response, which
GitHub Pages does for these parts -- so a loader that treated any interruption
as fatal had passed everything.

The loader now downloads each part whole and retries it from zero, up to four
times, before the engine sees a byte of it. More to the point, the page test's
server now cuts a part off halfway: once, which must be retried, and on every
request, which must fail with a reason. Run against the worker that shipped,
both cases fail with the live symptom; against the fix, both pass.

## Known limits

- `Model::forward` computes logits for every position; prefill needs only the
  last row, and lm_head is 31% of prefill's arithmetic. Narrowing it changes
  the contract the oracle dumps are written against.
- The wasm build has no threads. They need `SharedArrayBuffer`, so COOP/COEP
  headers on whatever serves the page; native reaches 42.8 tok/s at 8 threads
  against 23.96 at one.
- wasm and native differ in the last bits (finding 7); both pass the oracle.
- KL and decisive disagreement are computed on a strided sample, not every
  position. Full logits for the slice would be 800 MB.
- The tolerance rule lives in `compare.py`; the manifest records the numbers
  but not how they combine. Stamping the rule into the manifest would need the
  oracle regenerated.
- The generated-ids tripwire in `bench/results/` means something weaker under a
  quantized policy: differing ids are expected, and `eval/metrics.py` is the
  authority instead.
- The weights are not in the repository. The hosted demo's 129 MB file is
  rebuilt from Hugging Face by CI on each deploy and published only with the
  GitHub Pages site, so the demo depends on both staying reachable.
