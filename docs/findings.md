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

## 12. "No FMA" was a rule the build only enforced on x86

The numerics contract has always said no fused multiply-add: `acc += a * b`
rounds twice in the scalar backend, and every SIMD backend must round the same
way. `-ffp-contract=off` sat on the AVX2 and wasm backend files, and that was
enough on the machine the engine was built on -- GCC in ISO mode does not
contract, and x86-64 without `-mfma` has nothing to contract into.

Built on an Apple M4 with Apple clang, it was not enough. Clang's default is
`-ffp-contract=on`, arm64 has `fmadd` as baseline, and `backend_scalar.cpp`
compiled to 30 fused multiply-adds. The engine still passed the oracle, which
is the problem: tolerance cannot see this. Bit-identity can -- the native
build parted from the wasm scalar build at `block.0.attn.qkv`, the first
matmul, where on x86 the two part at the first `tanh`. With the flag global,
the arm64 build parts from wasm at `block.0.attn.probs`, the first `exp`: the
libm boundary of finding 7, and nothing earlier. On x86 GCC the change is a
no-op.

The contract now lives where the build can enforce it everywhere, and
`GPT2_NUMERICS=fast` (finding 16) is the one place it is lifted on purpose.

## 13. At 1024 positions, no fp32 implementation meets the fp32 rule

The oracle's longest prompt was 90 tokens, the eval's windows stop at 512, and
the demo runs to 1024. A sixth oracle run, `context`, fills the window with the
first 1024 ids of the pinned eval corpus. The engine failed it on 8 tensors,
`logits` at 857% of budget -- and passed the KV-cached check of position 1023
at 39%.

Against the same model run in float64, on every failing tensor the engine was
the closer of the two to the exact answer, by 1.3x to 4x, and its own q.k
arithmetic was 3x more accurate than PyTorch's. The first miss was a score of
-0.04 produced by terms summing to 148, a 3,700x cancellation that turns
`qkv`'s in-tolerance difference into an out-of-tolerance one. And against
float64, neither implementation meets the rule at this length: PyTorch fp32
misses on 8 tensors (worst 1237%), the engine on 4 (worst 971%) -- 5, 2 and 3
elements out of 786,432 in three of them, and 0.016% of logits, every one a
value near zero made by cancellation. Finding 1 again, at long context.

The obvious fix -- pass a tensor that sits no farther from float64 than
PyTorch fp32 does -- was built and is not sound. The engine was closer on 15 of
the 16 tensors near the limit and lost `ln_f.out` by 20%: its own LayerNorm
arithmetic was at 1.5% of budget, the loss inherited entirely from where
rounding happened to fall upstream, where LayerNorm's mean subtraction turns a
-5.5 into a -0.35 and keeps the absolute error. Two independent roundings,
compared 123 times, will lose one by chance. Passing it would have meant a
hand-picked factor.

So the `context` run's reference is the float64 forward, and what gates is its
KV-cached row 1023, under the rule unchanged: that row attends over the keys
and values of every earlier position, so their errors reach it. The
whole-sequence comparison runs and is reported, with PyTorch fp32's own figure
beside each miss, but does not gate. The five original runs are untouched.

## 14. Most of a prefill was arithmetic nothing read

Two things `Model::forward` computed and threw away. Logits for every position,
when generation reads only the last: lm_head is 31% of a 128-token prefill.
And attention over masked positions -- scores for keys a query may not see,
then `dst += 0 * v` for each of them, half of a prefill's probs @ v.

`Logits::Last` computes only the final row. Each logit is still one dot
product over the same span, so the row is the same bits; the dump tool's
`--last-logits` proves it against the oracle, with the record naming the row
it holds so `compare.py` slices the reference there, and three self-tests
showing a missing or wrong row is caught. The upper triangle of the scores is
now computed only when tapped -- the oracle dumps keep their checksums -- and
the weighted sum stops at the last visible key. Attention also runs threaded,
one item per head and query, each writing only its own row.

Every one of these is bit-identical to the engine before it: all 738 tensors,
native and both wasm backends, whole-sequence and KV-cached, at 1, 3 and 8
threads, and 50 generated tokens on three prompts. On an Apple M4 Max -- on
battery, so as a ratio only, measured back to back:

    prefill, T=128, native scalar     1 thread   1768 ->  648 ms    2.7x
                                      8 threads   310 ->  126 ms    2.5x

## 15. Threads sped up prefill and did nothing for decode

The wasm build now has threads -- a second build with pthreads, loaded when
the page is cross-origin isolated, which on GitHub Pages a small service
worker arranges (`web/coi-sw.js`). It is bit-identical to the single-threaded
build at 1 and 8 threads, in Node and on the page, which shows exactly
`gpt2_generate`'s 50 tokens at 8 threads. Under Node on the M4 Max:

| wasm_simd128 | Prefill | Decode / token | Tokens / s |
|---|---:|---:|---:|
| single-threaded build | 843 ms | 10.4 ms | 59.3 |
| threaded build, 1 thread | 885 ms | 11.3 ms | 55.3 |
| threaded build, 8 threads | 165 ms | 11.7 ms | 77.9 |

Prefill scales 5.4x. Decode does not move, here or natively, where 8 threads
decode at 17.6 ms against 16.2 at one. A decode step is one row, so its ~70
`parallel_for` calls are each a few microseconds of work behind a wake-up and
a join of the same order. A pool that spins rather than sleeps between calls
is the next thing to try.

The page met one surprise on the way: WebCrypto refuses to hash shared memory,
and the threaded build's memory is a SharedArrayBuffer, so the sha256 check of
the weights threw. The threaded page copies the file out once to hash it.

## 16. Fused multiply-add bought little where it could be measured

`GPT2_NUMERICS=fast` lifts finding 12's rule on purpose: FMA in the AVX2
backend, relaxed-SIMD madd in wasm, compiler contraction in scalar. A fast
build names itself `<backend>-fast` everywhere a backend is recorded and is
judged by tolerance and `eval/metrics.py` rather than bit-identity. It changes
720 of 738 tensors, passes the oracle, and against the exact fp32 engine on the
eval slice scores perplexity x1.00000, top-1 100.000%, KL 0.000000, natively
and in wasm.

And it is barely faster. On the M4 Max, prefill 654 -> 637 ms native and
853 -> 803 ms in wasm; decode, which streams weights, unchanged. The case it
was built for is AVX2, where the FMA path has 37 fused instructions against 0
in the exact build. This round of work was on arm64, which cannot run it; CI
can, and holds it to the oracle on every push -- it passes, whole-sequence and
KV-cached. How much faster it is there is still unmeasured: a shared CI runner
is no place to time anything.

## Known limits

- The 1024-position oracle run gates only its last position; its
  whole-sequence comparison is reported, because no fp32 implementation meets
  the rule there (finding 13).
- Threads do not speed up decode, native or wasm (finding 15).
- The AVX2 backend since findings 12-16 has been validated in CI only --
  against the oracle, bit-identical to native scalar, 50/50 greedy -- not on
  the machine its benchmarks came from, so `bench/results/` has no AVX2 figure
  for these changes, and the fast tier's AVX2 speed is unmeasured.
- The page test exercises the threaded path; the fallback for a browser that
  refuses isolation is the single-threaded module `test_web.mjs` covers, not a
  page-level check.
- There is no NEON backend: arm64 runs the scalar one natively.
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
