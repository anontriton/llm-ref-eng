# Showing its work: a GPT-2 engine proven correct at every step

*Iverson Lai · September 2026*

This is the story of building GPT-2 from scratch three times: in PyTorch, in
dependency-free C++, and in WebAssembly that runs in a browser tab. The more
interesting part is how each version was proven to compute the same thing as
the one before it. The [README](../README.md) has the numbers and commands,
and [findings.md](findings.md) has the detailed measurements. This article
explains why the project was built the way it was.

**Try it:** [anontriton.github.io/llm-ref-eng](https://anontriton.github.io/llm-ref-eng/)
runs GPT-2 entirely in your browser.

---

## The problem with "it works"

Suppose you write a neural network by hand and it generates fluent English.
Does that mean it's correct?

Not necessarily. Numerical code rarely fails loudly. If you transpose a matrix
the wrong way, use a slightly different formula for an activation function, or
scale attention by the wrong constant, the program still runs, the output is
still a list of plausible numbers, and the model often still writes readable
text. It's just a slightly worse model than the one you meant to build, and
nothing in the output tells you so.

GPT-2 has several traps of exactly this kind. Its checkpoint stores some weight
matrices transposed relative to PyTorch's usual convention. Its GELU is the
tanh approximation, not the exact erf form, and the two differ by up to
4.7e-4. Attention is scaled by 1/√64 (the per-head dimension), not by
1/√768. Getting any of these wrong still produces text.

So the goal of this project was never just to build an engine that runs. The
goal was to build one that can **prove** it computes GPT-2, and keeps proving
it while it's rewritten for speed, compressed, and moved to a different
platform.

## Background: what running GPT-2 involves

For readers new to language models, here's the minimum needed for the rest of
the article.

GPT-2 is a language model OpenAI released in 2019. Its input is a sequence of
**tokens**, which are words or pieces of words, each identified by a number
between 0 and 50,256. Its output is 50,257 scores, one for every possible next
token. Generating text means picking a token from those scores, appending it
to the input, and running the model again.

The smallest GPT-2 has 124 million **weights**, numbers learned during
training and distributed as a 548 MB file on Hugging Face. Running the model
(called **inference**) is a fixed sequence of arithmetic on those weights:

1. Each token becomes a vector of 768 numbers: a learned embedding for the
   token plus a learned embedding for its position.
2. Those vectors pass through 12 layers. Each layer has **attention**, which
   mixes in information from earlier tokens, and an **MLP**, which is two
   matrix multiplications with a nonlinear function between them. Each part
   is preceded by a normalization step and wrapped in a residual connection
   (its output is added to its input).
3. A final normalization and one more matrix multiplication turn the last
   vector into the 50,257 scores, called **logits**.

Almost all of the computing time goes to matrix multiplication. This project
doesn't train anything. It uses OpenAI's weights and implements everything
needed to run them.

## The plan: three implementations, strictly in order

The project was split into phases, and each phase had to meet written exit
criteria before the next one could start:

0. **Foundations:** download the weights, pin them by checksum, and write a
   GPT-2 by hand in PyTorch.
1. **Oracle:** turn that PyTorch version into a source of truth that other
   implementations can be checked against.
2. **Correct C++:** write a C++ engine and make it match the oracle, with no
   optimization allowed yet.
3. **Performance:** make it fast, re-validating after every change.
4. **Quantization:** shrink the weights and measure what that costs.
5. **WebAssembly:** run the engine in a browser.
6. **Documentation:** proven by running every documented command in a
   clean clone.

The order matters most between phases 2 and 3. Optimizing code that has never
been shown correct means you can't tell your own optimization bugs from bugs
that were already there. Proving correctness first means each optimization
only has to show one thing: that it changed nothing.

## Phase 0: a reference written by hand

The first version is [`reference/model.py`](../reference/model.py), GPT-2 in
226 lines of PyTorch. It uses PyTorch's basic operations (matrix multiply,
softmax) but not its GPT-2 implementation. The model's structure, the weight
loading, and every one of the traps above are written out explicitly.

It was checked once against Hugging Face's `GPT2LMHeadModel`, the widely used
implementation. Over four prompts the largest logit difference was 7.6e-5, and
both produced the same 50 tokens in a row when generating greedily (always
taking the top-scoring token).

From that point, Hugging Face drops out. It serves only to validate the
reference. The hand-written reference becomes the thing everything else is
measured against, because it's code this project owns and can read, and it
can be modified to record whatever the later stages need.

The weights are pinned too. [`scripts/weights.lock.json`](../scripts/weights.lock.json)
records the Hugging Face commit to download from (never "latest") and a
SHA-256 checksum for every file. If the files upstream ever change, the
download fails instead of silently producing different results.

## Phase 1: the oracle

Checking only the final output isn't enough, for two reasons. When it's wrong,
the cause could be anywhere in roughly a hundred operations. And a small error
early in the network can shrink by the end and appear only on some inputs.

So the reference records **every intermediate result**: embeddings, each
layer's normalizations, the combined query/key/value projection, the attention
scores, the attention output, the MLP, each layer's output, and the final
logits. Across five fixed prompts that's 615 arrays, saved as `.npy` files.
This collection is the **oracle**. A committed
[manifest](../oracle/manifest.json) records a checksum for each array along
with the conditions of the run: which weights, which prompt tokens, which data
type, which tolerance rule.

[`oracle/compare.py`](../oracle/compare.py) is the only program allowed to
decide pass or fail. It does three things:

- **It checks provenance before values.** A dump produced from different
  weights or different prompts is a failure, not a warning, because comparing
  numbers from two different runs proves nothing.
- **It walks the arrays in the order the model computes them** and reports the
  *first* one that's out of tolerance. If layer 3 is wrong, every later layer
  is wrong because of it, and the useful fact is "layer 3."
- **It refuses any quantized configuration.** A number-by-number comparison
  only makes sense between two full-precision implementations.

The comparison tool is tested as well. [`oracle/test_compare.py`](../oracle/test_compare.py)
feeds it deliberately broken dumps (values just over tolerance, NaNs, wrong
shapes, stale files, wrong weights, wrong prompts, quantized settings) and
confirms it rejects each one, and that it reports the earliest divergence, not
the largest. A checker that has never been shown to fail can't be trusted when
it passes.

## Phase 2: a correct C++ engine

The C++ engine is a line-by-line port of the reference, in C++17 with no
external libraries: no BLAS, no ML framework. It writes out the same 615
arrays in the same layout, so the same `compare.py` judges it.

This phase produced the project's first real finding, and it was about the
test rather than the engine.

### The first tolerance rule could not be met

The initial rule allowed each value an absolute error below 1e-4 *and* a
relative error below 0.1%. The engine failed 87 of 615 arrays, while choosing
the same top token as the reference at every position.

The failures depended on the size of the values, not on correctness. GPT-2
has a few "outlier" dimensions whose values reach about 2650. At that size,
the gap between one 32-bit float and the next representable one (one **ulp**)
is 2.4e-4. That's larger than the 1e-4 limit, so the rule was demanding closer
agreement than 32-bit floats can express. Passing would have required adding
up numbers in exactly the order PyTorch's matrix multiply does, which is an
implementation detail, not a property of correct code. The clearest example:
one array failed with an error of 9 ulps while another passed with 103 ulps,
only because the first held values about 800 times larger.

Before loosening anything, the project checked whether the engine was simply
sloppy. Each matrix multiplication was recomputed in 64-bit precision as a
ground truth, from identical inputs, and both implementations were measured
against it. The engine was as accurate as PyTorch or more accurate: on one
array it had 0.06 times PyTorch's error. The remaining gap between them was
two legitimate summation orders drifting apart across 12 layers.

Only then was the rule changed, to the form NumPy's `allclose` uses:

    |engine − reference| ≤ 1e-4 + 1e-3 · |reference|

The absolute term covers values near zero and the relative term covers large
ones. Several alternatives were measured against the same dump before
choosing: raising the absolute limit left 6 failures, and a rule based on
counting ulps failed 313 arrays or more. The adopted rule passes all 615, the
worst at 52% of its allowed error. That leaves margin for later changes
without being so loose it would miss a real bug. Three new self-tests confirm
it still catches real errors, including a masked attention value that should
be exactly zero but leaks 5e-4.

### Summation order

The first engine summed each dot product in one long chain of up to 3072
terms, and error grows with the length of the chain. Two changes cut the worst
error from 6.1e-3 to 2.5e-3 and also made the engine faster:

- Matrix multiplication sums in a balanced binary tree (**pairwise
  summation**), so error grows with log(n) instead of n.
- Every dot product and sum accumulates in exactly **8 separate running
  totals** (called lanes), combined at the end in a fixed order.

The number 8 was chosen for later phases. One AVX2 register holds 8 floats,
and two WebAssembly SIMD registers hold 8 floats. Fixing the order in the
engine's backend contract meant the future SIMD versions could produce
*exactly* the same bits as the plain C++ version, not just close results.
Accumulating in 64-bit doubles would also have passed, and was rejected for
that reason: no SIMD backend could reproduce it.

Phase 2 closed with all 615 arrays passing, the worst at 52% of budget, and 50
of 50 greedy tokens identical.

## Phase 3: fast, without changing a single bit

The correct engine took **8.8 seconds per token**. Four optimizations, in a
fixed order:

1. **KV cache.** When generating token 101, the attention keys and values for
   tokens 1 through 100 were already computed on earlier steps. Storing them
   means each step only processes the new token.
2. **Blocked matrix multiply.** Reordering the loops so that the data being
   used stays in the CPU's small, fast caches instead of being re-read from
   main memory.
3. **AVX2 SIMD.** Using CPU instructions that operate on 8 floats at once.
4. **Threads.** Splitting the work across CPU cores.

Measured on a 4-core laptop CPU (i5-1135G7), with a 128-token prompt and 128
generated tokens:

| Step | Prefill | Decode per token | Tokens/s |
|---|---:|---:|---:|
| Correct, no cache | 5784 ms | 8781.5 ms | 0.114 |
| + KV cache | 5825 ms | 48.5 ms | 10.68 |
| + blocked GEMM | 4561 ms | 47.5 ms | 12.09 |
| + AVX2 | 1573 ms | 29.7 ms | 23.94 |
| + 8 threads | **493 ms** | **19.8 ms** | **42.55** |

That's 373 times the throughput. The table isn't the main result, though.
**Every step produced output bit-for-bit identical to the step before it.**
That was verified by comparing engine dumps against engine dumps, not by
checking that each stayed within tolerance of PyTorch. The final engine sits
at exactly the same 52.0% of budget it reported at the end of Phase 2.

Two decisions made that possible:

- **The AVX2 backend refuses FMA** (fused multiply-add), even though the CPU
  supports it and it's faster. FMA rounds once where a separate multiply and
  add round twice, so it changes results. The browser's SIMD has no FMA at
  all, so using it here would also have split the native and browser engines
  apart.
- **SIMD sits behind an abstraction layer.** The kernels call a small backend
  interface ([`backend.h`](../engine/include/gpt2/backend/backend.h)) instead
  of CPU-specific instructions directly, so the AVX2 backend and the later
  WebAssembly backend are interchangeable implementations of one contract.

Threads aren't allowed to affect results either: all 615 arrays are
bit-identical at 1, 3 and 8 threads, and a unit test pins that.

Every step also left a benchmark result, a JSON file tagged with the commit it
measured, in [`bench/results/`](../bench/results/).

## Phase 4: making it smaller, and knowing what that cost

The weights take about 500 MB as 32-bit floats. **Quantization** stores them
with fewer bits. Here that's int8: each weight becomes an 8-bit integer, plus
one scale factor per output row of each matrix. The file shrinks to roughly a
quarter.

Quantization changes every number on purpose, so the layer-by-layer oracle no
longer applies, and `compare.py` refuses to run on it. A quantized model is
judged on its behavior instead, by [`eval/metrics.py`](../eval/metrics.py),
over a fixed 4,096-token sample of WikiText-2 (Wikipedia text), against the
full-precision engine:

- **Perplexity:** how well the model predicts real text. Lower is better.
- **Top-1 agreement:** how often it picks the same most likely token as the
  full-precision model.
- **KL divergence:** how much the entire distribution of scores moved, not
  just the top pick.
- **Decisive disagreement:** how often it overrules a token the full-precision
  model was *confident* about, meaning it preferred its choice by more than
  0.05 probability.

The last metric was added after measuring. The first int8 engine agreed on the
top token 97.48% of the time, below a guessed 98% threshold. Looking at the
positions where they disagreed, the full-precision model's own gap between its
first and second choice had a median of 0.00093, against 0.106 everywhere
else. The disagreements were almost entirely near-ties that the original model
was indifferent about. A raw agreement rate counts those the same as real
errors, so the top-1 threshold was lowered to a measured 97%, and the check
that matters became decisive disagreement. The shipped int8 model has zero
decisive disagreements across 256 sampled positions.

### Why one metric isn't enough

The first int8 version kept one table in full precision: `wte`, the token
embedding table. In GPT-2 this table does two jobs. It converts tokens into
vectors at the start, and (reused, or "tied") it converts the final vector
back into scores at the end.

Quantizing `wte` too would bring the file from 243 MB to 128 MB. Measured:

| | Perplexity | Top-1 agreement | Mean KL |
|---|---:|---:|---:|
| int8, `wte` full precision | ×0.998 | 97.48% | 0.0012 |
| int8, `wte` int8 | ×1.015 | 83.02% | 0.0414 |

Perplexity rose only 1.5%, which alone would have looked acceptable. But
**17% of the model's top picks changed**, and the distribution moved 35 times
further. Measuring with a single metric would have shipped a noticeably
different model.

### Finding the damage

A simulation ([`reference/wte_sim.py`](../reference/wte_sim.py)) quantized
each of the table's two jobs separately. The input side took int8 with no
measurable harm. All of the damage was on the output side.

The cause was the input to that final multiplication. The last normalization
step produces a few hidden dimensions with enormous values; one averages 201,
against a median of 0.35. Any rounding error in those columns of the table is
multiplied roughly two hundredfold, and because those columns contain the
largest values, they also set the scale for every row.

The fix was to keep just **8 of the 768 columns** in full precision and
quantize the rest. The result, `int8-wte-o8`, is 129.3 MB with perplexity
×0.998, 97.38% top-1 agreement, and zero decisive disagreements. The
simulation had predicted its perplexity to five decimal places. It also
decodes 1.15 times faster, because the final multiplication no longer reads a
full-precision table.

### int4, built and rejected

4-bit quantization was implemented with GPTQ, a method that compensates for
rounding error as it goes. It cut the damage of plain rounding about
four-fold, and still failed every criterion: perplexity ×1.042, 86.1% top-1
agreement, and 1.56% decisive disagreement, meaning the model overruled
answers the original was confident about. Four bits doesn't fit a model this
small on these criteria. The thresholds stayed where they were and int4 isn't
in the engine.

## Phase 5: the browser

The same C++ was compiled to **WebAssembly** (WASM) with Emscripten, adding a
third backend for WebAssembly's 128-bit SIMD instructions. Because of the
8-lane contract from Phase 2, that backend produces results bit-identical to
the plain WebAssembly build across all 615 arrays.

The browser build is *not* bit-identical to the native one, and the reason is
outside the engine. GELU calls `tanh` from the platform's math library. On
Linux that's glibc, which returns the correctly rounded result. Under
Emscripten it's musl, which is off by up to 2 ulps on 23% of inputs. The two
builds agree exactly up to the first GELU in layer 0 and differ slightly after
it. Both pass the oracle, native at 52.0% of budget and WebAssembly at 58.1%.

Single-threaded, the WebAssembly build runs at 74% of native AVX2 speed. In
Chrome, with the 129 MB int8 model, decoding takes **21 ms per token**.

### The tokenizer

The C++ engine takes token IDs and never handles text. The browser needs to
convert text to tokens and back, so GPT-2's tokenizer was written by hand in
191 lines of JavaScript ([`web/tokenizer.js`](../web/tokenizer.js)).

It has no oracle of its own, so it's checked three ways against Hugging Face's
token IDs: a committed set of test cases, every committed prompt, and a round
trip over 24,576 tokens of the evaluation text. The round trip works because
GPT-2's tokenizer is lossless: decoding the IDs gives back the exact original
text, so encoding that text again must reproduce the same IDs. Two edge cases
were settled by checking Hugging Face's output rather than guessing, including
that JavaScript's definition of whitespace differs from Python's in two
characters. The test was then shown to fail on six deliberately planted bugs
before it was trusted.

### The deploy

The live demo is published to GitHub Pages by a CI workflow
([`pages.yml`](../.github/workflows/pages.yml)) that, on a clean machine,
downloads and verifies the weights, builds the engine, runs the oracle
comparison, runs the quantized evaluation, and loads the page in headless
Chrome. The site is only published if all of that passes. The page itself
checks the SHA-256 of the weights it downloads against the exact file the
evaluation measured.

Even so, the first live deploy failed on the first page load. The CDN had cut
off one 32 MiB weight part mid-download. No test had ever used a network that
drops connections, so a loader that treated any interruption as fatal had
passed everything. The loader now retries each part from the start, and the
page test's server deliberately cuts off a part to prove the retry works, as
well as to prove that a part that keeps failing produces a clear error.

## What it adds up to

| Stage | Held to | Checked by |
|---|---|---|
| PyTorch reference | Hugging Face's GPT-2 | `reference/validate_hf.py` |
| C++ engine | the reference, at all 615 steps | `oracle/compare.py` |
| Each optimization | the version before it, bit for bit | engine dump vs engine dump |
| AVX2 and WASM SIMD | plain C++ on the same platform, bit for bit | fixed 8-lane order, no FMA |
| int8 weights | the full-precision engine, on four metrics | `eval/metrics.py` |
| JavaScript tokenizer | Hugging Face's token IDs | `web/test_tokenizer.mjs` |
| Browser demo | the command-line engine | headless Chrome in CI |

Each link in that chain is a program that can fail, and each has been shown
to fail when something is wrong.

## Lessons

**Test your tests.** Two of the most consequential findings were about the
checks, not the engine: a tolerance rule that no 32-bit implementation could
meet, and a quality metric that would have approved a model that changed 17%
of its answers. Both were only visible because the checks were questioned
before they were trusted.

**Change a threshold only on evidence.** Every threshold in the project was
revised at most once, and only after a measurement showed what the old one was
actually testing. That's also what made it credible to reject int4 instead of
lowering the bar until it passed.

**Decide the order of operations up front.** Fixing the summation order
before any optimization existed is why every backend on a platform, at every
thread count, produces identical numbers. Deciding it later would have meant choosing
between speed and reproducibility.

**Record the conditions of every measurement.** Checksums on weights, commit
hashes on benchmarks, and a manifest on every dump turned "the numbers
changed" from a mystery into a lookup. When one benchmark run came out 2.4
times slower, the cause turned out to be the laptop running on battery. The
benchmark tool now records the power state too.

## Known limits

- The engine computes scores for every position during the initial prompt
  pass, though only the last one is needed. Fixing it would change the output
  format the oracle checks.
- The browser build is single-threaded. Threads in WebAssembly need
  `SharedArrayBuffer`, which needs special HTTP headers on the host. Native
  reaches 42.6 tokens/s with 8 threads against 24.0 with one.
- Native and browser results differ in the last bits, because of `tanh`.

The full list is at the end of [findings.md](findings.md#known-limits).

## Reproducing it

Everything above can be rerun from a fresh clone. `./demo.sh` goes from
nothing to a running browser demo in one command, and
[`docs/maintaining.md`](maintaining.md) has the full sequence: oracle,
benchmarks, quantization and its evaluation, and the WebAssembly checks. In a
clean clone, the regenerated oracle matched all 615 committed checksums, and
the three weight files matched byte for byte.
