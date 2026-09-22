# GPT-2 from scratch: PyTorch, C++, WebAssembly

A GPT-2 124M inference engine built three times: a hand-written PyTorch
reference, a dependency-free C++ engine, and a WebAssembly build that runs in
the browser. The point is not that it runs. It is that every stage is **proven
against the one before it** -- the C++ engine is compared layer by layer
against the reference, 615 activations across five prompts, and no
optimization was allowed to land without passing that comparison again.

- **Correct:** every one of 615 intermediate tensors within tolerance of the
  PyTorch reference, the worst at 52% of its budget; 50 greedy tokens identical.
- **Fast:** 0.11 → 42.6 tokens/s natively through a KV cache, blocked GEMM,
  AVX2 and threads -- 373x, with every step bit-identical to the one before.
- **Small:** int8 weights at 129 MB, a quarter of fp32, judged on perplexity,
  top-1 agreement, KL and decisive disagreement against the fp32 engine.
- **In the browser:** the same C++ compiled to wasm SIMD, 21 ms per token in
  Chrome, with a hand-written tokenizer matching Hugging Face's ids exactly.

No framework, no BLAS, no model code from Hugging Face in anything that runs --
HF appears only as a check on the hand-written reference, never as the
reference.

## Try it

    ./demo.sh

One command, from a fresh clone. It checks the prerequisites first -- Python
3.10+, CMake, Ninja and [Emscripten](https://emscripten.org/docs/getting_started/downloads.html)
-- and says how to install any that are missing before downloading anything.
Then it sets up a Python environment, downloads and checksum-verifies GPT-2,
converts and quantizes it to the 129 MB browser file, compiles the C++ engine
to WebAssembly, and opens the page. The first run downloads about 1.5 GB and
takes a few minutes; after that it just serves the page. `./demo.sh --check`
only checks the prerequisites.

The page loads the weights into a Web Worker, tokenizes in JavaScript, and
streams tokens from the wasm engine -- all on your machine. At temperature 0 it
prints exactly what the validated engine produces.

Each step is an ordinary command in the repository; [Build and
verify](#build-and-verify) has them, along with the native engine, the oracle
and the benchmarks.

## A five-minute tour

For reading rather than running -- the files that carry the idea, in the order
the project built them:

| Look at | Why it matters |
|---|---|
| [`reference/model.py`](reference/model.py) | GPT-2 written out by hand in 226 lines of PyTorch -- the source of truth everything else is compared against |
| [`oracle/compare.py`](oracle/compare.py) | the judge: checks provenance, applies the tolerance rule, reports the first divergence in forward order |
| [`engine/src/model.cpp`](engine/src/model.cpp) | the C++ forward pass, a line-by-line port of the reference, with a tap for every tensor the oracle checks |
| [`engine/include/gpt2/backend/backend.h`](engine/include/gpt2/backend/backend.h) | the SIMD seam, and the 8-lane contract that makes every backend bit-identical |
| [`engine/src/backend_avx2.cpp`](engine/src/backend_avx2.cpp) | AVX2 that reproduces scalar results exactly -- and why it refuses FMA to do so |
| [`eval/metrics.py`](eval/metrics.py) | judging a quantized model with four metrics, because perplexity alone would have shipped a broken one |
| [`web/tokenizer.js`](web/tokenizer.js) | GPT-2's tokenizer in 191 lines of dependency-free JavaScript |
| [`docs/findings.md`](docs/findings.md) | what building it taught, with the measurements |

The commit history is written to be read too: each commit says what it proved
and how, and each optimization carries its benchmark.

## How correctness is proven

The reference in [`reference/model.py`](reference/model.py) is GPT-2 written
out by hand in PyTorch, checked once against Hugging Face's `GPT2LMHeadModel`:
max logit error 7.6e-5 over four prompts, 50 of 50 greedy tokens identical.

From then on the reference is the **oracle**. It dumps every intermediate
activation -- embeddings, each block's layer norms, QKV, attention scores and
outputs, MLP, residuals, final logits -- for five fixed prompts, 615 tensors,
each with a sha256 in a committed [manifest](oracle/manifest.json). The C++
engine dumps the same tensors in the same layout, and
[`oracle/compare.py`](oracle/compare.py) is the only thing that decides whether
they match. It walks them in forward order and reports the *first* divergence,
because once block 3 is wrong everything after it is an echo.

| Stage | Held to | By |
|---|---|---|
| PyTorch reference | Hugging Face `GPT2LMHeadModel` | `reference/validate_hf.py` |
| C++ engine, every backend | the reference, layer by layer | `oracle/compare.py`, `oracle/check_greedy.py` |
| Each optimization | the engine before it, bit for bit | engine dumps diffed against engine dumps |
| AVX2 and wasm SIMD | scalar, bit for bit | a fixed 8-lane accumulation order, no FMA |
| int8 weights | the fp32 engine, on a pinned WikiText-2 slice | `eval/metrics.py` |
| JavaScript tokenizer | Hugging Face's ids | `web/test_tokenizer.mjs` |
| Browser build | the command-line engine | `web/test_web.mjs`, and the page in headless Chrome |

The comparison is itself tested -- [`oracle/test_compare.py`](oracle/test_compare.py)
proves it rejects above-tolerance error, NaN, shape drift, stale dumps, wrong
weights, wrong prompts and quantized policies -- and the tolerance rule
(`|engine − reference| ≤ 1e-4 + 1e-3·|reference|`) was set from measurement,
not taste. The first rule could not be met by *any* fp32 implementation; the
story is [finding 1](docs/findings.md#1-the-original-tolerance-rule-could-not-be-met-in-fp32).

## Results

### Performance

Prompt of 128 tokens, 128 generated, on a 4-core i5-1135G7. Each step was
oracle-validated and left a commit-tagged benchmark in
[`bench/results/`](bench/results/).

| Step | Prefill | Decode / token | Tokens / s |
|---|---:|---:|---:|
| Scalar, no cache | 5784 ms | 8781.5 ms | 0.114 |
| + KV cache | 5825 ms | 48.5 ms | 10.68 |
| + blocked GEMM | 4561 ms | 47.5 ms | 12.09 |
| + AVX2 | 1573 ms | 29.7 ms | 23.94 |
| + 8 threads | **493 ms** | **19.8 ms** | **42.55** |

The same engine as WebAssembly, single-threaded, beside native single-threaded:

| Build | Tokens / s | Decode / token |
|---|---:|---:|
| native AVX2 | 23.96 | 28.9 ms |
| wasm SIMD (Node) | 17.70 | 36.9 ms |
| wasm SIMD, int8-wte-o8 (Node) | 23.11 | 23.5 ms |
| wasm SIMD, int8-wte-o8 (Chrome) | -- | 21 ms |

### Quantization

Weight-only int8, symmetric, one scale per output channel, judged against the
fp32 engine on a pinned 4,088-position WikiText-2 slice:

| Weights | Size | Perplexity | Top-1 agreement | Mean KL | Decisive disagreement |
|---|---:|---:|---:|---:|---:|
| fp32 | 497.8 MB | 36.3366 | -- | -- | -- |
| int8, embedding fp32 | 243.3 MB | ×0.99844 | 97.480% | 0.00117 | 0 / 256 |
| **int8-wte-o8** | **129.3 MB** | ×0.99805 | 97.383% | 0.00135 | 0 / 256 |
| int8, everything | 127.7 MB | ×1.01536 | 83.02% | 0.04139 | -- |
| int4 (GPTQ, group 64) | -- | ×1.04198 | 86.106% | 0.04138 | 1.562% |

The last two fail. Quantizing the whole embedding table changes 17% of the
model's choices while perplexity barely moves -- the reason there is more than
one metric. The fix, `int8-wte-o8`, keeps just 8 of its 768 columns in fp32
([finding 8](docs/findings.md#8-int8s-damage-in-wte-was-8-columns-of-lm_head)).
int4 was built, measured, and rejected: four bits does not fit a 124M model on
these criteria, and no threshold was moved to pretend otherwise.

## Build and verify

Needs Python 3 with `torch`, `numpy` and `transformers` (for the reference and
the checks only), CMake, a C++17 compiler, and Emscripten plus Node for the
wasm build. From the repository root:

    .venv/bin/python scripts/download_weights.py          # checksum-verified
    .venv/bin/python reference/validate_hf.py             # reference vs HF
    .venv/bin/python reference/dump.py                    # build the oracle
    .venv/bin/python scripts/convert_weights.py --verify
    .venv/bin/python scripts/export_runs.py

    cmake -S engine -B engine/build-avx2 -G Ninja -DGPT2_BACKEND=avx2
    cmake --build engine/build-avx2
    ctest --test-dir engine/build-avx2
    engine/build-avx2/tools/gpt2_dump --threads 8
    .venv/bin/python oracle/compare.py engine/dumps/manifest.json
    .venv/bin/python oracle/check_greedy.py --kv-cache --engine engine/build-avx2/tools/gpt2_generate

`GPT2_BACKEND` is `scalar` (the default), `avx2`, or `wasm_simd128` through
`web/build.sh`. The full sequence -- benchmarks, quantization and its eval, the
wasm checks -- and the procedure for changing the engine without breaking what
it proves are in [docs/maintaining.md](docs/maintaining.md).

## Repository

| Directory | Contents |
|---|---|
| [`reference/`](reference/) | the hand-written PyTorch GPT-2, the oracle dumper, quantization simulations |
| [`oracle/`](oracle/) | the committed manifest, `compare.py`, and its self-tests |
| [`engine/`](engine/) | the C++17 engine: `include/gpt2/`, `src/`, `tests/`, `tools/` |
| [`bench/`](bench/) | the benchmark harness and every result, tagged by commit |
| [`eval/`](eval/) | the quantized-model judge, the pinned corpus, results |
| [`web/`](web/) | the wasm build, the tokenizer, the demo page |
| [`scripts/`](scripts/) | weight download, conversion, quantization, pinned-input export |
| [`docs/`](docs/) | findings, and how to maintain the repository |

Each directory's README says what lives there and how to run it. The
project's rules -- tolerances, phase order, what may and may not be optimized
before what -- are in [CLAUDE.md](CLAUDE.md).

## What building it taught

The long versions are in [docs/findings.md](docs/findings.md).

1. **The first tolerance rule was unsatisfiable.** GPT-2's residual stream has
   dimensions near 2650, where one fp32 ulp is 2.4e-4 -- larger than the 1e-4
   absolute limit. The rule was demanding PyTorch's summation order, not
   correctness.
2. **The engine was not the less accurate one.** Against a float64 ground
   truth from identical inputs, its matmuls matched or beat PyTorch's.
3. **Summation order is the whole game.** Every reduction runs across exactly
   8 lanes, fixed in the backend contract, which is why AVX2 and wasm SIMD
   reproduce scalar bit for bit and every optimization kept the same numbers.
4. **Perplexity alone would have shipped a broken model.** Quantizing the
   embedding table moved perplexity 1.5% and changed 17% of the model's picks.
5. **All of that damage was 8 columns.** The final layer norm has a few
   outputs 200x the median; keeping those columns of the table in fp32 halves
   the download at no measurable cost.
6. **wasm cannot match native bit for bit** while `tanh` comes from the
   platform's libm: musl's is off by up to 2 ulp where glibc's is exact. Both
   pass the oracle; SIMD is held to its own platform's scalar build.
7. **A benchmark on battery measures the clock.** One matrix came out 2.4x
   slow with nothing in the results to say why; the harness now records the
   power state.

## Status

Complete: all seven phases, in order, each closed on evidence recorded in
[CLAUDE.md](CLAUDE.md) -- foundations, the oracle, a correct C++ engine,
performance, quantization, WebAssembly, and these docs. Known limits are at
the end of [docs/findings.md](docs/findings.md#known-limits).

## License

[MIT](LICENSE). The GPT-2 weights are not in this repository;
`scripts/download_weights.py` fetches them from Hugging Face, where OpenAI
released them under the MIT license as well.
