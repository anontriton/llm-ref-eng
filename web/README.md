# web/ -- the engine in WebAssembly, and the browser demo

The same `engine/` sources compiled with Emscripten. There is no web fork of
the engine: what the browser needed went into the engine behind the backend
abstraction (`src/backend_wasm_simd128.cpp`) or into a thin driver
(`engine/tools/gpt2_web.cpp`), and the wasm build is held to the same PyTorch
oracle as native, run under Node.

    web/build.sh wasm_simd128                          # -> web/build/engine-wasm_simd128/
    .venv/bin/python scripts/quantize_weights.py --wte-outliers 8
    python3 web/pack.py                                # -> web/build/site/
    web/serve.sh                                       # -> http://localhost:8000/

`./demo.sh`, at the repository root, runs all of that from a fresh clone --
prerequisites checked first, finished steps skipped. The same site is live at
**https://anontriton.github.io/llm-ref-eng/**, published by CI (below).

`build.sh [scalar|wasm_simd128]` configures `engine/` with `emcmake` and builds
every tool and test as a `.js` + `.wasm` pair; it finds `emcc` even where Arch
leaves it off PATH. `pack.py` assembles the static site and `serve.sh` serves
it on localhost -- the same directory GitHub Pages publishes.

## What is here

| File | Role |
|---|---|
| `build.sh` | build a backend |
| `pack.py`, `weights.lock.json` | assemble the static site; the one weight file it may contain |
| `serve.sh` | serve the site on localhost |
| `index.html` | the page: prompt, sampling controls, streamed output, timings |
| `worker.js` | downloads and verifies the weights, tokenizes, runs prefill and decode, streams text |
| `engine.js` | `engine/tools/gpt2_web.cpp`'s C API wrapped for JavaScript |
| `sampling.js` | greedy (argmax, first index on ties), temperature + top-k, a seeded RNG |
| `tokenizer.js` | GPT-2's byte-level BPE, hand-written, no dependencies |
| `test_web.mjs` | holds the browser module to the command-line engine |
| `test_page.mjs` | the built site in headless Chrome -- the last gate before a deploy |
| `test_tokenizer.mjs`, `tokenizer_cases.json` | holds the tokenizer to Hugging Face's ids |

`engine.js`, `sampling.js` and `tokenizer.js` run unchanged in the page and in
Node, so the tests exercise the code the page runs, not a copy of it.

## Correctness

**The wasm engine passes the oracle.** Under Node, with the tools reading the
real filesystem (`-sNODERAWFS`), the Phase 2 proof runs unchanged:

    node web/build/engine-wasm_simd128/tools/gpt2_dump.js --out web/build/dumps
    .venv/bin/python oracle/compare.py web/build/dumps/manifest.json
    .venv/bin/python oracle/check_greedy.py --kv-cache \
        --engine web/build/engine-wasm_simd128/tools/gpt2_generate.js
    ctest --test-dir web/build/engine-wasm_simd128

615/615 tensors, the worst at 58.1% of budget; 50/50 greedy tokens, with and
without the KV cache; 6/6 unit tests, with `test_threading` disabled (below).

**58.1%, not native's 52.0%, and that is libm, not the engine.** Native and
wasm agree bit for bit through embeddings, layer norms, QKV, attention and
softmax, and part at the first `std::tanh`, in GELU: Emscripten's musl is off
by up to 2 ulp on 23% of inputs where glibc's `tanhf` is correctly rounded.
The measurement is [finding 7](../docs/findings.md#7-wasm-cannot-be-bit-identical-to-native-while-the-kernels-call-libm).

**So wasm_simd128 is held to wasm scalar, bit for bit** -- the standard AVX2
met against native scalar. The eight accumulator lanes fixed in Phase 2 are
two `v128` here, the reduction tree is the scalar one step for step, there is
no fused multiply-add (wasm_simd128 has none; the build also passes
`-ffp-contract=off`), and `max()` stays scalar, as in every backend, because
`f32x4.max` treats NaN and signed zero differently from `>`. Only that one file
is built with `-msimd128`, and `wasm-dis` confirms the module has no relaxed-simd
op, no fused multiply-add and no `f32x4.max`.

| wasm_simd128 against wasm scalar | Result |
|---|---|
| oracle dumps, sha256 | 615/615 identical; 615/615 through the KV cache |
| `oracle/compare.py`, scalar as reference | 0.0% of budget |
| int8 path (`gpt2_eval`: nll, top-1, 64 rows of logits) | byte-identical |

**The browser module matches the command-line engine.** `node web/test_web.mjs`
loads `gpt2_web.mjs` through `engine.js`, streams the weights in 1 MiB chunks
the way a fetch body arrives, and requires 50 greedy ids on three oracle
prompts to equal `gpt2_generate.js`'s -- on the fp32 file, which chains back to
`check_greedy.py`, and on `int8-wte-o8`. It also checks that seeded sampling
repeats, and that one token past the 1024-token context is an error with a
reason after which the engine still works.

**The page shows exactly that output.** `node web/test_page.mjs` serves the
built site and drives it in headless Chrome over the DevTools protocol, with no
dependencies: the page loads and shows the verified sha256; greedy on "The
capital of France is" puts on screen exactly the text `gpt2_generate`'s ids
decode to; Stop ends a sampling run; a reload loads the weights from the
browser's cache; one flipped byte in one weight part is refused, and Generate
stays disabled; a part cut off halfway is retried, and one that never arrives
fails with a reason; a phone asks before downloading and lays out in 390 px;
and nothing reaches the console. Throttled to 30 MB/s in an earlier run, the
progress bar climbed to 129.3 MB over 4.3 s.

## The tokenizer

The engine takes ids, never text, so the browser tokenizes in JavaScript --
the one component here with no oracle behind it. `node web/test_tokenizer.mjs`
holds it to four things, all committed:

| Check | Against |
|---|---|
| fixture | 494 cases, Hugging Face's ids: hand-picked edge cases, a seeded fuzz set, decodes that cut a character in half |
| prompts | the oracle's 5 prompts and the benchmark prompt, text and ids both pinned |
| corpus | decode then re-encode the 24,576 committed WikiText-2 ids -- exact, in 60 ms |
| stream | the streaming decoder, one id at a time, equals a whole decode on every sequence |

The corpus check is the strongest, and needs no committed text: byte-level BPE
is lossless, so `decode(ids)` *is* the text the ids were cut from, and encoding
it again must reproduce Hugging Face's tokenization token for token. Hugging
Face writes the fixture (`scripts/export_tokenizer_cases.py`) and does nothing
else, the role `GPT2LMHeadModel` has for the model. The test was shown to fail
on six planted bugs before it was trusted; see
[finding 10](../docs/findings.md#10-the-tokenizers-strongest-test-needed-no-committed-text).

## The weights: int8-wte-o8, 129 MB

The download is the browser's real cost, and Phase 4's int8 file kept the
154 MB embedding table in fp32. `int8-wte-o8` quantizes it too, except the 8
columns where the final layer norm's output is largest, which are zeroed in the
int8 table and stored whole beside it: lm_head adds an 8-wide fp32 dot per
logit, and the embedding lookup overwrites 8 values. Why 8 columns, and why
that works, is [finding 8](../docs/findings.md#8-int8s-damage-in-wte-was-8-columns-of-lm_head).

| Against the fp32 engine | int8 (243.3 MB) | int8-wte-o8 (129.3 MB) | Simulated |
|---|---:|---:|---:|
| perplexity ratio | ×0.99844 | ×0.99805 | ×0.99805 |
| top-1 agreement | 97.480% | 97.383% | 97.407% |
| mean KL | 0.001168 | 0.001345 | 0.001327 |
| decisive disagreement | 0 / 256 | 0 / 256 | -- |

Native scalar and AVX2 are byte-identical on it, as are wasm scalar and
wasm_simd128; native and wasm differ by at most 9.2e-5 in a logit, the libm
difference, with top-1 identical at every position. The loader checks every
property the kernel relies on -- including that the int8 table really is zero
in those columns -- and names the policy from what it finds.

## Performance

Single-threaded, prompt 128 tokens, 128 generated, KV cache, on AC; results in
`bench/results/`. `bench/run.py --tool .../gpt2_bench.js` runs a wasm build
under Node and records it as `wasm32`.

| fp32 | Tokens / s | Prefill | Decode / token |
|---|---:|---:|---:|
| native scalar | 12.26 | 4424 ms | 47.40 ms |
| native AVX2 | 23.96 | 1676 ms | 28.87 ms |
| wasm scalar | 7.66 | 7726 ms | 70.75 ms |
| wasm_simd128 | 17.70 | 2550 ms | 36.86 ms |

| Quantized | int8: tokens / s | decode | int8-wte-o8: tokens / s | decode |
|---|---:|---:|---:|---:|
| native AVX2 | 32.30 | 18.75 ms | 36.36 | 15.26 ms |
| wasm_simd128 | 21.77 | 27.02 ms | 23.11 | 23.45 ms |

- **wasm_simd128 reaches 74% of native AVX2** in fp32. Prefill, which is
  compute-bound, pays for half the vector width (1.52x behind); decode, bound
  by weight traffic, is closer (1.28x).
- **SIMD is 2.3x over wasm scalar**, and int8 helps wasm as it does native,
  all of it in decode. `int8-wte-o8` is faster again: lm_head stops streaming
  fp32.
- **In Chrome** the page decodes at 21 ms/token, about 47 tokens/s, a little
  faster than the same module under Node; prefill takes 94 ms for 5 tokens.
  The weights load in 250 ms from a local server.

## Hosting

**https://anontriton.github.io/llm-ref-eng/** is published by
`.github/workflows/pages.yml`, and only after the whole gate passes on that
commit, on a clean runner, from the checkpoint up: the wasm unit tests, the
oracle comparison and 50 greedy steps on the shipped backend, the tokenizer and
web-module tests, the shipped weights' eval against fp32, `pack.py`'s pin
check, and `test_page.mjs`. Pull requests run the same gate and publish
nothing.

- **The weights are cut into 32 MiB parts.** GitHub refuses files over 100 MB;
  the file is 129 MB. `model/weights.json` lists the parts in order, with the
  whole file's size and sha256.
- **The shipped file is pinned.** `pack.py` refuses to build the site from any
  file but the one in `weights.lock.json` -- the file
  `eval/results/20260922T231318Z-1bb173c-int8-wte-o8.json` measured, rebuilt in
  CI byte for byte -- and the page hashes what it downloaded before the engine
  sees a byte, and says so on screen.
- **Each part is downloaded whole and retried if its connection drops** -- up
  to four attempts, restarting that part from zero -- and only then handed to
  the engine or stored. The first live deploy lost a part to a cold CDN
  mid-download; `test_page.mjs` now cuts a part off halfway, once and forever,
  and requires a retry and a clear failure respectively.
- **Repeat visits load from the browser's cache**, keyed by that sha256, so a
  new file can never be served from an old one's entries. A part is cached
  only once it is complete.
- **Phones ask first.** 129 MB and a few hundred MB of memory is a lot to
  take unasked.

## Decisions

- **The weights stream straight into wasm memory.** `gpt2_blob_alloc(n)`
  reserves the buffer `Weights::from_blob()` will own, and the worker copies
  each fetched chunk into it as it arrives. 129 MB is never held twice, and the
  progress bar is real. `Weights::load(path)` is `from_blob` after reading a
  file, so there is one parser.
- **Everything slow runs in a worker**, and yields between tokens through a
  `MessageChannel` -- not `setTimeout`, which clamps to 4 ms once nested -- so
  Stop takes effect on the next token.
- **Greedy is the validated mode.** At temperature 0 the page's output is the
  engine's, argmax with the same tie rule as `gpt2_generate`. Temperature and
  top-k (40 by default, as in GPT-2's own samples) are a demo feature, seeded so
  a run repeats.
- **No threads.** The engine's pool is `std::thread`, which Emscripten supports
  only with pthreads, which need `SharedArrayBuffer`, which needs COOP/COEP
  headers from whatever serves the page. The pool never exists at one thread,
  so the build degrades cleanly; `test_threading` is disabled rather than
  dropped, so ctest still lists it. Native reaches 42.8 tokens/s at 8 threads,
  so this is the largest remaining gap.
- **Node-only link flags are per tool.** `NODERAWFS` and `EXIT_RUNTIME` apply
  to the command-line tools and tests; `gpt2_web` has neither, so the same
  module runs in a browser worker and in Node.
