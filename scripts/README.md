# scripts/ -- weights, conversion, and pinned inputs

Weights:

- `download_weights.py`   fetch the GPT-2 124M checkpoint and tokenizer files
                          into `weights/` (gitignored) and verify them against
                          `weights.lock.json`. `--pin` created that lockfile
                          once; every run since is a verification.
- `weights.lock.json`     the Hugging Face commit the files come from -- never
                          `main`, so upstream changes cannot reach this repo --
                          and a pinned sha256 and byte count for all 5 files.
- `convert_weights.py`    the checkpoint -> `weights/gpt2-124m.bin`, the flat
                          file the C++ engine loads (format documented in the
                          script). Carries the source checkpoint's sha256 so the
                          engine's dump manifest can prove which weights it ran
                          on. `--verify` re-reads the output bit-exact.
- `quantize_weights.py`   fp32 file -> int8, per output channel:
                          `gpt2-124m-int8.bin` (243 MB, embedding fp32), or with
                          `--wte-outliers 8`, `gpt2-124m-int8-wte-o8.bin`
                          (129 MB), the file the browser demo loads.

Pinned inputs -- the engine does not tokenize, so ids travel as data. Each of
these writes a committed file; re-running them is a change to what every result
was measured on, not a setup step:

- `export_runs.py`             the oracle's prompts and ids ->
                               `engine/runs.tsv` (derived, gitignored).
- `export_bench_prompt.py`     the benchmark prompt -> `bench/prompts.tsv`.
- `export_eval_corpus.py`      the WikiText-2 slices -> `eval/corpus.tsv` and
                               `eval/calib.tsv`. Needs the network, and
                               re-fetching can return different rows: do not
                               run it to "set up".
- `export_tokenizer_cases.py`  HF's ids for the JavaScript tokenizer's edge
                               cases -> `web/tokenizer_cases.json`.

Conversion is where CLAUDE.md's gotchas bite, all confirmed against the real
checkpoint: `c_attn.weight` is `[768, 2304]` (a fused QKV that splits into
three 768 chunks), `mlp.c_fc.weight` is `[768, 3072]` -- [in, out], the
transpose of `nn.Linear` -- and there is no `lm_head.weight` key at all,
because it is tied to `wte`. The checkpoint also carries `h.{i}.attn.bias`,
HF's precomputed causal mask rather than a learned weight; it is dropped.
