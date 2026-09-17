# scripts/ -- weight acquisition and conversion

- `download_weights.py`  [done] fetch the GPT-2 124M checkpoint into `weights/`
                         (gitignored) and verify it against `weights.lock.json`.
                         Run `--pin` once to create that lockfile; every run
                         after that is a verification.
- `weights.lock.json`    [done] pinned sha256 + byte count for all 5 files.
- `convert_weights.py`   [Phase 2] convert the checkpoint into the flat binary
                         format the C++ engine loads. Deferred deliberately:
                         the on-disk layout should be chosen once the engine's
                         loader and kernels exist, not guessed at now.

Conversion is where the CLAUDE.md gotchas bite, all three confirmed against the
real checkpoint: `c_attn.weight` is `[768, 2304]` (a fused QKV that splits into
three 768 chunks), `mlp.c_fc.weight` is `[768, 3072]` -- i.e. [in, out], the
transpose of `nn.Linear` -- and there is no `lm_head.weight` key at all, because
it is tied to `wte`. The checkpoint also carries a `h.{i}.attn.bias` buffer,
which is HF's precomputed causal mask, not a learned weight; we drop it.
