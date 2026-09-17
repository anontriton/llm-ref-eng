# reference/ -- hand-written PyTorch GPT-2

This is the source of truth for numerics. Everything downstream is compared
against what this produces.

- `model.py`       [done] GPT-2 124M forward pass, written by hand. No HF model
                   code. Takes an optional `tap(name, tensor)` callback that
                   Phase 1's dumper uses to capture every intermediate.
- `weights.py`     [done] read the safetensors checkpoint straight into
                   `model.py`. Needs no transpose: `model.Linear` stores its
                   weight [in, out], which is the checkpoint's own layout.
- `validate_hf.py` [done] check `model.py` against HF `GPT2LMHeadModel`. This is
                   the ONLY place HF model code may be called.
- `dump.py`        [done] run the fixed prompt set and write every tapped
                   activation to `oracle/activations/<run>/` plus
                   `oracle/manifest.json`. Single-threaded for reproducibility.

Readability wins over cleverness here: this file is documentation for the C++
port as much as it is code.

Phase: 0-1.
