# reference/ -- hand-written PyTorch GPT-2

This is the source of truth for numerics. Everything downstream is compared
against what this produces.

- `model.py`     GPT-2 124M forward pass, written by hand. No HF model code.
- `weights.py`   load converted weights into `model.py` (Conv1D transpose, fused
                 QKV split, tied lm_head -- see CLAUDE.md gotchas).
- `dump.py`      run a prompt and write every intermediate activation to
                 `oracle/activations/` plus `oracle/manifest.json`.
- `validate_hf.py` one-time check of `model.py` against HF `GPT2LMHeadModel`.
                 This is the ONLY place HF model code may be called.

Readability wins over cleverness here: this file is documentation for the C++
port as much as it is code.

Phase: 0-1.
