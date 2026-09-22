# reference/ -- hand-written PyTorch GPT-2

The source of truth for numerics. Everything downstream is compared against
what this produces.

- `model.py`       GPT-2 124M forward pass, written by hand. No HF model code.
                   Takes an optional `tap(name, tensor)` callback that the
                   dumper uses to capture every intermediate.
- `weights.py`     read the safetensors checkpoint straight into `model.py`.
                   Needs no transpose: `model.Linear` stores its weight
                   [in, out], which is the checkpoint's own layout.
- `validate_hf.py` check `model.py` against HF `GPT2LMHeadModel`. The ONLY
                   place HF model code is called.
- `dump.py`        run the fixed prompt set and write every tapped activation
                   to `oracle/activations/<run>/` plus `oracle/manifest.json`.
                   Single-threaded for reproducibility.

Quantization is tried here, in PyTorch, before it is written in C++ -- deciding
a scheme is cheap in torch and expensive in a kernel:

- `quant_sim.py`   perplexity of a quantization scheme on the eval slice. Chose
                   int8 per-channel, and predicted the engine's perplexity to
                   five decimal places before the kernel existed.
- `gptq.py`        int4 with GPTQ's error compensation. Implemented, measured,
                   and rejected; it lives here and not in the engine.
- `wte_sim.py`     which use of the embedding table takes int8's damage, and
                   what representation of it could ship to a browser. Produced
                   `int8-wte-o8`, see `docs/findings.md`, 8.

    .venv/bin/python reference/validate_hf.py
    .venv/bin/python reference/dump.py

Readability wins over cleverness here: `model.py` is documentation for the C++
port as much as it is code.
