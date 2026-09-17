# scripts/ -- weight acquisition and conversion

- `download_weights.py`  fetch the GPT-2 124M checkpoint into `weights/`
                         (gitignored), verify checksum.
- `convert_weights.py`   convert the checkpoint into the flat binary format the
                         C++ engine loads: a small header + contiguous fp32
                         tensors in a fixed order, with names matching the
                         reference.

Conversion is where the CLAUDE.md gotchas bite: Conv1D weights are stored
transposed relative to `nn.Linear`, `attn.c_attn` is a fused 768 -> 2304 QKV
that splits into three 768 chunks, and `lm_head` is tied to `wte` so there is no
separate output matrix in the checkpoint.

Phase: 0.
