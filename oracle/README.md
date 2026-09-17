# oracle/ -- reference activations and the comparison harness

The oracle is the contract between the PyTorch reference and the C++ engine.

- `activations/`    `.npy` dumps (gitignored -- regenerate with
                    `reference/dump.py`)
- `manifest.json`   what a dump means: model id, weight checksum, prompt token
                    ids, dtype, tensor names + shapes, tolerance policy
- `compare.py`      the sole authority on pass/fail

## Rules
- The engine writes `.npy`; only Python reads it.
- A dump whose manifest disagrees with the run being compared is a failure, not
  a warning.
- Comparison walks tensors in forward order and reports the FIRST divergence.
  A mismatch at `block.3.attn.scores` makes every later mismatch noise.

## Tensor naming
Stable and hierarchical, forward order:

    embed.out
    block.{i}.ln_1.out
    block.{i}.attn.qkv
    block.{i}.attn.scores
    block.{i}.attn.out
    block.{i}.ln_2.out
    block.{i}.mlp.fc.out
    block.{i}.mlp.out
    block.{i}.out
    ln_f.out
    logits

## Tolerances
fp32 phases: max abs error < 1e-4; relative error < 1e-3 above a magnitude
floor; top-1 token identical for 50 consecutive greedy steps.

Quantized phases: layer-wise matching is void. Use perplexity on a fixed
WikiText-2 slice, top-1 agreement rate vs fp32, and KL divergence of logits.

Phase: 1 onward, re-run on EVERY optimization commit.
