# oracle/ -- reference activations and the comparison harness

The oracle is the contract between the PyTorch reference and the C++ engine.

- `activations/<run>/*.npy`  the dumps (gitignored -- regenerate with
                             `reference/dump.py`)
- `manifest.json`            what the dumps mean; the single index over all runs
- `compare.py`               the sole authority on pass/fail
- `test_compare.py`          proves `compare.py` rejects what it should

## Rules
- The engine writes `.npy`; only Python reads it.
- A dump whose manifest disagrees with the run being compared is a failure, not
  a warning. Mismatched weights, config, prompt tokens, or dtype exit 2 before
  a single number is compared.
- Comparison walks tensors in forward order and reports the FIRST divergence.
  A mismatch at `block.3.attn.scores` makes every later mismatch an echo.
- Every `.npy` is checksummed against the manifest, so a stale dump left over
  from a previous build cannot quietly pass.

## Runs
Five fixed prompts, chosen to stress different things:

| run | T | why |
|---|---|---|
| `capital` | 5 | short and boring; the everyday case |
| `unicorns` | 17 | mid-length prose |
| `code` | 7 | non-prose token distribution |
| `newline` | 1 | degenerate: a 1x1 attention matrix |
| `long` | 90 | position embeddings well past the start of the table |

123 tensors per run, 615 total, ~105 MB.

## Tensor naming
Stable and hierarchical, recorded in forward order (`order` in the manifest):

    embed.out
    block.{i}.ln_1.out
    block.{i}.attn.qkv          fused 768 -> 2304, before the split
    block.{i}.attn.scores       q@k.T / sqrt(64), BEFORE the causal mask
    block.{i}.attn.probs        after mask + softmax
    block.{i}.attn.out          after c_proj, before the residual add
    block.{i}.ln_2.out
    block.{i}.mlp.fc.out        after c_fc, before the activation
    block.{i}.mlp.act.out       after gelu_new
    block.{i}.mlp.out           after c_proj, before the residual add
    block.{i}.out               the block's residual stream output
    ln_f.out
    logits

`attn.scores` is tapped *before* masking so the oracle holds only finite
numbers. Its manifest record carries `"region": "causal_lower_triangle"`, and
`compare.py` only trusts that region -- the part any implementation must
compute. How an engine spells "masked" (`-inf`, `-1e9`, or simply never
computing the entry) is its own business, and shows up in `attn.probs`, where
it actually matters.

Splitting the MLP into `fc.out` / `act.out` / `out` is deliberate: a wrong GELU
is the most likely silent porting bug, and this isolates it to one tensor.

## Manifest
One file covering every run. Records the model config and the weight file's
sha256, the prompt and its exact `input_ids`, the dtype, the tolerance policy
in force, the producing environment (torch/numpy/python versions, thread
count), and for each tensor its order, shape, sha256, and min/max/absmax.

Dumps are single-threaded on purpose: multithreaded reductions can reassociate
and shift the last couple of bits, and an oracle that moves is not an oracle.

## Tolerances
fp32 phases: max abs error < 1e-4; relative error < 1e-3 above a magnitude
floor of 1e-2; top-1 token identical for 50 consecutive greedy steps.

Quantized phases: layer-wise matching is void, and `compare.py` refuses to run
if either manifest declares a non-fp32 policy. Use perplexity on a fixed
WikiText-2 slice, top-1 agreement rate vs fp32, and KL divergence of logits.

## Usage

    python reference/dump.py                          # build the oracle
    python oracle/compare.py engine/dumps/manifest.json
    python oracle/compare.py <candidate> --run capital -v
    python oracle/compare.py <candidate> --allow-subset   # during bring-up
    python oracle/test_compare.py                     # test the tester

Exit codes: 0 pass, 1 numeric divergence, 2 provenance or usage error.

Phase: 1 onward, re-run on EVERY optimization commit.
