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
Five fixed prompts, chosen to stress different things, and one run that fills
the context window:

| run | T | why |
|---|---|---|
| `capital` | 5 | short and boring; the everyday case |
| `unicorns` | 17 | mid-length prose |
| `code` | 7 | non-prose token distribution |
| `newline` | 1 | degenerate: a 1x1 attention matrix |
| `long` | 90 | position embeddings well past the start of the table |
| `context` | 1024 | the whole window: every row of `wpe`, attention over a long past |

123 tensors per run, 738 total, ~2.1 GB -- `context` is nearly all of it, each
attention tensor 12 x 1024 x 1024. Its ids are the first 1024 of the pinned
eval corpus (`eval/corpus.tsv`), checked against `eval/corpus.json`.

`context` is different in two ways, both recorded in its manifest entry. Its
reference is the model run in **float64** (`"reference_dtype": "float64"`),
rounded to float32 for storage, with each tensor's `fp32_budget` recording how
much of the rule PyTorch's own fp32 forward uses against it. And
`"whole_sequence": "report"`: a whole-sequence dump of it is compared and its
misses reported, not failed, while a KV-cached dump -- row 1023, which attends
over every earlier position's keys and values -- is gated as usual. At 1024
positions no fp32 implementation meets the rule at every element, PyTorch's
included; [finding 13](../docs/findings.md#13-at-1024-positions-no-fp32-implementation-meets-the-fp32-rule)
has the measurements and the criterion that was tried first and rejected. A
shape, NaN or provenance failure fails in either mode, and the marking is read
from the reference only: a candidate cannot declare itself report-only.

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
fp32 phases: per element, `|candidate - reference| <= 1e-4 + 1e-3 * |reference|`
(the numpy.allclose form), plus top-1 token identical for 50 consecutive greedy
steps (`check_greedy.py`). The two numbers combine rather than acting as
independent limits: at GPT-2's outlier magnitudes (~2650) one fp32 ulp already
exceeds 1e-4. The manifest's `rel_floor` now only gates the `max_rel` figure
`compare.py` reports as a diagnostic; it does not decide pass/fail.

A tensor may hold one position of a sequence rather than all of them: every
tensor of a KV-cached dump (the run's `kv_row`), or the logits of a forward run
under `Logits::Last` (the record's own `row`, from `gpt2_dump --last-logits`).
`compare.py` slices the reference at that position.

`compare.py` reports each passing run's worst tensor as a percentage of its
budget. That headroom is the number to watch across Phase 3: an optimization
that moves it sharply has changed the arithmetic, even if it still passes.

Quantized phases: layer-wise matching is void, and `compare.py` refuses to run
if either manifest declares a non-fp32 policy. Use perplexity on a fixed
WikiText-2 slice, top-1 agreement rate vs fp32, and KL divergence of logits.

## Usage

    python reference/dump.py                          # build the oracle
    python oracle/compare.py engine/dumps/manifest.json
    python oracle/compare.py <candidate> --run capital -v
    python oracle/compare.py <candidate> --allow-subset   # during bring-up
    python oracle/test_compare.py                     # test the tester
    python oracle/identical.py <dump> <dump> --verify  # bit-identity, engine vs engine

Exit codes: 0 pass, 1 numeric divergence, 2 provenance or usage error.

`identical.py` asks the stricter question `compare.py` does not: did two engine
runs produce exactly the same bits? It is how "this optimization, this backend,
this thread count changes nothing" is checked rather than argued.

Built in Phase 1, and re-run on every optimization commit since -- native and,
from Phase 5, wasm.
