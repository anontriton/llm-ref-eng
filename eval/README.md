# eval/ -- the quantized-phase harness

Built BEFORE anything was quantized, so Phase 4 had something to be judged by.

`oracle/compare.py` cannot do this job and says so: it refuses a non-fp32
policy outright, because a layer-wise tolerance stops meaning anything once the
arithmetic changes representation. An int8 engine is not a worse fp32 engine,
it is a different function that has to behave like the same model. So this is a
second authority, for a second kind of question.

## Running

    .venv/bin/python eval/run.py --out eval/runs/fp32
    .venv/bin/python eval/run.py --out eval/runs/int8 \
                                 --weights weights/gpt2-124m-int8.bin \
                                 --reference eval/runs/fp32

`engine/tools/gpt2_eval` measures and dumps, `metrics.py` judges, `run.py` adds
provenance and writes the committed summary. The same three-way split `bench/`
uses, and for the same reason: the number that decides pass or fail should be
computed by the side that is not also the thing being tested.

The policy is read off the weight file's header, never passed in. A run cannot
claim a precision it did not use.

## The corpus is pinned

    eval/corpus.tsv    8192 token ids, in the format engine/src/runs.cpp reads
    eval/corpus.json   where the text came from, and its checksums

"Perplexity on a fixed WikiText-2 slice" means nothing if the slice moves.
Rows 0-399 of the raw test split, joined in order, tokenized once, committed.
`scripts/export_eval_corpus.py` regenerates it and needs the network; nothing
else here does.

The ids live in the TSV and the provenance in the JSON, neither duplicating the
other, because the engine parses no JSON -- `gpt2/json.h` is a writer only.
Both sides checksum the same little-endian int32 bytes, so a run and the corpus
file can be held against each other.

## What is measured

Four metrics, because each is blind to what the others catch.

- **perplexity** -- did the model get worse at the language? Absolute, and
  comparable to the outside world.
- **top-1 agreement** -- does it pick the same words? What a user sees.
- **KL divergence** -- how far did the whole distribution move? A model one
  nudge away from choosing differently everywhere still scores perfectly on
  top-1; only this sees it coming.
- **decisive disagreement** -- of the positions where the two differ, how many
  did fp32 actually have an opinion about? Added in Phase 4, once there was
  something to measure.

Perplexity and agreement use every predicted position. KL and decisive
disagreement need whole distributions, so they use the strided sample the
engine dumps logits for -- all 4088 positions would be 800 MB.

### Why the fourth one exists

int8 scored 97.48% top-1 against a threshold of 98% that had been written down
before any int8 engine existed. Rather than move the bar to taste, the
disagreements were measured:

    fp32's own top-1 vs top-2 margin, at flipped positions    median 0.00093
    fp32's own top-1 vs top-2 margin, at agreeing positions   median 0.10553

113x. The flips sit exactly where the reference model was itself indifferent.
A raw rate cannot tell that from a real error, so the gate became decisive
disagreement -- a flip counts only when fp32 preferred its pick by more than
0.05 probability -- and the rate stays reported with its bar calibrated to what
a sound engine achieves.

## Validating the harness

    .venv/bin/python eval/validate_ppl.py eval/runs/fp32

`metrics.py` comparing a run to itself scores perfectly and proves only that
the wiring is consistent. An off-by-one in which token is the target would
survive that and then follow every quantized engine around. So the perplexity
is also recomputed in `reference/model.py`, the PyTorch model Phase 0 validated
against HuggingFace:

    reference  36.336585
    engine     36.336588      relative difference 1.0e-07

## Results

`eval/runs/` holds the dumps and is gitignored -- the sampled logits alone are
50 MB and regenerate from the committed corpus. `eval/results/*.json` is the
committed, commit-tagged summary, the same arrangement as `bench/results/`.

Built at the end of Phase 3, before anything was quantized. It judged int8 in
Phase 4 and `int8-wte-o8` in Phase 5, and both results are in `results/`. int4
never reached the engine; `reference/gptq.py` scored it with the same
metric definitions, and it failed every threshold here.
