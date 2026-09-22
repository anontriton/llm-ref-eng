// Choosing the next token from a row of logits.
//
// Greedy is the one the engine is validated with: argmax, first index on a
// tie, exactly as gpt2_generate's std::max_element picks -- so a greedy demo
// run is the same sequence the oracle checks. Sampling is for the demo only;
// seeded, so a run can be repeated.

export function argmax(logits) {
  let best = 0;
  for (let i = 1; i < logits.length; ++i) {
    if (logits[i] > logits[best]) best = i;
  }
  return best;
}

// mulberry32: small, fast, and enough for picking tokens. Returns [0, 1).
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Temperature 0 is greedy. Otherwise keep the top k logits, soften them by
// the temperature, and draw. top-k 40 is what GPT-2's own samples used.
export function sample(logits, { temperature = 0, topK = 40 }, random) {
  if (temperature <= 0) return argmax(logits);

  // The k largest, kept sorted descending. k is small, the vocabulary is not,
  // so an insertion into a short list beats sorting 50,257 entries.
  const k = Math.max(1, Math.min(topK, logits.length));
  const idx = [];
  for (let i = 0; i < logits.length; ++i) {
    const v = logits[i];
    if (idx.length === k && v <= logits[idx[k - 1]]) continue;
    let j = idx.length === k ? k - 1 : idx.length;
    while (j > 0 && logits[idx[j - 1]] < v) {
      idx[j] = idx[j - 1];
      --j;
    }
    idx[j] = i;
  }

  const top = logits[idx[0]];
  const weights = idx.map((i) => Math.exp((logits[i] - top) / temperature));
  let r = random() * weights.reduce((a, b) => a + b, 0);
  for (let j = 0; j < idx.length; ++j) {
    r -= weights[j];
    if (r < 0) return idx[j];
  }
  return idx[idx.length - 1];
}
