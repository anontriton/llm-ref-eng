// The demo's engine thread. Everything slow happens here -- the 129 MB
// download, prefill, decoding -- so the page stays responsive and can stop a
// generation part way.
//
// Page -> worker:  {type: "load"}
//                  {type: "generate", prompt, maxTokens, temperature, topK, seed}
//                  {type: "stop"}
// Worker -> page:  {type: "progress", what, loaded, total}
//                  {type: "ready", policy, backend, nCtx, vocabSize, loadMs, weightsBytes}
//                  {type: "start", promptTokens}
//                  {type: "text", text}
//                  {type: "done", reason, promptTokens, generated, prefillMs, decodeMs}
//                  {type: "error", message}
import createGpt2 from "../build/engine-wasm_simd128/tools/gpt2_web.mjs";
import { Engine } from "../engine.js";
import { rng, sample } from "../sampling.js";
import { ENDOFTEXT, Tokenizer } from "../tokenizer.js";

const WEIGHTS = new URL("../../weights/gpt2-124m-int8-wte-o8.bin", import.meta.url);
const VOCAB = new URL("../../weights/gpt2-124m/vocab.json", import.meta.url);
const MERGES = new URL("../../weights/gpt2-124m/merges.txt", import.meta.url);

let engine = null;
let tokenizer = null;
let stopping = false;
let busy = false;

const post = (msg) => self.postMessage(msg);

// A fetch body as an async iterable of chunks, reporting progress. Written
// out rather than `for await (... of response.body)`, which not every browser
// supports yet.
async function* chunks(response, what, total) {
  const reader = response.body.getReader();
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    loaded += value.length;
    post({ type: "progress", what, loaded, total });
    yield value;
  }
}

async function fetchOk(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url.pathname}: HTTP ${response.status} -- ` +
                    "is the server running from the repo root, and the file built?");
  }
  return response;
}

async function load() {
  const t0 = performance.now();
  const [vocab, merges] = await Promise.all([
    fetchOk(VOCAB).then((r) => r.json()),
    fetchOk(MERGES).then((r) => r.text()),
  ]);
  tokenizer = new Tokenizer(vocab, merges);

  engine = await Engine.create(createGpt2);
  const response = await fetchOk(WEIGHTS);
  const total = Number(response.headers.get("Content-Length"));
  if (total > 0) {
    await engine.loadWeights(total, chunks(response, "weights", total));
  } else {
    // No length to reserve against: take the whole body first. Twice the
    // memory for a moment, but only when the server will not say how big.
    const bytes = new Uint8Array(await response.arrayBuffer());
    await engine.loadWeights(bytes.length, [bytes]);
  }
  post({
    type: "ready", policy: engine.policy, backend: engine.backend,
    nCtx: engine.nCtx, vocabSize: engine.vocabSize,
    loadMs: performance.now() - t0, weightsBytes: total,
  });
}

// Let queued messages -- a stop, above all -- run between tokens. A
// MessageChannel round trip, not setTimeout, which clamps to 4 ms once nested.
const channel = new MessageChannel();
const yieldToEvents = () => new Promise((resolve) => {
  channel.port1.onmessage = resolve;
  channel.port2.postMessage(null);
});

async function generate({ prompt, maxTokens, temperature, topK, seed }) {
  const eot = tokenizer.encoder.get(ENDOFTEXT);
  // An empty prompt starts from <|endoftext|>, the way GPT-2 begins a document.
  const ids = prompt.length > 0 ? tokenizer.encode(prompt) : [eot];
  if (ids.length >= engine.nCtx) {
    throw new Error(`the prompt is ${ids.length} tokens; the context holds ${engine.nCtx}`);
  }
  post({ type: "start", promptTokens: ids.length });

  engine.reset();
  const random = rng(seed);
  const text = tokenizer.streamDecoder();
  const t0 = performance.now();
  let logits = engine.forward(ids);
  const prefillMs = performance.now() - t0;

  let generated = 0;
  let reason = "length";
  const t1 = performance.now();
  for (;;) {
    const next = sample(logits, { temperature, topK }, random);
    if (next === eot) { reason = "endoftext"; break; }
    ++generated;
    const piece = text.push(next);
    if (piece) post({ type: "text", text: piece });
    if (generated >= maxTokens) break;
    if (engine.position >= engine.nCtx) { reason = "context"; break; }
    await yieldToEvents();
    if (stopping) { reason = "stopped"; break; }
    logits = engine.forward([next]);
  }
  const tail = text.flush();
  if (tail) post({ type: "text", text: tail });
  // Decode time per token counts the steps that ran a forward: every token
  // after the first, which came out of prefill.
  post({
    type: "done", reason, promptTokens: ids.length, generated, prefillMs,
    decodeMs: generated > 1 ? (performance.now() - t1) / (generated - 1) : 0,
  });
}

self.onmessage = async ({ data }) => {
  if (data.type === "stop") {
    stopping = true;
    return;
  }
  if (busy) return;
  busy = true;
  stopping = false;
  try {
    if (data.type === "load") await load();
    else if (data.type === "generate") await generate(data);
  } catch (e) {
    post({ type: "error", message: e.message || String(e) });
  } finally {
    busy = false;
  }
};
