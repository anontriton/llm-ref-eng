// The demo's engine thread. Everything slow happens here -- the 129 MB
// download, prefill, decoding -- so the page stays responsive and can stop a
// generation part way.
//
// Page -> worker:  {type: "load"}
//                  {type: "generate", prompt, maxTokens, temperature, topK, seed}
//                  {type: "stop"}
// Worker -> page:  {type: "progress", what, loaded, total}
//                  {type: "ready", policy, backend, nCtx, vocabSize, loadMs,
//                   weightsBytes, sha256, fromCache}
//                  {type: "start", promptTokens}
//                  {type: "text", text}
//                  {type: "done", reason, promptTokens, generated, prefillMs, decodeMs}
//                  {type: "error", message}
//
// Paths are the static site's layout, which web/pack.py assembles -- the same
// directory whether web/serve.sh serves it or GitHub Pages does.
import createGpt2 from "./wasm/gpt2_web.mjs";
import { Engine } from "./engine.js";
import { rng, sample } from "./sampling.js";
import { ENDOFTEXT, Tokenizer } from "./tokenizer.js";

const MODEL = new URL("./model/", import.meta.url);

let engine = null;
let tokenizer = null;
let stopping = false;
let busy = false;

const post = (msg) => self.postMessage(msg);

// The weights' parts, in order, as one stream of chunks -- read from the
// browser's cache when an earlier visit stored them, fetched and stored
// otherwise. The cache is keyed by the file's sha256, so a new file can never
// be served from an old one's entries; old entries are deleted.
async function openCache(sha) {
  if (typeof caches === "undefined") return null;  // not a secure context
  try {
    const name = `gpt2-weights-${sha}`;
    for (const key of await caches.keys()) {
      if (key.startsWith("gpt2-weights-") && key !== name) await caches.delete(key);
    }
    return await caches.open(name);
  } catch {
    return null;  // storage refused (private mode, quota): just download
  }
}

async function* weightChunks(manifest, cache, status) {
  let loaded = 0;
  for (const part of manifest.parts) {
    const url = new URL(part.file, MODEL);
    let response = cache ? await cache.match(url) : undefined;
    let body;
    if (response) {
      body = response.body;
    } else {
      status.fromCache = false;
      response = await fetchOk(url);
      body = response.body;
      if (cache) {
        // One copy of the stream feeds the engine, the other the cache.
        const [mine, stored] = body.tee();
        body = mine;
        cache.put(url, new Response(stored, { headers: response.headers })).catch(() => {});
      }
    }
    const reader = body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      loaded += value.length;
      post({ type: "progress", what: "weights", loaded, total: manifest.bytes,
             fromCache: status.fromCache });
      yield value;
    }
  }
}

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function fetchOk(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url.pathname}: HTTP ${response.status} -- ` +
                    "was the site built with web/pack.py?");
  }
  return response;
}

async function load() {
  const t0 = performance.now();
  const [vocab, merges, manifest] = await Promise.all([
    fetchOk(new URL("vocab.json", MODEL)).then((r) => r.json()),
    fetchOk(new URL("merges.txt", MODEL)).then((r) => r.text()),
    fetchOk(new URL("weights.json", MODEL)).then((r) => r.json()),
  ]);
  tokenizer = new Tokenizer(vocab, merges);

  engine = await Engine.create(createGpt2);
  const cache = await openCache(manifest.sha256);
  const status = { fromCache: true };
  await engine.loadWeights(manifest.bytes, weightChunks(manifest, cache, status), null,
    async (bytes) => {
      post({ type: "progress", what: "verify", loaded: manifest.bytes, total: manifest.bytes });
      // The file the eval measured, or nothing: web/pack.py already refused
      // to publish any other, and this refuses to run one that changed since.
      const got = await sha256Hex(bytes);
      if (got !== manifest.sha256) {
        if (cache) await caches.delete(`gpt2-weights-${manifest.sha256}`).catch(() => {});
        throw new Error(`the downloaded weights do not match their checksum ` +
                        `(got ${got.slice(0, 12)}…, expected ${manifest.sha256.slice(0, 12)}…)`);
      }
    });
  post({
    type: "ready", policy: engine.policy, backend: engine.backend,
    nCtx: engine.nCtx, vocabSize: engine.vocabSize,
    loadMs: performance.now() - t0, weightsBytes: manifest.bytes,
    sha256: manifest.sha256, fromCache: status.fromCache,
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
