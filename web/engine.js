// The wasm engine from JavaScript: engine/tools/gpt2_web.cpp's C API, wrapped.
//
// Shared by the demo's worker and by web/test_web.mjs, so what the test holds
// to gpt2_generate is the code the page runs, not a copy of it.

export class Engine {
  // factory: the default export of gpt2_web.mjs.
  static async create(factory) {
    return new Engine(await factory());
  }

  constructor(module) {
    this.m = module;
    this.vocabSize = 0;
    this.nCtx = 0;
    this.idsPtr = 0;
    this.idsCap = 0;
  }

  error() {
    return this.m.UTF8ToString(this.m._gpt2_error());
  }

  // Stream `total` bytes of weight file into the engine. `chunks` is any
  // async iterable of Uint8Array -- a fetch body reader in the page, a file
  // stream in Node. Each chunk is copied into wasm memory as it arrives, so
  // the file is never held twice.
  //
  // `verify`, if given, is awaited with a view of the complete file before the
  // engine parses it; throwing from it aborts the load. The page uses it to
  // check the sha256 without a second 129 MB copy.
  async loadWeights(total, chunks, onProgress, verify) {
    const ptr = this.m._gpt2_blob_alloc(total);
    if (!ptr) throw new Error(this.error());
    let at = 0;
    for await (const chunk of chunks) {
      if (at + chunk.length > total) throw new Error(`weights: more than the ${total} bytes promised`);
      // HEAPU8 is re-read every time: growing memory replaces the view.
      this.m.HEAPU8.set(chunk, ptr + at);
      at += chunk.length;
      onProgress?.(at, total);
    }
    if (at !== total) throw new Error(`weights: got ${at} of ${total} bytes`);
    if (verify) await verify(this.m.HEAPU8.subarray(ptr, ptr + total));
    if (!this.m._gpt2_load()) throw new Error(this.error());
    this.vocabSize = this.m._gpt2_vocab_size();
    this.nCtx = this.m._gpt2_n_ctx();
    this.policy = this.m.UTF8ToString(this.m._gpt2_policy());
    this.backend = this.m.UTF8ToString(this.m._gpt2_backend());
  }

  reset() {
    this.m._gpt2_reset();
  }

  // Tokens the engine holds; the position the next one takes.
  get position() {
    return this.m._gpt2_position();
  }

  // Append ids through the KV cache; returns the last position's logits. The
  // array is a view of wasm memory, valid until the next call into the engine.
  forward(ids) {
    if (ids.length > this.idsCap) {
      if (this.idsPtr) this.m._free(this.idsPtr);
      this.idsCap = Math.max(ids.length, 1024);
      this.idsPtr = this.m._malloc(this.idsCap * 4);
    }
    this.m.HEAP32.set(ids, this.idsPtr >> 2);
    const p = this.m._gpt2_forward(this.idsPtr, ids.length);
    if (!p) throw new Error(this.error());
    return this.m.HEAPF32.subarray(p >> 2, (p >> 2) + this.vocabSize);
  }
}
