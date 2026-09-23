// The browser's entry point: a C API over the engine, for web/worker.js.
//
// Like the other tools this is a thin driver, not a second engine -- Weights,
// Model and KVCache do all the work, the same code the oracle validates. What
// it adds is the shape a JavaScript caller needs:
//
//   gpt2_blob_alloc(n)   reserve n bytes for the weight file and return where
//                        they live in wasm memory. The page streams the
//                        download straight into them, so the 129 MB file is
//                        never held twice.
//   gpt2_load()          parse those bytes (Weights::from_blob) and build the
//                        model and a full-context KV cache.
//   gpt2_reset()         forget the sequence; the next forward starts at 0.
//   gpt2_forward(p, n)   append n token ids through the cache and return the
//                        last position's logits, vocab_size floats. That row
//                        is the distribution over the next token; sampling
//                        from it is the caller's business.
//
// Errors never cross the boundary as exceptions: a failing call returns 0 or
// null and gpt2_error() says why.
//
// Built only under Emscripten, as an ES module that runs in a browser worker
// and in Node -- which is how web/test_web.mjs holds it to gpt2_generate.
#include <emscripten/emscripten.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

#include "gpt2/backend/backend.h"
#include "gpt2/kv_cache.h"
#include "gpt2/model.h"
#include "gpt2/weights.h"

namespace {

std::vector<unsigned char> g_blob;
std::unique_ptr<gpt2::Weights> g_weights;
std::unique_ptr<gpt2::Model> g_model;
std::unique_ptr<gpt2::KVCache> g_cache;
std::vector<float> g_last;  // the last row of the most recent forward
std::string g_error;

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE const char* gpt2_error() { return g_error.c_str(); }

EMSCRIPTEN_KEEPALIVE uint8_t* gpt2_blob_alloc(uint32_t n) {
  try {
    g_blob.assign(n, 0);
    return g_blob.data();
  } catch (const std::exception& e) {
    g_error = std::string("cannot allocate the weight buffer: ") + e.what();
    return nullptr;
  }
}

EMSCRIPTEN_KEEPALIVE int gpt2_load() {
  try {
    g_cache.reset();
    g_model.reset();
    g_weights = std::make_unique<gpt2::Weights>(
        gpt2::Weights::from_blob(std::move(g_blob), "downloaded weights"));
    g_blob = {};
    g_model = std::make_unique<gpt2::Model>(*g_weights);
    g_cache = std::make_unique<gpt2::KVCache>(g_weights->config());
    return 1;
  } catch (const std::exception& e) {
    g_error = e.what();
    return 0;
  }
}

EMSCRIPTEN_KEEPALIVE int gpt2_vocab_size() {
  return g_weights ? g_weights->config().vocab_size : 0;
}
EMSCRIPTEN_KEEPALIVE int gpt2_n_ctx() { return g_weights ? g_weights->config().n_ctx : 0; }
EMSCRIPTEN_KEEPALIVE const char* gpt2_policy() {
  return g_weights ? g_weights->policy().c_str() : "";
}
EMSCRIPTEN_KEEPALIVE const char* gpt2_backend() { return gpt2::backend::name(); }

// Positions the cache holds: the index the next token will take.
EMSCRIPTEN_KEEPALIVE int gpt2_position() { return g_cache ? g_cache->size() : 0; }

EMSCRIPTEN_KEEPALIVE void gpt2_reset() {
  if (g_cache) g_cache->clear();
}

EMSCRIPTEN_KEEPALIVE const float* gpt2_forward(const int32_t* ids, int n) {
  try {
    if (!g_model) throw std::runtime_error("no weights loaded");
    const std::vector<int32_t> input(ids, ids + n);
    const std::vector<float> logits = g_model->forward(input, *g_cache);
    const size_t vocab = static_cast<size_t>(g_weights->config().vocab_size);
    g_last.assign(logits.end() - static_cast<std::ptrdiff_t>(vocab), logits.end());
    return g_last.data();
  } catch (const std::exception& e) {
    g_error = e.what();
    return nullptr;
  }
}

}  // extern "C"
