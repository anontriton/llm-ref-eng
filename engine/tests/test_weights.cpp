// The weight loader, and the strictness that is the point of it.
//
// A transpose or a renamed key must fail loudly here rather than become "the
// logits are slightly wrong" twelve layers later. Needs the converted weight
// file; skips cleanly when it is absent so a fresh clone can still run ctest.
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "check.h"
#include "gpt2/weights.h"

namespace {

std::string repo_path(const char* rel) { return std::string(GPT2_REPO_ROOT) + "/" + rel; }

bool throws(const std::string& path) {
  try {
    gpt2::Weights::load(path);
    return false;
  } catch (const std::runtime_error&) {
    return true;
  }
}

}  // namespace

int main() {
  // A file that does not exist, and a file that is not ours, must both throw.
  check::ok(throws(repo_path("weights/definitely-not-here.bin")),
            "missing file throws");
  {
    const std::string junk = repo_path("engine/build/junk-weights.bin");
    std::ofstream f(junk, std::ios::binary);
    f << "NOTGPT2!" << std::string(200, '\0');
    f.close();
    check::ok(throws(junk), "bad magic throws");
    std::remove(junk.c_str());
  }

  const std::string path = repo_path("weights/gpt2-124m.bin");
  std::ifstream probe(path, std::ios::binary);
  if (!probe) {
    std::printf("test_weights: SKIP -- no %s (run scripts/convert_weights.py)\n",
                path.c_str());
    return check::report("test_weights");
  }
  probe.close();

  const gpt2::Weights w = gpt2::Weights::load(path);
  const gpt2::Config& c = w.config();

  check::equal(c.n_layer, 12, "n_layer");
  check::equal(c.n_head, 12, "n_head");
  check::equal(c.d_model, 768, "d_model");
  check::equal(c.d_ff, 3072, "d_ff");
  check::equal(c.n_ctx, 1024, "n_ctx");
  check::equal(c.vocab_size, 50257, "vocab_size");
  check::equal(c.d_head(), 64, "d_head is 64, which sets the attention scale");
  check::close(c.layer_norm_eps, 1e-5, 0.0, "layer_norm_eps is exactly 1e-5");

  check::equal(static_cast<long long>(w.size()), 148, "tensor count");
  check::equal(static_cast<long long>(w.source_sha256().size()), 64,
               "source checksum is 64 hex characters");

  // Shapes are where the HF Conv1D transpose gotcha would show up.
  check::ok(w.get("wte").shape == std::vector<int64_t>({50257, 768}), "wte shape");
  check::ok(w.get("wpe").shape == std::vector<int64_t>({1024, 768}), "wpe shape");
  check::ok(w.get("h.0.attn.c_attn.weight").shape == std::vector<int64_t>({768, 2304}),
            "c_attn is [d_model, 3*d_model] -- fused QKV, checkpoint orientation");
  check::ok(w.get("h.0.mlp.c_fc.weight").shape == std::vector<int64_t>({768, 3072}),
            "c_fc is [in, out], the transpose of nn.Linear");
  check::ok(w.get("h.0.mlp.c_proj.weight").shape == std::vector<int64_t>({3072, 768}),
            "mlp c_proj is [d_ff, d_model]");

  // The checkpoint's causal-mask buffer is not a weight and must not be loaded.
  check::ok(!w.has("h.0.attn.bias"), "HF's precomputed causal mask is dropped");
  // lm_head is tied to wte; there is no separate output matrix.
  check::ok(!w.has("lm_head.weight"), "no separate lm_head (it is tied to wte)");

  // Every layer resolved to real pointers.
  for (int i = 0; i < c.n_layer; ++i) {
    const gpt2::LayerWeights& L = w.layer(i);
    check::ok(L.ln_1_w && L.ln_1_b && L.c_attn_w && L.c_attn_b && L.attn_proj_w &&
              L.attn_proj_b && L.ln_2_w && L.ln_2_b && L.c_fc_w && L.c_fc_b &&
              L.mlp_proj_w && L.mlp_proj_b,
              "layer " + std::to_string(i) + " fully resolved");
  }
  check::ok(w.wte() && w.wpe() && w.ln_f_w() && w.ln_f_b(), "top-level weights resolved");

  // LayerNorm weights are near 1 and biases near 0 -- a cheap sanity check that
  // the data block is aligned with the directory rather than shifted.
  double sum = 0.0;
  for (int j = 0; j < c.d_model; ++j) sum += w.layer(0).ln_1_w[j];
  check::close(sum / c.d_model, 0.22, 0.25, "ln_1 weights are O(0.1-1), not garbage");

  return check::report("test_weights");
}
