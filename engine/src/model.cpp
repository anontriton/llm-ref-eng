#include "gpt2/model.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "gpt2/backend/backend.h"
#include "gpt2/ops.h"

namespace gpt2 {
namespace {

void emit(const Tap& tap, const std::string& name, const float* data,
          std::vector<int64_t> shape) {
  if (tap) tap(name, data, std::move(shape));
}

}  // namespace

Model::Model(const Weights& weights) : w_(weights), cfg_(weights.config()) {}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids,
                                  const Tap& tap) const {
  const int T = static_cast<int>(input_ids.size());
  const int C = cfg_.d_model;

  if (T == 0) throw std::runtime_error("forward: empty input");
  if (T > cfg_.n_ctx) {
    throw std::runtime_error("forward: sequence length " + std::to_string(T) +
                             " exceeds context " + std::to_string(cfg_.n_ctx));
  }

  // Token embedding plus learned absolute position embedding.
  std::vector<float> x(static_cast<size_t>(T) * C);
  for (int t = 0; t < T; ++t) {
    const int32_t id = input_ids[static_cast<size_t>(t)];
    if (id < 0 || id >= cfg_.vocab_size) {
      throw std::runtime_error("forward: token id " + std::to_string(id) +
                               " out of range at position " + std::to_string(t));
    }
    const float* tok = w_.wte() + static_cast<size_t>(id) * C;
    const float* pos = w_.wpe() + static_cast<size_t>(t) * C;
    float* dst = x.data() + static_cast<size_t>(t) * C;
    for (int i = 0; i < C; ++i) dst[i] = tok[i] + pos[i];
  }
  emit(tap, "embed.out", x.data(), {1, T, C});

  for (int i = 0; i < cfg_.n_layer; ++i) block(i, x.data(), T, tap);

  // Final layer norm, written in place: nothing downstream needs the input.
  ops::layernorm(x.data(), w_.ln_f_w(), w_.ln_f_b(), x.data(), T, C,
                 static_cast<float>(cfg_.layer_norm_eps));
  emit(tap, "ln_f.out", x.data(), {1, T, C});

  // lm_head is tied to wte -- there is no separate output matrix.
  std::vector<float> logits(static_cast<size_t>(T) * cfg_.vocab_size);
  ops::linear_tied(x.data(), w_.wte(), logits.data(), T, C, cfg_.vocab_size);
  emit(tap, "logits", logits.data(), {1, T, cfg_.vocab_size});

  return logits;
}

void Model::block(int i, float* x, int T, const Tap& tap) const {
  const LayerWeights& L = w_.layer(i);
  const int C = cfg_.d_model;
  const int H = cfg_.n_head;
  const int dh = cfg_.d_head();
  const int F = cfg_.d_ff;
  const std::string p = "block." + std::to_string(i);
  const size_t TC = static_cast<size_t>(T) * C;

  // ---- attention ----------------------------------------------------------
  std::vector<float> normed(TC);
  ops::layernorm(x, L.ln_1_w, L.ln_1_b, normed.data(), T, C, static_cast<float>(cfg_.layer_norm_eps));
  emit(tap, p + ".ln_1.out", normed.data(), {1, T, C});

  // Fused QKV: 768 -> 2304. Row t holds q, k, v back to back, each C wide, so a
  // single head's slice is contiguous and every dot product below is too.
  std::vector<float> qkv(static_cast<size_t>(T) * 3 * C);
  ops::linear(normed.data(), L.c_attn_w, L.c_attn_b, qkv.data(), T, C, 3 * C);
  emit(tap, p + ".attn.qkv", qkv.data(), {1, T, 3 * C});

  const auto q_at = [&](int t, int h) { return qkv.data() + (static_cast<size_t>(t) * 3 * C) + static_cast<size_t>(h) * dh; };
  const auto k_at = [&](int t, int h) { return q_at(t, h) + C; };
  const auto v_at = [&](int t, int h) { return q_at(t, h) + 2 * C; };

  // Scale by 1/sqrt(d_head) = 1/sqrt(64), NOT 1/sqrt(d_model).
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  std::vector<float> scores(static_cast<size_t>(H) * T * T);
  for (int h = 0; h < H; ++h) {
    for (int qi = 0; qi < T; ++qi) {
      float* row = scores.data() + ((static_cast<size_t>(h) * T) + qi) * T;
      for (int kj = 0; kj < T; ++kj) {
        row[kj] = backend::dot(q_at(qi, h), k_at(kj, h), static_cast<size_t>(dh)) * scale;
      }
    }
  }
  // Tapped before masking, so the oracle holds finite numbers. The comparison
  // only trusts the causal lower triangle; how an engine spells "masked" shows
  // up in .probs, where it actually matters.
  emit(tap, p + ".attn.scores", scores.data(), {1, H, T, T});

  // Causal mask: position t may not attend to anything after t.
  const float neg_inf = -std::numeric_limits<float>::infinity();
  for (int h = 0; h < H; ++h) {
    for (int qi = 0; qi < T; ++qi) {
      float* row = scores.data() + ((static_cast<size_t>(h) * T) + qi) * T;
      for (int kj = qi + 1; kj < T; ++kj) row[kj] = neg_inf;
    }
  }

  ops::softmax_rows(scores.data(), H * T, T);
  emit(tap, p + ".attn.probs", scores.data(), {1, H, T, T});

  // probs @ v, then merge heads back into [T, C].
  std::vector<float> merged(TC, 0.0f);
  for (int h = 0; h < H; ++h) {
    for (int qi = 0; qi < T; ++qi) {
      const float* prow = scores.data() + ((static_cast<size_t>(h) * T) + qi) * T;
      float* dst = merged.data() + static_cast<size_t>(qi) * C + static_cast<size_t>(h) * dh;
      // Every j, not just j <= qi: masked positions carry an exact zero, so the
      // sum is identical either way. Skipping them is a Phase 3 optimization.
      for (int kj = 0; kj < T; ++kj) {
        backend::axpy(prow[kj], v_at(kj, h), dst, static_cast<size_t>(dh));
      }
    }
  }

  std::vector<float> attn_out(TC);
  ops::linear(merged.data(), L.attn_proj_w, L.attn_proj_b, attn_out.data(), T, C, C);
  emit(tap, p + ".attn.out", attn_out.data(), {1, T, C});

  backend::add(attn_out.data(), x, TC);  // residual

  // ---- mlp ----------------------------------------------------------------
  ops::layernorm(x, L.ln_2_w, L.ln_2_b, normed.data(), T, C, static_cast<float>(cfg_.layer_norm_eps));
  emit(tap, p + ".ln_2.out", normed.data(), {1, T, C});

  std::vector<float> hidden(static_cast<size_t>(T) * F);
  ops::linear(normed.data(), L.c_fc_w, L.c_fc_b, hidden.data(), T, C, F);
  emit(tap, p + ".mlp.fc.out", hidden.data(), {1, T, F});

  ops::gelu_new(hidden.data(), hidden.data(), hidden.size());
  emit(tap, p + ".mlp.act.out", hidden.data(), {1, T, F});

  std::vector<float> mlp_out(TC);
  ops::linear(hidden.data(), L.mlp_proj_w, L.mlp_proj_b, mlp_out.data(), T, F, C);
  emit(tap, p + ".mlp.out", mlp_out.data(), {1, T, C});

  backend::add(mlp_out.data(), x, TC);  // residual

  emit(tap, p + ".out", x, {1, T, C});
}

}  // namespace gpt2
