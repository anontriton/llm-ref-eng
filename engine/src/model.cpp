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
  if (T == 0) throw std::runtime_error("forward: empty input");
  if (T > cfg_.n_ctx) {
    throw std::runtime_error("forward: sequence length " + std::to_string(T) +
                             " exceeds context " + std::to_string(cfg_.n_ctx));
  }

  // A cache scoped to this call, sized to exactly this sequence. It is thrown
  // away on return, so the call is stateless as before -- but the arithmetic
  // below runs through the same path the incremental form uses, which is what
  // keeps the two from drifting.
  KVCache scratch(cfg_, T);
  return forward_impl(input_ids, 0, scratch, tap);
}

std::vector<float> Model::forward(const std::vector<int32_t>& new_ids,
                                  KVCache& cache, const Tap& tap) const {
  return forward_impl(new_ids, cache.size(), cache, tap);
}

std::vector<float> Model::forward_impl(const std::vector<int32_t>& new_ids,
                                       int start_pos, KVCache& cache,
                                       const Tap& tap) const {
  const int T_new = static_cast<int>(new_ids.size());
  const int C = cfg_.d_model;
  const int total = start_pos + T_new;

  if (T_new == 0) throw std::runtime_error("forward: empty input");
  if (total > cfg_.n_ctx) {
    throw std::runtime_error("forward: sequence length " +
                             std::to_string(total) + " exceeds context " +
                             std::to_string(cfg_.n_ctx));
  }
  if (cache.size() != start_pos) {
    throw std::runtime_error("forward: cache holds " +
                             std::to_string(cache.size()) +
                             " positions, expected " +
                             std::to_string(start_pos));
  }
  cache.extend(T_new);

  // Token embedding plus learned absolute position embedding. The position is
  // absolute, so a token decoded as the 130th of a sequence reads wpe row 130
  // whether it arrived alone or as part of a batch.
  std::vector<float> x(static_cast<size_t>(T_new) * C);
  for (int t = 0; t < T_new; ++t) {
    const int32_t id = new_ids[static_cast<size_t>(t)];
    if (id < 0 || id >= cfg_.vocab_size) {
      throw std::runtime_error("forward: token id " + std::to_string(id) +
                               " out of range at position " +
                               std::to_string(start_pos + t));
    }
    const float* tok = w_.wte() + static_cast<size_t>(id) * C;
    const float* pos = w_.wpe() + static_cast<size_t>(start_pos + t) * C;
    float* dst = x.data() + static_cast<size_t>(t) * C;
    for (int i = 0; i < C; ++i) dst[i] = tok[i] + pos[i];
  }
  emit(tap, "embed.out", x.data(), {1, T_new, C});

  for (int i = 0; i < cfg_.n_layer; ++i) {
    block(i, x.data(), start_pos, T_new, cache, tap);
  }

  // Final layer norm, written in place: nothing downstream needs the input.
  ops::layernorm(x.data(), w_.ln_f_w(), w_.ln_f_b(), x.data(), T_new, C,
                 static_cast<float>(cfg_.layer_norm_eps));
  emit(tap, "ln_f.out", x.data(), {1, T_new, C});

  // lm_head is tied to wte -- there is no separate output matrix.
  std::vector<float> logits(static_cast<size_t>(T_new) * cfg_.vocab_size);
  ops::linear_tied(x.data(), w_.wte(), logits.data(), T_new, C,
                   cfg_.vocab_size);
  emit(tap, "logits", logits.data(), {1, T_new, cfg_.vocab_size});

  return logits;
}

void Model::block(int i, float* x, int start_pos, int T_new, KVCache& cache,
                  const Tap& tap) const {
  const LayerWeights& L = w_.layer(i);
  const int C = cfg_.d_model;
  const int H = cfg_.n_head;
  const int dh = cfg_.d_head();
  const int F = cfg_.d_ff;
  const int total = start_pos + T_new;
  const std::string p = "block." + std::to_string(i);
  const size_t TC = static_cast<size_t>(T_new) * C;

  // ---- attention ----------------------------------------------------------
  std::vector<float> normed(TC);
  ops::layernorm(x, L.ln_1_w, L.ln_1_b, normed.data(), T_new, C, static_cast<float>(cfg_.layer_norm_eps));
  emit(tap, p + ".ln_1.out", normed.data(), {1, T_new, C});

  // Fused QKV: 768 -> 2304. Row t holds q, k, v back to back, each C wide, so a
  // single head's slice is contiguous and every dot product below is too.
  // Computed for the new rows only; earlier rows are already in the cache.
  std::vector<float> qkv(static_cast<size_t>(T_new) * 3 * C);
  ops::linear(normed.data(), L.c_attn_w, L.c_attn_b, qkv.data(), T_new, C, 3 * C);
  emit(tap, p + ".attn.qkv", qkv.data(), {1, T_new, 3 * C});

  // Hand the new k and v to the cache. Everything downstream reads k and v
  // from there, so prefill and decode take the same path through attention.
  for (int t = 0; t < T_new; ++t) {
    const float* row = qkv.data() + static_cast<size_t>(t) * 3 * C;
    float* kdst = cache.k(i, start_pos + t);
    float* vdst = cache.v(i, start_pos + t);
    for (int c = 0; c < C; ++c) {
      kdst[c] = row[C + c];
      vdst[c] = row[2 * C + c];
    }
  }

  // q is indexed locally (new rows only); k and v absolutely (all positions).
  const auto q_at = [&](int t, int h) { return qkv.data() + (static_cast<size_t>(t) * 3 * C) + static_cast<size_t>(h) * dh; };
  const auto k_at = [&](int t, int h) { return cache.k(i, t) + static_cast<size_t>(h) * dh; };
  const auto v_at = [&](int t, int h) { return cache.v(i, t) + static_cast<size_t>(h) * dh; };

  // Scale by 1/sqrt(d_head) = 1/sqrt(64), NOT 1/sqrt(d_model).
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  // [H, T_new, total]: every new query against every position so far.
  std::vector<float> scores(static_cast<size_t>(H) * T_new * total);
  for (int h = 0; h < H; ++h) {
    for (int qi = 0; qi < T_new; ++qi) {
      float* row = scores.data() + ((static_cast<size_t>(h) * T_new) + qi) * total;
      for (int kj = 0; kj < total; ++kj) {
        row[kj] = backend::dot(q_at(qi, h), k_at(kj, h), static_cast<size_t>(dh)) * scale;
      }
    }
  }
  // Tapped before masking, so the oracle holds finite numbers. The comparison
  // only trusts the causal lower triangle; how an engine spells "masked" shows
  // up in .probs, where it actually matters.
  emit(tap, p + ".attn.scores", scores.data(), {1, H, T_new, total});

  // Causal mask: position t may not attend to anything after t. Row qi is at
  // absolute position start_pos + qi, so during single-token decode the row is
  // entirely unmasked -- there is nothing after the token being generated.
  const float neg_inf = -std::numeric_limits<float>::infinity();
  for (int h = 0; h < H; ++h) {
    for (int qi = 0; qi < T_new; ++qi) {
      float* row = scores.data() + ((static_cast<size_t>(h) * T_new) + qi) * total;
      for (int kj = start_pos + qi + 1; kj < total; ++kj) row[kj] = neg_inf;
    }
  }

  ops::softmax_rows(scores.data(), H * T_new, total);
  emit(tap, p + ".attn.probs", scores.data(), {1, H, T_new, total});

  // probs @ v, then merge heads back into [T_new, C].
  std::vector<float> merged(TC, 0.0f);
  for (int h = 0; h < H; ++h) {
    for (int qi = 0; qi < T_new; ++qi) {
      const float* prow = scores.data() + ((static_cast<size_t>(h) * T_new) + qi) * total;
      float* dst = merged.data() + static_cast<size_t>(qi) * C + static_cast<size_t>(h) * dh;
      // Every j, not just j <= qi: masked positions carry an exact zero, so the
      // sum is identical either way. Skipping them is a Phase 3 optimization.
      for (int kj = 0; kj < total; ++kj) {
        backend::axpy(prow[kj], v_at(kj, h), dst, static_cast<size_t>(dh));
      }
    }
  }

  std::vector<float> attn_out(TC);
  ops::linear(merged.data(), L.attn_proj_w, L.attn_proj_b, attn_out.data(), T_new, C, C);
  emit(tap, p + ".attn.out", attn_out.data(), {1, T_new, C});

  backend::add(attn_out.data(), x, TC);  // residual

  // ---- mlp ----------------------------------------------------------------
  ops::layernorm(x, L.ln_2_w, L.ln_2_b, normed.data(), T_new, C, static_cast<float>(cfg_.layer_norm_eps));
  emit(tap, p + ".ln_2.out", normed.data(), {1, T_new, C});

  std::vector<float> hidden(static_cast<size_t>(T_new) * F);
  ops::linear(normed.data(), L.c_fc_w, L.c_fc_b, hidden.data(), T_new, C, F);
  emit(tap, p + ".mlp.fc.out", hidden.data(), {1, T_new, F});

  ops::gelu_new(hidden.data(), hidden.data(), hidden.size());
  emit(tap, p + ".mlp.act.out", hidden.data(), {1, T_new, F});

  std::vector<float> mlp_out(TC);
  ops::linear(hidden.data(), L.mlp_proj_w, L.mlp_proj_b, mlp_out.data(), T_new, F, C);
  emit(tap, p + ".mlp.out", mlp_out.data(), {1, T_new, C});

  backend::add(mlp_out.data(), x, TC);  // residual

  emit(tap, p + ".out", x, {1, T_new, C});
}

}  // namespace gpt2
