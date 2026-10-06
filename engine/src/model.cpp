#include "gpt2/model.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>

#include "gpt2/backend/backend.h"
#include "gpt2/ops.h"
#include "gpt2/threading.h"

namespace gpt2 {
namespace {

// The name and shape are built only when something is listening. Decode runs
// untapped, and spelling out a dozen names and shape vectors per block, per
// token, only to drop them was most of its allocations.
void emit(const Tap& tap, const std::string& prefix, const char* suffix,
          const float* data, std::initializer_list<int64_t> shape) {
  if (tap) tap(prefix + suffix, data, std::vector<int64_t>(shape));
}

}  // namespace

struct Model::Workspace {
  std::vector<float> normed;   // [T_new, C], ln_1 then ln_2
  std::vector<float> qkv;      // [T_new, 3C]
  std::vector<float> scores;   // [H, T_new, total], scores then probs
  std::vector<float> merged;   // [T_new, C], heads merged back
  std::vector<float> proj;     // [T_new, C], attn.out then mlp.out
  std::vector<float> hidden;   // [T_new, d_ff]
};

Model::Model(const Weights& weights) : w_(weights), cfg_(weights.config()) {}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids,
                                  const Tap& tap, Logits logits) const {
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
  return forward_impl(input_ids, 0, scratch, tap, logits);
}

std::vector<float> Model::forward(const std::vector<int32_t>& new_ids,
                                  KVCache& cache, const Tap& tap,
                                  Logits logits) const {
  return forward_impl(new_ids, cache.size(), cache, tap, logits);
}

std::vector<float> Model::forward_impl(const std::vector<int32_t>& new_ids,
                                       int start_pos, KVCache& cache,
                                       const Tap& tap, Logits which) const {
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
    const float* pos = w_.wpe() + static_cast<size_t>(start_pos + t) * C;
    float* dst = x.data() + static_cast<size_t>(t) * C;
    // Dequantizes when the table is int8; a plain copy otherwise.
    w_.embed(id, dst);
    for (int i = 0; i < C; ++i) dst[i] += pos[i];
  }
  emit(tap, "", "embed.out", x.data(), {1, T_new, C});

  const size_t TC = static_cast<size_t>(T_new) * C;
  Workspace ws;
  ws.normed.resize(TC);
  ws.qkv.resize(3 * TC);
  ws.scores.resize(static_cast<size_t>(cfg_.n_head) * T_new * total);
  ws.merged.resize(TC);
  ws.proj.resize(TC);
  ws.hidden.resize(static_cast<size_t>(T_new) * cfg_.d_ff);

  for (int i = 0; i < cfg_.n_layer; ++i) {
    block(i, x.data(), start_pos, T_new, cache, ws, tap);
  }

  // Final layer norm, written in place: nothing downstream needs the input.
  ops::layernorm(x.data(), w_.ln_f_w(), w_.ln_f_b(), x.data(), T_new, C,
                 static_cast<float>(cfg_.layer_norm_eps));
  emit(tap, "", "ln_f.out", x.data(), {1, T_new, C});

  // lm_head is tied to wte -- there is no separate output matrix. Under
  // Logits::Last only the final row goes through it; each logit is still the
  // one dot product it always was, so that row is the same bits either way.
  const int rows = which == Logits::Last ? 1 : T_new;
  const float* xin = x.data() + static_cast<size_t>(T_new - rows) * C;
  std::vector<float> logits(static_cast<size_t>(rows) * cfg_.vocab_size);
  ops::linear_tied(xin, w_.wte(), logits.data(), rows, C, cfg_.vocab_size);
  emit(tap, "", "logits", logits.data(), {1, rows, cfg_.vocab_size});

  return logits;
}

void Model::block(int i, float* x, int start_pos, int T_new, KVCache& cache,
                  Workspace& ws, const Tap& tap) const {
  const LayerWeights& L = w_.layer(i);
  const int C = cfg_.d_model;
  const int H = cfg_.n_head;
  const int dh = cfg_.d_head();
  const int F = cfg_.d_ff;
  const int total = start_pos + T_new;
  const std::string p = "block." + std::to_string(i);
  const size_t TC = static_cast<size_t>(T_new) * C;

  // ---- attention ----------------------------------------------------------
  float* normed = ws.normed.data();
  ops::layernorm(x, L.ln_1_w, L.ln_1_b, normed, T_new, C, static_cast<float>(cfg_.layer_norm_eps));
  emit(tap, p, ".ln_1.out", normed, {1, T_new, C});

  // Fused QKV: 768 -> 2304. Row t holds q, k, v back to back, each C wide, so a
  // single head's slice is contiguous and every dot product below is too.
  // Computed for the new rows only; earlier rows are already in the cache.
  float* qkv = ws.qkv.data();
  ops::linear(normed, L.c_attn_w, L.c_attn_b, qkv, T_new, C, 3 * C);
  emit(tap, p, ".attn.qkv", qkv, {1, T_new, 3 * C});

  // Hand the new k and v to the cache. Everything downstream reads k and v
  // from there, so prefill and decode take the same path through attention.
  for (int t = 0; t < T_new; ++t) {
    const float* row = qkv + static_cast<size_t>(t) * 3 * C;
    float* kdst = cache.k(i, start_pos + t);
    float* vdst = cache.v(i, start_pos + t);
    for (int c = 0; c < C; ++c) {
      kdst[c] = row[C + c];
      vdst[c] = row[2 * C + c];
    }
  }

  // q is indexed locally (new rows only); k and v absolutely (all positions).
  const auto q_at = [&](int t, int h) { return qkv + (static_cast<size_t>(t) * 3 * C) + static_cast<size_t>(h) * dh; };
  const auto k_at = [&](int t, int h) { return cache.k(i, t) + static_cast<size_t>(h) * dh; };
  const auto v_at = [&](int t, int h) { return cache.v(i, t) + static_cast<size_t>(h) * dh; };
  float* scores = ws.scores.data();
  const auto row_at = [&](int h, int qi) { return scores + ((static_cast<size_t>(h) * T_new) + qi) * total; };

  // Scale by 1/sqrt(d_head) = 1/sqrt(64), NOT 1/sqrt(d_model).
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  // Attention runs as H * T_new independent rows -- one query of one head --
  // and each row writes only its own slice of `scores` and of `merged`. That
  // disjointness is the same property linear() is threaded on, so the result
  // does not depend on the thread count or the schedule. It used to run on the
  // calling thread alone, which at long context is most of the work.

  // [H, T_new, total]: every new query against every position so far. A query
  // at absolute position start_pos + qi may see keys 0..start_pos + qi; the
  // rest is masked below and never read. It is computed only when tapped, so
  // the oracle's dump keeps the finite upper triangle it always had -- the
  // comparison ignores it, but engine-to-engine checksums do not.
  threads::parallel_for(H * T_new, [&](int item) {
    const int h = item / T_new;
    const int qi = item % T_new;
    float* row = row_at(h, qi);
    const int end = tap ? total : start_pos + qi + 1;
    for (int kj = 0; kj < end; ++kj) {
      row[kj] = backend::dot(q_at(qi, h), k_at(kj, h), static_cast<size_t>(dh)) * scale;
    }
  });
  // Tapped before masking, so the oracle holds finite numbers. The comparison
  // only trusts the causal lower triangle; how an engine spells "masked" shows
  // up in .probs, where it actually matters.
  emit(tap, p, ".attn.scores", scores, {1, H, T_new, total});

  // Mask, softmax, then probs @ v, merging heads back into [T_new, C].
  //
  // Causal mask: position t may not attend to anything after t. Row qi is at
  // absolute position start_pos + qi, so during single-token decode the row is
  // entirely unmasked -- there is nothing after the token being generated.
  //
  // The weighted sum stops at the last visible key. Masked positions carry an
  // exact zero, and they come after every visible one, so leaving out the
  // trailing `dst += 0 * v` terms changes nothing but the time: a prefill used
  // to spend half its probs @ v on them.
  const float neg_inf = -std::numeric_limits<float>::infinity();
  float* merged = ws.merged.data();
  threads::parallel_for(H * T_new, [&](int item) {
    const int h = item / T_new;
    const int qi = item % T_new;
    const int last = start_pos + qi;
    float* row = row_at(h, qi);
    for (int kj = last + 1; kj < total; ++kj) row[kj] = neg_inf;
    ops::softmax_rows(row, 1, total);

    float* dst = merged + static_cast<size_t>(qi) * C + static_cast<size_t>(h) * dh;
    std::fill(dst, dst + dh, 0.0f);
    for (int kj = 0; kj <= last; ++kj) {
      backend::axpy(row[kj], v_at(kj, h), dst, static_cast<size_t>(dh));
    }
  });
  emit(tap, p, ".attn.probs", scores, {1, H, T_new, total});

  float* proj = ws.proj.data();
  ops::linear(merged, L.attn_proj_w, L.attn_proj_b, proj, T_new, C, C);
  emit(tap, p, ".attn.out", proj, {1, T_new, C});

  backend::add(proj, x, TC);  // residual

  // ---- mlp ----------------------------------------------------------------
  ops::layernorm(x, L.ln_2_w, L.ln_2_b, normed, T_new, C, static_cast<float>(cfg_.layer_norm_eps));
  emit(tap, p, ".ln_2.out", normed, {1, T_new, C});

  float* hidden = ws.hidden.data();
  const size_t TF = static_cast<size_t>(T_new) * F;
  ops::linear(normed, L.c_fc_w, L.c_fc_b, hidden, T_new, C, F);
  emit(tap, p, ".mlp.fc.out", hidden, {1, T_new, F});

  ops::gelu_new(hidden, hidden, TF);
  emit(tap, p, ".mlp.act.out", hidden, {1, T_new, F});

  ops::linear(hidden, L.mlp_proj_w, L.mlp_proj_b, proj, T_new, F, C);
  emit(tap, p, ".mlp.out", proj, {1, T_new, C});

  backend::add(proj, x, TC);  // residual

  emit(tap, p, ".out", x, {1, T_new, C});
}

}  // namespace gpt2
