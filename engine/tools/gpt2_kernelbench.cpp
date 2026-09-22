// Per-shape matmul timings, which whole-model numbers cannot give you.
//
// bench/ measures prefill and decode. That is the right unit for a claim about
// the engine, and the wrong one for deciding what to change: a 5% whole-model
// move can be one kernel getting much faster while another gets slower, and
// the two only separate here.
//
// This existed as a scratch file during Phase 3 and earned its place. The first
// attempt at blocked GEMM tiled output columns in both matmuls, which moved
// prefill by a believable-looking 5% -- and per shape it was four regressions
// paying for one large win:
//
//     qkv       768->2304    973 ms -> 1015 ms
//     attn_proj  768->768    310 ms ->  333 ms
//     c_fc      768->3072   1585 ms -> 1553 ms
//     mlp_proj  3072->768   1586 ms -> 1892 ms
//     lm_head  768->50257   1214 ms ->  582 ms
//
// linear()'s W is row-major along n_out, so a column strip reads it strided;
// linear_tied()'s Wt is the transpose, where the same strip is contiguous. Two
// kernels, opposite right answers, invisible in the aggregate.
//
// Runs the real kernels on the real weights, so it reports whatever precision
// the file holds rather than a synthetic stand-in.
#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "gpt2/backend/backend.h"
#include "gpt2/model.h"
#include "gpt2/ops.h"
#include "gpt2/threading.h"
#include "gpt2/weights.h"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Deterministic, and in the range activations actually occupy: a matmul's cost
// does not depend on the values, but denormals would make it lie.
void fill(std::vector<float>& v, int seed) {
  uint32_t s = static_cast<uint32_t>(seed) * 2654435761u + 1u;
  for (float& x : v) {
    s = s * 1664525u + 1013904223u;
    x = static_cast<float>(static_cast<int32_t>(s >> 8) % 2000 - 1000) * 0.001f;
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights_path = "weights/gpt2-124m.bin";
  int rows = 128;
  int repeat = 2;
  int threads = 1;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s: %s needs a value\n", argv[0], what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--weights") weights_path = next("--weights");
    else if (arg == "--rows") rows = std::stoi(next("--rows"));
    else if (arg == "--repeat") repeat = std::stoi(next("--repeat"));
    else if (arg == "--threads") threads = std::stoi(next("--threads"));
    else if (arg == "-h" || arg == "--help") {
      std::fprintf(stderr,
                   "usage: %s [--weights FILE] [--rows N] [--repeat N] "
                   "[--threads N]\n"
                   "  --rows     sequence length to simulate (default 128);\n"
                   "             use 1 to see what decode pays\n"
                   "  --repeat   timed passes, best of (default 2)\n",
                   argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "%s: unknown argument %s\n", argv[0], arg.c_str());
      return 2;
    }
  }
  if (rows < 1 || repeat < 1) {
    std::fprintf(stderr, "%s: --rows and --repeat must be >= 1\n", argv[0]);
    return 2;
  }

  try {
    gpt2::threads::set_count(threads);
    const gpt2::Weights weights = gpt2::Weights::load(weights_path);
    const gpt2::Config& cfg = weights.config();
    const gpt2::LayerWeights& L = weights.layer(0);
    const int d = cfg.d_model, f = cfg.d_ff, v = cfg.vocab_size;

    struct Case {
      const char* name;
      const gpt2::Matrix* W;
      int n_in, n_out, per_forward;   // how many of these one forward runs
      bool tied;
    };
    const Case cases[] = {
      {"qkv",       &L.c_attn_w,    d, 3 * d, cfg.n_layer, false},
      {"attn_proj", &L.attn_proj_w, d, d,     cfg.n_layer, false},
      {"c_fc",      &L.c_fc_w,      d, f,     cfg.n_layer, false},
      {"mlp_proj",  &L.mlp_proj_w,  f, d,     cfg.n_layer, false},
      {"lm_head",   &weights.wte(), d, v,     1,           true},
    };

    std::printf("%-10s %6s %-14s %10s %12s %10s\n",
                "kernel", "dtype", "shape", "ms each", "ms/forward", "GFLOP/s");

    double total = 0.0;
    for (const Case& c : cases) {
      std::vector<float> x(static_cast<size_t>(rows) * c.n_in);
      std::vector<float> out(static_cast<size_t>(rows) * c.n_out);
      std::vector<float> bias(static_cast<size_t>(c.n_out), 0.01f);
      fill(x, c.n_in);

      double best = 0.0;
      for (int r = 0; r < repeat; ++r) {
        const Clock::time_point t0 = Clock::now();
        if (c.tied) {
          gpt2::ops::linear_tied(x.data(), *c.W, out.data(), rows, c.n_in, c.n_out);
        } else {
          gpt2::ops::linear(x.data(), *c.W, bias.data(), out.data(),
                            rows, c.n_in, c.n_out);
        }
        const double e = ms_since(t0);
        if (r == 0 || e < best) best = e;
      }

      const double per_forward = best * c.per_forward;
      total += per_forward;
      const double gflop = 2.0 * rows * c.n_in * c.n_out / 1e9;
      std::printf("%-10s %6s %5d->%-7d %10.2f %12.1f %10.2f\n",
                  c.name, c.W->quantized() ? "int8" : "fp32",
                  c.n_in, c.n_out, best, per_forward, gflop / (best / 1000.0));
    }
    std::printf("%-10s %6s %-14s %10s %12.1f\n", "TOTAL",
                weights.policy(), "", "", total);
    std::printf("\nbackend %s, %d thread(s), rows %d\n",
                gpt2::backend::name(), gpt2::threads::count(), rows);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}
