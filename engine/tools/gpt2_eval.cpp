// Phase 4's measuring stick.
//
// CLAUDE.md voids layer-wise matching for quantized runs and names three
// replacements: perplexity on a fixed WikiText-2 slice, top-1 agreement
// against the fp32 engine, and KL divergence of logits against fp32. None of
// them can be checked by oracle/compare.py, which refuses a non-fp32 policy
// outright -- so this is built before anything is quantized, the same way the
// benchmark was built before anything was optimized.
//
// This tool measures and dumps; eval/metrics.py judges. The split is the one
// the oracle already uses, and for the same reason: the number that decides
// pass or fail should be computed by the side that is not also the thing being
// tested.
//
// What it writes, per run:
//
//   nll.npy            f32 [P]      per-position negative log-likelihood of
//                                   the true next token
//   top1.npy           i32 [P]      argmax at each predicted position
//   logits_sample.npy  f32 [S, V]   full logits, every --kl-stride positions
//   sample_index.npy   i32 [S]      which predicted positions those are
//   manifest.json                   provenance, and the policy in force
//
// Perplexity is exp(mean(nll)), which Python computes from nll.npy rather than
// trusting the summary printed here. Full logits at every position would be
// 800 MB for this slice, hence the stride: KL needs whole distributions, and a
// few hundred of them is enough to see a quantizer's damage.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "gpt2/backend/backend.h"
#include "gpt2/json.h"
#include "gpt2/model.h"
#include "gpt2/npy.h"
#include "gpt2/runs.h"
#include "gpt2/sha256.h"
#include "gpt2/threading.h"
#include "gpt2/weights.h"

namespace {

using Clock = std::chrono::steady_clock;

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--weights FILE] [--corpus FILE] [--out DIR]\n"
               "          [--window N] [--windows N] [--kl-stride N]\n"
               "          [--threads N]\n"
               "Writes nll/top1/logits_sample/sample_index .npy plus a "
               "manifest;\nprints a JSON summary on stdout.\n", argv0);
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights_path = "weights/gpt2-124m.bin";
  std::string corpus_path = "eval/corpus.tsv";
  std::string out_dir = "eval/runs/current";
  int window = 512;
  int windows = 8;
  int kl_stride = 16;
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
    else if (arg == "--corpus") corpus_path = next("--corpus");
    else if (arg == "--out") out_dir = next("--out");
    else if (arg == "--window") window = std::stoi(next("--window"));
    else if (arg == "--windows") windows = std::stoi(next("--windows"));
    else if (arg == "--kl-stride") kl_stride = std::stoi(next("--kl-stride"));
    else if (arg == "--threads") threads = std::stoi(next("--threads"));
    else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
    else {
      std::fprintf(stderr, "%s: unknown argument %s\n", argv[0], arg.c_str());
      return 2;
    }
  }

  if (window < 2 || windows < 1 || kl_stride < 1) {
    std::fprintf(stderr,
                 "%s: --window >= 2, --windows >= 1, --kl-stride >= 1\n",
                 argv[0]);
    return 2;
  }

  try {
    gpt2::threads::set_count(threads);

    const std::vector<gpt2::Run> corpus = gpt2::load_runs(corpus_path);
    if (corpus.empty()) throw std::runtime_error("empty corpus: " + corpus_path);
    const gpt2::Run& slice = corpus.front();
    const std::vector<int32_t>& ids = slice.input_ids;

    const size_t needed = static_cast<size_t>(window) * windows;
    if (ids.size() < needed) {
      throw std::runtime_error(
          "corpus holds " + std::to_string(ids.size()) + " ids, need " +
          std::to_string(needed) + " for " + std::to_string(windows) +
          " windows of " + std::to_string(window));
    }

    // Hashed over the raw little-endian int32 the corpus exporter hashes, so
    // a run and eval/corpus.json can be checked against each other. Only the
    // ids actually consumed: a run that reads a prefix should say so.
    const std::string ids_sha =
        gpt2::Sha256::hex_of(ids.data(), needed * sizeof(int32_t));

    const gpt2::Weights weights = gpt2::Weights::load(weights_path);
    const gpt2::Model model(weights);
    const int vocab = weights.config().vocab_size;

    // Read off the weight file, never passed in. A run that had to be told
    // which policy it was under would eventually be told wrong, and the whole
    // point of this harness is to be believed about that.
    const std::string policy = weights.policy();

    // Non-overlapping windows. Each one predicts its own positions 1..L-1 from
    // the context inside it, so the first token of a window is never scored --
    // it has nothing before it to be predicted from. Overlapping windows with
    // a stride would score every token with more context and read lower, but
    // this is a fixed yardstick for comparing engines to each other, not a
    // number to put beside a paper's.
    const int per_window = window - 1;
    const int predictions = per_window * windows;

    std::vector<float> nll(static_cast<size_t>(predictions));
    std::vector<int32_t> top1(static_cast<size_t>(predictions));
    std::vector<int32_t> sample_index;
    std::vector<float> sample_logits;
    for (int p = 0; p < predictions; p += kl_stride) sample_index.push_back(p);
    sample_logits.reserve(sample_index.size() * static_cast<size_t>(vocab));

    double nll_sum = 0.0;
    size_t next_sample = 0;
    const Clock::time_point t0 = Clock::now();

    for (int w = 0; w < windows; ++w) {
      std::fprintf(stderr, "\r  window %d/%d ", w + 1, windows);
      const std::vector<int32_t> chunk(
          ids.begin() + static_cast<long>(w) * window,
          ids.begin() + static_cast<long>(w + 1) * window);
      const std::vector<float> logits = model.forward(chunk);

      for (int t = 0; t + 1 < window; ++t) {
        const float* row = logits.data() + static_cast<size_t>(t) * vocab;
        const int target = chunk[static_cast<size_t>(t) + 1];
        const int p = w * per_window + t;

        // log softmax, shifted by the max so the exponentials cannot overflow.
        const float m = gpt2::backend::max(row, static_cast<size_t>(vocab));
        double acc = 0.0;
        for (int v = 0; v < vocab; ++v) acc += std::exp(row[v] - m);
        const double logsumexp = static_cast<double>(m) + std::log(acc);
        const double value = logsumexp - static_cast<double>(row[target]);

        nll[static_cast<size_t>(p)] = static_cast<float>(value);
        nll_sum += value;
        top1[static_cast<size_t>(p)] = static_cast<int32_t>(
            std::max_element(row, row + vocab) - row);

        if (next_sample < sample_index.size() &&
            sample_index[next_sample] == p) {
          sample_logits.insert(sample_logits.end(), row, row + vocab);
          ++next_sample;
        }
      }
    }
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::fprintf(stderr, "\r%*s\r", 24, "");

    const double mean_nll = nll_sum / predictions;
    const double perplexity = std::exp(mean_nll);

    gpt2::npy::write_f32(out_dir + "/nll.npy", nll.data(), {predictions});
    gpt2::npy::write_i32(out_dir + "/top1.npy", top1.data(), {predictions});
    gpt2::npy::write_i32(out_dir + "/sample_index.npy", sample_index.data(),
                         {static_cast<int64_t>(sample_index.size())});
    gpt2::npy::write_f32(out_dir + "/logits_sample.npy", sample_logits.data(),
                         {static_cast<int64_t>(sample_index.size()), vocab});

    {
      std::ofstream mf(out_dir + "/manifest.json");
      if (!mf) throw std::runtime_error("cannot write " + out_dir + "/manifest.json");
      gpt2::JsonWriter j(mf);
      j.begin_object();
      j.kv("schema", 1);
      j.kv("producer", std::string("engine/tools/gpt2_eval"));
      j.kv("policy", policy);
      j.kv("backend", std::string(gpt2::backend::name()));
      j.kv("threads", gpt2::threads::count());
      j.key("model");
      j.begin_object();
      j.kv("name", std::string("gpt2-124m"));
      j.kv("weights_sha256", weights.source_sha256());
      j.kv("vocab_size", vocab);
      j.end_object();
      j.key("corpus");
      j.begin_object();
      j.kv("name", slice.name);
      j.kv("description", slice.prompt);
      j.kv("ids_sha256", ids_sha);
      j.kv("n_tokens", static_cast<long long>(needed));
      j.end_object();
      j.key("eval");
      j.begin_object();
      j.kv("window", window);
      j.kv("windows", windows);
      j.kv("predictions", predictions);
      j.kv("kl_stride", kl_stride);
      j.kv("samples", static_cast<long long>(sample_index.size()));
      j.end_object();
      j.key("summary");
      j.begin_object();
      j.kv("mean_nll", mean_nll);
      j.kv("perplexity", perplexity);
      j.kv("elapsed_ms", elapsed_ms);
      j.end_object();
      j.end_object();
      mf << "\n";
    }

    std::printf("{\n");
    std::printf("  \"policy\": \"%s\",\n", policy.c_str());
    std::printf("  \"backend\": \"%s\",\n", gpt2::backend::name());
    std::printf("  \"threads\": %d,\n", gpt2::threads::count());
    std::printf("  \"window\": %d,\n", window);
    std::printf("  \"windows\": %d,\n", windows);
    std::printf("  \"predictions\": %d,\n", predictions);
    std::printf("  \"samples\": %lld,\n",
                static_cast<long long>(sample_index.size()));
    std::printf("  \"mean_nll\": %.9f,\n", mean_nll);
    std::printf("  \"perplexity\": %.6f,\n", perplexity);
    std::printf("  \"elapsed_ms\": %.3f,\n", elapsed_ms);
    std::printf("  \"out\": \"%s\"\n}\n", out_dir.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}
