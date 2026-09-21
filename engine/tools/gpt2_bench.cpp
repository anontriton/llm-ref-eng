// The benchmark driver: prefill and decode, timed.
//
// Built before Phase 3 starts, so every optimization has a baseline to move.
// This tool measures and prints; bench/run.py wraps it with the provenance
// (commit SHA, dirty flag, machine) that makes a number traceable. Same split
// as the oracle: C++ produces, Python judges and records.
//
// Two numbers, because they scale differently and Phase 3 attacks them in a
// specific order:
//
//   prefill  -- one forward over the whole prompt. Compute-bound on the
//               matmuls; what blocked GEMM, AVX2, and threads go after.
//   decode   -- the per-token cost of extending the sequence. Without
//               --kv-cache every step re-runs the entire prefix through the
//               full forward pass, so the number is enormous and grows with
//               each step; with it, each step processes one token. Both paths
//               stay reachable because the comparison between them is the
//               point.
//
// The generated ids are printed too. They are not a benchmark metric; they are
// a tripwire. An "optimization" that changes what the engine decodes shows up
// here as a different id sequence, without waiting for the oracle run.
#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "gpt2/kv_cache.h"
#include "gpt2/model.h"
#include "gpt2/runs.h"
#include "gpt2/weights.h"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int argmax(const float* row, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i) {
    if (row[i] > row[best]) best = i;
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights_path = "weights/gpt2-124m.bin";
  std::string prompts_path = "bench/prompts.tsv";
  std::string run_name;
  int generate = 128;
  int prefill_repeat = 3;
  bool kv_cache = false;

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
    else if (arg == "--prompts") prompts_path = next("--prompts");
    else if (arg == "--run") run_name = next("--run");
    else if (arg == "--generate") generate = std::stoi(next("--generate"));
    else if (arg == "--kv-cache") kv_cache = true;
    else if (arg == "--prefill-repeat") {
      prefill_repeat = std::stoi(next("--prefill-repeat"));
    } else if (arg == "-h" || arg == "--help") {
      std::fprintf(stderr,
                   "usage: %s [--weights FILE] [--prompts FILE] [--run NAME]\n"
                   "          [--generate N] [--prefill-repeat N] "
                   "[--kv-cache]\n"
                   "Prints one JSON object of timings on stdout; progress on "
                   "stderr.\n", argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "%s: unknown argument %s\n", argv[0], arg.c_str());
      return 2;
    }
  }

  if (generate < 1 || prefill_repeat < 1) {
    std::fprintf(stderr, "%s: --generate and --prefill-repeat must be >= 1\n",
                 argv[0]);
    return 2;
  }

  try {
    const std::vector<gpt2::Run> runs = gpt2::load_runs(prompts_path);
    const gpt2::Run* run = nullptr;
    if (run_name.empty()) {
      run = &runs.front();
    } else {
      for (const gpt2::Run& r : runs) {
        if (r.name == run_name) run = &r;
      }
    }
    if (run == nullptr) throw std::runtime_error("no such run: " + run_name);

    const Clock::time_point load_t0 = Clock::now();
    const gpt2::Weights weights = gpt2::Weights::load(weights_path);
    const gpt2::Model model(weights);
    const double load_ms = ms_since(load_t0);

    const int vocab = weights.config().vocab_size;
    const std::vector<int32_t>& prompt = run->input_ids;
    const int prompt_tokens = static_cast<int>(prompt.size());

    // Prefill, best of `prefill_repeat`. The minimum, not the mean: every
    // source of noise here (scheduling, frequency, a cold page) adds time and
    // none removes it, so the fastest run is the closest to the machine's
    // actual capability. The first iteration also warms the weight pages, so a
    // repeat count of 1 measures something systematically slower.
    gpt2::KVCache cache(weights.config());
    double prefill_ms = 0.0;
    for (int r = 0; r < prefill_repeat; ++r) {
      std::fprintf(stderr, "\r  prefill %d/%d ", r + 1, prefill_repeat);
      // Each repeat must start from an empty cache, or it would be measuring a
      // prefill that had already been done.
      cache.clear();
      const Clock::time_point t0 = Clock::now();
      const std::vector<float> logits =
          kv_cache ? model.forward(prompt, cache) : model.forward(prompt);
      const double elapsed = ms_since(t0);
      // Keep the compiler from deciding the forward pass is dead code.
      if (logits.empty()) throw std::runtime_error("empty logits");
      if (r == 0 || elapsed < prefill_ms) prefill_ms = elapsed;
    }

    // Decode. Greedy, so the id sequence is deterministic and comparable
    // across commits.
    std::vector<int32_t> ids = prompt;
    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(generate));

    // The cache is left holding the prompt by the last prefill repeat, so
    // decode picks up exactly where a real caller would.
    std::vector<int32_t> pending;
    const Clock::time_point decode_t0 = Clock::now();
    for (int s = 0; s < generate; ++s) {
      std::fprintf(stderr, "\r  decode %d/%d  ", s + 1, generate);
      const std::vector<float> logits =
          kv_cache ? model.forward(pending.empty()
                                       ? std::vector<int32_t>{ids.back()}
                                       : pending,
                                   cache)
                   : model.forward(ids);
      const size_t rows = kv_cache ? 1 : ids.size();
      const float* last =
          logits.data() + (rows - 1) * static_cast<size_t>(vocab);
      const int next_id = argmax(last, vocab);
      generated.push_back(next_id);
      ids.push_back(next_id);
      pending.assign(1, next_id);
    }
    const double decode_ms = ms_since(decode_t0);
    std::fprintf(stderr, "\r%*s\r", 24, "");

    const double decode_ms_per_token = decode_ms / generate;
    // End-to-end throughput: the tokens a caller asked for, over the time they
    // waited for them. Prefill is in the denominator because the caller waits
    // for it; it is reported separately so the two can be told apart.
    const double tokens_per_sec = generate / ((prefill_ms + decode_ms) / 1000.0);

    std::printf("{\n");
    std::printf("  \"run\": \"%s\",\n", run->name.c_str());
    std::printf("  \"prompt_tokens\": %d,\n", prompt_tokens);
    std::printf("  \"generate\": %d,\n", generate);
    std::printf("  \"prefill_repeat\": %d,\n", prefill_repeat);
    std::printf("  \"kv_cache\": %s,\n", kv_cache ? "true" : "false");
    std::printf("  \"weights_load_ms\": %.3f,\n", load_ms);
    std::printf("  \"prefill_ms\": %.3f,\n", prefill_ms);
    std::printf("  \"decode_total_ms\": %.3f,\n", decode_ms);
    std::printf("  \"decode_ms_per_token\": %.3f,\n", decode_ms_per_token);
    std::printf("  \"tokens_per_sec\": %.6f,\n", tokens_per_sec);
    std::printf("  \"generated_ids\": [");
    for (size_t i = 0; i < generated.size(); ++i) {
      std::printf("%s%d", i ? ", " : "", generated[i]);
    }
    std::printf("]\n}\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}
