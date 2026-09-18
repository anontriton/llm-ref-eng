// Greedy decoding, for the third of CLAUDE.md's fp32 tolerance criteria:
// "Top-1 token identical for 50 consecutive greedy steps".
//
// Layer-wise comparison proves the arithmetic tracks the reference. This proves
// the thing a user would actually notice -- that the engine picks the same
// words -- which is a different claim, and the one that survives Phase 4 when
// layer-wise matching is void.
//
// No KV cache: every step re-runs the full prefix through the same forward pass
// the oracle validated. That is slow and deliberate; the cache arrives in
// Phase 3, and it will have this output to prove itself against.
#include <algorithm>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "gpt2/model.h"
#include "gpt2/runs.h"
#include "gpt2/weights.h"

int main(int argc, char** argv) {
  std::string weights_path = "weights/gpt2-124m.bin";
  std::string runs_path = "engine/runs.tsv";
  std::string run_name;
  int steps = 50;

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
    else if (arg == "--runs") runs_path = next("--runs");
    else if (arg == "--run") run_name = next("--run");
    else if (arg == "--steps") steps = std::stoi(next("--steps"));
    else if (arg == "-h" || arg == "--help") {
      std::fprintf(stderr,
                   "usage: %s [--weights FILE] [--runs FILE] --run NAME "
                   "[--steps N]\n"
                   "Prints JSON: the prompt ids and the ids greedily decoded "
                   "after them.\n", argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "%s: unknown argument %s\n", argv[0], arg.c_str());
      return 2;
    }
  }

  try {
    const std::vector<gpt2::Run> runs = gpt2::load_runs(runs_path);
    const gpt2::Run* run = nullptr;
    if (run_name.empty()) {
      run = &runs.front();
    } else {
      for (const gpt2::Run& r : runs) {
        if (r.name == run_name) run = &r;
      }
    }
    if (run == nullptr) throw std::runtime_error("no such run: " + run_name);

    const gpt2::Weights weights = gpt2::Weights::load(weights_path);
    const gpt2::Model model(weights);
    const int vocab = weights.config().vocab_size;

    std::vector<int32_t> ids = run->input_ids;
    const size_t prompt_len = ids.size();
    std::vector<int32_t> generated;

    for (int s = 0; s < steps; ++s) {
      const std::vector<float> logits = model.forward(ids);
      // Only the final position matters: it is the distribution over what
      // comes next.
      const float* last = logits.data() +
                          (ids.size() - 1) * static_cast<size_t>(vocab);
      const int32_t next = static_cast<int32_t>(
          std::max_element(last, last + vocab) - last);
      generated.push_back(next);
      ids.push_back(next);
      std::fprintf(stderr, "\r  step %d/%d", s + 1, steps);
    }
    std::fprintf(stderr, "\r%*s\r", 24, "");

    std::cout << "{\"run\": \"" << run->name << "\", \"steps\": " << steps
              << ", \"prompt_ids\": [";
    for (size_t i = 0; i < prompt_len; ++i) {
      std::cout << (i ? ", " : "") << run->input_ids[i];
    }
    std::cout << "], \"generated_ids\": [";
    for (size_t i = 0; i < generated.size(); ++i) {
      std::cout << (i ? ", " : "") << generated[i];
    }
    std::cout << "]}" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
