// Run the engine over the oracle's fixed prompts and dump every activation.
//
// This is the Phase 2 exit criterion in executable form: run it, then run
// oracle/compare.py over what it wrote. Paths default to the repo layout, so
// from the repo root the whole check is
//
//     engine/build/tools/gpt2_dump
//     .venv/bin/python oracle/compare.py engine/dumps/manifest.json
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "gpt2/dump.h"
#include "gpt2/kv_cache.h"
#include "gpt2/model.h"
#include "gpt2/runs.h"
#include "gpt2/weights.h"

namespace {

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--weights FILE] [--runs FILE] [--out DIR] "
               "[--run NAME]... [--quiet]\n"
               "  --weights  flat weight file   (default weights/gpt2-124m.bin)\n"
               "  --runs     run definitions    (default engine/runs.tsv)\n"
               "  --out      dump directory     (default engine/dumps)\n"
               "  --run      only this run, repeatable\n"
      "  --kv-cache decode the last token through the KV cache and dump\n"
      "             only that row (compare.py slices the reference)\n",
               argv0);
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights_path = "weights/gpt2-124m.bin";
  std::string runs_path = "engine/runs.tsv";
  std::string out_dir = "engine/dumps";
  std::vector<std::string> only;
  bool quiet = false;
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
    else if (arg == "--runs") runs_path = next("--runs");
    else if (arg == "--out") out_dir = next("--out");
    else if (arg == "--run") only.push_back(next("--run"));
    else if (arg == "--quiet") quiet = true;
    else if (arg == "--kv-cache") kv_cache = true;
    else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
    else {
      std::fprintf(stderr, "%s: unknown argument %s\n", argv[0], arg.c_str());
      usage(argv[0]);
      return 2;
    }
  }

  try {
    if (!quiet) std::cout << "loading " << weights_path << std::endl;
    const gpt2::Weights weights = gpt2::Weights::load(weights_path);
    const gpt2::Config& cfg = weights.config();
    if (!quiet) {
      std::cout << "  " << weights.size() << " tensors, " << cfg.n_layer
                << " layers, d_model " << cfg.d_model << ", vocab "
                << cfg.vocab_size << "\n"
                << "  weights sha256 " << weights.source_sha256().substr(0, 16)
                << "..." << std::endl;
    }

    const std::vector<gpt2::Run> runs = gpt2::load_runs(runs_path);
    const gpt2::Model model(weights);
    gpt2::Dumper dumper(out_dir, weights);

    int done = 0;
    for (const gpt2::Run& run : runs) {
      if (!only.empty() &&
          std::find(only.begin(), only.end(), run.name) == only.end()) {
        continue;
      }
      const auto t0 = std::chrono::steady_clock::now();
      const int T = static_cast<int>(run.input_ids.size());
      if (kv_cache) {
        // Prefill everything but the last token untapped, then decode that
        // last token through the cache and dump only what it produced.
        //
        // Causality is what makes this a fair comparison: position T-1 attends
        // only to positions <= T-1, so its activations are identical whether
        // the earlier tokens were processed alongside it or cached beforehand.
        // The reference's whole-sequence dump therefore already contains the
        // right answer, at row T-1 -- no second oracle needed.
        gpt2::KVCache cache(cfg, T);
        std::vector<int32_t> last = run.input_ids;
        if (T > 1) {
          const std::vector<int32_t> prefix(run.input_ids.begin(),
                                            run.input_ids.end() - 1);
          model.forward(prefix, cache);
          last.assign(run.input_ids.end() - 1, run.input_ids.end());
        }
        const gpt2::Tap tap = dumper.begin(run, T - 1);
        model.forward(last, cache, tap);
        dumper.end();
      } else {
        const gpt2::Tap tap = dumper.begin(run);
        model.forward(run.input_ids, tap);
        dumper.end();
      }
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count();
      ++done;
      if (!quiet) {
        std::printf("  %-10s T=%-4zu %5lld ms\n", run.name.c_str(),
                    run.input_ids.size(), static_cast<long long>(ms));
      }
    }

    if (done == 0) {
      std::fprintf(stderr, "no runs matched\n");
      return 2;
    }

    dumper.write_manifest();
    if (!quiet) {
      std::cout << "\nwrote " << dumper.tensor_count() << " tensors across "
                << done << " runs\nmanifest -> " << out_dir << "/manifest.json"
                << std::endl;
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
