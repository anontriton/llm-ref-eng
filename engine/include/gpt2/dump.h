// Write the engine's activations in the oracle layout.
//
// The contract is set by oracle/README.md and implemented on the other side by
// reference/dump.py: a directory of .npy tensors plus a manifest.json that pins
// what the run means. This writes the same layout so oracle/compare.py can diff
// the two without knowing which produced which.
//
// The manifest is not decoration. It carries the weight checksum, the exact
// input_ids, the shapes, a per-tensor hash, and the tolerance policy in force;
// a dump whose manifest disagrees with the run being compared is a failure, not
// a warning.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpt2/model.h"
#include "gpt2/runs.h"
#include "gpt2/weights.h"

namespace gpt2 {

class Dumper {
 public:
  Dumper(std::string out_dir, const Weights& weights);

  // Start a run. Creates its directory, removes stale .npy files, and returns
  // the tap to hand to Model::forward.
  //
  // `kv_row` marks a dump produced through the KV cache, holding a single
  // position's activations rather than the whole sequence's. It is the
  // absolute position that row sits at, and the manifest carries it so
  // oracle/compare.py knows to slice the reference at the same row instead of
  // failing on the shape. -1 means an ordinary whole-sequence dump.
  Tap begin(const Run& run, int kv_row = -1);

  // Seal the run started by begin().
  void end();

  // Write <out_dir>/manifest.json covering every run sealed so far.
  void write_manifest() const;

  size_t tensor_count() const { return total_tensors_; }

 private:
  struct Record {
    int order = 0;
    std::string name;
    std::string file;
    std::vector<int64_t> shape;
    std::string sha256;
    double min = 0.0;
    double max = 0.0;
    double absmax = 0.0;
    bool causal_region = false;
  };

  struct RunDump {
    Run run;
    int kv_row = -1;
    std::vector<Record> tensors;
  };

  std::string out_dir_;
  const Weights& weights_;
  std::vector<RunDump> runs_;
  RunDump current_;
  bool open_ = false;
  size_t total_tensors_ = 0;
};

}  // namespace gpt2
