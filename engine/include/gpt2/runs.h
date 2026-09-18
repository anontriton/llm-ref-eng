// Read the run definitions the engine is asked to reproduce.
//
// The engine does not tokenize. The oracle already pins exact input_ids per
// run, and re-deriving them in C++ would introduce a second tokenizer whose
// disagreements would look like engine bugs. scripts/export_runs.py lifts them
// out of oracle/manifest.json into a TSV this reads:
//
//     name <TAB> escaped_prompt <TAB> id,id,id,...
//
// The prompt is carried only so the engine's manifest can echo it; nothing
// computes on it. Escapes are \\, \n, \r, \t.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gpt2 {

struct Run {
  std::string name;
  std::string prompt;
  std::vector<int32_t> input_ids;
};

// Throws std::runtime_error on a missing file or a malformed line.
std::vector<Run> load_runs(const std::string& path);

}  // namespace gpt2
