#include "gpt2/runs.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gpt2 {
namespace {

std::string unescape(const std::string& s, const std::string& where) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '\\') {
      out.push_back(s[i]);
      continue;
    }
    if (++i >= s.size()) throw std::runtime_error(where + ": trailing backslash");
    switch (s[i]) {
      case 'n': out.push_back('\n'); break;
      case 'r': out.push_back('\r'); break;
      case 't': out.push_back('\t'); break;
      case '\\': out.push_back('\\'); break;
      default:
        throw std::runtime_error(where + ": unknown escape \\" + std::string(1, s[i]));
    }
  }
  return out;
}

}  // namespace

std::vector<Run> load_runs(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("runs: cannot open " + path +
                             " (run scripts/export_runs.py)");
  }

  std::vector<Run> runs;
  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (line.empty() || line[0] == '#') continue;
    const std::string where = path + ":" + std::to_string(lineno);

    const size_t t1 = line.find('\t');
    const size_t t2 = t1 == std::string::npos ? t1 : line.find('\t', t1 + 1);
    if (t2 == std::string::npos) throw std::runtime_error(where + ": expected 3 fields");

    Run run;
    run.name = line.substr(0, t1);
    run.prompt = unescape(line.substr(t1 + 1, t2 - t1 - 1), where);

    std::istringstream ids(line.substr(t2 + 1));
    std::string tok;
    while (std::getline(ids, tok, ',')) {
      if (tok.empty()) continue;
      try {
        run.input_ids.push_back(static_cast<int32_t>(std::stol(tok)));
      } catch (const std::exception&) {
        throw std::runtime_error(where + ": bad token id " + tok);
      }
    }
    if (run.input_ids.empty()) throw std::runtime_error(where + ": no token ids");
    runs.push_back(std::move(run));
  }
  if (runs.empty()) throw std::runtime_error(path + ": no runs");
  return runs;
}
}  // namespace gpt2
