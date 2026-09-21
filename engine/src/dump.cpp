#include "gpt2/dump.h"

#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <set>
#include <stdexcept>

#include "gpt2/backend/backend.h"
#include "gpt2/json.h"
#include "gpt2/npy.h"
#include "gpt2/sha256.h"

namespace gpt2 {
namespace {

namespace fs = std::filesystem;

std::string utc_now() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S+00:00", &tm);
  return std::string(buf);
}

}  // namespace

Dumper::Dumper(std::string out_dir, const Weights& weights)
    : out_dir_(std::move(out_dir)), weights_(weights) {
  fs::create_directories(fs::path(out_dir_) / "activations");
}

Tap Dumper::begin(const Run& run, int kv_row) {
  if (open_) throw std::runtime_error("dump: previous run was not ended");
  current_ = RunDump{run, kv_row, {}};
  open_ = true;

  const fs::path dir = fs::path(out_dir_) / "activations" / run.name;
  fs::create_directories(dir);
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() == ".npy") fs::remove(entry.path());
  }

  // Captured by reference: the tap outlives nothing, it is consumed inside the
  // forward call that begin()/end() bracket.
  auto seen = std::make_shared<std::set<std::string>>();
  return [this, dir, seen](const std::string& name, const float* data,
                           const std::vector<int64_t>& shape) {
    if (!seen->insert(name).second) {
      throw std::runtime_error("dump: tensor " + name + " tapped twice in one forward");
    }

    const int64_t n = std::accumulate(shape.begin(), shape.end(), int64_t{1},
                                      std::multiplies<int64_t>());

    Record rec;
    rec.order = static_cast<int>(current_.tensors.size());
    rec.name = name;
    rec.file = name + ".npy";
    rec.shape = shape;
    // oracle/compare.py hashes the raw contiguous float32 bytes, not the .npy
    // container, so that is exactly what goes in here.
    rec.sha256 = Sha256::hex_of(data, static_cast<size_t>(n) * sizeof(float));

    double mn = data[0], mx = data[0], am = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      const double v = data[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      const double a = std::fabs(v);
      if (a > am) am = a;
    }
    rec.min = mn;
    rec.max = mx;
    rec.absmax = am;

    // Raw attention scores are only meaningful below the diagonal; the manifest
    // carries the rule so the comparison needs no special case.
    const std::string suffix = ".attn.scores";
    rec.causal_region = name.size() >= suffix.size() &&
                        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;

    npy::write_f32((dir / rec.file).string(), data, shape);
    current_.tensors.push_back(std::move(rec));
    ++total_tensors_;
  };
}

void Dumper::end() {
  if (!open_) throw std::runtime_error("dump: end() without begin()");
  runs_.push_back(std::move(current_));
  current_ = RunDump{};
  open_ = false;
}

void Dumper::write_manifest() const {
  const fs::path path = fs::path(out_dir_) / "manifest.json";
  std::ofstream out(path);
  if (!out) throw std::runtime_error("dump: cannot write " + path.string());

  const Config& cfg = weights_.config();
  JsonWriter j(out);

  j.begin_object();
  j.kv("schema", 1);
  j.kv("producer", std::string("engine/tools/gpt2_dump"));
  j.kv("created", utc_now());

  j.key("model");
  j.begin_object();
  j.kv("name", std::string("gpt2-124m"));
  j.kv("weights_sha256", weights_.source_sha256());
  j.key("config");
  j.begin_object();
  j.kv("n_layer", cfg.n_layer);
  j.kv("n_head", cfg.n_head);
  j.kv("d_model", cfg.d_model);
  j.kv("d_ff", cfg.d_ff);
  j.kv("n_ctx", cfg.n_ctx);
  j.kv("vocab_size", cfg.vocab_size);
  j.kv("layer_norm_eps", cfg.layer_norm_eps);
  j.end_object();
  j.end_object();

  j.kv("dtype", std::string("float32"));

  // The CLAUDE.md fp32 policy, echoed so the dump carries the rules it was
  // meant to be judged by. Declaring anything else here voids layer-wise
  // comparison, which is the intended behaviour once Phase 4 quantizes.
  j.key("tolerance");
  j.begin_object();
  j.kv("max_abs", 1e-4);
  j.kv("max_rel", 1e-3);
  j.kv("rel_floor", 1e-2);
  j.kv("policy", std::string("fp32"));
  j.end_object();

  j.key("environment");
  j.begin_object();
  j.kv("engine", std::string("gpt2 c++ engine"));
  j.kv("backend", std::string(backend::name()));
  j.kv("cxx", std::string(
#if defined(__clang__)
      "clang " __clang_version__
#elif defined(__GNUC__)
      "gcc " __VERSION__
#else
      "unknown"
#endif
      ));
  j.kv("threads", 1);
  j.end_object();

  j.key("runs");
  j.begin_array();
  for (const RunDump& rd : runs_) {
    j.begin_object();
    j.kv("name", rd.run.name);
    j.kv("prompt", rd.run.prompt);
    j.key("input_ids");
    j.begin_array();
    for (int32_t id : rd.run.input_ids) j.num(static_cast<long long>(id));
    j.end_array();
    j.kv("n_tokens", static_cast<long long>(rd.run.input_ids.size()));
    // Present only on a cached dump: the absolute position these tensors
    // hold. compare.py slices the reference at this row.
    if (rd.kv_row >= 0) j.kv("kv_row", static_cast<long long>(rd.kv_row));
    j.kv("dir", "activations/" + rd.run.name);
    j.key("tensors");
    j.begin_array();
    for (const Record& r : rd.tensors) {
      j.begin_object();
      j.kv("order", r.order);
      j.kv("name", r.name);
      j.kv("file", r.file);
      j.key("shape");
      j.begin_array();
      for (int64_t d : r.shape) j.num(static_cast<long long>(d));
      j.end_array();
      j.kv("dtype", std::string("float32"));
      j.kv("sha256", r.sha256);
      j.kv("min", r.min);
      j.kv("max", r.max);
      j.kv("absmax", r.absmax);
      if (r.causal_region) j.kv("region", std::string("causal_lower_triangle"));
      j.end_object();
    }
    j.end_array();
    j.end_object();
  }
  j.end_array();
  j.end_object();
  out << "\n";

  if (!out) throw std::runtime_error("dump: short write to " + path.string());
}

}  // namespace gpt2
