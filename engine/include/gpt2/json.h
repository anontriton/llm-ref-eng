// A minimal JSON writer -- writer only, no parser.
//
// The engine emits one document, the dump manifest, and reads none: run
// definitions arrive as TSV and weights as a flat binary precisely so that no
// JSON parser has to exist in the engine core.
#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace gpt2 {

class JsonWriter {
 public:
  explicit JsonWriter(std::ostream& out) : out_(out) {}

  void begin_object();
  void end_object();
  void begin_array();
  void end_array();

  void key(const std::string& k);
  void str(const std::string& v);
  void num(long long v);
  void num(double v);

  void kv(const std::string& k, const std::string& v) { key(k); str(v); }
  void kv(const std::string& k, long long v) { key(k); num(v); }
  void kv(const std::string& k, int v) { key(k); num(static_cast<long long>(v)); }
  void kv(const std::string& k, double v) { key(k); num(v); }

  // Shortest decimal form that reads back as the same double. The manifest's
  // numbers are compared for equality by oracle/compare.py, so a lossy
  // %g would be a provenance failure waiting to happen.
  static std::string number_to_string(double v);
  static std::string escape(const std::string& s);

 private:
  void prepare_value();
  void indent();

  std::ostream& out_;
  std::vector<bool> first_;   // per open container: is the next item the first?
  bool after_key_ = false;
};

}  // namespace gpt2
