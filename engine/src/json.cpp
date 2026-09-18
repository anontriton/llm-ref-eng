#include "gpt2/json.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gpt2 {

std::string JsonWriter::escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));  // UTF-8 passes through
        }
    }
  }
  out.push_back('"');
  return out;
}

std::string JsonWriter::number_to_string(double v) {
  char buf[64];
  // Walk up in precision and stop at the first form that parses back exactly.
  // 17 significant digits always round-trips a double; most values need far
  // fewer, and the short form keeps the manifest readable.
  for (int prec = 6; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
    if (std::strtod(buf, nullptr) == v) break;
  }
  return std::string(buf);
}

void JsonWriter::indent() {
  out_ << "\n";
  for (size_t i = 0; i < first_.size(); ++i) out_ << "  ";
}

void JsonWriter::prepare_value() {
  if (after_key_) {
    after_key_ = false;
    return;
  }
  if (!first_.empty()) {
    if (!first_.back()) out_ << ",";
    first_.back() = false;
    indent();
  }
}

void JsonWriter::begin_object() {
  prepare_value();
  out_ << "{";
  first_.push_back(true);
}

void JsonWriter::end_object() {
  const bool empty = first_.back();
  first_.pop_back();
  if (!empty) indent();
  out_ << "}";
}

void JsonWriter::begin_array() {
  prepare_value();
  out_ << "[";
  first_.push_back(true);
}

void JsonWriter::end_array() {
  const bool empty = first_.back();
  first_.pop_back();
  if (!empty) indent();
  out_ << "]";
}

void JsonWriter::key(const std::string& k) {
  prepare_value();
  out_ << escape(k) << ": ";
  after_key_ = true;
}

void JsonWriter::str(const std::string& v) {
  prepare_value();
  out_ << escape(v);
}

void JsonWriter::num(long long v) {
  prepare_value();
  out_ << v;
}

void JsonWriter::num(double v) {
  prepare_value();
  out_ << number_to_string(v);
}

}  // namespace gpt2
