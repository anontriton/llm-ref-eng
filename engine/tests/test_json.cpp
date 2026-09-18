// The manifest writer.
//
// oracle/compare.py parses this output and compares numbers for equality, so
// escaping and round-tripping are correctness concerns, not cosmetics. The
// `newline` run's prompt is a bare "\n", which would break a naive writer.
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string>

#include "check.h"
#include "gpt2/json.h"

int main() {
  using gpt2::JsonWriter;

  check::equal(JsonWriter::escape("plain"), "\"plain\"", "plain string");
  check::equal(JsonWriter::escape("\n"), "\"\\n\"", "bare newline (the `newline` run)");
  check::equal(JsonWriter::escape("a\"b\\c"), "\"a\\\"b\\\\c\"", "quote and backslash");
  check::equal(JsonWriter::escape("\t\r"), "\"\\t\\r\"", "tab and carriage return");
  check::equal(JsonWriter::escape("\x01"), "\"\\u0001\"", "control character");
  // UTF-8 passes through untouched rather than being escaped to \u.
  check::equal(JsonWriter::escape("caf\xc3\xa9"), "\"caf\xc3\xa9\"", "utf-8 byte pass-through");

  // Numbers must read back as the same double. 1e-5 is the live case:
  // layer_norm_eps is compared exactly against the reference manifest.
  check::equal(JsonWriter::number_to_string(1e-5), "1e-05", "layer_norm_eps stays 1e-05");
  check::equal(JsonWriter::number_to_string(0.0001), "0.0001", "max_abs tolerance");
  check::equal(JsonWriter::number_to_string(0.0), "0", "zero");
  for (double v : {1e-5, 1e-4, 1e-3, 0.1, 1.0 / 3.0, 3.14159265358979,
                   -2317.253662109375, 1.0000000116860974e-05}) {
    check::ok(std::strtod(JsonWriter::number_to_string(v).c_str(), nullptr) == v,
              "round-trips: " + JsonWriter::number_to_string(v));
  }

  // A nested document, to check commas and nesting come out parseable.
  std::ostringstream out;
  {
    JsonWriter j(out);
    j.begin_object();
    j.kv("schema", 1);
    j.key("runs");
    j.begin_array();
    j.begin_object();
    j.kv("name", std::string("newline"));
    j.kv("prompt", std::string("\n"));
    j.key("input_ids");
    j.begin_array();
    j.num(198LL);
    j.end_array();
    j.end_object();
    j.end_array();
    j.key("empty_object");
    j.begin_object();
    j.end_object();
    j.key("empty_array");
    j.begin_array();
    j.end_array();
    j.end_object();
  }
  const std::string doc = out.str();
  check::ok(doc.find("\"prompt\": \"\\n\"") != std::string::npos,
            "escaped prompt appears in the document");
  // Elements are indented one per line, the same shape json.dumps(indent=2)
  // gives the reference manifest -- so the two are diffable side by side.
  check::ok(doc.find("\"input_ids\": [") != std::string::npos, "array opens after its key");
  check::ok(doc.find("198") != std::string::npos, "array element present");
  check::ok(doc.find("\"empty_object\": {}") != std::string::npos, "empty object");
  check::ok(doc.find("\"empty_array\": []") != std::string::npos, "empty array");
  check::ok(doc.find(",,") == std::string::npos, "no doubled commas");
  check::equal(static_cast<long long>(std::count(doc.begin(), doc.end(), '{')),
               static_cast<long long>(std::count(doc.begin(), doc.end(), '}')),
               "braces balance");

  return check::report("test_json");
}
