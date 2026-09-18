// The .npy header the engine writes must be the one NumPy expects.
//
// The engine never reads .npy -- comparison lives in Python -- so a malformed
// header would only surface as an unreadable dump. These are the exact bytes
// numpy.lib.format writes for the same shapes.
#include <string>

#include "check.h"
#include "gpt2/npy.h"

int main() {
  using gpt2::npy::header_f32;

  const std::string h = header_f32({1, 5, 768});
  check::equal(h.find("{'descr': '<f4', 'fortran_order': False, 'shape': "), 0,
               "dict prefix");
  check::ok(h.find("(1, 5, 768), }") != std::string::npos, "shape tuple");
  check::equal(static_cast<long long>((10 + h.size()) % 64), 0,
               "header padded so data starts 64-byte aligned");
  check::equal(static_cast<long long>(h.back()), static_cast<long long>('\n'),
               "header ends with newline");

  // NumPy spells a one-element tuple "(5,)", not "(5, )".
  const std::string one = header_f32({5});
  check::ok(one.find("(5,), }") != std::string::npos, "1-D shape tuple is (5,)");
  check::equal(static_cast<long long>((10 + one.size()) % 64), 0,
               "1-D header alignment");

  const std::string four = header_f32({1, 12, 90, 90});
  check::ok(four.find("(1, 12, 90, 90), }") != std::string::npos,
            "4-D shape tuple (attention scores)");
  check::equal(static_cast<long long>((10 + four.size()) % 64), 0,
               "4-D header alignment");

  return check::report("test_npy");
}
