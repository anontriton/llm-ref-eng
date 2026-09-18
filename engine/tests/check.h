// A test harness small enough to not be a dependency.
//
// The engine core takes no external dependencies, and its tests inherit that:
// one executable per file, registered with ctest, failing with a non-zero exit
// and a message that says what was expected.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace check {

inline int failures = 0;
inline int checks = 0;

inline void ok(bool cond, const std::string& what) {
  ++checks;
  if (!cond) {
    ++failures;
    std::printf("  FAIL  %s\n", what.c_str());
  }
}

inline void close(double got, double want, double tol, const std::string& what) {
  ++checks;
  const double d = std::fabs(got - want);
  if (!(d <= tol)) {
    ++failures;
    std::printf("  FAIL  %s: got %.10g, want %.10g (diff %.3e > %.3e)\n",
                what.c_str(), got, want, d, tol);
  }
}

inline void equal(long long got, long long want, const std::string& what) {
  ++checks;
  if (got != want) {
    ++failures;
    std::printf("  FAIL  %s: got %lld, want %lld\n", what.c_str(), got, want);
  }
}

inline void equal(const std::string& got, const std::string& want,
                  const std::string& what) {
  ++checks;
  if (got != want) {
    ++failures;
    std::printf("  FAIL  %s:\n        got  %s\n        want %s\n",
                what.c_str(), got.c_str(), want.c_str());
  }
}

inline int report(const char* name) {
  if (failures == 0) {
    std::printf("%s: %d checks passed\n", name, checks);
    return 0;
  }
  std::printf("%s: %d of %d checks FAILED\n", name, failures, checks);
  return 1;
}

}  // namespace check
