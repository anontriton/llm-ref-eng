// The thread pool's contract, which is narrower than "it runs things".
//
// The engine's claim is that thread count is not a numerical parameter: eight
// threads produce the same bits as one. oracle/compare.py proves that for the
// whole model; this pins the two properties underneath it -- every work item
// runs exactly once, and a computation split across items does not depend on
// how many threads ran them.
#include <atomic>
#include <numeric>
#include <string>
#include <vector>

#include "check.h"
#include "gpt2/threading.h"

namespace {

void test_default_is_serial() {
  check::equal(gpt2::threads::count(), 1, "serial until asked otherwise");
}

void test_every_item_runs_exactly_once() {
  for (int workers : {1, 2, 3, 8, 13}) {
    gpt2::threads::set_count(workers);
    const int n = 1000;
    std::vector<std::atomic<int>> seen(n);
    for (auto& s : seen) s.store(0);

    gpt2::threads::parallel_for(n, [&](int i) {
      seen[i].fetch_add(1);
    });

    int wrong = 0;
    for (int i = 0; i < n; ++i) {
      if (seen[i].load() != 1) ++wrong;
    }
    check::equal(wrong, 0,
                 "every item ran exactly once at " + std::to_string(workers) +
                     " threads");
  }
}

// Disjoint output, so the schedule cannot reach the result. This is the shape
// every parallel site in the engine uses: each item owns a slice and nothing
// is reduced across items.
void test_result_does_not_depend_on_thread_count() {
  const int n = 977;  // prime, so no thread count divides it evenly
  std::vector<float> reference(n);

  for (int workers : {1, 2, 3, 8, 13}) {
    gpt2::threads::set_count(workers);
    std::vector<float> out(n, 0.0f);

    gpt2::threads::parallel_for(n, [&](int i) {
      float acc = 0.0f;
      for (int k = 0; k < 64; ++k) {
        acc += static_cast<float>(i + 1) * 0.1f / static_cast<float>(k + 3);
      }
      out[static_cast<size_t>(i)] = acc;
    });

    if (workers == 1) {
      reference = out;
      continue;
    }
    int differing = 0;
    for (int i = 0; i < n; ++i) {
      if (out[static_cast<size_t>(i)] != reference[static_cast<size_t>(i)]) {
        ++differing;
      }
    }
    check::equal(differing, 0,
                 "bit-identical to the serial run at " +
                     std::to_string(workers) + " threads");
  }
}

void test_edges() {
  gpt2::threads::set_count(4);

  int ran = 0;
  gpt2::threads::parallel_for(0, [&](int) { ++ran; });
  check::equal(ran, 0, "n == 0 runs nothing");

  gpt2::threads::parallel_for(1, [&](int i) { ran += i + 1; });
  check::equal(ran, 1, "n == 1 runs once");

  gpt2::threads::set_count(0);
  check::equal(gpt2::threads::count(), 1, "count is clamped to at least 1");

  gpt2::threads::set_count(-5);
  check::equal(gpt2::threads::count(), 1, "negative count is clamped too");
}

// The pool is rebuilt when the count changes; doing that repeatedly should not
// leak threads or deadlock.
void test_resize_is_safe() {
  for (int workers : {1, 8, 2, 8, 1, 4}) {
    gpt2::threads::set_count(workers);
    std::atomic<int> total{0};
    gpt2::threads::parallel_for(256, [&](int i) { total.fetch_add(i); });
    check::equal(total.load(), 255 * 256 / 2,
                 "sum after resizing to " + std::to_string(workers));
  }
  gpt2::threads::set_count(1);
}

}  // namespace

int main() {
  test_default_is_serial();
  test_every_item_runs_exactly_once();
  test_result_does_not_depend_on_thread_count();
  test_edges();
  test_resize_is_safe();
  return check::report("test_threading");
}
