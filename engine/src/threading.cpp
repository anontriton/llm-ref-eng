#include "gpt2/threading.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace gpt2::threads {
namespace {

class Pool {
 public:
  explicit Pool(int workers) {
    workers_.reserve(static_cast<size_t>(workers));
    for (int i = 0; i < workers; ++i) {
      workers_.emplace_back([this] { loop(); });
    }
  }

  ~Pool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++epoch_;
    }
    start_.notify_all();
    for (std::thread& t : workers_) t.join();
  }

  void run(int n, const std::function<void(int)>& body) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      body_ = &body;
      items_ = n;
      next_.store(0, std::memory_order_relaxed);
      pending_ = static_cast<int>(workers_.size());
      ++epoch_;
    }
    start_.notify_all();

    // The calling thread is one of the workers; it would otherwise sit idle
    // waiting for the others.
    drain(body, n);

    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return pending_ == 0; });
    body_ = nullptr;
  }

 private:
  // Claim indices until they run out. Dynamic rather than a fixed split, so an
  // item that happens to be slower does not leave a core idle. It does not
  // affect the result: the items are disjoint.
  void drain(const std::function<void(int)>& body, int n) {
    for (;;) {
      const int i = next_.fetch_add(1, std::memory_order_relaxed);
      if (i >= n) return;
      body(i);
    }
  }

  void loop() {
    uint64_t seen = 0;
    for (;;) {
      const std::function<void(int)>* body = nullptr;
      int n = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        start_.wait(lock, [this, seen] { return stop_ || epoch_ != seen; });
        if (stop_) return;
        seen = epoch_;
        body = body_;
        n = items_;
      }

      if (body != nullptr) drain(*body, n);

      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--pending_ == 0) done_.notify_one();
      }
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable start_;
  std::condition_variable done_;
  const std::function<void(int)>* body_ = nullptr;
  std::atomic<int> next_{0};
  int items_ = 0;
  int pending_ = 0;
  uint64_t epoch_ = 0;
  bool stop_ = false;
};

int g_count = 1;
Pool* g_pool = nullptr;

}  // namespace

int hardware_count() {
  const unsigned n = std::thread::hardware_concurrency();
  return n == 0 ? 1 : static_cast<int>(n);
}

void set_count(int n) {
  n = std::max(1, n);
  if (n == g_count) return;
  delete g_pool;
  g_pool = nullptr;
  g_count = n;
  // n includes the calling thread, so the pool holds n - 1.
  if (n > 1) g_pool = new Pool(n - 1);
}

int count() { return g_count; }

void parallel_for(int n, const std::function<void(int)>& body) {
  if (n <= 0) return;
  if (g_count == 1 || g_pool == nullptr || n == 1) {
    for (int i = 0; i < n; ++i) body(i);
    return;
  }
  g_pool->run(n, body);
}

}  // namespace gpt2::threads
