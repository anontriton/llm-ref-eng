// A fixed pool and a parallel_for, and nothing else.
//
// Phase 3's last item. The engine takes no external dependencies, so this is
// std::thread and a condition variable rather than a library.
//
// Two properties matter more than speed:
//
// Determinism. Work items are disjoint -- each one owns a tile of the output
// and no two threads touch the same element -- so which thread runs which item,
// and in what order, cannot change the result. Thread count is therefore not a
// numerical parameter: 8 threads must produce the same bits as 1. That is not
// automatic for a parallel GEMM, and it is why nothing here splits a reduction.
//
// A pool rather than threads per call. linear() is called around fifty times
// per forward pass; spawning threads each time would cost more than the work.
//
// Serial by default. count() is 1 until something sets it, and at 1
// parallel_for runs inline on the calling thread with no pool, no locks, and
// no threads created -- byte for byte the path the engine had before this
// existed.
#pragma once

#include <functional>

namespace gpt2::threads {

// Worker count, including the calling thread. Clamped to at least 1.
// Setting it is not thread-safe and is meant to happen once, at startup,
// before any forward pass.
void set_count(int n);
int count();

// hardware_concurrency(), or 1 if that is unknown.
int hardware_count();

// Runs body(i) for every i in [0, n), and returns once all of them have
// finished. At count() == 1 this is a plain loop on the calling thread.
//
// `body` must not throw and must not touch state another item touches; that
// disjointness is what makes the result independent of the schedule.
void parallel_for(int n, const std::function<void(int)>& body);

}  // namespace gpt2::threads
