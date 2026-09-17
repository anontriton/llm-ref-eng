# engine/ -- C++ inference engine

C++17, no external dependencies in the core.

- `include/gpt2/`  public headers (tensor, model config, ops, backend)
- `src/`           implementation, one translation unit per concern
- `tests/`         unit tests for kernels and weight loading

The engine dumps its own activations in the oracle layout so `oracle/compare.py`
can diff them against the PyTorch reference. The engine does NOT read `.npy` --
it only writes it. Comparison lives in Python.

SIMD sits behind a backend abstraction (`include/gpt2/backend/`): a scalar
backend for correctness, AVX2 for Phase 3, wasm_simd128 for Phase 5. Kernel
logic never contains raw intrinsics.

Build (once cmake is installed):

    cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=RelWithDebInfo
    cmake --build engine/build -j

Phase: 2 onward.
