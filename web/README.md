# web/ -- Emscripten build and demo

Compiles the same `engine/` sources to WebAssembly. No engine fork: if the web
build needs a change, it goes into the engine behind the backend abstraction.

- `build.sh`    emcc invocation
- `demo/`       demo page (tokenizer + generation UI)

The wasm_simd128 backend slots in where AVX2 sits on native, which is why the
abstraction targets 128-bit lanes.

Oracle validation still applies: the wasm build is compared against the same
PyTorch dumps, run in Node.

Phase: 5. Requires the Emscripten SDK (`emcc`), not installed yet.
