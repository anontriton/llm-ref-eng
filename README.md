# GPT-2 124M Inference Engine

A from-scratch GPT-2 124M inference engine, built in three stages -- a
hand-written PyTorch reference, a C++ engine, and a WebAssembly build -- where
the C++ engine is continuously validated layer-by-layer against the PyTorch
reference.

See [CLAUDE.md](CLAUDE.md) for project invariants, tolerance policy, and phase
order. Each directory has its own README describing what lives there.

    reference/   hand-written PyTorch GPT-2 + activation dumping
    engine/      C++ engine (src/, include/gpt2/, tests/)
    oracle/      dumped reference activations + manifest + compare.py
    bench/       benchmark harness + committed, commit-tagged results
    web/         Emscripten build + demo page
    scripts/     weight download, format conversion

Status: skeleton only. Phase 0 (foundations) has not started.
