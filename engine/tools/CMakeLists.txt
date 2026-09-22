# Phase 2: the dump driver. Its output is what oracle/compare.py judges.
add_executable(gpt2_dump gpt2_dump.cpp)
target_link_libraries(gpt2_dump PRIVATE gpt2)

# Greedy decoding, for the top-1-over-50-steps criterion.
add_executable(gpt2_generate gpt2_generate.cpp)
target_link_libraries(gpt2_generate PRIVATE gpt2)

# Phase 3's baseline: prefill and decode timings. bench/run.py wraps it.
add_executable(gpt2_bench gpt2_bench.cpp)
target_link_libraries(gpt2_bench PRIVATE gpt2)

# Phase 4: perplexity / top-1 / KL dumps. eval/metrics.py judges them.
add_executable(gpt2_eval gpt2_eval.cpp)
target_link_libraries(gpt2_eval PRIVATE gpt2)

# Per-shape matmul timings, for deciding what to change. bench/ says whether
# the engine got faster; this says which kernel did.
add_executable(gpt2_kernelbench gpt2_kernelbench.cpp)
target_link_libraries(gpt2_kernelbench PRIVATE gpt2)
