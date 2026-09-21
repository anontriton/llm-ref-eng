# Phase 2: the dump driver. Its output is what oracle/compare.py judges.
add_executable(gpt2_dump gpt2_dump.cpp)
target_link_libraries(gpt2_dump PRIVATE gpt2)

# Greedy decoding, for the top-1-over-50-steps criterion.
add_executable(gpt2_generate gpt2_generate.cpp)
target_link_libraries(gpt2_generate PRIVATE gpt2)

# Phase 3's baseline: prefill and decode timings. bench/run.py wraps it.
add_executable(gpt2_bench gpt2_bench.cpp)
target_link_libraries(gpt2_bench PRIVATE gpt2)
