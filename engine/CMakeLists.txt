cmake_minimum_required(VERSION 3.16)
project(gpt2_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Determinism beats speed until Phase 3: no fast-math, no reassociation.
add_compile_options(-Wall -Wextra -fno-fast-math)

# Which SIMD backend to build. Exactly one backend_*.cpp is compiled -- they
# define the same symbols -- and the kernels never name the choice, they just
# call gpt2::backend::*.
set(GPT2_BACKEND "scalar" CACHE STRING "SIMD backend: scalar or avx2")
set_property(CACHE GPT2_BACKEND PROPERTY STRINGS scalar avx2)

file(GLOB GPT2_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(FILTER GPT2_SOURCES EXCLUDE REGEX "/backend_[^/]*\\.cpp$")

set(GPT2_BACKEND_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/backend_${GPT2_BACKEND}.cpp")
if(NOT EXISTS "${GPT2_BACKEND_SOURCE}")
  message(FATAL_ERROR "GPT2_BACKEND=${GPT2_BACKEND}: no ${GPT2_BACKEND_SOURCE}")
endif()
list(APPEND GPT2_SOURCES "${GPT2_BACKEND_SOURCE}")

# Only the backend gets the ISA flags. The kernels stay baseline, which is what
# keeps SIMD behind the abstraction rather than spread through ops.cpp.
#
# -mavx2 without -mfma, plus -ffp-contract=off: a fused multiply-add rounds
# once where the scalar backend rounds twice, so allowing it -- whether written
# as an intrinsic or contracted by the compiler -- would make the two backends
# disagree in the last bits. They are required to agree.
if(GPT2_BACKEND STREQUAL "avx2")
  set_source_files_properties("${GPT2_BACKEND_SOURCE}" PROPERTIES
    COMPILE_OPTIONS "-mavx2;-ffp-contract=off")
endif()

message(STATUS "gpt2: backend ${GPT2_BACKEND}")

if(GPT2_SOURCES)
  add_library(gpt2 ${GPT2_SOURCES})
  target_include_directories(gpt2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

  add_subdirectory(tools)

  enable_testing()
  add_subdirectory(tests)
else()
  message(STATUS "gpt2: no sources in src/ yet (skeleton) -- nothing to build.")
endif()
