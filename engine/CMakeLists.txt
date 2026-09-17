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

file(GLOB GPT2_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

if(GPT2_SOURCES)
  add_library(gpt2 ${GPT2_SOURCES})
  target_include_directories(gpt2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

  enable_testing()
  add_subdirectory(tests)
else()
  message(STATUS "gpt2: no sources in src/ yet (skeleton) -- nothing to build.")
endif()
