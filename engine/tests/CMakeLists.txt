# Phase 2: one executable per test file, registered with ctest.
file(GLOB GPT2_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(test_src ${GPT2_TEST_SOURCES})
  get_filename_component(test_name "${test_src}" NAME_WE)
  add_executable(${test_name} "${test_src}")
  target_link_libraries(${test_name} PRIVATE gpt2)
  # Tests that need generated artifacts (the converted weight file) locate
  # them from here rather than depending on the working directory.
  target_compile_definitions(${test_name} PRIVATE
    GPT2_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../..")
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Without -pthread, Emscripten's std::thread constructor throws, so the pool
# cannot exist and there is nothing for test_threading to test. Disabled rather
# than dropped, so ctest still lists it -- as not run -- until the wasm build
# gets threads.
if(EMSCRIPTEN)
  set_tests_properties(test_threading PROPERTIES DISABLED TRUE)
endif()
