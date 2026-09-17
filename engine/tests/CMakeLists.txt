# Phase 2: one executable per test file, registered with ctest.
file(GLOB GPT2_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(test_src ${GPT2_TEST_SOURCES})
  get_filename_component(test_name "${test_src}" NAME_WE)
  add_executable(${test_name} "${test_src}")
  target_link_libraries(${test_name} PRIVATE gpt2)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
