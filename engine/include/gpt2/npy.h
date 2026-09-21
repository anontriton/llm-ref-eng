// Write a .npy file. Writing only -- reading .npy stays in Python, where the
// comparison lives.
//
// Format: NumPy v1.0. Magic, version, a u16 header length, then a Python dict
// literal padded so the data starts on a 64-byte boundary, then raw
// little-endian data. Contiguous C-order only.
//
// float32 for activations, int32 for token ids -- Phase 4's eval dumps a
// top-1 id per position, and writing those as floats because they happen to
// fit would be the kind of cleverness that is wrong the first time a vocab
// crosses 2^24.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gpt2::npy {

// Throws std::runtime_error if the file cannot be written.
void write_f32(const std::string& path, const float* data,
               const std::vector<int64_t>& shape);
void write_i32(const std::string& path, const int32_t* data,
               const std::vector<int64_t>& shape);

// The header bytes for a given dtype and shape, exposed so a unit test can
// check them against what NumPy itself emits without needing a reader.
std::string header(const char* descr, const std::vector<int64_t>& shape);
inline std::string header_f32(const std::vector<int64_t>& shape) {
  return header("<f4", shape);
}

}  // namespace gpt2::npy
