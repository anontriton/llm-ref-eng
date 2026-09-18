// Write a .npy file. Writing only -- reading .npy stays in Python, where the
// comparison lives.
//
// Format: NumPy v1.0. Magic, version, a u16 header length, then a Python dict
// literal padded so the data starts on a 64-byte boundary, then raw
// little-endian data. Only contiguous C-order float32 is supported, which is
// the only thing this engine ever produces.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gpt2::npy {

// Throws std::runtime_error if the file cannot be written.
void write_f32(const std::string& path, const float* data,
               const std::vector<int64_t>& shape);

// The header bytes for a given shape, exposed so a unit test can check them
// against what NumPy itself emits without needing a reader.
std::string header_f32(const std::vector<int64_t>& shape);

}  // namespace gpt2::npy
