// SHA-256, because the dump manifest has to carry a checksum per tensor and the
// engine core takes no external dependencies.
//
// oracle/compare.py hashes the tensor's raw contiguous float32 bytes -- not the
// .npy file around them -- so that is what Sha256::update() is fed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace gpt2 {

class Sha256 {
 public:
  Sha256();
  void update(const void* data, size_t len);
  std::string hex();  // finalizes; the object must not be reused afterwards

  static std::string hex_of(const void* data, size_t len);

 private:
  void compress(const unsigned char* block);

  uint32_t state_[8];
  uint64_t bit_len_ = 0;
  unsigned char buf_[64];
  size_t buf_len_ = 0;
};

}  // namespace gpt2
