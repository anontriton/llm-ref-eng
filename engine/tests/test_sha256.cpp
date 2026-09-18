// SHA-256 against the standard published vectors.
//
// The dump manifest's per-tensor checksum is what lets oracle/compare.py detect
// a stale dump, so a wrong hash here would disable a real safety net.
#include <string>
#include <vector>

#include "check.h"
#include "gpt2/sha256.h"

int main() {
  using gpt2::Sha256;

  check::equal(Sha256::hex_of("", 0),
               "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
               "empty string");
  check::equal(Sha256::hex_of("abc", 3),
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
               "\"abc\"");
  check::equal(
      Sha256::hex_of("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 56),
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
      "56-byte message (spans the padding boundary)");

  // A million 'a' -- exercises the streaming path over many blocks.
  {
    Sha256 h;
    const std::string chunk(1000, 'a');
    for (int i = 0; i < 1000; ++i) h.update(chunk.data(), chunk.size());
    check::equal(h.hex(),
                 "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
                 "one million 'a'");
  }

  // Streaming in awkward pieces must equal hashing in one go: the dumper feeds
  // whole tensors, but the buffering logic is where an off-by-one would hide.
  {
    std::vector<unsigned char> data(1000);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<unsigned char>(i * 7);
    const std::string all = Sha256::hex_of(data.data(), data.size());
    for (size_t split : {size_t{1}, size_t{63}, size_t{64}, size_t{65}, size_t{500}}) {
      Sha256 h;
      h.update(data.data(), split);
      h.update(data.data() + split, data.size() - split);
      check::equal(h.hex(), all, "split at " + std::to_string(split));
    }
  }

  return check::report("test_sha256");
}
