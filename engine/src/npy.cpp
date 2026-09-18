#include "gpt2/npy.h"

#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace gpt2::npy {

std::string header_f32(const std::vector<int64_t>& shape) {
  std::string dims;
  for (size_t i = 0; i < shape.size(); ++i) {
    dims += std::to_string(shape[i]);
    // NumPy writes a one-element tuple as "(5,)"; anything else is comma-space
    // separated. Matching its spelling keeps the files byte-identical to what
    // np.save would have produced, which makes them trivially diffable.
    if (shape.size() == 1 || i + 1 < shape.size()) dims += ", ";
  }
  if (shape.size() == 1) dims.resize(dims.size() - 1);  // "(5, )" -> "(5,)"

  std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                     dims + "), }";

  // Magic(6) + version(2) + header length(2) + dict + '\n' padded to 64.
  const size_t prefix = 10;
  size_t total = prefix + dict.size() + 1;
  const size_t padded = (total + 63) / 64 * 64;
  dict.append(padded - total, ' ');
  dict.push_back('\n');
  return dict;
}

void write_f32(const std::string& path, const float* data,
               const std::vector<int64_t>& shape) {
  const std::string dict = header_f32(shape);
  if (dict.size() > 0xffff) throw std::runtime_error("npy: header too long");

  const int64_t n = std::accumulate(shape.begin(), shape.end(), int64_t{1},
                                    std::multiplies<int64_t>());

  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("npy: cannot write " + path);

  const char magic[8] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
  out.write(magic, sizeof(magic));

  const unsigned char len[2] = {static_cast<unsigned char>(dict.size() & 0xff),
                                static_cast<unsigned char>((dict.size() >> 8) & 0xff)};
  out.write(reinterpret_cast<const char*>(len), 2);
  out.write(dict.data(), static_cast<std::streamsize>(dict.size()));
  out.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(n) * static_cast<std::streamsize>(sizeof(float)));

  if (!out) throw std::runtime_error("npy: short write to " + path);
}

}  // namespace gpt2::npy
