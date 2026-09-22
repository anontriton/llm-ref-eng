#include "gpt2/weights.h"

#include <cstring>
#include <fstream>
#include <sstream>

namespace gpt2 {
namespace {

constexpr char kMagic[8] = {'G', 'P', 'T', '2', 'W', 'T', 'S', '1'};
constexpr uint32_t kVersion = 1;
constexpr size_t kNameField = 64;
constexpr size_t kMaxDims = 4;
constexpr size_t kEntrySize = 128;
constexpr size_t kHeaderFixed = 8 + 16 + 32 + 32;
constexpr uint32_t kDtypeF32 = 0;
constexpr uint32_t kDtypeI8 = 1;
constexpr uint32_t kDtypeI32 = 2;

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("weights: " + what);
}

// Little-endian reads. Spelled out byte by byte rather than memcpy'ing into a
// struct so the format does not depend on this compiler's padding or on the
// host being little-endian.
uint32_t read_u32(const unsigned char* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t read_u64(const unsigned char* p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | p[static_cast<size_t>(i)];
  return v;
}

double read_f64(const unsigned char* p) {
  uint64_t bits = read_u64(p);
  double out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

std::string to_hex(const unsigned char* p, size_t n) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  out.reserve(n * 2);
  for (size_t i = 0; i < n; ++i) {
    out.push_back(digits[p[i] >> 4]);
    out.push_back(digits[p[i] & 0xf]);
  }
  return out;
}

}  // namespace

Weights Weights::load(const std::string& path) {
  Weights w;
  w.path_ = path;

  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) fail("cannot open " + path + " (run scripts/convert_weights.py)");
  const std::streamsize size = in.tellg();
  if (size < static_cast<std::streamsize>(kHeaderFixed)) fail(path + ": file is too small");
  in.seekg(0);
  w.blob_.resize(static_cast<size_t>(size));
  if (!in.read(reinterpret_cast<char*>(w.blob_.data()), size)) fail(path + ": short read");

  const unsigned char* b = w.blob_.data();
  if (std::memcmp(b, kMagic, sizeof(kMagic)) != 0) fail(path + ": bad magic");

  const uint32_t version = read_u32(b + 8);
  if (version != kVersion) {
    std::ostringstream m;
    m << path << ": version " << version << ", expected " << kVersion;
    fail(m.str());
  }
  const uint32_t n_tensors = read_u32(b + 12);
  const uint32_t data_start = read_u32(b + 16);

  w.config_.n_layer = static_cast<int>(read_u32(b + 24));
  w.config_.n_head = static_cast<int>(read_u32(b + 28));
  w.config_.d_model = static_cast<int>(read_u32(b + 32));
  w.config_.d_ff = static_cast<int>(read_u32(b + 36));
  w.config_.n_ctx = static_cast<int>(read_u32(b + 40));
  w.config_.vocab_size = static_cast<int>(read_u32(b + 44));
  w.config_.layer_norm_eps = read_f64(b + 48);
  w.source_sha256_ = to_hex(b + 56, 32);
  w.quant_ = read_u32(b + 20);

  if (w.config_.n_head == 0 || w.config_.d_model % w.config_.n_head != 0) {
    fail(path + ": d_model is not divisible by n_head");
  }

  const size_t dir_end = kHeaderFixed + static_cast<size_t>(n_tensors) * kEntrySize;
  if (dir_end > w.blob_.size() || data_start < dir_end) fail(path + ": truncated directory");

  for (uint32_t i = 0; i < n_tensors; ++i) {
    const unsigned char* e = b + kHeaderFixed + static_cast<size_t>(i) * kEntrySize;
    const char* raw = reinterpret_cast<const char*>(e);
    const size_t name_len = strnlen(raw, kNameField);
    std::string name(raw, name_len);

    const uint32_t ndim = read_u32(e + kNameField);
    const uint32_t dtype = read_u32(e + kNameField + 4);
    if (dtype != kDtypeF32 && dtype != kDtypeI8 && dtype != kDtypeI32) {
      fail(name + ": unsupported dtype");
    }
    if (ndim == 0 || ndim > kMaxDims) fail(name + ": bad ndim");

    WeightView view;
    view.numel = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
      const int64_t dim = static_cast<int64_t>(read_u64(e + kNameField + 8 + 8 * d));
      view.shape.push_back(dim);
      view.numel *= dim;
    }
    const uint64_t offset = read_u64(e + kNameField + 40);
    const uint64_t nbytes = read_u64(e + kNameField + 48);
    const size_t elem = dtype == kDtypeI8 ? 1 : 4;
    if (nbytes != static_cast<uint64_t>(view.numel) * elem) {
      fail(name + ": byte count disagrees with shape");
    }
    const size_t start = data_start + offset;
    if (start + nbytes > w.blob_.size()) fail(name + ": data runs past end of file");

    if (dtype == kDtypeI8) {
      view.qdata = reinterpret_cast<const int8_t*>(w.blob_.data() + start);
    } else if (dtype == kDtypeI32) {
      view.idata = reinterpret_cast<const int32_t*>(w.blob_.data() + start);
    } else {
      view.data = reinterpret_cast<const float*>(w.blob_.data() + start);
    }
    w.views_.emplace(std::move(name), std::move(view));
  }

  w.resolve();
  return w;
}

void Weights::embed(int32_t id, float* dst) const {
  const int d = config_.d_model;
  if (id < 0 || id >= config_.vocab_size) fail("embed: token id out of range");
  const size_t row = static_cast<size_t>(id) * d;
  if (!wte_.quantized()) {
    const float* src = wte_.f32 + row;
    for (int i = 0; i < d; ++i) dst[i] = src[i];
    return;
  }
  // Per-row scale, which is the same scale the tied lm_head uses for this
  // token's output channel -- one number describes the row in both directions.
  const int8_t* q = wte_.i8 + row;
  const float s = wte_.scale[id];
  for (int i = 0; i < d; ++i) dst[i] = static_cast<float>(q[i]) * s;
  // The int8 table holds zeros in the outlier columns; the real values are
  // kept whole.
  const float* o = wte_.outlier + static_cast<size_t>(id) * wte_.n_outlier;
  for (int k = 0; k < wte_.n_outlier; ++k) dst[wte_.outlier_cols[k]] = o[k];
}

// wte's outlier columns, if the file holds them. Every property the kernels
// rely on is checked here rather than assumed there: the columns ascend and
// are in range, the fp32 block is [vocab, k], and the int8 table is exactly
// zero in those columns -- lm_head's int8 dot product runs over all d_model
// terms and counts on those contributing nothing.
void Weights::resolve_wte_outliers() {
  const bool has_cols = has("wte.outlier_cols");
  if (has_cols != has("wte.outlier")) {
    fail("wte.outlier_cols and wte.outlier come together or not at all");
  }
  if (!has_cols) return;
  if (!wte_.quantized()) fail("wte.outlier_cols: wte is float32, nothing to hold back");

  const int d = config_.d_model;
  const WeightView& cv = get("wte.outlier_cols");
  if (cv.idata == nullptr || cv.shape.size() != 1 || cv.numel < 1 || cv.numel > d) {
    fail("wte.outlier_cols: expected int32 [k], 1 <= k <= d_model");
  }
  const int k = static_cast<int>(cv.numel);
  for (int i = 0; i < k; ++i) {
    const int32_t c = cv.idata[i];
    if (c < 0 || c >= d) fail("wte.outlier_cols: column out of range");
    if (i > 0 && c <= cv.idata[i - 1]) fail("wte.outlier_cols: not strictly ascending");
  }
  const WeightView& ov = get("wte.outlier");
  if (ov.data == nullptr ||
      ov.shape != std::vector<int64_t>{config_.vocab_size, k}) {
    fail("wte.outlier: expected float32 [vocab_size, k]");
  }
  for (int64_t j = 0; j < config_.vocab_size; ++j) {
    const int8_t* q = wte_.i8 + static_cast<size_t>(j) * d;
    for (int i = 0; i < k; ++i) {
      if (q[cv.idata[i]] != 0) fail("wte: nonzero int8 value in an outlier column");
    }
  }
  wte_.outlier_cols = cv.idata;
  wte_.outlier = ov.data;
  wte_.n_outlier = k;
}

const WeightView& Weights::get(const std::string& name) const {
  auto it = views_.find(name);
  if (it == views_.end()) fail("missing tensor " + name);
  return it->second;
}

void Weights::resolve() {
  const int d = config_.d_model;
  const int f = config_.d_ff;

  // Look a tensor up and insist on its shape. The shape check is what catches
  // the HF Conv1D transpose gotcha, which is otherwise invisible for the square
  // [768, 768] projections.
  auto need = [&](const std::string& name, std::vector<int64_t> want) -> const float* {
    const WeightView& v = get(name);
    if (v.shape != want) {
      std::ostringstream m;
      m << name << ": shape [";
      for (size_t i = 0; i < v.shape.size(); ++i) m << (i ? ", " : "") << v.shape[i];
      m << "], expected [";
      for (size_t i = 0; i < want.size(); ++i) m << (i ? ", " : "") << want[i];
      m << "]";
      fail(m.str());
    }
    if (v.data == nullptr) fail(name + ": expected float32");
    return v.data;
  };

  // A matrix, in whichever representation the file carries. When int8 it must
  // come with a scale of exactly one entry per output channel; a missing or
  // mis-sized scale is a corrupt file, not something to work around.
  auto need_matrix = [&](const std::string& name, std::vector<int64_t> want,
                         int64_t channels) -> Matrix {
    const WeightView& v = get(name);
    if (v.shape != want) {
      std::ostringstream m;
      m << name << ": shape [";
      for (size_t i = 0; i < v.shape.size(); ++i) m << (i ? ", " : "") << v.shape[i];
      m << "], expected [";
      for (size_t i = 0; i < want.size(); ++i) m << (i ? ", " : "") << want[i];
      m << "]";
      fail(m.str());
    }
    Matrix out;
    if (v.idata != nullptr) fail(name + ": int32 is not a weight dtype");
    if (!v.quantized()) {
      out.f32 = v.data;
      return out;
    }
    out.i8 = v.qdata;
    const WeightView& sv = get(name + ".scale");
    if (sv.data == nullptr) fail(name + ".scale: must be float32");
    if (sv.shape != std::vector<int64_t>{channels}) {
      fail(name + ".scale: expected " + std::to_string(channels) + " entries");
    }
    out.scale = sv.data;
    return out;
  };

  wte_ = need_matrix("wte", {config_.vocab_size, d}, config_.vocab_size);
  resolve_wte_outliers();

  // Named from what was found, not from a flag: the header says int8, the
  // tensors say how much of the model it covers.
  if (quant_ == 0) {
    policy_ = "fp32";
  } else if (!wte_.quantized()) {
    policy_ = "int8";
  } else if (wte_.n_outlier == 0) {
    policy_ = "int8-wte";
  } else {
    policy_ = "int8-wte-o" + std::to_string(wte_.n_outlier);
  }
  wpe_ = need("wpe", {config_.n_ctx, d});
  ln_f_w_ = need("ln_f.weight", {d});
  ln_f_b_ = need("ln_f.bias", {d});

  layers_.resize(static_cast<size_t>(config_.n_layer));
  for (int i = 0; i < config_.n_layer; ++i) {
    const std::string p = "h." + std::to_string(i) + ".";
    LayerWeights& L = layers_[static_cast<size_t>(i)];
    L.ln_1_w = need(p + "ln_1.weight", {d});
    L.ln_1_b = need(p + "ln_1.bias", {d});
    L.c_attn_w = need_matrix(p + "attn.c_attn.weight",
                             {d, 3 * static_cast<int64_t>(d)}, 3 * static_cast<int64_t>(d));
    L.c_attn_b = need(p + "attn.c_attn.bias", {3 * static_cast<int64_t>(d)});
    L.attn_proj_w = need_matrix(p + "attn.c_proj.weight", {d, d}, d);
    L.attn_proj_b = need(p + "attn.c_proj.bias", {d});
    L.ln_2_w = need(p + "ln_2.weight", {d});
    L.ln_2_b = need(p + "ln_2.bias", {d});
    L.c_fc_w = need_matrix(p + "mlp.c_fc.weight", {d, f}, f);
    L.c_fc_b = need(p + "mlp.c_fc.bias", {f});
    L.mlp_proj_w = need_matrix(p + "mlp.c_proj.weight", {f, d}, d);
    L.mlp_proj_b = need(p + "mlp.c_proj.bias", {d});
  }
}

}  // namespace gpt2
