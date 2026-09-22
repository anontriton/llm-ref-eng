// Kernels against independently computed values.
//
// Each test recomputes the same quantity in double precision, straight from the
// definition, and checks the fp32 kernel lands on it. That catches a wrong
// formula, which is the failure mode that matters here -- a wrong ordering
// shows up in the oracle comparison instead.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "check.h"
#include "gpt2/ops.h"

namespace {

// The erf formulation of GELU. Not what GPT-2 uses -- present only so the test
// below can prove the engine is NOT computing it.
double gelu_erf(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }

double gelu_tanh(double x) {
  const double c = std::sqrt(2.0 / M_PI);
  return 0.5 * x * (1.0 + std::tanh(c * (x + 0.044715 * x * x * x)));
}

void test_layernorm() {
  const int rows = 3, n = 8;
  std::vector<float> x(rows * n), w(n), b(n), out(rows * n);
  for (int i = 0; i < rows * n; ++i) x[i] = static_cast<float>(0.5 * i - 4.0);
  for (int j = 0; j < n; ++j) {
    w[j] = static_cast<float>(1.0 + 0.1 * j);
    b[j] = static_cast<float>(-0.2 * j);
  }
  const float eps = 1e-5f;
  gpt2::ops::layernorm(x.data(), w.data(), b.data(), out.data(), rows, n, eps);

  for (int r = 0; r < rows; ++r) {
    double mean = 0.0;
    for (int j = 0; j < n; ++j) mean += x[r * n + j];
    mean /= n;
    // Biased (population) variance: divide by N, not N-1.
    double var = 0.0;
    for (int j = 0; j < n; ++j) var += (x[r * n + j] - mean) * (x[r * n + j] - mean);
    var /= n;
    for (int j = 0; j < n; ++j) {
      const double want = (x[r * n + j] - mean) / std::sqrt(var + eps) * w[j] + b[j];
      check::close(out[r * n + j], want, 1e-5, "layernorm value");
    }
  }

  // With unit scale and zero shift a normalized row has mean 0 and variance 1.
  std::vector<float> w1(n, 1.0f), b0(n, 0.0f);
  gpt2::ops::layernorm(x.data(), w1.data(), b0.data(), out.data(), rows, n, 1e-12f);
  double mean = 0.0, sq = 0.0;
  for (int j = 0; j < n; ++j) { mean += out[j]; sq += out[j] * out[j]; }
  check::close(mean / n, 0.0, 1e-5, "normalized row has zero mean");
  check::close(sq / n, 1.0, 1e-4, "normalized row has unit variance");
}

void test_gelu() {
  std::vector<float> x, out;
  for (double v = -6.0; v <= 6.0; v += 0.25) x.push_back(static_cast<float>(v));
  out.resize(x.size());
  gpt2::ops::gelu_new(x.data(), out.data(), x.size());

  for (size_t i = 0; i < x.size(); ++i) {
    check::close(out[i], gelu_tanh(x[i]), 1e-5, "gelu_new matches the tanh form");
  }
  check::close(out[static_cast<size_t>(24)], 0.0, 1e-7, "gelu_new(0) == 0");

  // The gotcha, asserted rather than commented. The two formulations peak
  // 4.73e-4 apart at |x| ~ 2.7 -- mid-range, not in the tails, where both
  // converge -- which is 4.7x the 1e-4 per-layer tolerance. Swapping them is a
  // silent accuracy bug, so the test pins the gap rather than trusting a
  // comment about it.
  double worst = 0.0, worst_at = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    const double d = std::fabs(gelu_tanh(x[i]) - gelu_erf(x[i]));
    if (d > worst) { worst = d; worst_at = x[i]; }
  }
  check::ok(worst > 4e-4,
            "tanh and erf GELU differ by more than tolerance (must not be swapped)");
  check::close(std::fabs(worst_at), 2.75, 0.3, "the two forms differ most near |x| ~ 2.7");
}

void test_softmax() {
  const int rows = 2, n = 5;
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                          0.5f, -1.0f, 2.5f, 0.0f, 1.5f};
  gpt2::ops::softmax_rows(x.data(), rows, n);
  for (int r = 0; r < rows; ++r) {
    double s = 0.0;
    for (int j = 0; j < n; ++j) {
      s += x[r * n + j];
      check::ok(x[r * n + j] > 0.0f, "softmax output is positive");
    }
    check::close(s, 1.0, 1e-6, "softmax row sums to 1");
  }

  // A masked position is -inf going in and exactly 0 coming out -- that is how
  // causal masking disappears from the probabilities.
  const float ninf = -std::numeric_limits<float>::infinity();
  std::vector<float> m = {1.0f, ninf, ninf, 2.0f, ninf, ninf};
  gpt2::ops::softmax_rows(m.data(), 2, 3);
  check::close(m[0], 1.0, 0.0, "lone unmasked entry gets all the mass");
  check::close(m[1], 0.0, 0.0, "masked entry is exactly zero");
  check::close(m[2], 0.0, 0.0, "masked entry is exactly zero");
  check::close(m[3], 1.0, 0.0, "second row, lone unmasked entry");
}

void test_linear() {
  const int rows = 3, n_in = 5, n_out = 4;
  std::vector<float> x(rows * n_in), W(n_in * n_out), b(n_out), out(rows * n_out);
  for (int i = 0; i < rows * n_in; ++i) x[i] = static_cast<float>(0.3 * i - 1.0);
  // W is [n_in, n_out] -- the checkpoint's orientation, not nn.Linear's.
  for (int i = 0; i < n_in * n_out; ++i) W[i] = static_cast<float>(0.1 * i - 0.5);
  for (int j = 0; j < n_out; ++j) b[j] = static_cast<float>(j);

  const gpt2::Matrix Wm{W.data(), nullptr, nullptr};
  gpt2::ops::linear(x.data(), Wm, b.data(), out.data(), rows, n_in, n_out);
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < n_out; ++j) {
      double want = b[j];
      for (int i = 0; i < n_in; ++i) want += x[r * n_in + i] * W[i * n_out + j];
      check::close(out[r * n_out + j], want, 1e-4, "linear: x @ W + b");
    }
  }

  // A null bias means no bias, not a crash.
  gpt2::ops::linear(x.data(), Wm, nullptr, out.data(), rows, n_in, n_out);
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < n_out; ++j) {
      double want = 0.0;
      for (int i = 0; i < n_in; ++i) want += x[r * n_in + i] * W[i * n_out + j];
      check::close(out[r * n_out + j], want, 1e-4, "linear without bias");
    }
  }

  // An inner dimension that is not a multiple of the accumulation block still
  // has to sum every term exactly once.
  {
    const int deep = 3072 + 37;
    std::vector<float> dx(deep, 1.0f), dW(deep, 1.0f), dout(1);
    const gpt2::Matrix dWm{dW.data(), nullptr, nullptr};
    gpt2::ops::linear(dx.data(), dWm, nullptr, dout.data(), 1, deep, 1);
    check::close(dout[0], static_cast<double>(deep), 1e-3,
                 "ragged inner dimension sums every term");
  }
}

void test_linear_tied() {
  const int rows = 2, n_in = 6, n_out = 3;
  std::vector<float> x(rows * n_in), Wt(n_out * n_in), out(rows * n_out);
  for (int i = 0; i < rows * n_in; ++i) x[i] = static_cast<float>(0.25 * i - 0.5);
  // Wt is [n_out, n_in]: the tied lm_head reads rows of wte.
  for (int i = 0; i < n_out * n_in; ++i) Wt[i] = static_cast<float>(-0.2 * i + 0.7);

  const gpt2::Matrix Wtm{Wt.data(), nullptr, nullptr};
  gpt2::ops::linear_tied(x.data(), Wtm, out.data(), rows, n_in, n_out);
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < n_out; ++j) {
      double want = 0.0;
      for (int i = 0; i < n_in; ++i) want += x[r * n_in + i] * Wt[j * n_in + i];
      check::close(out[r * n_out + j], want, 1e-5, "linear_tied: x @ Wt^T");
    }
  }
}

// int8 with outlier columns held back in fp32 (Phase 5's wte). The reference
// is the matrix the representation stands for -- q * scale everywhere except
// the outlier columns, which take their fp32 values -- so a kernel that forgot
// the outliers, or counted them twice, lands far off.
void test_linear_tied_outliers() {
  const int rows = 3, n_in = 20, n_out = 5;
  const std::vector<int32_t> cols = {2, 9, 17};
  const int k = static_cast<int>(cols.size());
  std::vector<float> x(rows * n_in), scale(n_out), outlier(n_out * k), out(rows * n_out);
  std::vector<int8_t> q(n_out * n_in);
  for (int i = 0; i < rows * n_in; ++i) x[i] = static_cast<float>(0.1 * (i % 7) - 0.3);
  // Make the outlier inputs large, as ln_f's are.
  for (int r = 0; r < rows; ++r) {
    for (int c : cols) x[r * n_in + c] = static_cast<float>(40.0 + r);
  }
  for (int i = 0; i < n_out * n_in; ++i) q[i] = static_cast<int8_t>((i * 37) % 255 - 127);
  for (int j = 0; j < n_out; ++j) {
    scale[j] = static_cast<float>(0.01 * (j + 1));
    for (int c : cols) q[j * n_in + c] = 0;  // what the loader requires
    for (int m = 0; m < k; ++m) outlier[j * k + m] = static_cast<float>(0.5 - 0.3 * m + 0.1 * j);
  }

  gpt2::Matrix Wtm;
  Wtm.i8 = q.data();
  Wtm.scale = scale.data();
  Wtm.outlier_cols = cols.data();
  Wtm.outlier = outlier.data();
  Wtm.n_outlier = k;
  gpt2::ops::linear_tied(x.data(), Wtm, out.data(), rows, n_in, n_out);
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < n_out; ++j) {
      double want = 0.0;
      for (int i = 0; i < n_in; ++i) {
        const auto it = std::find(cols.begin(), cols.end(), i);
        const double w = it == cols.end()
            ? static_cast<double>(q[j * n_in + i]) * scale[j]
            : static_cast<double>(outlier[j * k + (it - cols.begin())]);
        want += static_cast<double>(x[r * n_in + i]) * w;
      }
      check::close(out[r * n_out + j], want, 1e-4,
                   "linear_tied: int8 with fp32 outlier columns");
    }
  }
}

}  // namespace

int main() {
  test_layernorm();
  test_gelu();
  test_softmax();
  test_linear();
  test_linear_tied();
  test_linear_tied_outliers();
  return check::report("test_ops");
}
