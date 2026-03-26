#include "conv_types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct SampleCoord {
  int n;
  int h;
  int w;
  int c;
};

struct CheckStats {
  float max_abs_err = 0.0f;
  float max_rel_err = 0.0f;
  size_t worst_index = 0;
  bool passed = true;
};

void fill_random(std::vector<float>& data, std::mt19937& gen, float lo, float hi) {
  std::uniform_real_distribution<float> dist(lo, hi);
  for (float& v : data) {
    v = dist(gen);
  }
}

float evaluate_loss(const TensorNHWC& x,
                    const FilterKRSC& w,
                    const Conv2DParams& p,
                    const TensorNHWC& dy) {
  TensorNHWC y;
  cpu_fprop_nhwc(x, w, p, y);
  float loss = 0.0f;
  for (size_t i = 0; i < y.data.size(); ++i) {
    loss += y.data[i] * dy.data[i];
  }
  return loss;
}

bool within_tolerance(float analytic, float numeric, float abs_tol, float rel_tol) {
  const float abs_err = std::fabs(analytic - numeric);
  const float denom = std::max(std::max(std::fabs(analytic), std::fabs(numeric)), 1e-6f);
  const float rel_err = abs_err / denom;
  return (abs_err <= abs_tol) || (rel_err <= rel_tol);
}

void update_stats(CheckStats& stats, size_t idx, float analytic, float numeric) {
  const float abs_err = std::fabs(analytic - numeric);
  const float denom = std::max(std::max(std::fabs(analytic), std::fabs(numeric)), 1e-6f);
  const float rel_err = abs_err / denom;
  if (abs_err > stats.max_abs_err) {
    stats.max_abs_err = abs_err;
    stats.worst_index = idx;
  }
  if (rel_err > stats.max_rel_err) {
    stats.max_rel_err = rel_err;
  }
}

CheckStats check_weight_gradient(const TensorNHWC& x,
                                 FilterKRSC& w,
                                 const Conv2DParams& p,
                                 const TensorNHWC& dy,
                                 const FilterKRSC& dw,
                                 float eps,
                                 float abs_tol,
                                 float rel_tol) {
  CheckStats stats;
  for (size_t i = 0; i < w.data.size(); ++i) {
    const float orig = w.data[i];
    w.data[i] = orig + eps;
    const float loss_pos = evaluate_loss(x, w, p, dy);
    w.data[i] = orig - eps;
    const float loss_neg = evaluate_loss(x, w, p, dy);
    w.data[i] = orig;

    const float numeric = (loss_pos - loss_neg) / (2.0f * eps);
    const float analytic = dw.data[i];
    update_stats(stats, i, analytic, numeric);
    if (!within_tolerance(analytic, numeric, abs_tol, rel_tol)) {
      stats.passed = false;
      std::ostringstream oss;
      oss << "weight gradient mismatch at index " << i
          << " analytic=" << analytic
          << " numeric=" << numeric
          << " abs_err=" << std::fabs(analytic - numeric);
      throw std::runtime_error(oss.str());
    }
  }
  return stats;
}

CheckStats check_input_gradient(const TensorNHWC& x_ref,
                                const FilterKRSC& w,
                                const Conv2DParams& p,
                                const TensorNHWC& dy,
                                const TensorNHWC& dx,
                                float eps,
                                float abs_tol,
                                float rel_tol) {
  TensorNHWC x = x_ref;
  const std::vector<SampleCoord> samples = {
      {0, 0, 0, 0},
      {0, 0, 127, 1},
      {0, 127, 0, 2},
      {0, 127, 127, 0},
      {0, 1, 1, 1},
      {0, 32, 96, 2},
      {0, 48, 17, 0},
      {0, 63, 64, 1},
      {0, 64, 63, 2},
      {0, 64, 64, 0},
      {0, 96, 32, 1},
      {0, 126, 126, 2},
  };

  CheckStats stats;
  for (size_t i = 0; i < samples.size(); ++i) {
    const SampleCoord sample = samples[i];
    const size_t idx = idx_nhwc(sample.n, sample.h, sample.w, sample.c, x.h, x.w, x.c);
    const float orig = x.data[idx];
    x.data[idx] = orig + eps;
    const float loss_pos = evaluate_loss(x, w, p, dy);
    x.data[idx] = orig - eps;
    const float loss_neg = evaluate_loss(x, w, p, dy);
    x.data[idx] = orig;

    const float numeric = (loss_pos - loss_neg) / (2.0f * eps);
    const float analytic = dx.data[idx];
    update_stats(stats, idx, analytic, numeric);
    if (!within_tolerance(analytic, numeric, abs_tol, rel_tol)) {
      stats.passed = false;
      std::ostringstream oss;
      oss << "input gradient mismatch at sample " << i
          << " (h=" << sample.h << ", w=" << sample.w << ", c=" << sample.c << ")"
          << " analytic=" << analytic
          << " numeric=" << numeric
          << " abs_err=" << std::fabs(analytic - numeric);
      throw std::runtime_error(oss.str());
    }
  }
  return stats;
}

}  // namespace

int main() {
  try {
    Conv2DParams p;
    p.pad_h = 1;
    p.pad_w = 1;
    p.stride_h = 1;
    p.stride_w = 1;
    p.dilation_h = 1;
    p.dilation_w = 1;
    p.groups = 1;

    TensorNHWC x(1, 128, 128, 3);
    FilterKRSC w(3, 3, 3, 4);
    const ConvShape shape = infer_conv_shape(x, w, p);
    TensorNHWC dy(x.n, shape.ho, shape.wo, w.k);
    TensorNHWC dx(x.n, x.h, x.w, x.c);
    FilterKRSC dw(w.r, w.s, w.cin_per_group, w.k);

    std::mt19937 gen(123);
    fill_random(x.data, gen, -0.25f, 0.25f);
    fill_random(w.data, gen, -0.25f, 0.25f);
    fill_random(dy.data, gen, -0.25f, 0.25f);

    cpu_bprop_nhwc(dy, w, p, dx);
    cpu_grad_nhwc(x, dy, p, dw);

    const float eps = 1e-3f;
    const CheckStats dw_stats = check_weight_gradient(x, w, p, dy, dw, eps, 5e-2f, 1e-2f);
    const CheckStats dx_stats = check_input_gradient(x, w, p, dy, dx, eps, 5e-2f, 1e-2f);

    std::cout << "conv_grad_check passed\n";
    std::cout << "case: n=1 h=128 w=128 c=3 k=4 r=3 s=3 pad=1 stride=1 groups=1\n";
    std::cout << "dw max_abs_err=" << dw_stats.max_abs_err
              << " max_rel_err=" << dw_stats.max_rel_err << "\n";
    std::cout << "dx(sampled) max_abs_err=" << dx_stats.max_abs_err
              << " max_rel_err=" << dx_stats.max_rel_err << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "conv_grad_check failed: " << e.what() << "\n";
    return 1;
  }
}
