#include "src/conv_types.h"
#include "src/cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using U8QParams = nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>;
using I32QParams = nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>;

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;

  ~DeviceBuffer() {
    cudaFree(ptr);
  }

  void allocate(size_t count) {
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
  }
};

struct ErrorStats {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double rmse = 0.0;
  double max_rel = 0.0;
  double mean_rel = 0.0;
  double ref_rms = 0.0;
  double rel_rmse = 0.0;
};

U8QParams choose_u8_qparams(const float* data, size_t count) {
  float min_v = std::numeric_limits<float>::infinity();
  float max_v = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < count; ++i) {
    min_v = std::min(min_v, data[i]);
    max_v = std::max(max_v, data[i]);
  }

  U8QParams qp;
  qp.symmetric = false;
  if (!(min_v < max_v)) {
    qp.scale = 1.0f;
    qp.zero_point = 0;
    return qp;
  }

  qp.scale = (max_v - min_v) / 255.0f;
  const float zp_real = -min_v / qp.scale;
  const int zp = std::max(0, std::min(255, static_cast<int>(std::lrint(zp_real))));
  qp.zero_point = static_cast<uint8_t>(zp);
  return qp;
}

float quantize_to_u8(float value, const U8QParams& qp) {
  const float q = std::round(value / qp.scale) + static_cast<float>(qp.zero_point);
  const float q_clamped = std::min(255.0f, std::max(0.0f, q));
  return q_clamped;
}

std::vector<U8QParams> quantize_input_per_image(const TensorNHWC& src, TensorNHWC& dst) {
  dst = TensorNHWC(src.n, src.h, src.w, src.c);
  std::vector<U8QParams> qps(static_cast<size_t>(src.n));
  const size_t img_elems = static_cast<size_t>(src.h) * src.w * src.c;
  for (int n = 0; n < src.n; ++n) {
    const float* src_img = src.ptr() + static_cast<size_t>(n) * img_elems;
    float* dst_img = dst.ptr() + static_cast<size_t>(n) * img_elems;
    qps[static_cast<size_t>(n)] = choose_u8_qparams(src_img, img_elems);
    for (size_t i = 0; i < img_elems; ++i) {
      dst_img[i] = quantize_to_u8(src_img[i], qps[static_cast<size_t>(n)]);
    }
  }
  return qps;
}

U8QParams quantize_filters_global(const FilterKRSC& src, FilterKRSC& dst) {
  dst = FilterKRSC(src.r, src.s, src.cin_per_group, src.k, src.ay, src.ax);
  const U8QParams qp = choose_u8_qparams(src.ptr(), src.elements());
  for (size_t i = 0; i < src.elements(); ++i) {
    dst.data[i] = quantize_to_u8(src.data[i], qp);
  }
  return qp;
}

std::vector<float> dequantize_output(const TensorNHWCI32& y_i32,
                                     const std::vector<I32QParams>& out_qp) {
  std::vector<float> out(y_i32.elements());
  const size_t spatial = static_cast<size_t>(y_i32.h) * y_i32.w * y_i32.c;
  for (int n = 0; n < y_i32.n; ++n) {
    const float scale = out_qp[static_cast<size_t>(n)].scale;
    const size_t base = static_cast<size_t>(n) * spatial;
    for (size_t i = 0; i < spatial; ++i) {
      out[base + i] = static_cast<float>(y_i32.data[base + i]) * scale;
    }
  }
  return out;
}

ErrorStats compute_error_stats(const std::vector<float>& ref, const std::vector<float>& got) {
  if (ref.size() != got.size()) {
    throw std::runtime_error("size mismatch while computing error stats");
  }

  ErrorStats stats;
  double sum_abs = 0.0;
  double sum_sq = 0.0;
  double sum_rel = 0.0;
  double ref_sq = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double a = static_cast<double>(ref[i]);
    const double b = static_cast<double>(got[i]);
    const double abs_err = std::fabs(a - b);
    const double rel_err = abs_err / std::max(std::fabs(a), 1e-6);

    stats.max_abs = std::max(stats.max_abs, abs_err);
    stats.max_rel = std::max(stats.max_rel, rel_err);
    sum_abs += abs_err;
    sum_sq += abs_err * abs_err;
    sum_rel += rel_err;
    ref_sq += a * a;
  }

  const double denom = static_cast<double>(ref.size());
  stats.mean_abs = sum_abs / denom;
  stats.rmse = std::sqrt(sum_sq / denom);
  stats.mean_rel = sum_rel / denom;
  stats.ref_rms = std::sqrt(ref_sq / denom);
  stats.rel_rmse = stats.rmse / std::max(stats.ref_rms, 1e-12);
  return stats;
}

void fill_normal(TensorNHWC& x, FilterKRSC& w, uint32_t seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> input_dist(0.0f, 1.0f);
  const float fan_in = static_cast<float>(w.r * w.s * w.cin_per_group);
  const float weight_std = std::sqrt(2.0f / fan_in);
  std::normal_distribution<float> weight_dist(0.0f, weight_std);

  for (float& v : x.data) v = input_dist(gen);
  for (float& v : w.data) v = weight_dist(gen);
}

void print_stats(const char* tag, const ErrorStats& stats) {
  std::cout << tag
            << " max_abs=" << stats.max_abs
            << " mean_abs=" << stats.mean_abs
            << " rmse=" << stats.rmse
            << " max_rel=" << stats.max_rel
            << " mean_rel=" << stats.mean_rel
            << " ref_rms=" << stats.ref_rms
            << " rel_rmse=" << stats.rel_rmse
            << "\n";
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

    constexpr uint32_t seed = 42;
    TensorNHWC x(8, 16, 16, 32);
    FilterKRSC w(3, 3, 32, 32);
    fill_normal(x, w, seed);

    TensorNHWC x_q;
    FilterKRSC w_q;
    const std::vector<U8QParams> in_qp = quantize_input_per_image(x, x_q);
    const U8QParams f_qp = quantize_filters_global(w, w_q);

    TensorNHWC y_ref;
    cpu_fprop_nhwc(x, w, p, y_ref);

    TensorNHWCI32 y_q_cpu;
    std::vector<I32QParams> out_qp_cpu(static_cast<size_t>(x.n));
    cpu_fprop_nhwc_qi32<nnalgebra::DataType::LinQuantU8>(
        x_q, w_q, p, y_q_cpu, in_qp.data(), &f_qp, out_qp_cpu.data());
    const std::vector<float> y_deq_cpu = dequantize_output(y_q_cpu, out_qp_cpu);
    const ErrorStats cpu_stats = compute_error_stats(y_ref.data, y_deq_cpu);

    DeviceBuffer<float> d_x;
    DeviceBuffer<float> d_w;
    DeviceBuffer<int32_t> d_y;
    DeviceBuffer<U8QParams> d_in_qp;
    DeviceBuffer<U8QParams> d_f_qp;
    DeviceBuffer<I32QParams> d_out_qp;

    d_x.allocate(x_q.elements());
    d_w.allocate(w_q.elements());
    d_y.allocate(y_q_cpu.elements());
    d_in_qp.allocate(in_qp.size());
    d_f_qp.allocate(1);
    d_out_qp.allocate(out_qp_cpu.size());

    CUDA_CHECK(cudaMemcpy(d_x.ptr, x_q.ptr(), x_q.elements() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w.ptr, w_q.ptr(), w_q.elements() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_qp.ptr, in_qp.data(), in_qp.size() * sizeof(U8QParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_qp.ptr, &f_qp, sizeof(U8QParams), cudaMemcpyHostToDevice));

    launch_fprop_nhwc_qi32<nnalgebra::DataType::LinQuantU8>(
        d_x.ptr, d_w.ptr, d_y.ptr,
        x.n, x.h, x.w, x.c, w.r, w.s, w.k,
        p, d_in_qp.ptr, d_f_qp.ptr, d_out_qp.ptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    TensorNHWCI32 y_q_cuda(y_q_cpu.n, y_q_cpu.h, y_q_cpu.w, y_q_cpu.c);
    std::vector<I32QParams> out_qp_cuda(static_cast<size_t>(x.n));
    CUDA_CHECK(cudaMemcpy(y_q_cuda.ptr(), d_y.ptr, y_q_cuda.elements() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_qp_cuda.data(), d_out_qp.ptr, out_qp_cuda.size() * sizeof(I32QParams), cudaMemcpyDeviceToHost));

    const std::vector<float> y_deq_cuda = dequantize_output(y_q_cuda, out_qp_cuda);
    const ErrorStats cuda_stats = compute_error_stats(y_ref.data, y_deq_cuda);

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "quant_error_u8\n";
    std::cout << "seed=" << seed
              << " shape=N" << x.n
              << " H" << x.h
              << " W" << x.w
              << " C" << x.c
              << " K" << w.k
              << " R" << w.r
              << " S" << w.s
              << "\n";
    std::cout << "input_scale[min,max]=[";
    float min_in_scale = std::numeric_limits<float>::infinity();
    float max_in_scale = 0.0f;
    for (const U8QParams& qp : in_qp) {
      min_in_scale = std::min(min_in_scale, qp.scale);
      max_in_scale = std::max(max_in_scale, qp.scale);
    }
    std::cout << min_in_scale << ", " << max_in_scale << "] ";
    std::cout << "filter_scale=" << f_qp.scale << "\n";
    print_stats("cpu_dequant_vs_float:", cpu_stats);
    print_stats("cuda_dequant_vs_float:", cuda_stats);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "quant_error_u8 failed: " << e.what() << "\n";
    return 1;
  }
}
