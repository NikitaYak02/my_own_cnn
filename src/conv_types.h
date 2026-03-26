#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "nnalgebra_quantization.h"

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

struct Conv2DParams {
  int pad_h = 1;
  int pad_w = 1;
  int stride_h = 1;
  int stride_w = 1;
  int dilation_h = 1;
  int dilation_w = 1;
  int groups = 1;
};

struct BlockConv2DParams {
  Conv2DParams conv;
  int block_by = 1;
  int block_bx = 1;
};

struct TensorNHWC {
  int n = 0;
  int h = 0;
  int w = 0;
  int c = 0;
  std::vector<float> data;

  TensorNHWC() = default;
  TensorNHWC(int n_, int h_, int w_, int c_) : n(n_), h(h_), w(w_), c(c_), data(static_cast<size_t>(n_) * h_ * w_ * c_) {}

  size_t elements() const { return static_cast<size_t>(n) * h * w * c; }
  float* ptr() { return data.data(); }
  const float* ptr() const { return data.data(); }
};

// Filter memory is stored output-channel major: [K, R, S, C].
struct FilterKRSC {
  int r = 0;
  int s = 0;
  int cin_per_group = 0;
  int k = 0;
  std::vector<float> data;

  FilterKRSC() = default;
  FilterKRSC(int r_, int s_, int cin_per_group_, int k_) : r(r_), s(s_), cin_per_group(cin_per_group_), k(k_), data(static_cast<size_t>(r_) * s_ * cin_per_group_ * k_) {}

  size_t elements() const { return static_cast<size_t>(r) * s * cin_per_group * k; }
  float* ptr() { return data.data(); }
  const float* ptr() const { return data.data(); }
};

using FilterHWCN = FilterKRSC;

// Blocked filter memory is stored as [K, By, Bx, R, S, C].
struct BlockFilterKByBxRSC {
  int k = 0;
  int by = 0;
  int bx = 0;
  int r = 0;
  int s = 0;
  int cin_per_group = 0;
  std::vector<float> data;

  BlockFilterKByBxRSC() = default;
  BlockFilterKByBxRSC(int k_, int by_, int bx_, int r_, int s_, int cin_per_group_)
      : k(k_), by(by_), bx(bx_), r(r_), s(s_), cin_per_group(cin_per_group_),
        data(static_cast<size_t>(k_) * by_ * bx_ * r_ * s_ * cin_per_group_) {}

  size_t elements() const { return static_cast<size_t>(k) * by * bx * r * s * cin_per_group; }
  float* ptr() { return data.data(); }
  const float* ptr() const { return data.data(); }
};

struct TensorNHWCI32 {
  int n = 0;
  int h = 0;
  int w = 0;
  int c = 0;
  std::vector<int32_t> data;

  TensorNHWCI32() = default;
  TensorNHWCI32(int n_, int h_, int w_, int c_)
      : n(n_), h(h_), w(w_), c(c_),
        data(static_cast<size_t>(n_) * h_ * w_ * c_) {}

  size_t elements() const { return static_cast<size_t>(n) * h * w * c; }
  int32_t* ptr() { return data.data(); }
  const int32_t* ptr() const { return data.data(); }
};

struct ConvShape {
  int ho = 0;
  int wo = 0;
  int cin_group = 0;
  int kout_group = 0;
};

struct BlockConvShape {
  ConvShape base;
  int block_ho = 0;
  int block_wo = 0;
};

inline ConvShape infer_conv_shape(const TensorNHWC& x, const FilterKRSC& w, const Conv2DParams& p) {
  if (p.groups <= 0) throw std::runtime_error("groups must be > 0");
  if (x.c % p.groups != 0) throw std::runtime_error("input channels must be divisible by groups");
  if (w.k % p.groups != 0) throw std::runtime_error("output channels must be divisible by groups");
  const int cin_group = x.c / p.groups;
  if (cin_group != w.cin_per_group) throw std::runtime_error("filter cin_per_group mismatch");

  const int ho_num = x.h + 2 * p.pad_h - p.dilation_h * (w.r - 1) - 1;
  const int wo_num = x.w + 2 * p.pad_w - p.dilation_w * (w.s - 1) - 1;
  if (ho_num < 0 || wo_num < 0) throw std::runtime_error("invalid output size (negative numerator)");
  if (p.stride_h <= 0 || p.stride_w <= 0) throw std::runtime_error("stride must be > 0");
  const int ho = ho_num / p.stride_h + 1;
  const int wo = wo_num / p.stride_w + 1;
  if (ho <= 0 || wo <= 0) throw std::runtime_error("computed output size is <= 0");

  ConvShape shape;
  shape.ho = ho;
  shape.wo = wo;
  shape.cin_group = cin_group;
  shape.kout_group = w.k / p.groups;
  return shape;
}

inline BlockConvShape infer_block_conv_shape(const TensorNHWC& x,
                                             const BlockFilterKByBxRSC& w,
                                             const BlockConv2DParams& p) {
  if (p.block_by <= 0 || p.block_bx <= 0) {
    throw std::runtime_error("block_by and block_bx must be > 0");
  }
  if (w.by != p.block_by || w.bx != p.block_bx) {
    throw std::runtime_error("blocked filter block grid mismatch");
  }

  const FilterKRSC dense_w(w.r, w.s, w.cin_per_group, w.k);
  const ConvShape base = infer_conv_shape(x, dense_w, p.conv);
  if ((base.ho % p.block_by) != 0 || (base.wo % p.block_bx) != 0) {
    throw std::runtime_error("blocked convolution requires Ho/Wo divisible by block grid");
  }

  BlockConvShape shape;
  shape.base = base;
  shape.block_ho = base.ho / p.block_by;
  shape.block_wo = base.wo / p.block_bx;
  return shape;
}

__host__ __device__ inline size_t idx_nhwc(int n, int h, int w, int c, int H, int W, int C) {
  return ((static_cast<size_t>(n) * H + h) * W + w) * C + c;
}

__host__ __device__ inline size_t idx_krsc(int k, int r, int s, int c, int R, int S, int C) {
  return ((static_cast<size_t>(k) * R + r) * S + s) * C + c;
}

__host__ __device__ inline size_t idx_kbybxrsc(int k, int by, int bx, int r, int s, int c,
                                               int By, int Bx, int R, int S, int C) {
  return (((((static_cast<size_t>(k) * By + by) * Bx + bx) * R + r) * S + s) * C + c);
}

struct VerifyResult {
  float max_abs_err = 0.0f;
  float max_rel_err = 0.0f;
  size_t max_abs_idx = 0;
  size_t max_rel_idx = 0;
  bool passed = false;
};

struct BenchResult {
  float median_ms = 0.0f;
  float p90_ms = 0.0f;
  float gflops = 0.0f;
};

enum class GradKernelAlgo {
  GemmIm2Col = 0,
  Algo0Atomic = 1,
  Algo1Deterministic = 2,
};

void cpu_fprop_nhwc(const TensorNHWC& x, const FilterKRSC& w, const Conv2DParams& p, TensorNHWC& y);
void cpu_bprop_nhwc(const TensorNHWC& dy, const FilterKRSC& w, const Conv2DParams& p, TensorNHWC& dx);
void cpu_grad_nhwc(const TensorNHWC& x, const TensorNHWC& dy, const Conv2DParams& p, FilterKRSC& dw);
void cpu_block_fprop_nhwc(const TensorNHWC& x, const BlockFilterKByBxRSC& w, const BlockConv2DParams& p, TensorNHWC& y);
void cpu_block_bprop_nhwc(const TensorNHWC& dy, const BlockFilterKByBxRSC& w, const BlockConv2DParams& p, TensorNHWC& dx);
void cpu_block_grad_nhwc(const TensorNHWC& x, const TensorNHWC& dy, const BlockConv2DParams& p, BlockFilterKByBxRSC& dw);

namespace conv_quant_detail {

void cpu_fprop_nhwc_qi32_u8(const TensorNHWC& x, const FilterKRSC& w, const Conv2DParams& p,
                            TensorNHWCI32& y,
                            const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* in_qp,
                            const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* f_qp);
void cpu_fprop_nhwc_qi32_s5(const TensorNHWC& x, const FilterKRSC& w, const Conv2DParams& p,
                            TensorNHWCI32& y,
                            const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* in_qp,
                            const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* f_qp);
void cpu_block_fprop_nhwc_qi32_u8(const TensorNHWC& x, const BlockFilterKByBxRSC& w, const BlockConv2DParams& p,
                                  TensorNHWCI32& y,
                                  const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* in_qp,
                                  const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* f_qp);
void cpu_block_fprop_nhwc_qi32_s5(const TensorNHWC& x, const BlockFilterKByBxRSC& w, const BlockConv2DParams& p,
                                  TensorNHWCI32& y,
                                  const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* in_qp,
                                  const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* f_qp);

void launch_fprop_nhwc_qi32_u8(const float* d_x, const float* d_w, int32_t* d_y,
                               int n, int h, int w, int c, int r, int s, int k,
                               const Conv2DParams& p,
                               const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* in_qp,
                               const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* f_qp);
void launch_fprop_nhwc_qi32_s5(const float* d_x, const float* d_w, int32_t* d_y,
                               int n, int h, int w, int c, int r, int s, int k,
                               const Conv2DParams& p,
                               const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* in_qp,
                               const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* f_qp);
void launch_block_fprop_nhwc_qi32_u8(const float* d_x, const float* d_w, int32_t* d_y,
                                     int n, int h, int w, int c, int r, int s, int k,
                                     const BlockConv2DParams& p,
                                     const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* in_qp,
                                     const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* f_qp);
void launch_block_fprop_nhwc_qi32_s5(const float* d_x, const float* d_w, int32_t* d_y,
                                     int n, int h, int w, int c, int r, int s, int k,
                                     const BlockConv2DParams& p,
                                     const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* in_qp,
                                     const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* f_qp);

}  // namespace conv_quant_detail

template <nnalgebra::DataType Tin>
inline void cpu_fprop_nhwc_qi32(const TensorNHWC& x, const FilterKRSC& w, const Conv2DParams& p,
                                TensorNHWCI32& y,
                                const nnalgebra::QuantizationParameters<Tin>* in_qp,
                                const nnalgebra::QuantizationParameters<Tin>* f_qp) {
  static_assert(nnalgebra::kIsSupportedQuantizedInput<Tin>,
                "cpu_fprop_nhwc_qi32 supports only LinQuantU8 and LinQuantS5");
  if (!in_qp || !f_qp) throw std::runtime_error("quantized CPU fprop requires non-null quantization parameters");

  if constexpr (Tin == nnalgebra::DataType::LinQuantU8) {
    conv_quant_detail::cpu_fprop_nhwc_qi32_u8(x, w, p, y, in_qp, f_qp);
  } else {
    conv_quant_detail::cpu_fprop_nhwc_qi32_s5(x, w, p, y, in_qp, f_qp);
  }
}

template <nnalgebra::DataType Tin>
inline void cpu_block_fprop_nhwc_qi32(const TensorNHWC& x, const BlockFilterKByBxRSC& w, const BlockConv2DParams& p,
                                      TensorNHWCI32& y,
                                      const nnalgebra::QuantizationParameters<Tin>* in_qp,
                                      const nnalgebra::QuantizationParameters<Tin>* f_qp) {
  static_assert(nnalgebra::kIsSupportedQuantizedInput<Tin>,
                "cpu_block_fprop_nhwc_qi32 supports only LinQuantU8 and LinQuantS5");
  if (!in_qp || !f_qp) throw std::runtime_error("quantized blocked CPU fprop requires non-null quantization parameters");

  if constexpr (Tin == nnalgebra::DataType::LinQuantU8) {
    conv_quant_detail::cpu_block_fprop_nhwc_qi32_u8(x, w, p, y, in_qp, f_qp);
  } else {
    conv_quant_detail::cpu_block_fprop_nhwc_qi32_s5(x, w, p, y, in_qp, f_qp);
  }
}

void launch_fprop_nhwc(const float* d_x, const float* d_w, float* d_y,
                       int n, int h, int w, int c, int r, int s, int k,
                       const Conv2DParams& p);
void launch_bprop_nhwc(const float* d_dy, const float* d_w, float* d_dx,
                       int n, int h, int w, int c, int r, int s, int k,
                       const Conv2DParams& p);
void launch_grad_nhwc(const float* d_x, const float* d_dy, float* d_dw,
                      int n, int h, int w, int c, int r, int s, int k,
                      const Conv2DParams& p,
                      GradKernelAlgo algo = GradKernelAlgo::GemmIm2Col);
void launch_block_fprop_nhwc(const float* d_x, const float* d_w, float* d_y,
                             int n, int h, int w, int c, int r, int s, int k,
                             const BlockConv2DParams& p);
void launch_block_bprop_nhwc(const float* d_dy, const float* d_w, float* d_dx,
                             int n, int h, int w, int c, int r, int s, int k,
                             const BlockConv2DParams& p);
void launch_block_grad_nhwc(const float* d_x, const float* d_dy, float* d_dw,
                            int n, int h, int w, int c, int r, int s, int k,
                            const BlockConv2DParams& p,
                            GradKernelAlgo algo = GradKernelAlgo::GemmIm2Col);

template <nnalgebra::DataType Tin>
inline void launch_fprop_nhwc_qi32(const float* d_x, const float* d_w, int32_t* d_y,
                                   int n, int h, int w, int c, int r, int s, int k,
                                   const Conv2DParams& p,
                                   const nnalgebra::QuantizationParameters<Tin>* in_qp,
                                   const nnalgebra::QuantizationParameters<Tin>* f_qp) {
  static_assert(nnalgebra::kIsSupportedQuantizedInput<Tin>,
                "launch_fprop_nhwc_qi32 supports only LinQuantU8 and LinQuantS5");
  if (!in_qp || !f_qp) throw std::runtime_error("quantized CUDA fprop requires non-null quantization parameters");

  if constexpr (Tin == nnalgebra::DataType::LinQuantU8) {
    conv_quant_detail::launch_fprop_nhwc_qi32_u8(d_x, d_w, d_y, n, h, w, c, r, s, k, p, in_qp, f_qp);
  } else {
    conv_quant_detail::launch_fprop_nhwc_qi32_s5(d_x, d_w, d_y, n, h, w, c, r, s, k, p, in_qp, f_qp);
  }
}

template <nnalgebra::DataType Tin>
inline void launch_block_fprop_nhwc_qi32(const float* d_x, const float* d_w, int32_t* d_y,
                                         int n, int h, int w, int c, int r, int s, int k,
                                         const BlockConv2DParams& p,
                                         const nnalgebra::QuantizationParameters<Tin>* in_qp,
                                         const nnalgebra::QuantizationParameters<Tin>* f_qp) {
  static_assert(nnalgebra::kIsSupportedQuantizedInput<Tin>,
                "launch_block_fprop_nhwc_qi32 supports only LinQuantU8 and LinQuantS5");
  if (!in_qp || !f_qp) throw std::runtime_error("quantized blocked CUDA fprop requires non-null quantization parameters");

  if constexpr (Tin == nnalgebra::DataType::LinQuantU8) {
    conv_quant_detail::launch_block_fprop_nhwc_qi32_u8(d_x, d_w, d_y, n, h, w, c, r, s, k, p, in_qp, f_qp);
  } else {
    conv_quant_detail::launch_block_fprop_nhwc_qi32_s5(d_x, d_w, d_y, n, h, w, c, r, s, k, p, in_qp, f_qp);
  }
}

VerifyResult verify_tensors(const std::vector<float>& ref, const std::vector<float>& got, float abs_eps, float rel_eps);

BenchResult benchmark_cuda_op(const std::string& op_name,
                              int warmup,
                              int iters,
                              size_t flops,
                              const std::function<void()>& fn);

bool cudnn_is_available();
BenchResult cudnn_fprop_bench(const float* d_x, const float* d_w, float* d_y,
                              int n, int h, int w, int c, int r, int s, int k,
                              const Conv2DParams& p,
                              int warmup, int iters);
BenchResult cudnn_bprop_bench(const float* d_dy, const float* d_w, float* d_dx,
                              int n, int h, int w, int c, int r, int s, int k,
                              const Conv2DParams& p,
                              int warmup, int iters);
BenchResult cudnn_grad_bench(const float* d_x, const float* d_dy, float* d_dw,
                             int n, int h, int w, int c, int r, int s, int k,
                             const Conv2DParams& p,
                             int warmup, int iters);
BenchResult cudnn_block_fprop_bench(const float* d_x, const float* d_w, float* d_y,
                                    int n, int h, int w, int c, int r, int s, int k,
                                    const BlockConv2DParams& p,
                                    int warmup, int iters);
BenchResult cudnn_block_bprop_bench(const float* d_dy, const float* d_w, float* d_dx,
                                    int n, int h, int w, int c, int r, int s, int k,
                                    const BlockConv2DParams& p,
                                    int warmup, int iters);
BenchResult cudnn_block_grad_bench(const float* d_x, const float* d_dy, float* d_dw,
                                   int n, int h, int w, int c, int r, int s, int k,
                                   const BlockConv2DParams& p,
                                   int warmup, int iters);
