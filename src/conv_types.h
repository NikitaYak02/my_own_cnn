#pragma once

#include <algorithm>
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
  int ay = 1;
  int ax = 1;
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

// Filter memory is stored output-channel major: [K, R, S, C, Ay, Ax].
struct FilterKRSC {
  int r = 0;
  int s = 0;
  int cin_per_group = 0;
  int k = 0;
  int ay = 1;
  int ax = 1;
  std::vector<float> data;

  FilterKRSC() = default;
  FilterKRSC(int r_, int s_, int cin_per_group_, int k_, int ay_ = 1, int ax_ = 1)
      : r(r_), s(s_), cin_per_group(cin_per_group_), k(k_), ay(ay_), ax(ax_),
        data(static_cast<size_t>(r_) * s_ * cin_per_group_ * k_ * ay_ * ax_) {}

  size_t elements() const { return static_cast<size_t>(r) * s * cin_per_group * k * ay * ax; }
  float* ptr() { return data.data(); }
  const float* ptr() const { return data.data(); }
};

using FilterHWCN = FilterKRSC;

// Blocked filter memory is stored as [K, By, Bx, R, S, C, Ay, Ax].
struct BlockFilterKByBxRSC {
  int k = 0;
  int by = 0;
  int bx = 0;
  int r = 0;
  int s = 0;
  int cin_per_group = 0;
  int ay = 1;
  int ax = 1;
  std::vector<float> data;

  BlockFilterKByBxRSC() = default;
  BlockFilterKByBxRSC(int k_, int by_, int bx_, int r_, int s_, int cin_per_group_, int ay_ = 1, int ax_ = 1)
      : k(k_), by(by_), bx(bx_), r(r_), s(s_), cin_per_group(cin_per_group_), ay(ay_), ax(ax_),
        data(static_cast<size_t>(k_) * by_ * bx_ * r_ * s_ * cin_per_group_ * ay_ * ax_) {}

  size_t elements() const { return static_cast<size_t>(k) * by * bx * r * s * cin_per_group * ay * ax; }
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
  int base_ho = 0;
  int base_wo = 0;
  int ho = 0;
  int wo = 0;
  int ay = 1;
  int ax = 1;
  int cin_group = 0;
  int kout_group = 0;
};

struct BlockConvShape {
  ConvShape base;
  int block_ho = 0;
  int block_wo = 0;
};

namespace conv_runtime_detail {

constexpr int kGradSplitKOutputsPerWarp = 4;
constexpr int kGradSplitKTargetRowsPerChunk = 2048;
constexpr int kGradSplitKMaxChunks = 16;
constexpr size_t kGradSplitKMaxPartialElements = 8ull * 1024ull * 1024ull;
constexpr size_t kGradGemmTileTargetBytes = 64ull * 1024ull * 1024ull;
constexpr int kGradGemmMinRowsPerChunk = 2048;
constexpr size_t kConvScratchTargetBytes = 96ull * 1024ull * 1024ull;
constexpr int kConvMinRowsPerChunk = 1024;

inline int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

inline int select_grad_split_k(int rows, size_t slice_weights) {
  if (rows <= 0 || slice_weights == 0) return 1;

  int split_k = std::max(1, ceil_div(rows, kGradSplitKTargetRowsPerChunk));
  split_k = std::min(split_k, kGradSplitKMaxChunks);

  const size_t max_by_memory = std::max<size_t>(1, kGradSplitKMaxPartialElements / slice_weights);
  split_k = std::min(split_k, static_cast<int>(max_by_memory));
  return std::max(1, split_k);
}

inline int select_grad_gemm_rows_per_chunk(int rows, int kdim, int ncol) {
  if (rows <= 0) return 1;
  const size_t row_bytes = static_cast<size_t>(kdim + ncol) * sizeof(float);
  if (row_bytes == 0) return rows;

  int chunk_rows = static_cast<int>(kGradGemmTileTargetBytes / row_bytes);
  chunk_rows = std::max(kGradGemmMinRowsPerChunk, chunk_rows);
  chunk_rows = std::min(chunk_rows, rows);
  return std::max(1, chunk_rows);
}

inline int select_conv_rows_per_chunk(int rows, int kdim, int ncol) {
  if (rows <= 0) return 1;
  const size_t row_bytes = static_cast<size_t>(kdim + ncol) * sizeof(float);
  if (row_bytes == 0) return rows;

  int chunk_rows = static_cast<int>(kConvScratchTargetBytes / row_bytes);
  chunk_rows = std::max(1, chunk_rows);
  if (row_bytes <= kConvScratchTargetBytes) {
    chunk_rows = std::max(kConvMinRowsPerChunk, chunk_rows);
  }
  chunk_rows = std::min(chunk_rows, rows);
  return std::max(1, chunk_rows);
}

}  // namespace conv_runtime_detail

struct Conv2DRuntimeConfig {
  int n = 0;
  int h = 0;
  int w = 0;
  int c = 0;
  int r = 0;
  int s = 0;
  int k = 0;
  Conv2DParams params;
  ConvShape shape;
  int m = 0;
  int kdim = 0;
  int ncol = 0;
  int conv_rows_per_chunk = 0;
  int grad_gemm_rows_per_chunk = 0;
  int grad_slice_weights = 0;
  int grad_packed_weights = 0;
  int grad_split_k = 1;
  int grad_deterministic_rows_per_chunk = 1;
  int grad_reduce_blocks = 0;
  bool can_vec4 = false;
  size_t input_elements = 0;
  size_t output_elements = 0;
  size_t weight_elements = 0;
};

struct BlockConv2DRuntimeConfig {
  int n = 0;
  int h = 0;
  int w = 0;
  int c = 0;
  int r = 0;
  int s = 0;
  int k = 0;
  BlockConv2DParams params;
  BlockConvShape shape;
  int m_block = 0;
  int kdim = 0;
  int ncol = 0;
  int conv_rows_per_chunk = 0;
  int grad_gemm_rows_per_chunk = 0;
  int grad_slice_weights = 0;
  int grad_packed_weights = 0;
  int grad_split_k = 1;
  int grad_deterministic_rows_per_chunk = 1;
  int grad_reduce_blocks = 0;
  size_t per_group_slice = 0;
  size_t input_elements = 0;
  size_t output_elements = 0;
  size_t weight_elements = 0;
};

inline ConvShape infer_conv_shape(const TensorNHWC& x, const FilterKRSC& w, const Conv2DParams& p) {
  if (p.groups <= 0) throw std::runtime_error("groups must be > 0");
  if (p.ay <= 0 || p.ax <= 0) throw std::runtime_error("ay and ax must be > 0");
  if (x.c % p.groups != 0) throw std::runtime_error("input channels must be divisible by groups");
  if (w.k % p.groups != 0) throw std::runtime_error("output channels must be divisible by groups");
  const int cin_group = x.c / p.groups;
  if (cin_group != w.cin_per_group) throw std::runtime_error("filter cin_per_group mismatch");
  if (w.ay != p.ay || w.ax != p.ax) throw std::runtime_error("filter Ay/Ax mismatch");

  const int ho_num = x.h + 2 * p.pad_h - p.dilation_h * (w.r - 1) - 1;
  const int wo_num = x.w + 2 * p.pad_w - p.dilation_w * (w.s - 1) - 1;
  if (ho_num < 0 || wo_num < 0) throw std::runtime_error("invalid output size (negative numerator)");
  if (p.stride_h <= 0 || p.stride_w <= 0) throw std::runtime_error("stride must be > 0");
  const int base_ho = ho_num / p.stride_h + 1;
  const int base_wo = wo_num / p.stride_w + 1;
  if (base_ho <= 0 || base_wo <= 0) throw std::runtime_error("computed output size is <= 0");

  ConvShape shape;
  shape.base_ho = base_ho;
  shape.base_wo = base_wo;
  shape.ho = base_ho * p.ay;
  shape.wo = base_wo * p.ax;
  shape.ay = p.ay;
  shape.ax = p.ax;
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

  const FilterKRSC dense_w(w.r, w.s, w.cin_per_group, w.k, w.ay, w.ax);
  const ConvShape base = infer_conv_shape(x, dense_w, p.conv);
  if ((base.base_ho % p.block_by) != 0 || (base.base_wo % p.block_bx) != 0) {
    throw std::runtime_error("blocked convolution requires base Ho/Wo divisible by block grid");
  }

  BlockConvShape shape;
  shape.base = base;
  shape.block_ho = base.base_ho / p.block_by;
  shape.block_wo = base.base_wo / p.block_bx;
  return shape;
}

inline Conv2DRuntimeConfig make_conv2d_runtime_config(int n, int h, int w, int c,
                                                      const FilterKRSC& weights,
                                                      const Conv2DParams& params) {
  Conv2DRuntimeConfig cfg;
  cfg.n = n;
  cfg.h = h;
  cfg.w = w;
  cfg.c = c;
  cfg.r = weights.r;
  cfg.s = weights.s;
  cfg.k = weights.k;
  cfg.params = params;

  const TensorNHWC x_shape(n, h, w, c);
  cfg.shape = infer_conv_shape(x_shape, weights, params);

  cfg.m = n * cfg.shape.base_ho * cfg.shape.base_wo;
  cfg.kdim = weights.r * weights.s * cfg.shape.cin_group;
  cfg.ncol = cfg.shape.kout_group * params.ay * params.ax;
  cfg.conv_rows_per_chunk = conv_runtime_detail::select_conv_rows_per_chunk(cfg.m, cfg.kdim, cfg.ncol);
  cfg.grad_gemm_rows_per_chunk = conv_runtime_detail::select_grad_gemm_rows_per_chunk(cfg.m, cfg.kdim, cfg.ncol);
  cfg.grad_slice_weights = cfg.kdim * cfg.ncol;
  cfg.grad_packed_weights =
      weights.r * weights.s * cfg.shape.cin_group *
      conv_runtime_detail::ceil_div(cfg.shape.kout_group, conv_runtime_detail::kGradSplitKOutputsPerWarp);
  cfg.grad_split_k = conv_runtime_detail::select_grad_split_k(cfg.m, static_cast<size_t>(cfg.grad_slice_weights));
  cfg.grad_deterministic_rows_per_chunk = conv_runtime_detail::ceil_div(cfg.m, cfg.grad_split_k);
  cfg.grad_reduce_blocks = conv_runtime_detail::ceil_div(cfg.grad_slice_weights, 256);
  cfg.can_vec4 = (cfg.shape.cin_group % 4 == 0) && (cfg.shape.cin_group >= 4) && ((c % 4) == 0);
  cfg.input_elements = static_cast<size_t>(n) * h * w * c;
  cfg.output_elements = static_cast<size_t>(n) * cfg.shape.ho * cfg.shape.wo * weights.k;
  cfg.weight_elements = weights.elements();
  return cfg;
}

inline BlockConv2DRuntimeConfig make_block_conv2d_runtime_config(int n, int h, int w, int c,
                                                                 const BlockFilterKByBxRSC& weights,
                                                                 const BlockConv2DParams& params) {
  BlockConv2DRuntimeConfig cfg;
  cfg.n = n;
  cfg.h = h;
  cfg.w = w;
  cfg.c = c;
  cfg.r = weights.r;
  cfg.s = weights.s;
  cfg.k = weights.k;
  cfg.params = params;

  const TensorNHWC x_shape(n, h, w, c);
  cfg.shape = infer_block_conv_shape(x_shape, weights, params);

  cfg.m_block = n * cfg.shape.block_ho * cfg.shape.block_wo;
  cfg.kdim = weights.r * weights.s * cfg.shape.base.cin_group;
  cfg.ncol = cfg.shape.base.kout_group * params.conv.ay * params.conv.ax;
  cfg.conv_rows_per_chunk = conv_runtime_detail::select_conv_rows_per_chunk(cfg.m_block, cfg.kdim, cfg.ncol);
  cfg.grad_gemm_rows_per_chunk = conv_runtime_detail::select_grad_gemm_rows_per_chunk(cfg.m_block, cfg.kdim, cfg.ncol);
  cfg.grad_slice_weights = cfg.kdim * cfg.ncol;
  cfg.grad_packed_weights =
      weights.r * weights.s * cfg.shape.base.cin_group *
      conv_runtime_detail::ceil_div(cfg.shape.base.kout_group, conv_runtime_detail::kGradSplitKOutputsPerWarp);
  cfg.grad_split_k = conv_runtime_detail::select_grad_split_k(cfg.m_block, static_cast<size_t>(cfg.grad_slice_weights));
  cfg.grad_deterministic_rows_per_chunk = conv_runtime_detail::ceil_div(cfg.m_block, cfg.grad_split_k);
  cfg.grad_reduce_blocks = conv_runtime_detail::ceil_div(cfg.grad_slice_weights, 256);
  cfg.per_group_slice = static_cast<size_t>(cfg.kdim) * cfg.ncol;
  cfg.input_elements = static_cast<size_t>(n) * h * w * c;
  cfg.output_elements = static_cast<size_t>(n) * cfg.shape.base.ho * cfg.shape.base.wo * weights.k;
  cfg.weight_elements = weights.elements();
  return cfg;
}

__host__ __device__ inline size_t idx_nhwc(int n, int h, int w, int c, int H, int W, int C) {
  return ((static_cast<size_t>(n) * H + h) * W + w) * C + c;
}

__host__ __device__ inline size_t idx_krsc(int k, int r, int s, int c, int R, int S, int C) {
  return ((static_cast<size_t>(k) * R + r) * S + s) * C + c;
}

__host__ __device__ inline size_t idx_krsc(int k, int r, int s, int c, int ay, int ax,
                                           int R, int S, int C, int Ay, int Ax) {
  return ((((((static_cast<size_t>(k) * R + r) * S + s) * C + c) * Ay + ay) * Ax) + ax);
}

__host__ __device__ inline size_t idx_kbybxrsc(int k, int by, int bx, int r, int s, int c,
                                               int By, int Bx, int R, int S, int C) {
  return (((((static_cast<size_t>(k) * By + by) * Bx + bx) * R + r) * S + s) * C + c);
}

__host__ __device__ inline size_t idx_kbybxrsc(int k, int by, int bx, int r, int s, int c, int ay, int ax,
                                               int By, int Bx, int R, int S, int C, int Ay, int Ax) {
  return (((((((((static_cast<size_t>(k) * By + by) * Bx + bx) * R + r) * S + s) * C + c) * Ay + ay) * Ax) + ax));
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
  Algo2TiledAtomic = 3,
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
                            const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* f_qp,
                            nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp);
void cpu_fprop_nhwc_qi32_s5(const TensorNHWC& x, const FilterKRSC& w, const Conv2DParams& p,
                            TensorNHWCI32& y,
                            const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* in_qp,
                            const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* f_qp,
                            nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp);
void cpu_block_fprop_nhwc_qi32_u8(const TensorNHWC& x, const BlockFilterKByBxRSC& w, const BlockConv2DParams& p,
                                  TensorNHWCI32& y,
                                  const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* in_qp,
                                  const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* f_qp,
                                  nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp);
void cpu_block_fprop_nhwc_qi32_s5(const TensorNHWC& x, const BlockFilterKByBxRSC& w, const BlockConv2DParams& p,
                                  TensorNHWCI32& y,
                                  const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* in_qp,
                                  const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* f_qp,
                                  nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp);

void launch_fprop_nhwc_qi32_u8(const float* d_x, const float* d_w, int32_t* d_y,
                               int n, int h, int w, int c, int r, int s, int k,
                               const Conv2DParams& p,
                               const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* in_qp,
                               const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* f_qp,
                               nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp);
void launch_fprop_nhwc_qi32_s5(const float* d_x, const float* d_w, int32_t* d_y,
                               int n, int h, int w, int c, int r, int s, int k,
                               const Conv2DParams& p,
                               const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* in_qp,
                               const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* f_qp,
                               nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp);
void launch_block_fprop_nhwc_qi32_u8(const float* d_x, const float* d_w, int32_t* d_y,
                                     int n, int h, int w, int c, int r, int s, int k,
                                     const BlockConv2DParams& p,
                                     const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* in_qp,
                                     const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* f_qp,
                                     nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp);
void launch_block_fprop_nhwc_qi32_s5(const float* d_x, const float* d_w, int32_t* d_y,
                                     int n, int h, int w, int c, int r, int s, int k,
                                     const BlockConv2DParams& p,
                                     const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* in_qp,
                                     const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* f_qp,
                                     nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp);

}  // namespace conv_quant_detail

template <nnalgebra::DataType Tin>
inline void cpu_fprop_nhwc_qi32(const TensorNHWC& x, const FilterKRSC& w, const Conv2DParams& p,
                                TensorNHWCI32& y,
                                const nnalgebra::QuantizationParameters<Tin>* in_qp,
                                const nnalgebra::QuantizationParameters<Tin>* f_qp,
                                nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp = nullptr) {
  static_assert(nnalgebra::kIsSupportedQuantizedInput<Tin>,
                "cpu_fprop_nhwc_qi32 supports only LinQuantU8 and LinQuantS5");
  if (!in_qp || !f_qp) throw std::runtime_error("quantized CPU fprop requires non-null quantization parameters");

  if constexpr (Tin == nnalgebra::DataType::LinQuantU8) {
    conv_quant_detail::cpu_fprop_nhwc_qi32_u8(x, w, p, y, in_qp, f_qp, out_qp);
  } else {
    conv_quant_detail::cpu_fprop_nhwc_qi32_s5(x, w, p, y, in_qp, f_qp, out_qp);
  }
}

template <nnalgebra::DataType Tin>
inline void cpu_block_fprop_nhwc_qi32(const TensorNHWC& x, const BlockFilterKByBxRSC& w, const BlockConv2DParams& p,
                                      TensorNHWCI32& y,
                                      const nnalgebra::QuantizationParameters<Tin>* in_qp,
                                      const nnalgebra::QuantizationParameters<Tin>* f_qp,
                                      nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp = nullptr) {
  static_assert(nnalgebra::kIsSupportedQuantizedInput<Tin>,
                "cpu_block_fprop_nhwc_qi32 supports only LinQuantU8 and LinQuantS5");
  if (!in_qp || !f_qp) throw std::runtime_error("quantized blocked CPU fprop requires non-null quantization parameters");

  if constexpr (Tin == nnalgebra::DataType::LinQuantU8) {
    conv_quant_detail::cpu_block_fprop_nhwc_qi32_u8(x, w, p, y, in_qp, f_qp, out_qp);
  } else {
    conv_quant_detail::cpu_block_fprop_nhwc_qi32_s5(x, w, p, y, in_qp, f_qp, out_qp);
  }
}

void launch_fprop_nhwc(const float* d_x, const float* d_w, float* d_y,
                       int n, int h, int w, int c, int r, int s, int k,
                       const Conv2DParams& p);
void launch_fprop_nhwc(const float* d_x, const float* d_w, float* d_y,
                       const Conv2DRuntimeConfig& cfg);
void launch_bprop_nhwc(const float* d_dy, const float* d_w, float* d_dx,
                       int n, int h, int w, int c, int r, int s, int k,
                       const Conv2DParams& p);
void launch_bprop_nhwc(const float* d_dy, const float* d_w, float* d_dx,
                       const Conv2DRuntimeConfig& cfg);
void launch_grad_nhwc(const float* d_x, const float* d_dy, float* d_dw,
                      int n, int h, int w, int c, int r, int s, int k,
                      const Conv2DParams& p,
                      GradKernelAlgo algo = GradKernelAlgo::GemmIm2Col);
void launch_grad_nhwc(const float* d_x, const float* d_dy, float* d_dw,
                      const Conv2DRuntimeConfig& cfg,
                      GradKernelAlgo algo = GradKernelAlgo::GemmIm2Col);
void launch_block_fprop_nhwc(const float* d_x, const float* d_w, float* d_y,
                             int n, int h, int w, int c, int r, int s, int k,
                             const BlockConv2DParams& p);
void launch_block_fprop_nhwc(const float* d_x, const float* d_w, float* d_y,
                             const BlockConv2DRuntimeConfig& cfg);
void launch_block_bprop_nhwc(const float* d_dy, const float* d_w, float* d_dx,
                             int n, int h, int w, int c, int r, int s, int k,
                             const BlockConv2DParams& p);
void launch_block_bprop_nhwc(const float* d_dy, const float* d_w, float* d_dx,
                             const BlockConv2DRuntimeConfig& cfg);
void launch_block_grad_nhwc(const float* d_x, const float* d_dy, float* d_dw,
                            int n, int h, int w, int c, int r, int s, int k,
                            const BlockConv2DParams& p,
                            GradKernelAlgo algo = GradKernelAlgo::GemmIm2Col);
void launch_block_grad_nhwc(const float* d_x, const float* d_dy, float* d_dw,
                            const BlockConv2DRuntimeConfig& cfg,
                            GradKernelAlgo algo = GradKernelAlgo::GemmIm2Col);

void invalidate_conv_weight_cache(const float* d_w);
void invalidate_block_conv_weight_cache(const float* d_w);
void invalidate_all_conv_workspace_caches();

template <nnalgebra::DataType Tin>
inline void launch_fprop_nhwc_qi32(const float* d_x, const float* d_w, int32_t* d_y,
                                   int n, int h, int w, int c, int r, int s, int k,
                                   const Conv2DParams& p,
                                   const nnalgebra::QuantizationParameters<Tin>* in_qp,
                                   const nnalgebra::QuantizationParameters<Tin>* f_qp,
                                   nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp = nullptr) {
  static_assert(nnalgebra::kIsSupportedQuantizedInput<Tin>,
                "launch_fprop_nhwc_qi32 supports only LinQuantU8 and LinQuantS5");
  if (!in_qp || !f_qp) throw std::runtime_error("quantized CUDA fprop requires non-null quantization parameters");

  if constexpr (Tin == nnalgebra::DataType::LinQuantU8) {
    conv_quant_detail::launch_fprop_nhwc_qi32_u8(d_x, d_w, d_y, n, h, w, c, r, s, k, p, in_qp, f_qp, out_qp);
  } else {
    conv_quant_detail::launch_fprop_nhwc_qi32_s5(d_x, d_w, d_y, n, h, w, c, r, s, k, p, in_qp, f_qp, out_qp);
  }
}

template <nnalgebra::DataType Tin>
inline void launch_block_fprop_nhwc_qi32(const float* d_x, const float* d_w, int32_t* d_y,
                                         int n, int h, int w, int c, int r, int s, int k,
                                         const BlockConv2DParams& p,
                                         const nnalgebra::QuantizationParameters<Tin>* in_qp,
                                         const nnalgebra::QuantizationParameters<Tin>* f_qp,
                                         nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp = nullptr) {
  static_assert(nnalgebra::kIsSupportedQuantizedInput<Tin>,
                "launch_block_fprop_nhwc_qi32 supports only LinQuantU8 and LinQuantS5");
  if (!in_qp || !f_qp) throw std::runtime_error("quantized blocked CUDA fprop requires non-null quantization parameters");

  if constexpr (Tin == nnalgebra::DataType::LinQuantU8) {
    conv_quant_detail::launch_block_fprop_nhwc_qi32_u8(d_x, d_w, d_y, n, h, w, c, r, s, k, p, in_qp, f_qp, out_qp);
  } else {
    conv_quant_detail::launch_block_fprop_nhwc_qi32_s5(d_x, d_w, d_y, n, h, w, c, r, s, k, p, in_qp, f_qp, out_qp);
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
