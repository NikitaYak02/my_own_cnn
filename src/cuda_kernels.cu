#include "conv_types.h"
#include "cuda_utils.h"
#include "bmm.h"

#include <algorithm>
#include <cuda_runtime.h>

namespace {
struct Workspace {
  float* d_col = nullptr;
  size_t d_col_cap = 0;
  float* d_wg_all = nullptr;
  size_t d_wg_all_cap = 0;
  float* d_block_wg_all = nullptr;
  size_t d_block_wg_all_cap = 0;
  const float* packed_w_src = nullptr;
  int packed_r = 0;
  int packed_s = 0;
  int packed_cin_group = 0;
  int packed_k = 0;
  int packed_groups = 0;
  int packed_ay = 1;
  int packed_ax = 1;
  const float* packed_block_w_src = nullptr;
  int packed_block_by = 0;
  int packed_block_bx = 0;
  int packed_block_r = 0;
  int packed_block_s = 0;
  int packed_block_cin_group = 0;
  int packed_block_k = 0;
  int packed_block_groups = 0;
  int packed_block_ay = 1;
  int packed_block_ax = 1;
  float* d_ymat = nullptr;
  size_t d_ymat_cap = 0;
  float* d_dy_mat = nullptr;
  size_t d_dy_mat_cap = 0;
  float* d_dcol = nullptr;
  size_t d_dcol_cap = 0;
  float* d_dwg = nullptr;
  size_t d_dwg_cap = 0;
  float* d_grad_partials = nullptr;
  size_t d_grad_partials_cap = 0;

  ~Workspace() {
    cudaFree(d_col);
    cudaFree(d_wg_all);
    cudaFree(d_block_wg_all);
    cudaFree(d_ymat);
    cudaFree(d_dy_mat);
    cudaFree(d_dcol);
    cudaFree(d_dwg);
    cudaFree(d_grad_partials);
  }
};

void ensure_capacity(float** ptr, size_t* cap, size_t elements) {
  if (*cap >= elements) return;
  if (*ptr) CUDA_CHECK(cudaFree(*ptr));
  CUDA_CHECK(cudaMalloc(ptr, elements * sizeof(float)));
  *cap = elements;
}

constexpr int kGradSplitKBlockSize = 256;
constexpr int kGradSplitKWarpSize = 32;
constexpr int kGradSplitKWarpsPerBlock = kGradSplitKBlockSize / kGradSplitKWarpSize;
constexpr int kGradSplitKOutputsPerWarp = 4;
constexpr int kGradSplitKTargetRowsPerChunk = 2048;
constexpr int kGradSplitKMaxChunks = 16;
constexpr size_t kGradSplitKMaxPartialElements = 8ull * 1024ull * 1024ull;
constexpr int kGradTileRows = 64;
constexpr int kGradTileKDim = 8;
constexpr int kGradTileKOut = 8;
constexpr size_t kGradGemmTileTargetBytes = 64ull * 1024ull * 1024ull;
constexpr int kGradGemmMinRowsPerChunk = 2048;

int select_grad_split_k(int rows, size_t slice_weights) {
  if (rows <= 0 || slice_weights == 0) return 1;

  int split_k = std::max(1, (rows + kGradSplitKTargetRowsPerChunk - 1) / kGradSplitKTargetRowsPerChunk);
  split_k = std::min(split_k, kGradSplitKMaxChunks);

  const size_t max_by_memory = std::max<size_t>(1, kGradSplitKMaxPartialElements / slice_weights);
  split_k = std::min(split_k, static_cast<int>(max_by_memory));
  return std::max(1, split_k);
}

int select_grad_gemm_rows_per_chunk(int rows, int kdim, int ncol) {
  if (rows <= 0) return 1;
  const size_t row_bytes = static_cast<size_t>(kdim + ncol) * sizeof(float);
  if (row_bytes == 0) return rows;

  int chunk_rows = static_cast<int>(kGradGemmTileTargetBytes / row_bytes);
  chunk_rows = std::max(kGradGemmMinRowsPerChunk, chunk_rows);
  chunk_rows = std::min(chunk_rows, rows);
  return std::max(1, chunk_rows);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = kGradSplitKWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ __forceinline__ void decode_conv_row(int row, int ho, int wo,
                                                int& n_idx, int& ho_idx, int& wo_idx) {
  n_idx = row / (ho * wo);
  const int rem = row - n_idx * (ho * wo);
  ho_idx = rem / wo;
  wo_idx = rem - ho_idx * wo;
}

__device__ __forceinline__ void decode_kdim_index(int kdim_idx, int s, int cin_group,
                                                  int& rr, int& ss, int& ci) {
  ci = kdim_idx % cin_group;
  const int t0 = kdim_idx / cin_group;
  ss = t0 % s;
  rr = t0 / s;
}

__device__ __forceinline__ void decode_output_app_index(int ext_idx, int ay, int ax,
                                                        int& ko, int& ay_idx, int& ax_idx) {
  const int app = ay * ax;
  ko = ext_idx / app;
  const int rem = ext_idx - ko * app;
  ay_idx = rem / ax;
  ax_idx = rem - ay_idx * ax;
}

__global__ void pack_filter_krsc_group_kernel(const float* __restrict__ w,
                                              float* __restrict__ wg,
                                              int r, int s, int cin_group,
                                              int ay, int ax,
                                              int k_base, int kout_group);
__global__ void pack_block_filter_kbybxrsc_group_kernel(const float* __restrict__ w,
                                                        float* __restrict__ wg,
                                                        int by_count, int bx_count,
                                                        int r, int s, int cin_group,
                                                        int ay, int ax,
                                                        int block_y, int block_x,
                                                        int k_base, int kout_group);
__global__ void pack_nhwc_group_matrix_app_kernel(const float* __restrict__ src,
                                                  float* __restrict__ dst,
                                                  int n, int base_h, int base_w,
                                                  int ay, int ax,
                                                  int c, int c_base, int c_count);
__global__ void pack_nhwc_group_matrix_app_rows_kernel(const float* __restrict__ src,
                                                       float* __restrict__ dst,
                                                       int n, int base_h, int base_w,
                                                       int ay, int ax,
                                                       int c,
                                                       int row_start, int row_count,
                                                       int c_base, int c_count);
__global__ void unpack_matrix_to_nhwc_group_app_kernel(const float* __restrict__ src,
                                                       float* __restrict__ dst,
                                                       int n, int base_h, int base_w,
                                                       int ay, int ax,
                                                       int c, int c_base, int c_count);
__global__ void pack_nhwc_group_matrix_app_block_kernel(const float* __restrict__ src,
                                                        float* __restrict__ dst,
                                                        int n, int base_h, int base_w, int c,
                                                        int ho_start_base, int wo_start_base,
                                                        int block_ho_base, int block_wo_base,
                                                        int ay, int ax,
                                                        int c_base, int c_count);
__global__ void pack_nhwc_group_matrix_app_block_rows_kernel(const float* __restrict__ src,
                                                             float* __restrict__ dst,
                                                             int n, int base_h, int base_w, int c,
                                                             int ho_start_base, int wo_start_base,
                                                             int block_ho_base, int block_wo_base,
                                                             int ay, int ax,
                                                             int row_start, int row_count,
                                                             int c_base, int c_count);
__global__ void unpack_matrix_to_nhwc_group_app_block_kernel(const float* __restrict__ src,
                                                             float* __restrict__ dst,
                                                             int n, int base_h, int base_w, int c,
                                                             int ho_start_base, int wo_start_base,
                                                             int block_ho_base, int block_wo_base,
                                                             int ay, int ax,
                                                             int c_base, int c_count);

void ensure_packed_weights(Workspace& ws,
                           const float* d_w,
                           int r, int s, int cin_group, int k, int groups, int ay, int ax) {
  const size_t kdim = static_cast<size_t>(r) * s * cin_group;
  const size_t ncol = static_cast<size_t>(k / groups) * ay * ax;
  const size_t per_group = kdim * ncol;
  const size_t total = per_group * groups;
  ensure_capacity(&ws.d_wg_all, &ws.d_wg_all_cap, total);

  const bool need_repack =
      (ws.packed_w_src != d_w) ||
      (ws.packed_r != r) ||
      (ws.packed_s != s) ||
      (ws.packed_cin_group != cin_group) ||
      (ws.packed_k != k) ||
      (ws.packed_groups != groups) ||
      (ws.packed_ay != ay) ||
      (ws.packed_ax != ax);
  if (!need_repack) return;

  const int t = 256;
  for (int g = 0; g < groups; ++g) {
    const int kout_base = g * (k / groups);
    float* dst = ws.d_wg_all + g * per_group;
    const int blocks = static_cast<int>((per_group + t - 1) / t);
    pack_filter_krsc_group_kernel<<<blocks, t>>>(d_w, dst, r, s, cin_group, ay, ax, kout_base, k / groups);
  }
  CUDA_CHECK(cudaGetLastError());
  ws.packed_w_src = d_w;
  ws.packed_r = r;
  ws.packed_s = s;
  ws.packed_cin_group = cin_group;
  ws.packed_k = k;
  ws.packed_groups = groups;
  ws.packed_ay = ay;
  ws.packed_ax = ax;
}

size_t packed_block_weight_offset(int groups, int by_count, int bx_count,
                                  int group, int by, int bx,
                                  size_t per_group_slice) {
  (void)groups;
  return (((static_cast<size_t>(group) * by_count) + by) * bx_count + bx) * per_group_slice;
}

void ensure_packed_block_weights(Workspace& ws,
                                 const float* d_w,
                                 int by_count, int bx_count,
                                 int r, int s, int cin_group, int k, int groups, int ay, int ax) {
  const size_t kdim = static_cast<size_t>(r) * s * cin_group;
  const size_t ncol = static_cast<size_t>(k / groups) * ay * ax;
  const size_t per_group_slice = kdim * ncol;
  const size_t total = per_group_slice * groups * by_count * bx_count;
  ensure_capacity(&ws.d_block_wg_all, &ws.d_block_wg_all_cap, total);

  const bool need_repack =
      (ws.packed_block_w_src != d_w) ||
      (ws.packed_block_by != by_count) ||
      (ws.packed_block_bx != bx_count) ||
      (ws.packed_block_r != r) ||
      (ws.packed_block_s != s) ||
      (ws.packed_block_cin_group != cin_group) ||
      (ws.packed_block_k != k) ||
      (ws.packed_block_groups != groups) ||
      (ws.packed_block_ay != ay) ||
      (ws.packed_block_ax != ax);
  if (!need_repack) return;

  const int t = 256;
  for (int g = 0; g < groups; ++g) {
    const int kout_base = g * (k / groups);
    for (int by = 0; by < by_count; ++by) {
      for (int bx = 0; bx < bx_count; ++bx) {
        float* dst = ws.d_block_wg_all + packed_block_weight_offset(groups, by_count, bx_count, g, by, bx, per_group_slice);
        const int blocks = static_cast<int>((per_group_slice + t - 1) / t);
        pack_block_filter_kbybxrsc_group_kernel<<<blocks, t>>>(d_w, dst,
                                                               by_count, bx_count,
                                                               r, s, cin_group, ay, ax,
                                                               by, bx,
                                                               kout_base, k / groups);
      }
    }
  }
  CUDA_CHECK(cudaGetLastError());
  ws.packed_block_w_src = d_w;
  ws.packed_block_by = by_count;
  ws.packed_block_bx = bx_count;
  ws.packed_block_r = r;
  ws.packed_block_s = s;
  ws.packed_block_cin_group = cin_group;
  ws.packed_block_k = k;
  ws.packed_block_groups = groups;
  ws.packed_block_ay = ay;
  ws.packed_block_ax = ax;
}

Workspace& workspace() {
  static Workspace ws;
  return ws;
}

__global__ void im2col_nhwc_kernel(const float* __restrict__ x,
                                   float* __restrict__ col,
                                   int n, int h, int w, int c,
                                   int ho, int wo,
                                   int r, int s,
                                   int pad_h, int pad_w,
                                   int stride_h, int stride_w,
                                   int dilation_h, int dilation_w,
                                   int c_base, int cin_group) {
  const int m = n * ho * wo;
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * kdim;
  if (idx >= total) return;

  const int row = idx / kdim;
  const int col_idx = idx - row * kdim;

  const int ci = col_idx % cin_group;
  const int t = col_idx / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  const int n_idx = row / (ho * wo);
  const int rem = row - n_idx * (ho * wo);
  const int ho_idx = rem / wo;
  const int wo_idx = rem - ho_idx * wo;

  const int hi = ho_idx * stride_h - pad_h + rr * dilation_h;
  const int wi = wo_idx * stride_w - pad_w + ss * dilation_w;

  float v = 0.0f;
  if (hi >= 0 && hi < h && wi >= 0 && wi < w) {
    v = x[idx_nhwc(n_idx, hi, wi, c_base + ci, h, w, c)];
  }
  col[idx] = v;
}

__global__ void im2col_nhwc_vec4_kernel(const float* __restrict__ x,
                                        float* __restrict__ col,
                                        int n, int h, int w, int c,
                                        int ho, int wo,
                                        int r, int s,
                                        int pad_h, int pad_w,
                                        int stride_h, int stride_w,
                                        int dilation_h, int dilation_w,
                                        int c_base, int cin_group) {
  const int m = n * ho * wo;
  const int cin_group4 = cin_group / 4;
  const int kdim4 = r * s * cin_group4;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * kdim4;
  if (idx >= total) return;

  const int row = idx / kdim4;
  const int col_idx4 = idx - row * kdim4;

  const int ci4 = (col_idx4 % cin_group4) * 4;
  const int t = col_idx4 / cin_group4;
  const int ss = t % s;
  const int rr = t / s;

  const int n_idx = row / (ho * wo);
  const int rem = row - n_idx * (ho * wo);
  const int ho_idx = rem / wo;
  const int wo_idx = rem - ho_idx * wo;

  const int hi = ho_idx * stride_h - pad_h + rr * dilation_h;
  const int wi = wo_idx * stride_w - pad_w + ss * dilation_w;

  const int out_base = row * (r * s * cin_group) + t * cin_group + ci4;
  if (hi >= 0 && hi < h && wi >= 0 && wi < w) {
    const int x_idx = ((n_idx * h + hi) * w + wi) * c + (c_base + ci4);
    *reinterpret_cast<float4*>(col + out_base) = *reinterpret_cast<const float4*>(x + x_idx);
  } else {
    *reinterpret_cast<float4*>(col + out_base) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}

__global__ void pack_nhwc_group_matrix_kernel(const float* __restrict__ src,
                                              float* __restrict__ dst,
                                              int n, int h, int w, int c,
                                              int c_base, int c_count) {
  const int m = n * h * w;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * c_count;
  if (idx >= total) return;

  const int row = idx / c_count;
  const int col = idx - row * c_count;

  const int n_idx = row / (h * w);
  const int rem = row - n_idx * (h * w);
  const int h_idx = rem / w;
  const int w_idx = rem - h_idx * w;

  dst[idx] = src[idx_nhwc(n_idx, h_idx, w_idx, c_base + col, h, w, c)];
}

__global__ void pack_nhwc_group_matrix_app_kernel(const float* __restrict__ src,
                                                  float* __restrict__ dst,
                                                  int n, int base_h, int base_w,
                                                  int ay, int ax,
                                                  int c, int c_base, int c_count) {
  const int m = n * base_h * base_w;
  const int ncol = c_count * ay * ax;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * ncol;
  if (idx >= total) return;

  const int row = idx / ncol;
  const int ext_col = idx - row * ncol;

  int ko, ay_idx, ax_idx;
  decode_output_app_index(ext_col, ay, ax, ko, ay_idx, ax_idx);

  int n_idx, ho_base, wo_base;
  decode_conv_row(row, base_h, base_w, n_idx, ho_base, wo_base);
  dst[idx] = src[idx_nhwc(n_idx, ho_base * ay + ay_idx, wo_base * ax + ax_idx,
                          c_base + ko, base_h * ay, base_w * ax, c)];
}

__global__ void im2col_nhwc_rows_kernel(const float* __restrict__ x,
                                        float* __restrict__ col,
                                        int n, int h, int w, int c,
                                        int ho, int wo,
                                        int row_start, int row_count,
                                        int r, int s,
                                        int pad_h, int pad_w,
                                        int stride_h, int stride_w,
                                        int dilation_h, int dilation_w,
                                        int c_base, int cin_group) {
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = row_count * kdim;
  if (idx >= total) return;

  const int local_row = idx / kdim;
  const int col_idx = idx - local_row * kdim;
  const int row = row_start + local_row;

  const int ci = col_idx % cin_group;
  const int t = col_idx / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  int n_idx, ho_idx, wo_idx;
  decode_conv_row(row, ho, wo, n_idx, ho_idx, wo_idx);

  const int hi = ho_idx * stride_h - pad_h + rr * dilation_h;
  const int wi = wo_idx * stride_w - pad_w + ss * dilation_w;
  float v = 0.0f;
  if (hi >= 0 && hi < h && wi >= 0 && wi < w) {
    v = x[idx_nhwc(n_idx, hi, wi, c_base + ci, h, w, c)];
  }
  col[idx] = v;
}

__global__ void im2col_nhwc_vec4_rows_kernel(const float* __restrict__ x,
                                             float* __restrict__ col,
                                             int n, int h, int w, int c,
                                             int ho, int wo,
                                             int row_start, int row_count,
                                             int r, int s,
                                             int pad_h, int pad_w,
                                             int stride_h, int stride_w,
                                             int dilation_h, int dilation_w,
                                             int c_base, int cin_group) {
  const int cin_group4 = cin_group / 4;
  const int kdim4 = r * s * cin_group4;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = row_count * kdim4;
  if (idx >= total) return;

  const int local_row = idx / kdim4;
  const int col_idx4 = idx - local_row * kdim4;
  const int row = row_start + local_row;

  const int ci4 = (col_idx4 % cin_group4) * 4;
  const int t = col_idx4 / cin_group4;
  const int ss = t % s;
  const int rr = t / s;

  int n_idx, ho_idx, wo_idx;
  decode_conv_row(row, ho, wo, n_idx, ho_idx, wo_idx);

  const int hi = ho_idx * stride_h - pad_h + rr * dilation_h;
  const int wi = wo_idx * stride_w - pad_w + ss * dilation_w;
  const int out_base = local_row * (r * s * cin_group) + t * cin_group + ci4;
  if (hi >= 0 && hi < h && wi >= 0 && wi < w) {
    const int x_idx = ((n_idx * h + hi) * w + wi) * c + (c_base + ci4);
    *reinterpret_cast<float4*>(col + out_base) = *reinterpret_cast<const float4*>(x + x_idx);
  } else {
    *reinterpret_cast<float4*>(col + out_base) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}

__global__ void pack_nhwc_group_matrix_rows_kernel(const float* __restrict__ src,
                                                   float* __restrict__ dst,
                                                   int n, int h, int w, int c,
                                                   int row_start, int row_count,
                                                   int c_base, int c_count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = row_count * c_count;
  if (idx >= total) return;

  const int local_row = idx / c_count;
  const int col = idx - local_row * c_count;
  const int row = row_start + local_row;

  int n_idx, h_idx, w_idx;
  decode_conv_row(row, h, w, n_idx, h_idx, w_idx);
  dst[idx] = src[idx_nhwc(n_idx, h_idx, w_idx, c_base + col, h, w, c)];
}

__global__ void pack_nhwc_group_matrix_app_rows_kernel(const float* __restrict__ src,
                                                       float* __restrict__ dst,
                                                       int n, int base_h, int base_w,
                                                       int ay, int ax,
                                                       int c,
                                                       int row_start, int row_count,
                                                       int c_base, int c_count) {
  const int ncol = c_count * ay * ax;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = row_count * ncol;
  if (idx >= total) return;

  const int local_row = idx / ncol;
  const int ext_col = idx - local_row * ncol;
  const int row = row_start + local_row;

  int ko, ay_idx, ax_idx;
  decode_output_app_index(ext_col, ay, ax, ko, ay_idx, ax_idx);

  int n_idx, ho_base, wo_base;
  decode_conv_row(row, base_h, base_w, n_idx, ho_base, wo_base);
  dst[idx] = src[idx_nhwc(n_idx, ho_base * ay + ay_idx, wo_base * ax + ax_idx,
                          c_base + ko, base_h * ay, base_w * ax, c)];
}

__global__ void unpack_matrix_to_nhwc_group_kernel(const float* __restrict__ src,
                                                   float* __restrict__ dst,
                                                   int n, int h, int w, int c,
                                                   int c_base, int c_count) {
  const int m = n * h * w;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * c_count;
  if (idx >= total) return;

  const int row = idx / c_count;
  const int col = idx - row * c_count;

  const int n_idx = row / (h * w);
  const int rem = row - n_idx * (h * w);
  const int h_idx = rem / w;
  const int w_idx = rem - h_idx * w;

  dst[idx_nhwc(n_idx, h_idx, w_idx, c_base + col, h, w, c)] = src[idx];
}

__global__ void pack_filter_krsc_group_kernel(const float* __restrict__ w,
                                              float* __restrict__ wg,
                                              int r, int s, int cin_group,
                                              int ay, int ax,
                                              int k_base, int kout_group) {
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kout_group * ay * ax * kdim;
  if (idx >= total) return;

  const int ko_ext = idx / kdim;
  const int k_idx = idx - ko_ext * kdim;
  int ko = 0;
  int ay_idx = 0;
  int ax_idx = 0;
  decode_output_app_index(ko_ext, ay, ax, ko, ay_idx, ax_idx);

  const int ci = k_idx % cin_group;
  const int t = k_idx / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  wg[idx] = w[idx_krsc(k_base + ko, rr, ss, ci, ay_idx, ax_idx, r, s, cin_group, ay, ax)];
}

__global__ void unpack_filter_krsc_group_kernel(const float* __restrict__ wg,
                                                float* __restrict__ w,
                                                int r, int s, int cin_group,
                                                int ay, int ax,
                                                int k_total,
                                                int k_base, int kout_group) {
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kdim * kout_group * ay * ax;
  if (idx >= total) return;

  const int ncol = kout_group * ay * ax;
  const int row = idx / ncol;
  const int col_ext = idx - row * ncol;
  int ko = 0;
  int ay_idx = 0;
  int ax_idx = 0;
  decode_output_app_index(col_ext, ay, ax, ko, ay_idx, ax_idx);

  const int ci = row % cin_group;
  const int t = row / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  (void)k_total;
  w[idx_krsc(k_base + ko, rr, ss, ci, ay_idx, ax_idx, r, s, cin_group, ay, ax)] = wg[idx];
}

__global__ void pack_block_filter_kbybxrsc_group_kernel(const float* __restrict__ w,
                                                        float* __restrict__ wg,
                                                        int by_count, int bx_count,
                                                        int r, int s, int cin_group,
                                                        int ay, int ax,
                                                        int block_y, int block_x,
                                                        int k_base, int kout_group) {
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kout_group * ay * ax * kdim;
  if (idx >= total) return;

  const int ko_ext = idx / kdim;
  const int k_idx = idx - ko_ext * kdim;
  int ko = 0;
  int ay_idx = 0;
  int ax_idx = 0;
  decode_output_app_index(ko_ext, ay, ax, ko, ay_idx, ax_idx);

  const int ci = k_idx % cin_group;
  const int t = k_idx / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  wg[idx] = w[idx_kbybxrsc(k_base + ko, block_y, block_x, rr, ss, ci, ay_idx, ax_idx,
                           by_count, bx_count, r, s, cin_group, ay, ax)];
}

__global__ void unpack_block_filter_kbybxrsc_group_kernel(const float* __restrict__ wg,
                                                          float* __restrict__ w,
                                                          int by_count, int bx_count,
                                                          int r, int s, int cin_group,
                                                          int ay, int ax,
                                                          int block_y, int block_x,
                                                          int k_total,
                                                          int k_base, int kout_group) {
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kdim * kout_group * ay * ax;
  if (idx >= total) return;

  const int ncol = kout_group * ay * ax;
  const int row = idx / ncol;
  const int col_ext = idx - row * ncol;
  int ko = 0;
  int ay_idx = 0;
  int ax_idx = 0;
  decode_output_app_index(col_ext, ay, ax, ko, ay_idx, ax_idx);

  const int ci = row % cin_group;
  const int t = row / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  (void)k_total;
  w[idx_kbybxrsc(k_base + ko, block_y, block_x, rr, ss, ci, ay_idx, ax_idx,
                 by_count, bx_count, r, s, cin_group, ay, ax)] = wg[idx];
}

__global__ void im2col_nhwc_block_kernel(const float* __restrict__ x,
                                         float* __restrict__ col,
                                         int n, int h, int w, int c,
                                         int ho_start, int wo_start,
                                         int block_ho, int block_wo,
                                         int r, int s,
                                         int pad_h, int pad_w,
                                         int stride_h, int stride_w,
                                         int dilation_h, int dilation_w,
                                         int c_base, int cin_group) {
  const int m = n * block_ho * block_wo;
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * kdim;
  if (idx >= total) return;

  const int row = idx / kdim;
  const int col_idx = idx - row * kdim;

  const int ci = col_idx % cin_group;
  const int t = col_idx / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  const int n_idx = row / (block_ho * block_wo);
  const int rem = row - n_idx * (block_ho * block_wo);
  const int lho = rem / block_wo;
  const int lwo = rem - lho * block_wo;

  const int ho = ho_start + lho;
  const int wo = wo_start + lwo;
  const int hi = ho * stride_h - pad_h + rr * dilation_h;
  const int wi = wo * stride_w - pad_w + ss * dilation_w;

  float v = 0.0f;
  if (hi >= 0 && hi < h && wi >= 0 && wi < w) {
    v = x[idx_nhwc(n_idx, hi, wi, c_base + ci, h, w, c)];
  }
  col[idx] = v;
}

__global__ void pack_nhwc_group_matrix_block_kernel(const float* __restrict__ src,
                                                    float* __restrict__ dst,
                                                    int n, int h, int w, int c,
                                                    int ho_start, int wo_start,
                                                    int block_ho, int block_wo,
                                                    int c_base, int c_count) {
  const int m = n * block_ho * block_wo;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * c_count;
  if (idx >= total) return;

  const int row = idx / c_count;
  const int col = idx - row * c_count;

  const int n_idx = row / (block_ho * block_wo);
  const int rem = row - n_idx * (block_ho * block_wo);
  const int lho = rem / block_wo;
  const int lwo = rem - lho * block_wo;

  dst[idx] = src[idx_nhwc(n_idx, ho_start + lho, wo_start + lwo, c_base + col, h, w, c)];
}

__global__ void pack_nhwc_group_matrix_app_block_kernel(const float* __restrict__ src,
                                                        float* __restrict__ dst,
                                                        int n, int base_h, int base_w, int c,
                                                        int ho_start_base, int wo_start_base,
                                                        int block_ho_base, int block_wo_base,
                                                        int ay, int ax,
                                                        int c_base, int c_count) {
  const int m = n * block_ho_base * block_wo_base;
  const int ncol = c_count * ay * ax;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * ncol;
  if (idx >= total) return;

  const int row = idx / ncol;
  const int ext_col = idx - row * ncol;

  int ko, ay_idx, ax_idx;
  decode_output_app_index(ext_col, ay, ax, ko, ay_idx, ax_idx);

  const int n_idx = row / (block_ho_base * block_wo_base);
  const int rem = row - n_idx * (block_ho_base * block_wo_base);
  const int lho = rem / block_wo_base;
  const int lwo = rem - lho * block_wo_base;
  const int ho_base = ho_start_base + lho;
  const int wo_base = wo_start_base + lwo;

  dst[idx] = src[idx_nhwc(n_idx, ho_base * ay + ay_idx, wo_base * ax + ax_idx,
                          c_base + ko, base_h * ay, base_w * ax, c)];
}

__global__ void im2col_nhwc_block_rows_kernel(const float* __restrict__ x,
                                              float* __restrict__ col,
                                              int n, int h, int w, int c,
                                              int ho_start, int wo_start,
                                              int block_ho, int block_wo,
                                              int row_start, int row_count,
                                              int r, int s,
                                              int pad_h, int pad_w,
                                              int stride_h, int stride_w,
                                              int dilation_h, int dilation_w,
                                              int c_base, int cin_group) {
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = row_count * kdim;
  if (idx >= total) return;

  const int local_row = idx / kdim;
  const int col_idx = idx - local_row * kdim;
  const int row = row_start + local_row;

  const int ci = col_idx % cin_group;
  const int t = col_idx / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  const int n_idx = row / (block_ho * block_wo);
  const int rem = row - n_idx * (block_ho * block_wo);
  const int lho = rem / block_wo;
  const int lwo = rem - lho * block_wo;
  const int ho = ho_start + lho;
  const int wo = wo_start + lwo;
  const int hi = ho * stride_h - pad_h + rr * dilation_h;
  const int wi = wo * stride_w - pad_w + ss * dilation_w;

  float v = 0.0f;
  if (hi >= 0 && hi < h && wi >= 0 && wi < w) {
    v = x[idx_nhwc(n_idx, hi, wi, c_base + ci, h, w, c)];
  }
  col[idx] = v;
}

__global__ void pack_nhwc_group_matrix_block_rows_kernel(const float* __restrict__ src,
                                                         float* __restrict__ dst,
                                                         int n, int h, int w, int c,
                                                         int ho_start, int wo_start,
                                                         int block_ho, int block_wo,
                                                         int row_start, int row_count,
                                                         int c_base, int c_count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = row_count * c_count;
  if (idx >= total) return;

  const int local_row = idx / c_count;
  const int col = idx - local_row * c_count;
  const int row = row_start + local_row;

  const int n_idx = row / (block_ho * block_wo);
  const int rem = row - n_idx * (block_ho * block_wo);
  const int lho = rem / block_wo;
  const int lwo = rem - lho * block_wo;

  dst[idx] = src[idx_nhwc(n_idx, ho_start + lho, wo_start + lwo, c_base + col, h, w, c)];
}

__global__ void pack_nhwc_group_matrix_app_block_rows_kernel(const float* __restrict__ src,
                                                             float* __restrict__ dst,
                                                             int n, int base_h, int base_w, int c,
                                                             int ho_start_base, int wo_start_base,
                                                             int block_ho_base, int block_wo_base,
                                                             int ay, int ax,
                                                             int row_start, int row_count,
                                                             int c_base, int c_count) {
  const int ncol = c_count * ay * ax;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = row_count * ncol;
  if (idx >= total) return;

  const int local_row = idx / ncol;
  const int ext_col = idx - local_row * ncol;
  const int row = row_start + local_row;

  int ko, ay_idx, ax_idx;
  decode_output_app_index(ext_col, ay, ax, ko, ay_idx, ax_idx);

  const int n_idx = row / (block_ho_base * block_wo_base);
  const int rem = row - n_idx * (block_ho_base * block_wo_base);
  const int lho = rem / block_wo_base;
  const int lwo = rem - lho * block_wo_base;
  const int ho_base = ho_start_base + lho;
  const int wo_base = wo_start_base + lwo;

  dst[idx] = src[idx_nhwc(n_idx, ho_base * ay + ay_idx, wo_base * ax + ax_idx,
                          c_base + ko, base_h * ay, base_w * ax, c)];
}

__global__ void unpack_matrix_to_nhwc_group_block_kernel(const float* __restrict__ src,
                                                         float* __restrict__ dst,
                                                         int n, int h, int w, int c,
                                                         int ho_start, int wo_start,
                                                         int block_ho, int block_wo,
                                                         int c_base, int c_count) {
  const int m = n * block_ho * block_wo;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * c_count;
  if (idx >= total) return;

  const int row = idx / c_count;
  const int col = idx - row * c_count;

  const int n_idx = row / (block_ho * block_wo);
  const int rem = row - n_idx * (block_ho * block_wo);
  const int lho = rem / block_wo;
  const int lwo = rem - lho * block_wo;

  dst[idx_nhwc(n_idx, ho_start + lho, wo_start + lwo, c_base + col, h, w, c)] = src[idx];
}

__global__ void unpack_matrix_to_nhwc_group_app_block_kernel(const float* __restrict__ src,
                                                             float* __restrict__ dst,
                                                             int n, int base_h, int base_w, int c,
                                                             int ho_start_base, int wo_start_base,
                                                             int block_ho_base, int block_wo_base,
                                                             int ay, int ax,
                                                             int c_base, int c_count) {
  const int m = n * block_ho_base * block_wo_base;
  const int ncol = c_count * ay * ax;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * ncol;
  if (idx >= total) return;

  const int row = idx / ncol;
  const int ext_col = idx - row * ncol;

  int ko, ay_idx, ax_idx;
  decode_output_app_index(ext_col, ay, ax, ko, ay_idx, ax_idx);

  const int n_idx = row / (block_ho_base * block_wo_base);
  const int rem = row - n_idx * (block_ho_base * block_wo_base);
  const int lho = rem / block_wo_base;
  const int lwo = rem - lho * block_wo_base;
  const int ho_base = ho_start_base + lho;
  const int wo_base = wo_start_base + lwo;

  dst[idx_nhwc(n_idx, ho_base * ay + ay_idx, wo_base * ax + ax_idx,
               c_base + ko, base_h * ay, base_w * ax, c)] = src[idx];
}

__global__ void col2im_accum_nhwc_block_kernel(const float* __restrict__ dcol,
                                               float* __restrict__ dx,
                                               int n, int h, int w, int c,
                                               int ho_start, int wo_start,
                                               int block_ho, int block_wo,
                                               int r, int s,
                                               int pad_h, int pad_w,
                                               int stride_h, int stride_w,
                                               int dilation_h, int dilation_w,
                                               int c_base, int cin_group) {
  const int m = n * block_ho * block_wo;
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * kdim;
  if (idx >= total) return;

  const int row = idx / kdim;
  const int col_idx = idx - row * kdim;

  const int ci = col_idx % cin_group;
  const int t = col_idx / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  const int n_idx = row / (block_ho * block_wo);
  const int rem = row - n_idx * (block_ho * block_wo);
  const int lho = rem / block_wo;
  const int lwo = rem - lho * block_wo;

  const int ho = ho_start + lho;
  const int wo = wo_start + lwo;
  const int hi = ho * stride_h - pad_h + rr * dilation_h;
  const int wi = wo * stride_w - pad_w + ss * dilation_w;
  if (hi >= 0 && hi < h && wi >= 0 && wi < w) {
    atomicAdd(&dx[idx_nhwc(n_idx, hi, wi, c_base + ci, h, w, c)], dcol[idx]);
  }
}

__global__ void gather_input_block_nhwc_kernel(const float* __restrict__ src,
                                               float* __restrict__ dst,
                                               int n, int src_h, int src_w, int c,
                                               int hi_start, int wi_start,
                                               int local_h, int local_w) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n * local_h * local_w * c;
  if (idx >= total) return;

  const int ci = idx % c;
  const int t0 = idx / c;
  const int wi = t0 % local_w;
  const int t1 = t0 / local_w;
  const int hi = t1 % local_h;
  const int ni = t1 / local_h;

  const int src_hi = hi_start + hi;
  const int src_wi = wi_start + wi;
  float v = 0.0f;
  if (src_hi >= 0 && src_hi < src_h && src_wi >= 0 && src_wi < src_w) {
    v = src[idx_nhwc(ni, src_hi, src_wi, ci, src_h, src_w, c)];
  }
  dst[idx_nhwc(ni, hi, wi, ci, local_h, local_w, c)] = v;
}

__global__ void gather_output_block_nhwc_kernel(const float* __restrict__ src,
                                                float* __restrict__ dst,
                                                int n, int src_h, int src_w, int c,
                                                int ho_start, int wo_start,
                                                int block_ho, int block_wo) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n * block_ho * block_wo * c;
  if (idx >= total) return;

  const int ci = idx % c;
  const int t0 = idx / c;
  const int wo = t0 % block_wo;
  const int t1 = t0 / block_wo;
  const int ho = t1 % block_ho;
  const int ni = t1 / block_ho;

  dst[idx_nhwc(ni, ho, wo, ci, block_ho, block_wo, c)] =
      src[idx_nhwc(ni, ho_start + ho, wo_start + wo, ci, src_h, src_w, c)];
}

__global__ void scatter_output_block_nhwc_kernel(const float* __restrict__ src,
                                                 float* __restrict__ dst,
                                                 int n, int dst_h, int dst_w, int c,
                                                 int ho_start, int wo_start,
                                                 int block_ho, int block_wo) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n * block_ho * block_wo * c;
  if (idx >= total) return;

  const int ci = idx % c;
  const int t0 = idx / c;
  const int wo = t0 % block_wo;
  const int t1 = t0 / block_wo;
  const int ho = t1 % block_ho;
  const int ni = t1 / block_ho;

  dst[idx_nhwc(ni, ho_start + ho, wo_start + wo, ci, dst_h, dst_w, c)] =
      src[idx_nhwc(ni, ho, wo, ci, block_ho, block_wo, c)];
}

__global__ void scatter_add_input_block_nhwc_kernel(const float* __restrict__ src,
                                                    float* __restrict__ dst,
                                                    int n, int dst_h, int dst_w, int c,
                                                    int hi_start, int wi_start,
                                                    int local_h, int local_w) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n * local_h * local_w * c;
  if (idx >= total) return;

  const int ci = idx % c;
  const int t0 = idx / c;
  const int wi = t0 % local_w;
  const int t1 = t0 / local_w;
  const int hi = t1 % local_h;
  const int ni = t1 / local_h;

  const int dst_hi = hi_start + hi;
  const int dst_wi = wi_start + wi;
  if (dst_hi >= 0 && dst_hi < dst_h && dst_wi >= 0 && dst_wi < dst_w) {
    dst[idx_nhwc(ni, dst_hi, dst_wi, ci, dst_h, dst_w, c)] +=
        src[idx_nhwc(ni, hi, wi, ci, local_h, local_w, c)];
  }
}

__global__ void gather_block_filter_to_krsc_kernel(const float* __restrict__ src,
                                                   float* __restrict__ dst,
                                                   int by_count, int bx_count,
                                                   int r, int s, int cin_group, int k,
                                                   int block_y, int block_x) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = k * r * s * cin_group;
  if (idx >= total) return;

  const int ci = idx % cin_group;
  const int t0 = idx / cin_group;
  const int ss = t0 % s;
  const int t1 = t0 / s;
  const int rr = t1 % r;
  const int ko = t1 / r;

  dst[idx_krsc(ko, rr, ss, ci, r, s, cin_group)] =
      src[idx_kbybxrsc(ko, block_y, block_x, rr, ss, ci, by_count, bx_count, r, s, cin_group)];
}

__global__ void scatter_block_filter_from_krsc_kernel(const float* __restrict__ src,
                                                      float* __restrict__ dst,
                                                      int by_count, int bx_count,
                                                      int r, int s, int cin_group, int k,
                                                      int block_y, int block_x) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = k * r * s * cin_group;
  if (idx >= total) return;

  const int ci = idx % cin_group;
  const int t0 = idx / cin_group;
  const int ss = t0 % s;
  const int t1 = t0 / s;
  const int rr = t1 % r;
  const int ko = t1 / r;

  dst[idx_kbybxrsc(ko, block_y, block_x, rr, ss, ci, by_count, bx_count, r, s, cin_group)] =
      src[idx_krsc(ko, rr, ss, ci, r, s, cin_group)];
}

__global__ void col2im_accum_nhwc_kernel(const float* __restrict__ dcol,
                                         float* __restrict__ dx,
                                         int n, int h, int w, int c,
                                         int ho, int wo,
                                         int r, int s,
                                         int pad_h, int pad_w,
                                         int stride_h, int stride_w,
                                         int dilation_h, int dilation_w,
                                         int c_base, int cin_group) {
  const int m = n * ho * wo;
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * kdim;
  if (idx >= total) return;

  const int row = idx / kdim;
  const int col_idx = idx - row * kdim;

  const int ci = col_idx % cin_group;
  const int t = col_idx / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  const int n_idx = row / (ho * wo);
  const int rem = row - n_idx * (ho * wo);
  const int ho_idx = rem / wo;
  const int wo_idx = rem - ho_idx * wo;

  const int hi = ho_idx * stride_h - pad_h + rr * dilation_h;
  const int wi = wo_idx * stride_w - pad_w + ss * dilation_w;
  if (hi >= 0 && hi < h && wi >= 0 && wi < w) {
    atomicAdd(&dx[idx_nhwc(n_idx, hi, wi, c_base + ci, h, w, c)], dcol[idx]);
  }
}

__global__ void grad_filter_algo0_nhwc_kernel(const float* __restrict__ x,
                                              const float* __restrict__ dy,
                                              float* __restrict__ dw,
                                              int n, int h, int w, int c,
                                              int ho, int wo,
                                              int r, int s, int k,
                                              int pad_h, int pad_w,
                                              int stride_h, int stride_w,
                                              int dilation_h, int dilation_w,
                                              int cin_group, int kout_group) {
  const size_t total = static_cast<size_t>(n) * ho * wo * k;
  const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t idx = start; idx < total; idx += stride) {
    const int k_idx = static_cast<int>(idx % k);
    const size_t row = idx / k;
    const int n_idx = static_cast<int>(row / static_cast<size_t>(ho * wo));
    const int rem = static_cast<int>(row % static_cast<size_t>(ho * wo));
    const int ho_idx = rem / wo;
    const int wo_idx = rem - ho_idx * wo;

    const int group = k_idx / kout_group;
    const int cin_base = group * cin_group;
    const float dyv = dy[idx_nhwc(n_idx, ho_idx, wo_idx, k_idx, ho, wo, k)];

    for (int rr = 0; rr < r; ++rr) {
      const int hi = ho_idx * stride_h - pad_h + rr * dilation_h;
      if (hi < 0 || hi >= h) continue;
      for (int ss = 0; ss < s; ++ss) {
        const int wi = wo_idx * stride_w - pad_w + ss * dilation_w;
        if (wi < 0 || wi >= w) continue;
        for (int ci = 0; ci < cin_group; ++ci) {
          const float xv = x[idx_nhwc(n_idx, hi, wi, cin_base + ci, h, w, c)];
          atomicAdd(&dw[idx_krsc(k_idx, rr, ss, ci, r, s, cin_group)], xv * dyv);
        }
      }
    }
  }
}

__global__ void grad_filter_algo1_splitk_partials_nhwc_kernel(const float* __restrict__ x,
                                                              const float* __restrict__ dy,
                                                              float* __restrict__ partials,
                                                              int n, int h, int w, int c,
                                                              int ho, int wo,
                                                              int r, int s, int k,
                                                              int pad_h, int pad_w,
                                                              int stride_h, int stride_w,
                                                              int dilation_h, int dilation_w,
                                                              int cin_base, int cin_group,
                                                              int kout_base, int kout_group,
                                                              int rows_per_chunk) {
  const int warp_id = threadIdx.x / kGradSplitKWarpSize;
  const int lane = threadIdx.x % kGradSplitKWarpSize;
  const int slice_weights = r * s * cin_group * kout_group;
  const int k_groups_per_row = (kout_group + kGradSplitKOutputsPerWarp - 1) / kGradSplitKOutputsPerWarp;
  const int packed_weight_idx = blockIdx.x * kGradSplitKWarpsPerBlock + warp_id;
  const int total_packed_weights = r * s * cin_group * k_groups_per_row;
  if (packed_weight_idx >= total_packed_weights) return;

  const int split_idx = blockIdx.y;
  const int m = n * ho * wo;
  int row_begin = split_idx * rows_per_chunk;
  if (row_begin > m) row_begin = m;
  int row_end = row_begin + rows_per_chunk;
  if (row_end > m) row_end = m;

  const int k_group = packed_weight_idx % k_groups_per_row;
  const int k_row = packed_weight_idx / k_groups_per_row;
  const int ci = k_row % cin_group;
  const int t0 = k_row / cin_group;
  const int ss = t0 % s;
  const int rr = t0 / s;
  const int k_local_base = k_group * kGradSplitKOutputsPerWarp;

  float sums[kGradSplitKOutputsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int row = row_begin + lane; row < row_end; row += kGradSplitKWarpSize) {
    const int n_idx = row / (ho * wo);
    const int rem = row - n_idx * (ho * wo);
    const int ho_idx = rem / wo;
    const int wo_idx = rem - ho_idx * wo;

    const int hi = ho_idx * stride_h - pad_h + rr * dilation_h;
    const int wi = wo_idx * stride_w - pad_w + ss * dilation_w;
    if (hi < 0 || hi >= h || wi < 0 || wi >= w) continue;

    const float xv = x[idx_nhwc(n_idx, hi, wi, cin_base + ci, h, w, c)];
    #pragma unroll
    for (int kk = 0; kk < kGradSplitKOutputsPerWarp; ++kk) {
      const int k_local = k_local_base + kk;
      if (k_local >= kout_group) continue;
      const float dyv = dy[idx_nhwc(n_idx, ho_idx, wo_idx, kout_base + k_local, ho, wo, k)];
      sums[kk] += xv * dyv;
    }
  }

  #pragma unroll
  for (int kk = 0; kk < kGradSplitKOutputsPerWarp; ++kk) {
    sums[kk] = warp_reduce_sum(sums[kk]);
  }

  if (lane == 0) {
    const size_t partial_base = static_cast<size_t>(split_idx) * slice_weights + static_cast<size_t>(k_row) * kout_group;
    #pragma unroll
    for (int kk = 0; kk < kGradSplitKOutputsPerWarp; ++kk) {
      const int k_local = k_local_base + kk;
      if (k_local < kout_group) {
        partials[partial_base + k_local] = sums[kk];
      }
    }
  }
}

__global__ void reduce_grad_splitk_partials_kernel(const float* __restrict__ partials,
                                                   float* __restrict__ out,
                                                   int split_k,
                                                   int slice_weights) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= slice_weights) return;

  float sum = 0.0f;
  for (int split = 0; split < split_k; ++split) {
    sum += partials[static_cast<size_t>(split) * slice_weights + idx];
  }
  out[idx] = sum;
}

__global__ void grad_filter_tiled_atomic_nhwc_kernel(const float* __restrict__ x,
                                                     const float* __restrict__ dy,
                                                     float* __restrict__ dw,
                                                     int n, int h, int w, int c,
                                                     int ho, int wo,
                                                     int r, int s, int k,
                                                     int pad_h, int pad_w,
                                                     int stride_h, int stride_w,
                                                     int dilation_h, int dilation_w,
                                                     int cin_base, int cin_group,
                                                     int kout_base, int kout_group) {
  extern __shared__ float shared[];
  float* sh_x = shared;
  float* sh_dy = sh_x + kGradTileRows * kGradTileKDim;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int threads = blockDim.x * blockDim.y;
  const int m = n * ho * wo;
  const int kdim = r * s * cin_group;
  const int row_tile = blockIdx.z * kGradTileRows;
  const int rows_in_tile = min(kGradTileRows, m - row_tile);
  if (rows_in_tile <= 0) return;

  const int k_local = blockIdx.x * kGradTileKOut + threadIdx.x;
  const int kdim_idx = blockIdx.y * kGradTileKDim + threadIdx.y;

  for (int idx = tid; idx < rows_in_tile * kGradTileKDim; idx += threads) {
    const int m_local = idx / kGradTileKDim;
    const int tile_kdim = idx - m_local * kGradTileKDim;
    const int global_kdim = blockIdx.y * kGradTileKDim + tile_kdim;

    float xv = 0.0f;
    if (global_kdim < kdim) {
      int n_idx, ho_idx, wo_idx;
      decode_conv_row(row_tile + m_local, ho, wo, n_idx, ho_idx, wo_idx);

      int rr, ss, ci;
      decode_kdim_index(global_kdim, s, cin_group, rr, ss, ci);
      const int hi = ho_idx * stride_h - pad_h + rr * dilation_h;
      const int wi = wo_idx * stride_w - pad_w + ss * dilation_w;
      if (hi >= 0 && hi < h && wi >= 0 && wi < w) {
        xv = x[idx_nhwc(n_idx, hi, wi, cin_base + ci, h, w, c)];
      }
    }
    sh_x[m_local * kGradTileKDim + tile_kdim] = xv;
  }

  for (int idx = tid; idx < rows_in_tile * kGradTileKOut; idx += threads) {
    const int m_local = idx / kGradTileKOut;
    const int tile_k = idx - m_local * kGradTileKOut;
    const int global_k = blockIdx.x * kGradTileKOut + tile_k;

    float dyv = 0.0f;
    if (global_k < kout_group) {
      int n_idx, ho_idx, wo_idx;
      decode_conv_row(row_tile + m_local, ho, wo, n_idx, ho_idx, wo_idx);
      dyv = dy[idx_nhwc(n_idx, ho_idx, wo_idx, kout_base + global_k, ho, wo, k)];
    }
    sh_dy[m_local * kGradTileKOut + tile_k] = dyv;
  }

  __syncthreads();

  if (k_local >= kout_group || kdim_idx >= kdim) return;

  int rr, ss, ci;
  decode_kdim_index(kdim_idx, s, cin_group, rr, ss, ci);

  float sum = 0.0f;
  for (int m_local = 0; m_local < rows_in_tile; ++m_local) {
    sum += sh_x[m_local * kGradTileKDim + threadIdx.y] *
           sh_dy[m_local * kGradTileKOut + threadIdx.x];
  }

  if (sum != 0.0f) {
    atomicAdd(&dw[idx_krsc(kout_base + k_local, rr, ss, ci, r, s, cin_group)], sum);
  }
}

__global__ void block_grad_filter_algo0_nhwc_kernel(const float* __restrict__ x,
                                                    const float* __restrict__ dy,
                                                    float* __restrict__ dw,
                                                    int n, int h, int w, int c,
                                                    int ho, int wo,
                                                    int block_ho, int block_wo,
                                                    int by_count, int bx_count,
                                                    int r, int s, int k,
                                                    int pad_h, int pad_w,
                                                    int stride_h, int stride_w,
                                                    int dilation_h, int dilation_w,
                                                    int cin_group, int kout_group) {
  const size_t total = static_cast<size_t>(n) * ho * wo * k;
  const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t idx = start; idx < total; idx += stride) {
    const int k_idx = static_cast<int>(idx % k);
    const size_t row = idx / k;
    const int n_idx = static_cast<int>(row / static_cast<size_t>(ho * wo));
    const int rem = static_cast<int>(row % static_cast<size_t>(ho * wo));
    const int ho_idx = rem / wo;
    const int wo_idx = rem - ho_idx * wo;

    const int by = ho_idx / block_ho;
    const int bx = wo_idx / block_wo;
    const int group = k_idx / kout_group;
    const int cin_base = group * cin_group;
    const float dyv = dy[idx_nhwc(n_idx, ho_idx, wo_idx, k_idx, ho, wo, k)];

    for (int rr = 0; rr < r; ++rr) {
      const int hi = ho_idx * stride_h - pad_h + rr * dilation_h;
      if (hi < 0 || hi >= h) continue;
      for (int ss = 0; ss < s; ++ss) {
        const int wi = wo_idx * stride_w - pad_w + ss * dilation_w;
        if (wi < 0 || wi >= w) continue;
        for (int ci = 0; ci < cin_group; ++ci) {
          const float xv = x[idx_nhwc(n_idx, hi, wi, cin_base + ci, h, w, c)];
          atomicAdd(&dw[idx_kbybxrsc(k_idx, by, bx, rr, ss, ci,
                                     by_count, bx_count, r, s, cin_group)],
                    xv * dyv);
        }
      }
    }
  }
}

__global__ void block_grad_filter_algo1_splitk_partials_nhwc_kernel(const float* __restrict__ x,
                                                                    const float* __restrict__ dy,
                                                                    float* __restrict__ partials,
                                                                    int n, int h, int w, int c,
                                                                    int ho, int wo,
                                                                    int ho_start, int wo_start,
                                                                    int block_ho, int block_wo,
                                                                    int r, int s, int k,
                                                                    int pad_h, int pad_w,
                                                                    int stride_h, int stride_w,
                                                                    int dilation_h, int dilation_w,
                                                                    int cin_base, int cin_group,
                                                                    int kout_base, int kout_group,
                                                                    int rows_per_chunk) {
  const int warp_id = threadIdx.x / kGradSplitKWarpSize;
  const int lane = threadIdx.x % kGradSplitKWarpSize;
  const int slice_weights = r * s * cin_group * kout_group;
  const int k_groups_per_row = (kout_group + kGradSplitKOutputsPerWarp - 1) / kGradSplitKOutputsPerWarp;
  const int packed_weight_idx = blockIdx.x * kGradSplitKWarpsPerBlock + warp_id;
  const int total_packed_weights = r * s * cin_group * k_groups_per_row;
  if (packed_weight_idx >= total_packed_weights) return;

  const int split_idx = blockIdx.y;
  const int m = n * block_ho * block_wo;
  int row_begin = split_idx * rows_per_chunk;
  if (row_begin > m) row_begin = m;
  int row_end = row_begin + rows_per_chunk;
  if (row_end > m) row_end = m;

  const int k_group = packed_weight_idx % k_groups_per_row;
  const int k_row = packed_weight_idx / k_groups_per_row;
  const int ci = k_row % cin_group;
  const int t0 = k_row / cin_group;
  const int ss = t0 % s;
  const int rr = t0 / s;
  const int k_local_base = k_group * kGradSplitKOutputsPerWarp;

  float sums[kGradSplitKOutputsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int row = row_begin + lane; row < row_end; row += kGradSplitKWarpSize) {
    const int n_idx = row / (block_ho * block_wo);
    const int rem = row - n_idx * (block_ho * block_wo);
    const int lho = rem / block_wo;
    const int lwo = rem - lho * block_wo;
    const int ho_idx = ho_start + lho;
    const int wo_idx = wo_start + lwo;

    const int hi = ho_idx * stride_h - pad_h + rr * dilation_h;
    const int wi = wo_idx * stride_w - pad_w + ss * dilation_w;
    if (hi < 0 || hi >= h || wi < 0 || wi >= w) continue;

    const float xv = x[idx_nhwc(n_idx, hi, wi, cin_base + ci, h, w, c)];
    #pragma unroll
    for (int kk = 0; kk < kGradSplitKOutputsPerWarp; ++kk) {
      const int k_local = k_local_base + kk;
      if (k_local >= kout_group) continue;
      const float dyv = dy[idx_nhwc(n_idx, ho_idx, wo_idx, kout_base + k_local, ho, wo, k)];
      sums[kk] += xv * dyv;
    }
  }

  #pragma unroll
  for (int kk = 0; kk < kGradSplitKOutputsPerWarp; ++kk) {
    sums[kk] = warp_reduce_sum(sums[kk]);
  }

  if (lane == 0) {
    const size_t partial_base = static_cast<size_t>(split_idx) * slice_weights + static_cast<size_t>(k_row) * kout_group;
    #pragma unroll
    for (int kk = 0; kk < kGradSplitKOutputsPerWarp; ++kk) {
      const int k_local = k_local_base + kk;
      if (k_local < kout_group) {
        partials[partial_base + k_local] = sums[kk];
      }
    }
  }
}

__global__ void block_grad_filter_tiled_atomic_nhwc_kernel(const float* __restrict__ x,
                                                           const float* __restrict__ dy,
                                                           float* __restrict__ dw,
                                                           int n, int h, int w, int c,
                                                           int ho, int wo,
                                                           int ho_start, int wo_start,
                                                           int block_ho, int block_wo,
                                                           int by_count, int bx_count,
                                                           int r, int s, int k,
                                                           int pad_h, int pad_w,
                                                           int stride_h, int stride_w,
                                                           int dilation_h, int dilation_w,
                                                           int cin_base, int cin_group,
                                                           int kout_base, int kout_group) {
  extern __shared__ float shared[];
  float* sh_x = shared;
  float* sh_dy = sh_x + kGradTileRows * kGradTileKDim;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int threads = blockDim.x * blockDim.y;
  const int m = n * block_ho * block_wo;
  const int kdim = r * s * cin_group;
  const int row_tile = blockIdx.z * kGradTileRows;
  const int rows_in_tile = min(kGradTileRows, m - row_tile);
  if (rows_in_tile <= 0) return;

  const int by = ho_start / block_ho;
  const int bx = wo_start / block_wo;
  const int k_local = blockIdx.x * kGradTileKOut + threadIdx.x;
  const int kdim_idx = blockIdx.y * kGradTileKDim + threadIdx.y;

  for (int idx = tid; idx < rows_in_tile * kGradTileKDim; idx += threads) {
    const int m_local = idx / kGradTileKDim;
    const int tile_kdim = idx - m_local * kGradTileKDim;
    const int global_kdim = blockIdx.y * kGradTileKDim + tile_kdim;

    float xv = 0.0f;
    if (global_kdim < kdim) {
      const int row = row_tile + m_local;
      const int n_idx = row / (block_ho * block_wo);
      const int rem = row - n_idx * (block_ho * block_wo);
      const int lho = rem / block_wo;
      const int lwo = rem - lho * block_wo;
      const int ho_idx = ho_start + lho;
      const int wo_idx = wo_start + lwo;

      int rr, ss, ci;
      decode_kdim_index(global_kdim, s, cin_group, rr, ss, ci);
      const int hi = ho_idx * stride_h - pad_h + rr * dilation_h;
      const int wi = wo_idx * stride_w - pad_w + ss * dilation_w;
      if (hi >= 0 && hi < h && wi >= 0 && wi < w) {
        xv = x[idx_nhwc(n_idx, hi, wi, cin_base + ci, h, w, c)];
      }
    }
    sh_x[m_local * kGradTileKDim + tile_kdim] = xv;
  }

  for (int idx = tid; idx < rows_in_tile * kGradTileKOut; idx += threads) {
    const int m_local = idx / kGradTileKOut;
    const int tile_k = idx - m_local * kGradTileKOut;
    const int global_k = blockIdx.x * kGradTileKOut + tile_k;

    float dyv = 0.0f;
    if (global_k < kout_group) {
      const int row = row_tile + m_local;
      const int n_idx = row / (block_ho * block_wo);
      const int rem = row - n_idx * (block_ho * block_wo);
      const int lho = rem / block_wo;
      const int lwo = rem - lho * block_wo;
      const int ho_idx = ho_start + lho;
      const int wo_idx = wo_start + lwo;
      dyv = dy[idx_nhwc(n_idx, ho_idx, wo_idx, kout_base + global_k, ho, wo, k)];
    }
    sh_dy[m_local * kGradTileKOut + tile_k] = dyv;
  }

  __syncthreads();

  if (k_local >= kout_group || kdim_idx >= kdim) return;

  int rr, ss, ci;
  decode_kdim_index(kdim_idx, s, cin_group, rr, ss, ci);

  float sum = 0.0f;
  for (int m_local = 0; m_local < rows_in_tile; ++m_local) {
    sum += sh_x[m_local * kGradTileKDim + threadIdx.y] *
           sh_dy[m_local * kGradTileKOut + threadIdx.x];
  }

  if (sum != 0.0f) {
    atomicAdd(&dw[idx_kbybxrsc(kout_base + k_local, by, bx, rr, ss, ci,
                               by_count, bx_count, r, s, cin_group)],
              sum);
  }
}

}  // namespace

void gather_input_block_nhwc(const float* d_src, float* d_dst,
                             int n, int src_h, int src_w, int c,
                             int hi_start, int wi_start,
                             int local_h, int local_w) {
  const size_t total = static_cast<size_t>(n) * local_h * local_w * c;
  const int t = 256;
  const int blocks = static_cast<int>((total + t - 1) / t);
  gather_input_block_nhwc_kernel<<<blocks, t>>>(d_src, d_dst, n, src_h, src_w, c,
                                                hi_start, wi_start, local_h, local_w);
  CUDA_CHECK(cudaGetLastError());
}

void gather_output_block_nhwc(const float* d_src, float* d_dst,
                              int n, int src_h, int src_w, int c,
                              int ho_start, int wo_start,
                              int block_ho, int block_wo) {
  const size_t total = static_cast<size_t>(n) * block_ho * block_wo * c;
  const int t = 256;
  const int blocks = static_cast<int>((total + t - 1) / t);
  gather_output_block_nhwc_kernel<<<blocks, t>>>(d_src, d_dst, n, src_h, src_w, c,
                                                 ho_start, wo_start, block_ho, block_wo);
  CUDA_CHECK(cudaGetLastError());
}

void launch_grad_gemm_tiled_group_nhwc(const float* d_x, const float* d_dy, float* d_dw,
                                       int n, int h, int w, int c,
                                       int base_ho, int base_wo,
                                       int r, int s, int k,
                                       const Conv2DParams& p,
                                       int cin_base, int cin_group,
                                       int kout_base, int kout_group,
                                       bool can_vec4) {
  const int m = n * base_ho * base_wo;
  const int kdim = r * s * cin_group;
  const int ncol = kout_group * p.ay * p.ax;
  const int rows_per_chunk = select_grad_gemm_rows_per_chunk(m, kdim, ncol);

  Workspace& ws = workspace();
  ensure_capacity(&ws.d_col, &ws.d_col_cap, static_cast<size_t>(rows_per_chunk) * kdim);
  ensure_capacity(&ws.d_dy_mat, &ws.d_dy_mat_cap, static_cast<size_t>(rows_per_chunk) * ncol);
  ensure_capacity(&ws.d_dwg, &ws.d_dwg_cap, static_cast<size_t>(kdim) * ncol);
  float* d_col = ws.d_col;
  float* d_dy_mat = ws.d_dy_mat;
  float* d_dwg = ws.d_dwg;

  CUDA_CHECK(cudaMemset(d_dwg, 0, static_cast<size_t>(kdim) * ncol * sizeof(float)));

  const int t = 256;
  for (int row_start = 0; row_start < m; row_start += rows_per_chunk) {
    const int row_count = std::min(rows_per_chunk, m - row_start);

    if (can_vec4 && (cin_base % 4 == 0)) {
      const int total_vec = row_count * r * s * (cin_group / 4);
      const int blocks_vec = (total_vec + t - 1) / t;
      im2col_nhwc_vec4_rows_kernel<<<blocks_vec, t>>>(d_x, d_col,
                                                      n, h, w, c,
                                                      base_ho, base_wo,
                                                      row_start, row_count,
                                                      r, s,
                                                      p.pad_h, p.pad_w,
                                                      p.stride_h, p.stride_w,
                                                      p.dilation_h, p.dilation_w,
                                                      cin_base, cin_group);
    } else {
      const int total_col = row_count * kdim;
      const int blocks_col = (total_col + t - 1) / t;
      im2col_nhwc_rows_kernel<<<blocks_col, t>>>(d_x, d_col,
                                                 n, h, w, c,
                                                 base_ho, base_wo,
                                                 row_start, row_count,
                                                 r, s,
                                                 p.pad_h, p.pad_w,
                                                 p.stride_h, p.stride_w,
                                                 p.dilation_h, p.dilation_w,
                                                 cin_base, cin_group);
    }

    const int total_dy = row_count * ncol;
    const int blocks_dy = (total_dy + t - 1) / t;
    pack_nhwc_group_matrix_app_rows_kernel<<<blocks_dy, t>>>(d_dy, d_dy_mat,
                                                             n, base_ho, base_wo, p.ay, p.ax,
                                                             k, row_start, row_count,
                                                             kout_base, kout_group);

    bmm_matmul_accum(d_col, d_dy_mat, d_dwg,
                     1, kdim, ncol, row_count,
                     BMM_TRANSPOSE_YES, BMM_TRANSPOSE_NONE);
  }

  const int total_wg = kdim * ncol;
  const int blocks_wg = (total_wg + t - 1) / t;
  unpack_filter_krsc_group_kernel<<<blocks_wg, t>>>(d_dwg, d_dw, r, s, cin_group, p.ay, p.ax,
                                                    k, kout_base, kout_group);
  CUDA_CHECK(cudaGetLastError());
}

void launch_block_grad_gemm_tiled_group_nhwc(const float* d_x, const float* d_dy, float* d_dw,
                                             int n, int h, int w, int c,
                                             int base_ho, int base_wo,
                                             int ho_start_base, int wo_start_base,
                                             int block_ho_base, int block_wo_base,
                                             int by_count, int bx_count,
                                             int r, int s, int k,
                                             const Conv2DParams& p,
                                             int cin_base, int cin_group,
                                             int kout_base, int kout_group,
                                             int by, int bx) {
  const int m = n * block_ho_base * block_wo_base;
  const int kdim = r * s * cin_group;
  const int ncol = kout_group * p.ay * p.ax;
  const int rows_per_chunk = select_grad_gemm_rows_per_chunk(m, kdim, ncol);

  Workspace& ws = workspace();
  ensure_capacity(&ws.d_col, &ws.d_col_cap, static_cast<size_t>(rows_per_chunk) * kdim);
  ensure_capacity(&ws.d_dy_mat, &ws.d_dy_mat_cap, static_cast<size_t>(rows_per_chunk) * ncol);
  ensure_capacity(&ws.d_dwg, &ws.d_dwg_cap, static_cast<size_t>(kdim) * ncol);
  float* d_col = ws.d_col;
  float* d_dy_mat = ws.d_dy_mat;
  float* d_dwg = ws.d_dwg;

  CUDA_CHECK(cudaMemset(d_dwg, 0, static_cast<size_t>(kdim) * ncol * sizeof(float)));

  const int t = 256;
  for (int row_start = 0; row_start < m; row_start += rows_per_chunk) {
    const int row_count = std::min(rows_per_chunk, m - row_start);
    const int total_col = row_count * kdim;
    const int blocks_col = (total_col + t - 1) / t;
    im2col_nhwc_block_rows_kernel<<<blocks_col, t>>>(d_x, d_col,
                                                     n, h, w, c,
                                                     ho_start_base, wo_start_base,
                                                     block_ho_base, block_wo_base,
                                                     row_start, row_count,
                                                     r, s,
                                                     p.pad_h, p.pad_w,
                                                     p.stride_h, p.stride_w,
                                                     p.dilation_h, p.dilation_w,
                                                     cin_base, cin_group);

    const int total_dy = row_count * ncol;
    const int blocks_dy = (total_dy + t - 1) / t;
    pack_nhwc_group_matrix_app_block_rows_kernel<<<blocks_dy, t>>>(d_dy, d_dy_mat,
                                                                   n, base_ho, base_wo, k,
                                                                   ho_start_base, wo_start_base,
                                                                   block_ho_base, block_wo_base,
                                                                   p.ay, p.ax,
                                                                   row_start, row_count,
                                                                   kout_base, kout_group);

    bmm_matmul_accum(d_col, d_dy_mat, d_dwg,
                     1, kdim, ncol, row_count,
                     BMM_TRANSPOSE_YES, BMM_TRANSPOSE_NONE);
  }

  const int total_wg = kdim * ncol;
  const int blocks_wg = (total_wg + t - 1) / t;
  unpack_block_filter_kbybxrsc_group_kernel<<<blocks_wg, t>>>(d_dwg, d_dw,
                                                              by_count, bx_count,
                                                              r, s, cin_group, p.ay, p.ax,
                                                              by, bx,
                                                              k,
                                                              kout_base, kout_group);
  CUDA_CHECK(cudaGetLastError());
}

void scatter_output_block_nhwc(const float* d_src, float* d_dst,
                               int n, int dst_h, int dst_w, int c,
                               int ho_start, int wo_start,
                               int block_ho, int block_wo) {
  const size_t total = static_cast<size_t>(n) * block_ho * block_wo * c;
  const int t = 256;
  const int blocks = static_cast<int>((total + t - 1) / t);
  scatter_output_block_nhwc_kernel<<<blocks, t>>>(d_src, d_dst, n, dst_h, dst_w, c,
                                                  ho_start, wo_start, block_ho, block_wo);
  CUDA_CHECK(cudaGetLastError());
}

void scatter_add_input_block_nhwc(const float* d_src, float* d_dst,
                                  int n, int dst_h, int dst_w, int c,
                                  int hi_start, int wi_start,
                                  int local_h, int local_w) {
  const size_t total = static_cast<size_t>(n) * local_h * local_w * c;
  const int t = 256;
  const int blocks = static_cast<int>((total + t - 1) / t);
  scatter_add_input_block_nhwc_kernel<<<blocks, t>>>(d_src, d_dst, n, dst_h, dst_w, c,
                                                     hi_start, wi_start, local_h, local_w);
  CUDA_CHECK(cudaGetLastError());
}

void gather_block_filter_to_krsc(const float* d_src, float* d_dst,
                                 int by_count, int bx_count,
                                 int r, int s, int cin_group, int k,
                                 int block_y, int block_x) {
  const size_t total = static_cast<size_t>(k) * r * s * cin_group;
  const int t = 256;
  const int blocks = static_cast<int>((total + t - 1) / t);
  gather_block_filter_to_krsc_kernel<<<blocks, t>>>(d_src, d_dst,
                                                    by_count, bx_count,
                                                    r, s, cin_group, k,
                                                    block_y, block_x);
  CUDA_CHECK(cudaGetLastError());
}

void scatter_block_filter_from_krsc(const float* d_src, float* d_dst,
                                    int by_count, int bx_count,
                                    int r, int s, int cin_group, int k,
                                    int block_y, int block_x) {
  const size_t total = static_cast<size_t>(k) * r * s * cin_group;
  const int t = 256;
  const int blocks = static_cast<int>((total + t - 1) / t);
  scatter_block_filter_from_krsc_kernel<<<blocks, t>>>(d_src, d_dst,
                                                       by_count, bx_count,
                                                       r, s, cin_group, k,
                                                       block_y, block_x);
  CUDA_CHECK(cudaGetLastError());
}

void launch_fprop_nhwc(const float* d_x, const float* d_w, float* d_y,
                       int n, int h, int w, int c, int r, int s, int k,
                       const Conv2DParams& p) {
  TensorNHWC x_shape(n, h, w, c);
  FilterKRSC w_shape(r, s, c / p.groups, k, p.ay, p.ax);
  ConvShape sh = infer_conv_shape(x_shape, w_shape, p);

  const int m = n * sh.base_ho * sh.base_wo;
  const int kdim = r * s * sh.cin_group;
  const int ncol = sh.kout_group * p.ay * p.ax;
  const bool can_vec4 = (sh.cin_group % 4 == 0) && (sh.cin_group >= 4) && ((c % 4) == 0);

  Workspace& ws = workspace();
  ensure_capacity(&ws.d_col, &ws.d_col_cap, static_cast<size_t>(m) * kdim);
  ensure_capacity(&ws.d_ymat, &ws.d_ymat_cap, static_cast<size_t>(m) * ncol);
  ensure_packed_weights(ws, d_w, r, s, sh.cin_group, k, p.groups, p.ay, p.ax);
  float* d_col = ws.d_col;
  float* d_ymat = ws.d_ymat;

  const int t = 256;
  for (int g = 0; g < p.groups; ++g) {
    const int cin_base = g * sh.cin_group;
    const int kout_base = g * sh.kout_group;

    int total_col = m * kdim;
    int blocks_col = (total_col + t - 1) / t;
    if (can_vec4 && (cin_base % 4 == 0)) {
      const int total_vec = m * r * s * (sh.cin_group / 4);
      const int blocks_vec = (total_vec + t - 1) / t;
      im2col_nhwc_vec4_kernel<<<blocks_vec, t>>>(d_x, d_col, n, h, w, c,
                                                 sh.base_ho, sh.base_wo,
                                                 r, s,
                                                 p.pad_h, p.pad_w,
                                                 p.stride_h, p.stride_w,
                                                 p.dilation_h, p.dilation_w,
                                                 cin_base, sh.cin_group);
    } else {
      im2col_nhwc_kernel<<<blocks_col, t>>>(d_x, d_col, n, h, w, c,
                                            sh.base_ho, sh.base_wo,
                                            r, s,
                                            p.pad_h, p.pad_w,
                                            p.stride_h, p.stride_w,
                                            p.dilation_h, p.dilation_w,
                                            cin_base, sh.cin_group);
    }

    float* d_wg = ws.d_wg_all + static_cast<size_t>(g) * ncol * kdim;
    // Forward GEMM: [m, kdim] @ [ncol, kdim]^T -> [m, ncol]
    bmm_matmul(d_col, d_wg, d_ymat, 1, m, ncol, kdim, BMM_TRANSPOSE_NONE, BMM_TRANSPOSE_YES);

    int total_ym = m * ncol;
    int blocks_ym = (total_ym + t - 1) / t;
    unpack_matrix_to_nhwc_group_app_kernel<<<blocks_ym, t>>>(d_ymat, d_y,
                                                             n, sh.base_ho, sh.base_wo, p.ay, p.ax,
                                                             k, kout_base, sh.kout_group);
  }

  CUDA_CHECK(cudaGetLastError());
}

void launch_bprop_nhwc(const float* d_dy, const float* d_w, float* d_dx,
                       int n, int h, int w, int c, int r, int s, int k,
                       const Conv2DParams& p) {
  TensorNHWC x_shape(n, h, w, c);
  FilterKRSC w_shape(r, s, c / p.groups, k, p.ay, p.ax);
  ConvShape sh = infer_conv_shape(x_shape, w_shape, p);

  const int m = n * sh.base_ho * sh.base_wo;
  const int kdim = r * s * sh.cin_group;
  const int ncol = sh.kout_group * p.ay * p.ax;

  Workspace& ws = workspace();
  ensure_capacity(&ws.d_dy_mat, &ws.d_dy_mat_cap, static_cast<size_t>(m) * ncol);
  ensure_capacity(&ws.d_dcol, &ws.d_dcol_cap, static_cast<size_t>(m) * kdim);
  ensure_packed_weights(ws, d_w, r, s, sh.cin_group, k, p.groups, p.ay, p.ax);
  float* d_dy_mat = ws.d_dy_mat;
  float* d_dcol = ws.d_dcol;

  CUDA_CHECK(cudaMemset(d_dx, 0, static_cast<size_t>(n) * h * w * c * sizeof(float)));

  const int t = 256;

  for (int g = 0; g < p.groups; ++g) {
    const int cin_base = g * sh.cin_group;
    const int kout_base = g * sh.kout_group;

    int total_dy = m * ncol;
    int blocks_dy = (total_dy + t - 1) / t;
    pack_nhwc_group_matrix_app_kernel<<<blocks_dy, t>>>(d_dy, d_dy_mat,
                                                        n, sh.base_ho, sh.base_wo, p.ay, p.ax,
                                                        k, kout_base, sh.kout_group);

    float* d_wg = ws.d_wg_all + static_cast<size_t>(g) * ncol * kdim;
    // Input-gradient GEMM: [m, ncol] @ [ncol, kdim] -> [m, kdim]
    bmm_matmul(d_dy_mat, d_wg, d_dcol, 1, m, kdim, ncol, BMM_TRANSPOSE_NONE, BMM_TRANSPOSE_NONE);

    int total_dcol = m * kdim;
    int blocks_dcol = (total_dcol + t - 1) / t;
    col2im_accum_nhwc_kernel<<<blocks_dcol, t>>>(d_dcol, d_dx,
                                                 n, h, w, c,
                                                 sh.base_ho, sh.base_wo,
                                                 r, s,
                                                 p.pad_h, p.pad_w,
                                                 p.stride_h, p.stride_w,
                                                 p.dilation_h, p.dilation_w,
                                                 cin_base, sh.cin_group);
  }

  CUDA_CHECK(cudaGetLastError());
}

void launch_grad_nhwc(const float* d_x, const float* d_dy, float* d_dw,
                      int n, int h, int w, int c, int r, int s, int k,
                      const Conv2DParams& p,
                      GradKernelAlgo algo) {
  TensorNHWC x_shape(n, h, w, c);
  FilterKRSC w_shape(r, s, c / p.groups, k, p.ay, p.ax);
  ConvShape sh = infer_conv_shape(x_shape, w_shape, p);

  if ((p.ay != 1 || p.ax != 1) && algo != GradKernelAlgo::GemmIm2Col) {
    throw std::runtime_error("Ay/Ax currently support only grad_algo=gemm");
  }

  if (algo == GradKernelAlgo::Algo0Atomic) {
    CUDA_CHECK(cudaMemset(d_dw, 0, static_cast<size_t>(r) * s * sh.cin_group * k * p.ay * p.ax * sizeof(float)));
    const size_t total = static_cast<size_t>(n) * sh.ho * sh.wo * k;
    const int t = 256;
    const int blocks = static_cast<int>(std::min<size_t>((total + t - 1) / t, 65535));
    grad_filter_algo0_nhwc_kernel<<<blocks, t>>>(d_x, d_dy, d_dw,
                                                 n, h, w, c,
                                                 sh.ho, sh.wo,
                                                 r, s, k,
                                                 p.pad_h, p.pad_w,
                                                 p.stride_h, p.stride_w,
                                                 p.dilation_h, p.dilation_w,
                                                 sh.cin_group, sh.kout_group);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  if (algo == GradKernelAlgo::Algo1Deterministic) {
    const int m = n * sh.base_ho * sh.base_wo;
    const int slice_weights = r * s * sh.cin_group * sh.kout_group * p.ay * p.ax;
    const int packed_weights = r * s * sh.cin_group *
                               ((sh.kout_group + kGradSplitKOutputsPerWarp - 1) / kGradSplitKOutputsPerWarp);
    const int split_k = select_grad_split_k(m, static_cast<size_t>(slice_weights));
    const int rows_per_chunk = (m + split_k - 1) / split_k;

    Workspace& ws = workspace();
    ensure_capacity(&ws.d_grad_partials, &ws.d_grad_partials_cap, static_cast<size_t>(split_k) * slice_weights);
    ensure_capacity(&ws.d_dwg, &ws.d_dwg_cap, static_cast<size_t>(slice_weights));

    const dim3 block(kGradSplitKBlockSize);
    const dim3 grid((packed_weights + kGradSplitKWarpsPerBlock - 1) / kGradSplitKWarpsPerBlock, split_k);
    const int reduce_blocks = (slice_weights + 255) / 256;

    for (int g = 0; g < p.groups; ++g) {
      const int cin_base = g * sh.cin_group;
      const int kout_base = g * sh.kout_group;
      grad_filter_algo1_splitk_partials_nhwc_kernel<<<grid, block>>>(d_x, d_dy, ws.d_grad_partials,
                                                                      n, h, w, c,
                                                                      sh.base_ho, sh.base_wo,
                                                                      r, s, k,
                                                                      p.pad_h, p.pad_w,
                                                                      p.stride_h, p.stride_w,
                                                                      p.dilation_h, p.dilation_w,
                                                                      cin_base, sh.cin_group,
                                                                      kout_base, sh.kout_group,
                                                                      rows_per_chunk);
      reduce_grad_splitk_partials_kernel<<<reduce_blocks, 256>>>(ws.d_grad_partials, ws.d_dwg, split_k, slice_weights);
      unpack_filter_krsc_group_kernel<<<reduce_blocks, 256>>>(ws.d_dwg, d_dw, r, s, sh.cin_group, p.ay, p.ax,
                                                              k, kout_base, sh.kout_group);
    }
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  if (algo == GradKernelAlgo::Algo2TiledAtomic) {
    CUDA_CHECK(cudaMemset(d_dw, 0, static_cast<size_t>(r) * s * sh.cin_group * k * p.ay * p.ax * sizeof(float)));

    const int kdim = r * s * sh.cin_group;
    const dim3 block(kGradTileKOut, kGradTileKDim);
    const dim3 grid((sh.kout_group + kGradTileKOut - 1) / kGradTileKOut,
                    (kdim + kGradTileKDim - 1) / kGradTileKDim,
                    (n * sh.ho * sh.wo + kGradTileRows - 1) / kGradTileRows);
    const size_t shared_bytes = static_cast<size_t>(kGradTileRows) * (kGradTileKDim + kGradTileKOut) * sizeof(float);

    for (int g = 0; g < p.groups; ++g) {
      const int cin_base = g * sh.cin_group;
      const int kout_base = g * sh.kout_group;
      grad_filter_tiled_atomic_nhwc_kernel<<<grid, block, shared_bytes>>>(d_x, d_dy, d_dw,
                                                                           n, h, w, c,
                                                                           sh.base_ho, sh.base_wo,
                                                                           r, s, k,
                                                                           p.pad_h, p.pad_w,
                                                                           p.stride_h, p.stride_w,
                                                                           p.dilation_h, p.dilation_w,
                                                                           cin_base, sh.cin_group,
                                                                           kout_base, sh.kout_group);
    }
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  const bool can_vec4 = (sh.cin_group % 4 == 0) && (sh.cin_group >= 4) && (c % 4 == 0);

  CUDA_CHECK(cudaMemset(d_dw, 0, static_cast<size_t>(r) * s * sh.cin_group * k * p.ay * p.ax * sizeof(float)));

  for (int g = 0; g < p.groups; ++g) {
    const int cin_base = g * sh.cin_group;
    const int kout_base = g * sh.kout_group;
    launch_grad_gemm_tiled_group_nhwc(d_x, d_dy, d_dw,
                                      n, h, w, c,
                                      sh.base_ho, sh.base_wo,
                                      r, s, k,
                                      p,
                                      cin_base, sh.cin_group,
                                      kout_base, sh.kout_group,
                                      can_vec4);
  }

  CUDA_CHECK(cudaGetLastError());
}

void launch_block_fprop_nhwc(const float* d_x, const float* d_w, float* d_y,
                             int n, int h, int w, int c, int r, int s, int k,
                             const BlockConv2DParams& p) {
  TensorNHWC x_shape(n, h, w, c);
  BlockFilterKByBxRSC w_shape(k, p.block_by, p.block_bx, r, s, c / p.conv.groups, p.conv.ay, p.conv.ax);
  const BlockConvShape sh = infer_block_conv_shape(x_shape, w_shape, p);

  const int m_block = n * sh.block_ho * sh.block_wo;
  const int kdim = r * s * sh.base.cin_group;
  const int ncol = sh.base.kout_group * p.conv.ay * p.conv.ax;
  const size_t per_group_slice = static_cast<size_t>(kdim) * ncol;

  Workspace& ws = workspace();
  ensure_capacity(&ws.d_col, &ws.d_col_cap, static_cast<size_t>(m_block) * kdim);
  ensure_capacity(&ws.d_ymat, &ws.d_ymat_cap, static_cast<size_t>(m_block) * ncol);
  ensure_packed_block_weights(ws, d_w, p.block_by, p.block_bx, r, s, sh.base.cin_group, k, p.conv.groups, p.conv.ay, p.conv.ax);
  float* d_col = ws.d_col;
  float* d_ymat = ws.d_ymat;

  const int t = 256;
  for (int g = 0; g < p.conv.groups; ++g) {
    const int cin_base = g * sh.base.cin_group;
    const int kout_base = g * sh.base.kout_group;
    for (int by = 0; by < p.block_by; ++by) {
      const int ho_start = by * sh.block_ho;
      for (int bx = 0; bx < p.block_bx; ++bx) {
        const int wo_start = bx * sh.block_wo;
        const int total_col = m_block * kdim;
        const int blocks_col = (total_col + t - 1) / t;
        im2col_nhwc_block_kernel<<<blocks_col, t>>>(d_x, d_col,
                                                    n, h, w, c,
                                                    ho_start, wo_start,
                                                    sh.block_ho, sh.block_wo,
                                                    r, s,
                                                    p.conv.pad_h, p.conv.pad_w,
                                                    p.conv.stride_h, p.conv.stride_w,
                                                    p.conv.dilation_h, p.conv.dilation_w,
                                                    cin_base, sh.base.cin_group);

        float* d_wg = ws.d_block_wg_all + packed_block_weight_offset(p.conv.groups, p.block_by, p.block_bx,
                                                                     g, by, bx, per_group_slice);
        bmm_matmul(d_col, d_wg, d_ymat, 1, m_block, ncol, kdim, BMM_TRANSPOSE_NONE, BMM_TRANSPOSE_YES);

        const int total_ym = m_block * ncol;
        const int blocks_ym = (total_ym + t - 1) / t;
        unpack_matrix_to_nhwc_group_app_block_kernel<<<blocks_ym, t>>>(d_ymat, d_y,
                                                                       n, sh.base.base_ho, sh.base.base_wo, k,
                                                                       ho_start, wo_start,
                                                                       sh.block_ho, sh.block_wo,
                                                                       p.conv.ay, p.conv.ax,
                                                                       kout_base, sh.base.kout_group);
      }
    }
  }

  CUDA_CHECK(cudaGetLastError());
}

void launch_block_bprop_nhwc(const float* d_dy, const float* d_w, float* d_dx,
                             int n, int h, int w, int c, int r, int s, int k,
                             const BlockConv2DParams& p) {
  TensorNHWC x_shape(n, h, w, c);
  BlockFilterKByBxRSC w_shape(k, p.block_by, p.block_bx, r, s, c / p.conv.groups, p.conv.ay, p.conv.ax);
  const BlockConvShape sh = infer_block_conv_shape(x_shape, w_shape, p);

  const int m_block = n * sh.block_ho * sh.block_wo;
  const int kdim = r * s * sh.base.cin_group;
  const int ncol = sh.base.kout_group * p.conv.ay * p.conv.ax;
  const size_t per_group_slice = static_cast<size_t>(kdim) * ncol;

  Workspace& ws = workspace();
  ensure_capacity(&ws.d_dy_mat, &ws.d_dy_mat_cap, static_cast<size_t>(m_block) * ncol);
  ensure_capacity(&ws.d_dcol, &ws.d_dcol_cap, static_cast<size_t>(m_block) * kdim);
  ensure_packed_block_weights(ws, d_w, p.block_by, p.block_bx, r, s, sh.base.cin_group, k, p.conv.groups, p.conv.ay, p.conv.ax);
  float* d_dy_mat = ws.d_dy_mat;
  float* d_dcol = ws.d_dcol;

  CUDA_CHECK(cudaMemset(d_dx, 0, static_cast<size_t>(n) * h * w * c * sizeof(float)));

  const int t = 256;
  for (int g = 0; g < p.conv.groups; ++g) {
    const int cin_base = g * sh.base.cin_group;
    const int kout_base = g * sh.base.kout_group;
    for (int by = 0; by < p.block_by; ++by) {
      const int ho_start = by * sh.block_ho;
      for (int bx = 0; bx < p.block_bx; ++bx) {
        const int wo_start = bx * sh.block_wo;
        const int total_dy = m_block * ncol;
        const int blocks_dy = (total_dy + t - 1) / t;
        pack_nhwc_group_matrix_app_block_kernel<<<blocks_dy, t>>>(d_dy, d_dy_mat,
                                                                  n, sh.base.base_ho, sh.base.base_wo, k,
                                                                  ho_start, wo_start,
                                                                  sh.block_ho, sh.block_wo,
                                                                  p.conv.ay, p.conv.ax,
                                                                  kout_base, sh.base.kout_group);

        float* d_wg = ws.d_block_wg_all + packed_block_weight_offset(p.conv.groups, p.block_by, p.block_bx,
                                                                     g, by, bx, per_group_slice);
        bmm_matmul(d_dy_mat, d_wg, d_dcol, 1, m_block, kdim, ncol, BMM_TRANSPOSE_NONE, BMM_TRANSPOSE_NONE);

        const int total_dcol = m_block * kdim;
        const int blocks_dcol = (total_dcol + t - 1) / t;
        col2im_accum_nhwc_block_kernel<<<blocks_dcol, t>>>(d_dcol, d_dx,
                                                           n, h, w, c,
                                                           ho_start, wo_start,
                                                           sh.block_ho, sh.block_wo,
                                                           r, s,
                                                           p.conv.pad_h, p.conv.pad_w,
                                                           p.conv.stride_h, p.conv.stride_w,
                                                           p.conv.dilation_h, p.conv.dilation_w,
                                                           cin_base, sh.base.cin_group);
      }
    }
  }

  CUDA_CHECK(cudaGetLastError());
}

void launch_block_grad_nhwc(const float* d_x, const float* d_dy, float* d_dw,
                            int n, int h, int w, int c, int r, int s, int k,
                            const BlockConv2DParams& p,
                            GradKernelAlgo algo) {
  TensorNHWC x_shape(n, h, w, c);
  BlockFilterKByBxRSC w_shape(k, p.block_by, p.block_bx, r, s, c / p.conv.groups, p.conv.ay, p.conv.ax);
  const BlockConvShape sh = infer_block_conv_shape(x_shape, w_shape, p);

  if ((p.conv.ay != 1 || p.conv.ax != 1) && algo != GradKernelAlgo::GemmIm2Col) {
    throw std::runtime_error("Ay/Ax currently support only grad_algo=gemm");
  }

  if (algo == GradKernelAlgo::Algo0Atomic) {
    CUDA_CHECK(cudaMemset(d_dw, 0, static_cast<size_t>(k) * p.block_by * p.block_bx * r * s * sh.base.cin_group * p.conv.ay * p.conv.ax * sizeof(float)));
    const size_t total = static_cast<size_t>(n) * sh.base.ho * sh.base.wo * k;
    const int t = 256;
    const int blocks = static_cast<int>(std::min<size_t>((total + t - 1) / t, 65535));
    block_grad_filter_algo0_nhwc_kernel<<<blocks, t>>>(d_x, d_dy, d_dw,
                                                       n, h, w, c,
                                                       sh.base.ho, sh.base.wo,
                                                       sh.block_ho, sh.block_wo,
                                                       p.block_by, p.block_bx,
                                                       r, s, k,
                                                       p.conv.pad_h, p.conv.pad_w,
                                                       p.conv.stride_h, p.conv.stride_w,
                                                       p.conv.dilation_h, p.conv.dilation_w,
                                                       sh.base.cin_group, sh.base.kout_group);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  if (algo == GradKernelAlgo::Algo1Deterministic) {
    const int m_block = n * sh.block_ho * sh.block_wo;
    const int slice_weights = r * s * sh.base.cin_group * sh.base.kout_group * p.conv.ay * p.conv.ax;
    const int packed_weights = r * s * sh.base.cin_group *
                               ((sh.base.kout_group + kGradSplitKOutputsPerWarp - 1) / kGradSplitKOutputsPerWarp);
    const int split_k = select_grad_split_k(m_block, static_cast<size_t>(slice_weights));
    const int rows_per_chunk = (m_block + split_k - 1) / split_k;

    Workspace& ws = workspace();
    ensure_capacity(&ws.d_grad_partials, &ws.d_grad_partials_cap, static_cast<size_t>(split_k) * slice_weights);
    ensure_capacity(&ws.d_dwg, &ws.d_dwg_cap, static_cast<size_t>(slice_weights));

    const dim3 block(kGradSplitKBlockSize);
    const dim3 grid((packed_weights + kGradSplitKWarpsPerBlock - 1) / kGradSplitKWarpsPerBlock, split_k);
    const int reduce_blocks = (slice_weights + 255) / 256;

    for (int g = 0; g < p.conv.groups; ++g) {
      const int cin_base = g * sh.base.cin_group;
      const int kout_base = g * sh.base.kout_group;
      for (int by = 0; by < p.block_by; ++by) {
        const int ho_start = by * sh.block_ho;
        for (int bx = 0; bx < p.block_bx; ++bx) {
          const int wo_start = bx * sh.block_wo;
          block_grad_filter_algo1_splitk_partials_nhwc_kernel<<<grid, block>>>(d_x, d_dy, ws.d_grad_partials,
                                                                                n, h, w, c,
                                                                                sh.base.ho, sh.base.wo,
                                                                                ho_start, wo_start,
                                                                                sh.block_ho, sh.block_wo,
                                                                                r, s, k,
                                                                                p.conv.pad_h, p.conv.pad_w,
                                                                                p.conv.stride_h, p.conv.stride_w,
                                                                                p.conv.dilation_h, p.conv.dilation_w,
                                                                                cin_base, sh.base.cin_group,
                                                                                kout_base, sh.base.kout_group,
                                                                                rows_per_chunk);
          reduce_grad_splitk_partials_kernel<<<reduce_blocks, 256>>>(ws.d_grad_partials, ws.d_dwg, split_k, slice_weights);
          unpack_block_filter_kbybxrsc_group_kernel<<<reduce_blocks, 256>>>(ws.d_dwg, d_dw,
                                                                            p.block_by, p.block_bx,
                                                                            r, s, sh.base.cin_group, p.conv.ay, p.conv.ax,
                                                                            by, bx,
                                                                            k,
                                                                            kout_base, sh.base.kout_group);
        }
      }
    }
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  if (algo == GradKernelAlgo::Algo2TiledAtomic) {
    CUDA_CHECK(cudaMemset(d_dw, 0, static_cast<size_t>(k) * p.block_by * p.block_bx * r * s * sh.base.cin_group * p.conv.ay * p.conv.ax * sizeof(float)));

    const int kdim = r * s * sh.base.cin_group;
    const dim3 block(kGradTileKOut, kGradTileKDim);
    const dim3 grid((sh.base.kout_group + kGradTileKOut - 1) / kGradTileKOut,
                    (kdim + kGradTileKDim - 1) / kGradTileKDim,
                    (n * sh.block_ho * sh.block_wo + kGradTileRows - 1) / kGradTileRows);
    const size_t shared_bytes = static_cast<size_t>(kGradTileRows) * (kGradTileKDim + kGradTileKOut) * sizeof(float);

    for (int g = 0; g < p.conv.groups; ++g) {
      const int cin_base = g * sh.base.cin_group;
      const int kout_base = g * sh.base.kout_group;
      for (int by = 0; by < p.block_by; ++by) {
        const int ho_start = by * sh.block_ho;
        for (int bx = 0; bx < p.block_bx; ++bx) {
          const int wo_start = bx * sh.block_wo;
          block_grad_filter_tiled_atomic_nhwc_kernel<<<grid, block, shared_bytes>>>(d_x, d_dy, d_dw,
                                                                                     n, h, w, c,
                                                                                     sh.base.ho, sh.base.wo,
                                                                                     ho_start, wo_start,
                                                                                     sh.block_ho, sh.block_wo,
                                                                                     p.block_by, p.block_bx,
                                                                                     r, s, k,
                                                                                     p.conv.pad_h, p.conv.pad_w,
                                                                                     p.conv.stride_h, p.conv.stride_w,
                                                                                     p.conv.dilation_h, p.conv.dilation_w,
                                                                                     cin_base, sh.base.cin_group,
                                                                                     kout_base, sh.base.kout_group);
        }
      }
    }
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  CUDA_CHECK(cudaMemset(d_dw, 0, static_cast<size_t>(k) * p.block_by * p.block_bx * r * s * sh.base.cin_group * p.conv.ay * p.conv.ax * sizeof(float)));
  for (int g = 0; g < p.conv.groups; ++g) {
    const int cin_base = g * sh.base.cin_group;
    const int kout_base = g * sh.base.kout_group;
    for (int by = 0; by < p.block_by; ++by) {
      const int ho_start = by * sh.block_ho;
      for (int bx = 0; bx < p.block_bx; ++bx) {
        const int wo_start = bx * sh.block_wo;
        launch_block_grad_gemm_tiled_group_nhwc(d_x, d_dy, d_dw,
                                                n, h, w, c,
                                                sh.base.base_ho, sh.base.base_wo,
                                                ho_start, wo_start,
                                                sh.block_ho, sh.block_wo,
                                                p.block_by, p.block_bx,
                                                r, s, k,
                                                p.conv,
                                                cin_base, sh.base.cin_group,
                                                kout_base, sh.base.kout_group,
                                                by, bx);
      }
    }
  }

  CUDA_CHECK(cudaGetLastError());
}

namespace {

struct QuantizedWorkspace {
  int32_t* d_col = nullptr;
  size_t d_col_cap = 0;
  int32_t* d_wg = nullptr;
  size_t d_wg_cap = 0;
  int32_t* d_ymat = nullptr;
  size_t d_ymat_cap = 0;

  ~QuantizedWorkspace() {
    cudaFree(d_col);
    cudaFree(d_wg);
    cudaFree(d_ymat);
  }
};

void ensure_capacity(int32_t** ptr, size_t* cap, size_t elements) {
  if (*cap >= elements) return;
  if (*ptr) CUDA_CHECK(cudaFree(*ptr));
  CUDA_CHECK(cudaMalloc(ptr, elements * sizeof(int32_t)));
  *cap = elements;
}

template <nnalgebra::DataType Tin>
__global__ void im2col_nhwc_qi32_kernel(const float* __restrict__ x,
                                        int32_t* __restrict__ col,
                                        int n, int h, int w, int c,
                                        int ho, int wo,
                                        int r, int s,
                                        int pad_h, int pad_w,
                                        int stride_h, int stride_w,
                                        int dilation_h, int dilation_w,
                                        int c_base, int cin_group,
                                        const nnalgebra::QuantizationParameters<Tin>* in_qp) {
  const int m = n * ho * wo;
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * kdim;
  if (idx >= total) return;

  const int row = idx / kdim;
  const int col_idx = idx - row * kdim;

  const int ci = col_idx % cin_group;
  const int t = col_idx / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  const int n_idx = row / (ho * wo);
  const int rem = row - n_idx * (ho * wo);
  const int ho_idx = rem / wo;
  const int wo_idx = rem - ho_idx * wo;

  const int hi = ho_idx * stride_h - pad_h + rr * dilation_h;
  const int wi = wo_idx * stride_w - pad_w + ss * dilation_w;
  if (hi >= 0 && hi < h && wi >= 0 && wi < w) {
    const int32_t raw = static_cast<int32_t>(x[idx_nhwc(n_idx, hi, wi, c_base + ci, h, w, c)]);
    col[idx] = raw - nnalgebra::getZeroPoint(in_qp[n_idx]);
  } else {
    col[idx] = 0;
  }
}

template <nnalgebra::DataType Tin>
__global__ void im2col_nhwc_block_qi32_kernel(const float* __restrict__ x,
                                              int32_t* __restrict__ col,
                                              int n, int h, int w, int c,
                                              int ho_start, int wo_start,
                                              int block_ho, int block_wo,
                                              int r, int s,
                                              int pad_h, int pad_w,
                                              int stride_h, int stride_w,
                                              int dilation_h, int dilation_w,
                                              int c_base, int cin_group,
                                              const nnalgebra::QuantizationParameters<Tin>* in_qp) {
  const int m = n * block_ho * block_wo;
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * kdim;
  if (idx >= total) return;

  const int row = idx / kdim;
  const int col_idx = idx - row * kdim;

  const int ci = col_idx % cin_group;
  const int t = col_idx / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  const int n_idx = row / (block_ho * block_wo);
  const int rem = row - n_idx * (block_ho * block_wo);
  const int lho = rem / block_wo;
  const int lwo = rem - lho * block_wo;

  const int ho = ho_start + lho;
  const int wo = wo_start + lwo;
  const int hi = ho * stride_h - pad_h + rr * dilation_h;
  const int wi = wo * stride_w - pad_w + ss * dilation_w;
  if (hi >= 0 && hi < h && wi >= 0 && wi < w) {
    const int32_t raw = static_cast<int32_t>(x[idx_nhwc(n_idx, hi, wi, c_base + ci, h, w, c)]);
    col[idx] = raw - nnalgebra::getZeroPoint(in_qp[n_idx]);
  } else {
    col[idx] = 0;
  }
}

template <nnalgebra::DataType Tin>
__global__ void pack_filter_krsc_group_qi32_kernel(const float* __restrict__ w,
                                                   int32_t* __restrict__ wg,
                                                   int r, int s, int cin_group,
                                                   int k_base, int kout_group,
                                                   const nnalgebra::QuantizationParameters<Tin>* f_qp) {
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kout_group * kdim;
  if (idx >= total) return;

  const int ko = idx / kdim;
  const int k_idx = idx - ko * kdim;

  const int ci = k_idx % cin_group;
  const int t = k_idx / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  const int32_t raw = static_cast<int32_t>(w[idx_krsc(k_base + ko, rr, ss, ci, r, s, cin_group)]);
  wg[idx] = raw - nnalgebra::getZeroPoint(*f_qp);
}

template <nnalgebra::DataType Tin>
__global__ void pack_block_filter_kbybxrsc_group_qi32_kernel(const float* __restrict__ w,
                                                             int32_t* __restrict__ wg,
                                                             int by_count, int bx_count,
                                                             int r, int s, int cin_group,
                                                             int block_y, int block_x,
                                                             int k_base, int kout_group,
                                                             const nnalgebra::QuantizationParameters<Tin>* f_qp) {
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kout_group * kdim;
  if (idx >= total) return;

  const int ko = idx / kdim;
  const int k_idx = idx - ko * kdim;

  const int ci = k_idx % cin_group;
  const int t = k_idx / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  const int32_t raw = static_cast<int32_t>(w[idx_kbybxrsc(k_base + ko, block_y, block_x, rr, ss, ci,
                                                          by_count, bx_count, r, s, cin_group)]);
  wg[idx] = raw - nnalgebra::getZeroPoint(*f_qp);
}

__global__ void unpack_matrix_to_nhwc_group_i32_kernel(const int32_t* __restrict__ src,
                                                       int32_t* __restrict__ dst,
                                                       int n, int h, int w, int c,
                                                       int c_base, int c_count) {
  const int m = n * h * w;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * c_count;
  if (idx >= total) return;

  const int row = idx / c_count;
  const int col = idx - row * c_count;

  const int n_idx = row / (h * w);
  const int rem = row - n_idx * (h * w);
  const int h_idx = rem / w;
  const int w_idx = rem - h_idx * w;

  dst[idx_nhwc(n_idx, h_idx, w_idx, c_base + col, h, w, c)] = src[idx];
}

__global__ void unpack_matrix_to_nhwc_group_app_kernel(const float* __restrict__ src,
                                                       float* __restrict__ dst,
                                                       int n, int base_h, int base_w,
                                                       int ay, int ax,
                                                       int c, int c_base, int c_count) {
  const int m = n * base_h * base_w;
  const int ncol = c_count * ay * ax;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * ncol;
  if (idx >= total) return;

  const int row = idx / ncol;
  const int ext_col = idx - row * ncol;

  int ko, ay_idx, ax_idx;
  decode_output_app_index(ext_col, ay, ax, ko, ay_idx, ax_idx);

  int n_idx, ho_base, wo_base;
  decode_conv_row(row, base_h, base_w, n_idx, ho_base, wo_base);
  dst[idx_nhwc(n_idx, ho_base * ay + ay_idx, wo_base * ax + ax_idx,
               c_base + ko, base_h * ay, base_w * ax, c)] = src[idx];
}

__global__ void unpack_matrix_to_nhwc_group_block_i32_kernel(const int32_t* __restrict__ src,
                                                             int32_t* __restrict__ dst,
                                                             int n, int h, int w, int c,
                                                             int ho_start, int wo_start,
                                                             int block_ho, int block_wo,
                                                             int c_base, int c_count) {
  const int m = n * block_ho * block_wo;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * c_count;
  if (idx >= total) return;

  const int row = idx / c_count;
  const int col = idx - row * c_count;

  const int n_idx = row / (block_ho * block_wo);
  const int rem = row - n_idx * (block_ho * block_wo);
  const int lho = rem / block_wo;
  const int lwo = rem - lho * block_wo;

  dst[idx_nhwc(n_idx, ho_start + lho, wo_start + lwo, c_base + col, h, w, c)] = src[idx];
}

template <nnalgebra::DataType Tin>
void launch_fprop_nhwc_qi32_impl(const float* d_x, const float* d_w, int32_t* d_y,
                                 int n, int h, int w, int c, int r, int s, int k,
                                 const Conv2DParams& p,
                                 const nnalgebra::QuantizationParameters<Tin>* in_qp,
                                 const nnalgebra::QuantizationParameters<Tin>* f_qp) {
  if (p.ay != 1 || p.ax != 1) {
    throw std::runtime_error("quantized NHWC fprop does not support ay/ax != 1");
  }
  TensorNHWC x_shape(n, h, w, c);
  FilterKRSC w_shape(r, s, c / p.groups, k);
  const ConvShape sh = infer_conv_shape(x_shape, w_shape, p);

  const int m = n * sh.ho * sh.wo;
  const int kdim = r * s * sh.cin_group;
  const int ncol = sh.kout_group;
  const int t = 256;

  QuantizedWorkspace ws;
  ensure_capacity(&ws.d_col, &ws.d_col_cap, static_cast<size_t>(m) * kdim);
  ensure_capacity(&ws.d_wg, &ws.d_wg_cap, static_cast<size_t>(kdim) * ncol);
  ensure_capacity(&ws.d_ymat, &ws.d_ymat_cap, static_cast<size_t>(m) * ncol);

  CUDA_CHECK(cudaMemset(d_y, 0, static_cast<size_t>(n) * sh.ho * sh.wo * k * sizeof(int32_t)));

  for (int g = 0; g < p.groups; ++g) {
    const int cin_base = g * sh.cin_group;
    const int kout_base = g * sh.kout_group;

    const int total_col = m * kdim;
    const int blocks_col = (total_col + t - 1) / t;
    im2col_nhwc_qi32_kernel<Tin><<<blocks_col, t>>>(d_x, ws.d_col,
                                                    n, h, w, c,
                                                    sh.ho, sh.wo,
                                                    r, s,
                                                    p.pad_h, p.pad_w,
                                                    p.stride_h, p.stride_w,
                                                    p.dilation_h, p.dilation_w,
                                                    cin_base, sh.cin_group,
                                                    in_qp);

    const int total_wg = kdim * ncol;
    const int blocks_wg = (total_wg + t - 1) / t;
    pack_filter_krsc_group_qi32_kernel<Tin><<<blocks_wg, t>>>(d_w, ws.d_wg,
                                                              r, s, sh.cin_group,
                                                              kout_base, sh.kout_group,
                                                              f_qp);

    bmm_matmul_i32(ws.d_col, ws.d_wg, ws.d_ymat, 1, m, ncol, kdim,
                   BMM_TRANSPOSE_NONE, BMM_TRANSPOSE_YES);

    const int total_ym = m * ncol;
    const int blocks_ym = (total_ym + t - 1) / t;
    unpack_matrix_to_nhwc_group_i32_kernel<<<blocks_ym, t>>>(ws.d_ymat, d_y,
                                                             n, sh.ho, sh.wo, k,
                                                             kout_base, sh.kout_group);
  }

  CUDA_CHECK(cudaGetLastError());
}

template <nnalgebra::DataType Tin>
void launch_block_fprop_nhwc_qi32_impl(const float* d_x, const float* d_w, int32_t* d_y,
                                       int n, int h, int w, int c, int r, int s, int k,
                                       const BlockConv2DParams& p,
                                       const nnalgebra::QuantizationParameters<Tin>* in_qp,
                                       const nnalgebra::QuantizationParameters<Tin>* f_qp) {
  if (p.conv.ay != 1 || p.conv.ax != 1) {
    throw std::runtime_error("quantized blocked NHWC fprop does not support ay/ax != 1");
  }
  TensorNHWC x_shape(n, h, w, c);
  BlockFilterKByBxRSC w_shape(k, p.block_by, p.block_bx, r, s, c / p.conv.groups);
  const BlockConvShape sh = infer_block_conv_shape(x_shape, w_shape, p);

  const int m_block = n * sh.block_ho * sh.block_wo;
  const int kdim = r * s * sh.base.cin_group;
  const int ncol = sh.base.kout_group;
  const int t = 256;

  QuantizedWorkspace ws;
  ensure_capacity(&ws.d_col, &ws.d_col_cap, static_cast<size_t>(m_block) * kdim);
  ensure_capacity(&ws.d_wg, &ws.d_wg_cap, static_cast<size_t>(kdim) * ncol);
  ensure_capacity(&ws.d_ymat, &ws.d_ymat_cap, static_cast<size_t>(m_block) * ncol);

  CUDA_CHECK(cudaMemset(d_y, 0, static_cast<size_t>(n) * sh.base.ho * sh.base.wo * k * sizeof(int32_t)));

  for (int g = 0; g < p.conv.groups; ++g) {
    const int cin_base = g * sh.base.cin_group;
    const int kout_base = g * sh.base.kout_group;
    for (int by = 0; by < p.block_by; ++by) {
      const int ho_start = by * sh.block_ho;
      for (int bx = 0; bx < p.block_bx; ++bx) {
        const int wo_start = bx * sh.block_wo;

        const int total_col = m_block * kdim;
        const int blocks_col = (total_col + t - 1) / t;
        im2col_nhwc_block_qi32_kernel<Tin><<<blocks_col, t>>>(d_x, ws.d_col,
                                                              n, h, w, c,
                                                              ho_start, wo_start,
                                                              sh.block_ho, sh.block_wo,
                                                              r, s,
                                                              p.conv.pad_h, p.conv.pad_w,
                                                              p.conv.stride_h, p.conv.stride_w,
                                                              p.conv.dilation_h, p.conv.dilation_w,
                                                              cin_base, sh.base.cin_group,
                                                              in_qp);

        const int total_wg = kdim * ncol;
        const int blocks_wg = (total_wg + t - 1) / t;
        pack_block_filter_kbybxrsc_group_qi32_kernel<Tin><<<blocks_wg, t>>>(d_w, ws.d_wg,
                                                                            p.block_by, p.block_bx,
                                                                            r, s, sh.base.cin_group,
                                                                            by, bx,
                                                                            kout_base, sh.base.kout_group,
                                                                            f_qp);

        bmm_matmul_i32(ws.d_col, ws.d_wg, ws.d_ymat, 1, m_block, ncol, kdim,
                       BMM_TRANSPOSE_NONE, BMM_TRANSPOSE_YES);

        const int total_ym = m_block * ncol;
        const int blocks_ym = (total_ym + t - 1) / t;
        unpack_matrix_to_nhwc_group_block_i32_kernel<<<blocks_ym, t>>>(ws.d_ymat, d_y,
                                                                       n, sh.base.ho, sh.base.wo, k,
                                                                       ho_start, wo_start,
                                                                       sh.block_ho, sh.block_wo,
                                                                       kout_base, sh.base.kout_group);
      }
    }
  }

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

namespace conv_quant_detail {

void launch_fprop_nhwc_qi32_u8(const float* d_x, const float* d_w, int32_t* d_y,
                               int n, int h, int w, int c, int r, int s, int k,
                               const Conv2DParams& p,
                               const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* in_qp,
                               const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* f_qp) {
  launch_fprop_nhwc_qi32_impl(d_x, d_w, d_y, n, h, w, c, r, s, k, p, in_qp, f_qp);
}

void launch_fprop_nhwc_qi32_s5(const float* d_x, const float* d_w, int32_t* d_y,
                               int n, int h, int w, int c, int r, int s, int k,
                               const Conv2DParams& p,
                               const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* in_qp,
                               const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* f_qp) {
  launch_fprop_nhwc_qi32_impl(d_x, d_w, d_y, n, h, w, c, r, s, k, p, in_qp, f_qp);
}

void launch_block_fprop_nhwc_qi32_u8(const float* d_x, const float* d_w, int32_t* d_y,
                                     int n, int h, int w, int c, int r, int s, int k,
                                     const BlockConv2DParams& p,
                                     const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* in_qp,
                                     const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* f_qp) {
  launch_block_fprop_nhwc_qi32_impl(d_x, d_w, d_y, n, h, w, c, r, s, k, p, in_qp, f_qp);
}

void launch_block_fprop_nhwc_qi32_s5(const float* d_x, const float* d_w, int32_t* d_y,
                                     int n, int h, int w, int c, int r, int s, int k,
                                     const BlockConv2DParams& p,
                                     const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* in_qp,
                                     const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* f_qp) {
  launch_block_fprop_nhwc_qi32_impl(d_x, d_w, d_y, n, h, w, c, r, s, k, p, in_qp, f_qp);
}

}  // namespace conv_quant_detail
