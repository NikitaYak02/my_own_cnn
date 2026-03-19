#include "conv_types.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>

namespace {
constexpr int GEMM_TILE_M = 32;
constexpr int GEMM_TILE_N = 32;
constexpr int GEMM_TILE_K = 8;
constexpr int GEMM_THREAD_TILE_M = 2;
constexpr int GEMM_THREAD_TILE_N = 2;

struct Workspace {
  float* d_col = nullptr;
  size_t d_col_cap = 0;
  float* d_wg = nullptr;
  size_t d_wg_cap = 0;
  float* d_wg_all = nullptr;
  size_t d_wg_all_cap = 0;
  const float* packed_w_src = nullptr;
  int packed_r = 0;
  int packed_s = 0;
  int packed_cin_group = 0;
  int packed_k = 0;
  int packed_groups = 0;
  float* d_ymat = nullptr;
  size_t d_ymat_cap = 0;
  float* d_dy_mat = nullptr;
  size_t d_dy_mat_cap = 0;
  float* d_dcol = nullptr;
  size_t d_dcol_cap = 0;
  float* d_dwg = nullptr;
  size_t d_dwg_cap = 0;
  int* d_spatial_map = nullptr;
  size_t d_spatial_map_cap = 0;
  int* d_row_nhw_base = nullptr;
  size_t d_row_nhw_base_cap = 0;
  int* d_row_spatial_base = nullptr;
  size_t d_row_spatial_base_cap = 0;
  int* d_k_ci = nullptr;
  size_t d_k_ci_cap = 0;
  int* d_k_rs = nullptr;
  size_t d_k_rs_cap = 0;
  int map_h = 0;
  int map_w = 0;
  int map_ho = 0;
  int map_wo = 0;
  int map_r = 0;
  int map_s = 0;
  int map_pad_h = 0;
  int map_pad_w = 0;
  int map_stride_h = 0;
  int map_stride_w = 0;
  int map_dilation_h = 0;
  int map_dilation_w = 0;
  int rowmap_n = 0;
  int rowmap_ho = 0;
  int rowmap_wo = 0;
  int rowmap_h = 0;
  int rowmap_w = 0;
  int kmap_r = 0;
  int kmap_s = 0;
  int kmap_cin_group = 0;

  ~Workspace() {
    cudaFree(d_col);
    cudaFree(d_wg);
    cudaFree(d_wg_all);
    cudaFree(d_ymat);
    cudaFree(d_dy_mat);
    cudaFree(d_dcol);
    cudaFree(d_dwg);
    cudaFree(d_spatial_map);
    cudaFree(d_row_nhw_base);
    cudaFree(d_row_spatial_base);
    cudaFree(d_k_ci);
    cudaFree(d_k_rs);
  }
};

void ensure_capacity(float** ptr, size_t* cap, size_t elements) {
  if (*cap >= elements) return;
  if (*ptr) CUDA_CHECK(cudaFree(*ptr));
  CUDA_CHECK(cudaMalloc(ptr, elements * sizeof(float)));
  *cap = elements;
}

void ensure_capacity_int(int** ptr, size_t* cap, size_t elements) {
  if (*cap >= elements) return;
  if (*ptr) CUDA_CHECK(cudaFree(*ptr));
  CUDA_CHECK(cudaMalloc(ptr, elements * sizeof(int)));
  *cap = elements;
}

__global__ void pack_filter_hwcn_group_kernel(const float* __restrict__ w,
                                              float* __restrict__ wg,
                                              int r, int s, int cin_group,
                                              int k_total,
                                              int k_base, int kout_group);

__global__ void precompute_spatial_map_kernel(int* __restrict__ spatial_map,
                                              int h, int w, int ho, int wo,
                                              int r, int s,
                                              int pad_h, int pad_w,
                                              int stride_h, int stride_w,
                                              int dilation_h, int dilation_w);
__global__ void precompute_row_maps_kernel(int* __restrict__ row_nhw_base,
                                           int* __restrict__ row_spatial_base,
                                           int n, int h, int w, int ho, int wo, int rs);
__global__ void precompute_k_maps_kernel(int* __restrict__ k_ci,
                                         int* __restrict__ k_rs,
                                         int r, int s, int cin_group);

void ensure_packed_weights(Workspace& ws,
                           const float* d_w,
                           int r, int s, int cin_group, int k, int groups) {
  const size_t kdim = static_cast<size_t>(r) * s * cin_group;
  const size_t ncol = k / groups;
  const size_t per_group = kdim * ncol;
  const size_t total = per_group * groups;
  ensure_capacity(&ws.d_wg_all, &ws.d_wg_all_cap, total);

  const bool need_repack =
      (ws.packed_w_src != d_w) ||
      (ws.packed_r != r) ||
      (ws.packed_s != s) ||
      (ws.packed_cin_group != cin_group) ||
      (ws.packed_k != k) ||
      (ws.packed_groups != groups);
  if (!need_repack) return;

  const int t = 256;
  for (int g = 0; g < groups; ++g) {
    const int kout_base = g * static_cast<int>(ncol);
    float* dst = ws.d_wg_all + g * per_group;
    int blocks = static_cast<int>((per_group + t - 1) / t);
    pack_filter_hwcn_group_kernel<<<blocks, t>>>(d_w, dst, r, s, cin_group, k, kout_base, static_cast<int>(ncol));
  }
  CUDA_CHECK(cudaGetLastError());
  ws.packed_w_src = d_w;
  ws.packed_r = r;
  ws.packed_s = s;
  ws.packed_cin_group = cin_group;
  ws.packed_k = k;
  ws.packed_groups = groups;
}

void ensure_spatial_map(Workspace& ws,
                        int h, int w, int ho, int wo,
                        int r, int s,
                        int pad_h, int pad_w,
                        int stride_h, int stride_w,
                        int dilation_h, int dilation_w) {
  const size_t rs = static_cast<size_t>(r) * s;
  const size_t count = static_cast<size_t>(ho) * wo * rs;
  ensure_capacity_int(&ws.d_spatial_map, &ws.d_spatial_map_cap, count);

  const bool need_rebuild =
      (ws.map_h != h) || (ws.map_w != w) || (ws.map_ho != ho) || (ws.map_wo != wo) ||
      (ws.map_r != r) || (ws.map_s != s) ||
      (ws.map_pad_h != pad_h) || (ws.map_pad_w != pad_w) ||
      (ws.map_stride_h != stride_h) || (ws.map_stride_w != stride_w) ||
      (ws.map_dilation_h != dilation_h) || (ws.map_dilation_w != dilation_w);
  if (!need_rebuild) return;

  const int t = 256;
  const int blocks = static_cast<int>((count + t - 1) / t);
  precompute_spatial_map_kernel<<<blocks, t>>>(ws.d_spatial_map, h, w, ho, wo, r, s, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
  CUDA_CHECK(cudaGetLastError());

  ws.map_h = h;
  ws.map_w = w;
  ws.map_ho = ho;
  ws.map_wo = wo;
  ws.map_r = r;
  ws.map_s = s;
  ws.map_pad_h = pad_h;
  ws.map_pad_w = pad_w;
  ws.map_stride_h = stride_h;
  ws.map_stride_w = stride_w;
  ws.map_dilation_h = dilation_h;
  ws.map_dilation_w = dilation_w;
}

void ensure_row_maps(Workspace& ws, int n, int h, int w, int ho, int wo, int rs) {
  const size_t m = static_cast<size_t>(n) * ho * wo;
  ensure_capacity_int(&ws.d_row_nhw_base, &ws.d_row_nhw_base_cap, m);
  ensure_capacity_int(&ws.d_row_spatial_base, &ws.d_row_spatial_base_cap, m);

  const bool need_rebuild =
      (ws.rowmap_n != n) || (ws.rowmap_h != h) || (ws.rowmap_w != w) ||
      (ws.rowmap_ho != ho) || (ws.rowmap_wo != wo);
  if (!need_rebuild) return;

  const int t = 256;
  const int blocks = static_cast<int>((m + t - 1) / t);
  precompute_row_maps_kernel<<<blocks, t>>>(ws.d_row_nhw_base, ws.d_row_spatial_base, n, h, w, ho, wo, rs);
  CUDA_CHECK(cudaGetLastError());
  ws.rowmap_n = n;
  ws.rowmap_h = h;
  ws.rowmap_w = w;
  ws.rowmap_ho = ho;
  ws.rowmap_wo = wo;
}

void ensure_k_maps(Workspace& ws, int r, int s, int cin_group) {
  const size_t kdim = static_cast<size_t>(r) * s * cin_group;
  ensure_capacity_int(&ws.d_k_ci, &ws.d_k_ci_cap, kdim);
  ensure_capacity_int(&ws.d_k_rs, &ws.d_k_rs_cap, kdim);

  const bool need_rebuild =
      (ws.kmap_r != r) || (ws.kmap_s != s) || (ws.kmap_cin_group != cin_group);
  if (!need_rebuild) return;

  const int t = 256;
  const int blocks = static_cast<int>((kdim + t - 1) / t);
  precompute_k_maps_kernel<<<blocks, t>>>(ws.d_k_ci, ws.d_k_rs, r, s, cin_group);
  CUDA_CHECK(cudaGetLastError());
  ws.kmap_r = r;
  ws.kmap_s = s;
  ws.kmap_cin_group = cin_group;
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

__global__ void precompute_spatial_map_kernel(int* __restrict__ spatial_map,
                                              int h, int w, int ho, int wo,
                                              int r, int s,
                                              int pad_h, int pad_w,
                                              int stride_h, int stride_w,
                                              int dilation_h, int dilation_w) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int rs = r * s;
  const int total = ho * wo * rs;
  if (idx >= total) return;

  const int t = idx;
  const int rs_idx = t % rs;
  const int ow = (t / rs) % wo;
  const int oh = t / (rs * wo);
  const int rr = rs_idx / s;
  const int ss = rs_idx % s;

  const int hi = oh * stride_h - pad_h + rr * dilation_h;
  const int wi = ow * stride_w - pad_w + ss * dilation_w;
  spatial_map[idx] = (hi >= 0 && hi < h && wi >= 0 && wi < w) ? (hi * w + wi) : -1;
}

__global__ void precompute_row_maps_kernel(int* __restrict__ row_nhw_base,
                                           int* __restrict__ row_spatial_base,
                                           int n, int h, int w, int ho, int wo, int rs) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = n * ho * wo;
  if (idx >= m) return;
  const int n_idx = idx / (ho * wo);
  const int rem = idx - n_idx * (ho * wo);
  const int ho_idx = rem / wo;
  const int wo_idx = rem - ho_idx * wo;
  row_nhw_base[idx] = n_idx * h * w;
  row_spatial_base[idx] = (ho_idx * wo + wo_idx) * rs;
}

__global__ void precompute_k_maps_kernel(int* __restrict__ k_ci,
                                         int* __restrict__ k_rs,
                                         int r, int s, int cin_group) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int kdim = r * s * cin_group;
  if (idx >= kdim) return;
  k_ci[idx] = idx % cin_group;
  k_rs[idx] = idx / cin_group;
}

__global__ void im2col_nhwc_precomp_kernel(const float* __restrict__ x,
                                           const int* __restrict__ spatial_map,
                                           float* __restrict__ col,
                                           int n, int h, int w, int c,
                                           int ho, int wo, int r, int s,
                                           int c_base, int cin_group) {
  const int m = n * ho * wo;
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * kdim;
  if (idx >= total) return;

  const int row = idx / kdim;
  const int col_idx = idx - row * kdim;
  const int ci = col_idx % cin_group;
  const int rs_idx = col_idx / cin_group;

  const int n_idx = row / (ho * wo);
  const int ow = row % wo;
  const int oh = (row / wo) % ho;
  const int hw = spatial_map[(oh * wo + ow) * (r * s) + rs_idx];
  if (hw < 0) {
    col[idx] = 0.0f;
    return;
  }
  const int x_idx = ((n_idx * h * w + hw) * c) + (c_base + ci);
  col[idx] = x[x_idx];
}

__global__ void im2col_nhwc_precomp_vec4_kernel(const float* __restrict__ x,
                                                const int* __restrict__ spatial_map,
                                                float* __restrict__ col,
                                                int n, int h, int w, int c,
                                                int ho, int wo, int r, int s,
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
  const int rs_idx = col_idx4 / cin_group4;

  const int n_idx = row / (ho * wo);
  const int ow = row % wo;
  const int oh = (row / wo) % ho;
  const int hw = spatial_map[(oh * wo + ow) * (r * s) + rs_idx];

  const int out_base = row * (r * s * cin_group) + rs_idx * cin_group + ci4;
  if (hw < 0) {
    *reinterpret_cast<float4*>(col + out_base) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }
  const int x_idx = ((n_idx * h * w + hw) * c) + (c_base + ci4);
  *reinterpret_cast<float4*>(col + out_base) = *reinterpret_cast<const float4*>(x + x_idx);
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

__global__ void pack_filter_hwcn_group_kernel(const float* __restrict__ w,
                                              float* __restrict__ wg,
                                              int r, int s, int cin_group,
                                              int k_total,
                                              int k_base, int kout_group) {
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kdim * kout_group;
  if (idx >= total) return;

  const int row = idx / kout_group;
  const int col = idx - row * kout_group;

  const int ci = row % cin_group;
  const int t = row / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  wg[idx] = w[idx_hwcn(rr, ss, ci, k_base + col, s, cin_group, k_total)];
}

__global__ void unpack_filter_hwcn_group_kernel(const float* __restrict__ wg,
                                                float* __restrict__ w,
                                                int r, int s, int cin_group,
                                                int k_total,
                                                int k_base, int kout_group) {
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kdim * kout_group;
  if (idx >= total) return;

  const int row = idx / kout_group;
  const int col = idx - row * kout_group;

  const int ci = row % cin_group;
  const int t = row / cin_group;
  const int ss = t % s;
  const int rr = t / s;

  w[idx_hwcn(rr, ss, ci, k_base + col, s, cin_group, k_total)] = wg[idx];
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

__global__ void col2im_accum_nhwc_precomp_kernel(const float* __restrict__ dcol,
                                                 const int* __restrict__ spatial_map,
                                                 float* __restrict__ dx,
                                                 int n, int h, int w, int c,
                                                 int ho, int wo, int r, int s,
                                                 int c_base, int cin_group) {
  const int m = n * ho * wo;
  const int kdim = r * s * cin_group;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * kdim;
  if (idx >= total) return;

  const int row = idx / kdim;
  const int col_idx = idx - row * kdim;
  const int ci = col_idx % cin_group;
  const int rs_idx = col_idx / cin_group;

  const int n_idx = row / (ho * wo);
  const int ow = row % wo;
  const int oh = (row / wo) % ho;
  const int hw = spatial_map[(oh * wo + ow) * (r * s) + rs_idx];
  if (hw < 0) return;
  const int x_idx = ((n_idx * h * w + hw) * c) + (c_base + ci);
  atomicAdd(&dx[x_idx], dcol[idx]);
}

__global__ void fprop_implicit_gemm_kernel(const float* __restrict__ x,
                                           const float* __restrict__ wg,
                                           const int* __restrict__ row_nhw_base,
                                           const int* __restrict__ row_spatial_base,
                                           const int* __restrict__ k_ci,
                                           const int* __restrict__ k_rs,
                                           const int* __restrict__ spatial_map,
                                           float* __restrict__ ymat,
                                           int n, int h, int w, int c,
                                           int ho, int wo,
                                           int r, int s,
                                           int pad_h, int pad_w,
                                           int stride_h, int stride_w,
                                           int dilation_h, int dilation_w,
                                           int cin_base, int cin_group,
                                           int ncol,
                                           bool use_vec4) {
  __shared__ float As[GEMM_TILE_M][GEMM_TILE_K + 1];
  __shared__ float Bs[GEMM_TILE_K][GEMM_TILE_N + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  const int m = n * ho * wo;
  const int kdim = r * s * cin_group;
  const int block_row = blockIdx.y * GEMM_TILE_M;
  const int block_col = blockIdx.x * GEMM_TILE_N;
  const int row0 = block_row + ty * GEMM_THREAD_TILE_M;
  const int col0 = block_col + tx * GEMM_THREAD_TILE_N;

  float acc[GEMM_THREAD_TILE_M][GEMM_THREAD_TILE_N] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

  (void)pad_h;
  (void)pad_w;
  (void)stride_h;
  (void)stride_w;
  (void)dilation_h;
  (void)dilation_w;

  for (int k0 = 0; k0 < kdim; k0 += GEMM_TILE_K) {
    const int a_row = tid / GEMM_TILE_K;
    const int a_col = tid % GEMM_TILE_K;
    const int g_a_row = block_row + a_row;
    const int g_a_col = k0 + a_col;
    if (use_vec4 && (a_col % 4 == 0)) {
      if (g_a_row < m && (g_a_col + 3) < kdim) {
        const int nhw_base = row_nhw_base[g_a_row];
        const int sp_base = row_spatial_base[g_a_row];
        const int rs_idx = k_rs[g_a_col];
        const int ci = k_ci[g_a_col];
        const int hw = spatial_map[sp_base + rs_idx];
        if (hw >= 0) {
          const int x_base = ((nhw_base + hw) * c) + (cin_base + ci);
          const float4 v = *reinterpret_cast<const float4*>(x + x_base);
          As[a_row][a_col + 0] = v.x;
          As[a_row][a_col + 1] = v.y;
          As[a_row][a_col + 2] = v.z;
          As[a_row][a_col + 3] = v.w;
        } else {
          As[a_row][a_col + 0] = 0.0f;
          As[a_row][a_col + 1] = 0.0f;
          As[a_row][a_col + 2] = 0.0f;
          As[a_row][a_col + 3] = 0.0f;
        }
      } else {
        if (a_col + 0 < GEMM_TILE_K) As[a_row][a_col + 0] = 0.0f;
        if (a_col + 1 < GEMM_TILE_K) As[a_row][a_col + 1] = 0.0f;
        if (a_col + 2 < GEMM_TILE_K) As[a_row][a_col + 2] = 0.0f;
        if (a_col + 3 < GEMM_TILE_K) As[a_row][a_col + 3] = 0.0f;
      }
    } else if (!use_vec4) {
      if (g_a_row < m && g_a_col < kdim) {
        const int nhw_base = row_nhw_base[g_a_row];
        const int sp_base = row_spatial_base[g_a_row];
        const int rs_idx = k_rs[g_a_col];
        const int ci = k_ci[g_a_col];
        const int hw = spatial_map[sp_base + rs_idx];
        As[a_row][a_col] = (hw >= 0) ? x[((nhw_base + hw) * c) + (cin_base + ci)] : 0.0f;
      } else {
        As[a_row][a_col] = 0.0f;
      }
    }

    const int b_row = tid / GEMM_TILE_N;
    const int b_col = tid % GEMM_TILE_N;
    const int g_b_row = k0 + b_row;
    const int g_b_col = block_col + b_col;
    Bs[b_row][b_col] = (g_b_row < kdim && g_b_col < ncol) ? wg[g_b_row * ncol + g_b_col] : 0.0f;

    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < GEMM_TILE_K; ++kk) {
      const float a0 = (row0 + 0 < m) ? As[ty * GEMM_THREAD_TILE_M + 0][kk] : 0.0f;
      const float a1 = (row0 + 1 < m) ? As[ty * GEMM_THREAD_TILE_M + 1][kk] : 0.0f;
      const float b0 = (col0 + 0 < ncol) ? Bs[kk][tx * GEMM_THREAD_TILE_N + 0] : 0.0f;
      const float b1 = (col0 + 1 < ncol) ? Bs[kk][tx * GEMM_THREAD_TILE_N + 1] : 0.0f;
      acc[0][0] += a0 * b0;
      acc[0][1] += a0 * b1;
      acc[1][0] += a1 * b0;
      acc[1][1] += a1 * b1;
    }
    __syncthreads();
  }

  if (row0 + 0 < m && col0 + 0 < ncol) ymat[(row0 + 0) * ncol + (col0 + 0)] = acc[0][0];
  if (row0 + 0 < m && col0 + 1 < ncol) ymat[(row0 + 0) * ncol + (col0 + 1)] = acc[0][1];
  if (row0 + 1 < m && col0 + 0 < ncol) ymat[(row0 + 1) * ncol + (col0 + 0)] = acc[1][0];
  if (row0 + 1 < m && col0 + 1 < ncol) ymat[(row0 + 1) * ncol + (col0 + 1)] = acc[1][1];
}

template <bool TRANS_A, bool TRANS_B>
__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K,
                                  float alpha, float beta) {
  __shared__ float As[GEMM_TILE_M][GEMM_TILE_K + 1];
  __shared__ float Bs[GEMM_TILE_K][GEMM_TILE_N + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int block_row = blockIdx.y * GEMM_TILE_M;
  const int block_col = blockIdx.x * GEMM_TILE_N;
  const int row0 = block_row + ty * GEMM_THREAD_TILE_M;
  const int col0 = block_col + tx * GEMM_THREAD_TILE_N;

  float acc[GEMM_THREAD_TILE_M][GEMM_THREAD_TILE_N] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

  for (int k0 = 0; k0 < K; k0 += GEMM_TILE_K) {
    const int a_row = tid / GEMM_TILE_K;
    const int a_col = tid % GEMM_TILE_K;
    const int g_a_row = block_row + a_row;
    const int g_a_col = k0 + a_col;
    if (g_a_row < M && g_a_col < K) {
      As[a_row][a_col] = TRANS_A ? A[g_a_col * M + g_a_row] : A[g_a_row * K + g_a_col];
    } else {
      As[a_row][a_col] = 0.0f;
    }

    const int b_row = tid / GEMM_TILE_N;
    const int b_col = tid % GEMM_TILE_N;
    const int g_b_row = k0 + b_row;
    const int g_b_col = block_col + b_col;
    if (g_b_row < K && g_b_col < N) {
      Bs[b_row][b_col] = TRANS_B ? B[g_b_col * K + g_b_row] : B[g_b_row * N + g_b_col];
    } else {
      Bs[b_row][b_col] = 0.0f;
    }

    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < GEMM_TILE_K; ++kk) {
      const float a0 = (row0 + 0 < M) ? As[ty * GEMM_THREAD_TILE_M + 0][kk] : 0.0f;
      const float a1 = (row0 + 1 < M) ? As[ty * GEMM_THREAD_TILE_M + 1][kk] : 0.0f;
      const float b0 = (col0 + 0 < N) ? Bs[kk][tx * GEMM_THREAD_TILE_N + 0] : 0.0f;
      const float b1 = (col0 + 1 < N) ? Bs[kk][tx * GEMM_THREAD_TILE_N + 1] : 0.0f;
      acc[0][0] += a0 * b0;
      acc[0][1] += a0 * b1;
      acc[1][0] += a1 * b0;
      acc[1][1] += a1 * b1;
    }
    __syncthreads();
  }

  if (row0 + 0 < M && col0 + 0 < N) {
    const int idx = (row0 + 0) * N + (col0 + 0);
    C[idx] = alpha * acc[0][0] + beta * C[idx];
  }
  if (row0 + 0 < M && col0 + 1 < N) {
    const int idx = (row0 + 0) * N + (col0 + 1);
    C[idx] = alpha * acc[0][1] + beta * C[idx];
  }
  if (row0 + 1 < M && col0 + 0 < N) {
    const int idx = (row0 + 1) * N + (col0 + 0);
    C[idx] = alpha * acc[1][0] + beta * C[idx];
  }
  if (row0 + 1 < M && col0 + 1 < N) {
    const int idx = (row0 + 1) * N + (col0 + 1);
    C[idx] = alpha * acc[1][1] + beta * C[idx];
  }
}

void launch_gemm(const float* A, const float* B, float* C, int M, int N, int K, bool transA, bool transB, float alpha = 1.0f, float beta = 0.0f) {
  dim3 block(GEMM_TILE_N / GEMM_THREAD_TILE_N, GEMM_TILE_M / GEMM_THREAD_TILE_M);
  dim3 grid((N + GEMM_TILE_N - 1) / GEMM_TILE_N, (M + GEMM_TILE_M - 1) / GEMM_TILE_M);
  if (!transA && !transB) {
    gemm_tiled_kernel<false, false><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
  } else if (!transA && transB) {
    gemm_tiled_kernel<false, true><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
  } else if (transA && !transB) {
    gemm_tiled_kernel<true, false><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
  } else {
    gemm_tiled_kernel<true, true><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
  }
}

}  // namespace

void launch_fprop_nhwc(const float* d_x, const float* d_w, float* d_y,
                       int n, int h, int w, int c, int r, int s, int k,
                       const Conv2DParams& p,
                       bool use_implicit_precomp) {
  TensorNHWC x_shape(n, h, w, c);
  FilterHWCN w_shape(r, s, c / p.groups, k);
  ConvShape sh = infer_conv_shape(x_shape, w_shape, p);

  const int m = n * sh.ho * sh.wo;
  const int kdim = r * s * sh.cin_group;
  const int ncol = sh.kout_group;
  const int rs = r * s;
  const bool use_vec4 = (sh.cin_group % 4 == 0) && ((c % 4) == 0);

  Workspace& ws = workspace();
  ensure_capacity(&ws.d_ymat, &ws.d_ymat_cap, static_cast<size_t>(m) * ncol);
  ensure_packed_weights(ws, d_w, r, s, sh.cin_group, k, p.groups);
  ensure_spatial_map(ws, h, w, sh.ho, sh.wo, r, s, p.pad_h, p.pad_w, p.stride_h, p.stride_w, p.dilation_h, p.dilation_w);
  ensure_row_maps(ws, n, h, w, sh.ho, sh.wo, rs);
  ensure_k_maps(ws, r, s, sh.cin_group);
  float* d_ymat = ws.d_ymat;
  (void)use_implicit_precomp;

  dim3 block(GEMM_TILE_N / GEMM_THREAD_TILE_N, GEMM_TILE_M / GEMM_THREAD_TILE_M);
  dim3 grid((ncol + GEMM_TILE_N - 1) / GEMM_TILE_N, (m + GEMM_TILE_M - 1) / GEMM_TILE_M);

  for (int g = 0; g < p.groups; ++g) {
    const int cin_base = g * sh.cin_group;
    const int kout_base = g * sh.kout_group;
    float* d_wg = ws.d_wg_all + static_cast<size_t>(g) * kdim * ncol;
    const bool group_vec4 = use_vec4 && ((cin_base % 4) == 0);
    fprop_implicit_gemm_kernel<<<grid, block>>>(d_x, d_wg,
                                                 ws.d_row_nhw_base,
                                                 ws.d_row_spatial_base,
                                                 ws.d_k_ci,
                                                 ws.d_k_rs,
                                                 ws.d_spatial_map,
                                                 d_ymat,
                                                 n, h, w, c,
                                                 sh.ho, sh.wo,
                                                 r, s,
                                                 p.pad_h, p.pad_w,
                                                 p.stride_h, p.stride_w,
                                                 p.dilation_h, p.dilation_w,
                                                 cin_base, sh.cin_group, ncol, group_vec4);
    const int t = 256;
    int total_ym = m * ncol;
    int blocks_ym = (total_ym + t - 1) / t;
    unpack_matrix_to_nhwc_group_kernel<<<blocks_ym, t>>>(d_ymat, d_y, n, sh.ho, sh.wo, k, kout_base, sh.kout_group);
  }

  CUDA_CHECK(cudaGetLastError());
}

void launch_bprop_nhwc(const float* d_dy, const float* d_w, float* d_dx,
                       int n, int h, int w, int c, int r, int s, int k,
                       const Conv2DParams& p,
                       bool use_implicit_precomp) {
  TensorNHWC x_shape(n, h, w, c);
  FilterHWCN w_shape(r, s, c / p.groups, k);
  ConvShape sh = infer_conv_shape(x_shape, w_shape, p);

  const int m = n * sh.ho * sh.wo;
  const int kdim = r * s * sh.cin_group;
  const int ncol = sh.kout_group;

  Workspace& ws = workspace();
  ensure_capacity(&ws.d_dy_mat, &ws.d_dy_mat_cap, static_cast<size_t>(m) * ncol);
  ensure_capacity(&ws.d_dcol, &ws.d_dcol_cap, static_cast<size_t>(m) * kdim);
  ensure_packed_weights(ws, d_w, r, s, sh.cin_group, k, p.groups);
  if (use_implicit_precomp) {
    ensure_spatial_map(ws, h, w, sh.ho, sh.wo, r, s, p.pad_h, p.pad_w, p.stride_h, p.stride_w, p.dilation_h, p.dilation_w);
  }
  float* d_dy_mat = ws.d_dy_mat;
  float* d_dcol = ws.d_dcol;

  CUDA_CHECK(cudaMemset(d_dx, 0, static_cast<size_t>(n) * h * w * c * sizeof(float)));

  const int t = 256;

  for (int g = 0; g < p.groups; ++g) {
    const int cin_base = g * sh.cin_group;
    const int kout_base = g * sh.kout_group;

    int total_dy = m * ncol;
    int blocks_dy = (total_dy + t - 1) / t;
    pack_nhwc_group_matrix_kernel<<<blocks_dy, t>>>(d_dy, d_dy_mat, n, sh.ho, sh.wo, k, kout_base, sh.kout_group);

    float* d_wg = ws.d_wg_all + static_cast<size_t>(g) * kdim * ncol;

    launch_gemm(d_dy_mat, d_wg, d_dcol, m, kdim, ncol, false, true, 1.0f, 0.0f);

    int total_dcol = m * kdim;
    int blocks_dcol = (total_dcol + t - 1) / t;
    if (use_implicit_precomp) {
      col2im_accum_nhwc_precomp_kernel<<<blocks_dcol, t>>>(d_dcol, ws.d_spatial_map, d_dx, n, h, w, c, sh.ho, sh.wo, r, s, cin_base, sh.cin_group);
    } else {
      col2im_accum_nhwc_kernel<<<blocks_dcol, t>>>(d_dcol, d_dx,
                                                   n, h, w, c,
                                                   sh.ho, sh.wo,
                                                   r, s,
                                                   p.pad_h, p.pad_w,
                                                   p.stride_h, p.stride_w,
                                                   p.dilation_h, p.dilation_w,
                                                   cin_base, sh.cin_group);
    }
  }

  CUDA_CHECK(cudaGetLastError());
}

void launch_grad_nhwc(const float* d_x, const float* d_dy, float* d_dw,
                      int n, int h, int w, int c, int r, int s, int k,
                      const Conv2DParams& p,
                      bool use_implicit_precomp) {
  TensorNHWC x_shape(n, h, w, c);
  FilterHWCN w_shape(r, s, c / p.groups, k);
  ConvShape sh = infer_conv_shape(x_shape, w_shape, p);

  const int m = n * sh.ho * sh.wo;
  const int kdim = r * s * sh.cin_group;
  const int ncol = sh.kout_group;
  const bool can_vec4 = (sh.cin_group % 4 == 0) && (sh.cin_group >= 4) && (c % 4 == 0);

  Workspace& ws = workspace();
  ensure_capacity(&ws.d_col, &ws.d_col_cap, static_cast<size_t>(m) * kdim);
  ensure_capacity(&ws.d_dy_mat, &ws.d_dy_mat_cap, static_cast<size_t>(m) * ncol);
  ensure_capacity(&ws.d_dwg, &ws.d_dwg_cap, static_cast<size_t>(kdim) * ncol);
  if (use_implicit_precomp) {
    ensure_spatial_map(ws, h, w, sh.ho, sh.wo, r, s, p.pad_h, p.pad_w, p.stride_h, p.stride_w, p.dilation_h, p.dilation_w);
  }
  float* d_col = ws.d_col;
  float* d_dy_mat = ws.d_dy_mat;
  float* d_dwg = ws.d_dwg;

  CUDA_CHECK(cudaMemset(d_dw, 0, static_cast<size_t>(r) * s * sh.cin_group * k * sizeof(float)));

  const int t = 256;

  for (int g = 0; g < p.groups; ++g) {
    const int cin_base = g * sh.cin_group;
    const int kout_base = g * sh.kout_group;

    int total_col = m * kdim;
    int blocks_col = (total_col + t - 1) / t;
    if (use_implicit_precomp) {
      if (can_vec4 && (cin_base % 4 == 0)) {
        const int total_vec = m * r * s * (sh.cin_group / 4);
        const int blocks_vec = (total_vec + t - 1) / t;
        im2col_nhwc_precomp_vec4_kernel<<<blocks_vec, t>>>(d_x, ws.d_spatial_map, d_col, n, h, w, c, sh.ho, sh.wo, r, s, cin_base, sh.cin_group);
      } else {
        im2col_nhwc_precomp_kernel<<<blocks_col, t>>>(d_x, ws.d_spatial_map, d_col, n, h, w, c, sh.ho, sh.wo, r, s, cin_base, sh.cin_group);
      }
    } else {
      if (can_vec4 && (cin_base % 4 == 0)) {
        const int total_vec = m * r * s * (sh.cin_group / 4);
        const int blocks_vec = (total_vec + t - 1) / t;
        im2col_nhwc_vec4_kernel<<<blocks_vec, t>>>(d_x, d_col, n, h, w, c,
                                                   sh.ho, sh.wo,
                                                   r, s,
                                                   p.pad_h, p.pad_w,
                                                   p.stride_h, p.stride_w,
                                                   p.dilation_h, p.dilation_w,
                                                   cin_base, sh.cin_group);
      } else {
        im2col_nhwc_kernel<<<blocks_col, t>>>(d_x, d_col, n, h, w, c,
                                              sh.ho, sh.wo,
                                              r, s,
                                              p.pad_h, p.pad_w,
                                              p.stride_h, p.stride_w,
                                              p.dilation_h, p.dilation_w,
                                              cin_base, sh.cin_group);
      }
    }

    int total_dy = m * ncol;
    int blocks_dy = (total_dy + t - 1) / t;
    pack_nhwc_group_matrix_kernel<<<blocks_dy, t>>>(d_dy, d_dy_mat, n, sh.ho, sh.wo, k, kout_base, sh.kout_group);

    launch_gemm(d_col, d_dy_mat, d_dwg, kdim, ncol, m, true, false, 1.0f, 0.0f);

    int total_wg = kdim * ncol;
    int blocks_wg = (total_wg + t - 1) / t;
    unpack_filter_hwcn_group_kernel<<<blocks_wg, t>>>(d_dwg, d_dw, r, s, sh.cin_group, k, kout_base, sh.kout_group);
  }

  CUDA_CHECK(cudaGetLastError());
}
