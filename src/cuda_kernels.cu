#include "conv_types.h"
#include "cuda_utils.h"
#include "bmm.h"

#include <cuda_runtime.h>

namespace {
struct Workspace {
  float* d_col = nullptr;
  size_t d_col_cap = 0;
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

  ~Workspace() {
    cudaFree(d_col);
    cudaFree(d_wg_all);
    cudaFree(d_ymat);
    cudaFree(d_dy_mat);
    cudaFree(d_dcol);
    cudaFree(d_dwg);
  }
};

void ensure_capacity(float** ptr, size_t* cap, size_t elements) {
  if (*cap >= elements) return;
  if (*ptr) CUDA_CHECK(cudaFree(*ptr));
  CUDA_CHECK(cudaMalloc(ptr, elements * sizeof(float)));
  *cap = elements;
}

__global__ void pack_filter_krsc_group_kernel(const float* __restrict__ w,
                                              float* __restrict__ wg,
                                              int r, int s, int cin_group,
                                              int k_base, int kout_group);

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
    pack_filter_krsc_group_kernel<<<blocks, t>>>(d_w, dst, r, s, cin_group, kout_base, static_cast<int>(ncol));
  }
  CUDA_CHECK(cudaGetLastError());
  ws.packed_w_src = d_w;
  ws.packed_r = r;
  ws.packed_s = s;
  ws.packed_cin_group = cin_group;
  ws.packed_k = k;
  ws.packed_groups = groups;
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
                                              int k_base, int kout_group) {
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

  wg[idx] = w[idx_krsc(k_base + ko, rr, ss, ci, r, s, cin_group)];
}

__global__ void unpack_filter_krsc_group_kernel(const float* __restrict__ wg,
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

  w[idx_krsc(k_base + col, rr, ss, ci, r, s, cin_group)] = wg[idx];
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

}  // namespace

void launch_fprop_nhwc(const float* d_x, const float* d_w, float* d_y,
                       int n, int h, int w, int c, int r, int s, int k,
                       const Conv2DParams& p) {
  TensorNHWC x_shape(n, h, w, c);
  FilterKRSC w_shape(r, s, c / p.groups, k);
  ConvShape sh = infer_conv_shape(x_shape, w_shape, p);

  const int m = n * sh.ho * sh.wo;
  const int kdim = r * s * sh.cin_group;
  const int ncol = sh.kout_group;
  const bool can_vec4 = (sh.cin_group % 4 == 0) && (sh.cin_group >= 4) && ((c % 4) == 0);

  Workspace& ws = workspace();
  ensure_capacity(&ws.d_col, &ws.d_col_cap, static_cast<size_t>(m) * kdim);
  ensure_capacity(&ws.d_ymat, &ws.d_ymat_cap, static_cast<size_t>(m) * ncol);
  ensure_packed_weights(ws, d_w, r, s, sh.cin_group, k, p.groups);
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

    float* d_wg = ws.d_wg_all + static_cast<size_t>(g) * ncol * kdim;
    // Forward GEMM: [m, kdim] @ [ncol, kdim]^T -> [m, ncol]
    bmm_matmul(d_col, d_wg, d_ymat, 1, m, ncol, kdim, BMM_TRANSPOSE_NONE, BMM_TRANSPOSE_YES);

    int total_ym = m * ncol;
    int blocks_ym = (total_ym + t - 1) / t;
    unpack_matrix_to_nhwc_group_kernel<<<blocks_ym, t>>>(d_ymat, d_y, n, sh.ho, sh.wo, k, kout_base, sh.kout_group);
  }

  CUDA_CHECK(cudaGetLastError());
}

void launch_bprop_nhwc(const float* d_dy, const float* d_w, float* d_dx,
                       int n, int h, int w, int c, int r, int s, int k,
                       const Conv2DParams& p) {
  TensorNHWC x_shape(n, h, w, c);
  FilterKRSC w_shape(r, s, c / p.groups, k);
  ConvShape sh = infer_conv_shape(x_shape, w_shape, p);

  const int m = n * sh.ho * sh.wo;
  const int kdim = r * s * sh.cin_group;
  const int ncol = sh.kout_group;

  Workspace& ws = workspace();
  ensure_capacity(&ws.d_dy_mat, &ws.d_dy_mat_cap, static_cast<size_t>(m) * ncol);
  ensure_capacity(&ws.d_dcol, &ws.d_dcol_cap, static_cast<size_t>(m) * kdim);
  ensure_packed_weights(ws, d_w, r, s, sh.cin_group, k, p.groups);
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

    float* d_wg = ws.d_wg_all + static_cast<size_t>(g) * ncol * kdim;
    // Input-gradient GEMM: [m, ncol] @ [ncol, kdim] -> [m, kdim]
    bmm_matmul(d_dy_mat, d_wg, d_dcol, 1, m, kdim, ncol, BMM_TRANSPOSE_NONE, BMM_TRANSPOSE_NONE);

    int total_dcol = m * kdim;
    int blocks_dcol = (total_dcol + t - 1) / t;
    col2im_accum_nhwc_kernel<<<blocks_dcol, t>>>(d_dcol, d_dx,
                                                 n, h, w, c,
                                                 sh.ho, sh.wo,
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
                      const Conv2DParams& p) {
  TensorNHWC x_shape(n, h, w, c);
  FilterKRSC w_shape(r, s, c / p.groups, k);
  ConvShape sh = infer_conv_shape(x_shape, w_shape, p);

  const int m = n * sh.ho * sh.wo;
  const int kdim = r * s * sh.cin_group;
  const int ncol = sh.kout_group;
  const bool can_vec4 = (sh.cin_group % 4 == 0) && (sh.cin_group >= 4) && (c % 4 == 0);

  Workspace& ws = workspace();
  ensure_capacity(&ws.d_col, &ws.d_col_cap, static_cast<size_t>(m) * kdim);
  ensure_capacity(&ws.d_dy_mat, &ws.d_dy_mat_cap, static_cast<size_t>(m) * ncol);
  ensure_capacity(&ws.d_dwg, &ws.d_dwg_cap, static_cast<size_t>(kdim) * ncol);
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

    int total_dy = m * ncol;
    int blocks_dy = (total_dy + t - 1) / t;
    pack_nhwc_group_matrix_kernel<<<blocks_dy, t>>>(d_dy, d_dy_mat, n, sh.ho, sh.wo, k, kout_base, sh.kout_group);

    // Weight-gradient GEMM: [m, kdim]^T @ [m, ncol] -> [kdim, ncol]
    bmm_matmul(d_col, d_dy_mat, d_dwg, 1, kdim, ncol, m, BMM_TRANSPOSE_YES, BMM_TRANSPOSE_NONE);

    int total_wg = kdim * ncol;
    int blocks_wg = (total_wg + t - 1) / t;
    unpack_filter_krsc_group_kernel<<<blocks_wg, t>>>(d_dwg, d_dw, r, s, sh.cin_group, k, kout_base, sh.kout_group);
  }

  CUDA_CHECK(cudaGetLastError());
}
