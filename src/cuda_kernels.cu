#include "conv_types.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>

namespace {

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

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

__global__ void pack_filter_hwio_group_kernel(const float* __restrict__ w,
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

  wg[idx] = w[idx_hwio(rr, ss, ci, k_base + col, s, cin_group, k_total)];
}

__global__ void unpack_filter_hwio_group_kernel(const float* __restrict__ wg,
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

  w[idx_hwio(rr, ss, ci, k_base + col, s, cin_group, k_total)] = wg[idx];
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

template <bool TA, bool TB>
__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K,
                                  float alpha,
                                  float beta) {
  __shared__ float As[TILE_M][TILE_K + 1];
  __shared__ float Bs[TILE_K][TILE_N + 1];

  const int row = blockIdx.y * TILE_M + threadIdx.y;
  const int col = blockIdx.x * TILE_N + threadIdx.x;

  float acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += TILE_K) {
    const int a_k = k0 + threadIdx.x;
    const int b_k = k0 + threadIdx.y;

    float av = 0.0f;
    if (row < M && a_k < K) {
      av = TA ? A[a_k * M + row] : A[row * K + a_k];
    }
    As[threadIdx.y][threadIdx.x] = av;

    float bv = 0.0f;
    if (col < N && b_k < K) {
      bv = TB ? B[col * K + b_k] : B[b_k * N + col];
    }
    Bs[threadIdx.y][threadIdx.x] = bv;

    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < TILE_K; ++kk) {
      acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    const int out_idx = row * N + col;
    C[out_idx] = alpha * acc + beta * C[out_idx];
  }
}

void launch_gemm(const float* A, const float* B, float* C, int M, int N, int K, bool transA, bool transB, float alpha = 1.0f, float beta = 0.0f) {
  dim3 block(TILE_N, TILE_M);
  dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

  if (!transA && !transB) {
    gemm_tiled_kernel<false, false><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
  } else if (transA && !transB) {
    gemm_tiled_kernel<true, false><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
  } else if (!transA && transB) {
    gemm_tiled_kernel<false, true><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
  } else {
    gemm_tiled_kernel<true, true><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
  }
}

}  // namespace

void launch_fprop_nhwc(const float* d_x, const float* d_w, float* d_y,
                       int n, int h, int w, int c, int r, int s, int k,
                       const Conv2DParams& p) {
  TensorNHWC x_shape(n, h, w, c);
  FilterHWIO w_shape(r, s, c / p.groups, k);
  ConvShape sh = infer_conv_shape(x_shape, w_shape, p);

  const int m = n * sh.ho * sh.wo;
  const int kdim = r * s * sh.cin_group;
  const int ncol = sh.kout_group;

  float* d_col = nullptr;
  float* d_wg = nullptr;
  float* d_ymat = nullptr;

  CUDA_CHECK(cudaMalloc(&d_col, static_cast<size_t>(m) * kdim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wg, static_cast<size_t>(kdim) * ncol * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ymat, static_cast<size_t>(m) * ncol * sizeof(float)));

  const int t = 256;

  for (int g = 0; g < p.groups; ++g) {
    const int cin_base = g * sh.cin_group;
    const int kout_base = g * sh.kout_group;

    int total_col = m * kdim;
    int blocks_col = (total_col + t - 1) / t;
    im2col_nhwc_kernel<<<blocks_col, t>>>(d_x, d_col, n, h, w, c,
                                          sh.ho, sh.wo,
                                          r, s,
                                          p.pad_h, p.pad_w,
                                          p.stride_h, p.stride_w,
                                          p.dilation_h, p.dilation_w,
                                          cin_base, sh.cin_group);

    int total_wg = kdim * ncol;
    int blocks_wg = (total_wg + t - 1) / t;
    pack_filter_hwio_group_kernel<<<blocks_wg, t>>>(d_w, d_wg, r, s, sh.cin_group, k, kout_base, sh.kout_group);

    launch_gemm(d_col, d_wg, d_ymat, m, ncol, kdim, false, false, 1.0f, 0.0f);

    int total_ym = m * ncol;
    int blocks_ym = (total_ym + t - 1) / t;
    unpack_matrix_to_nhwc_group_kernel<<<blocks_ym, t>>>(d_ymat, d_y, n, sh.ho, sh.wo, k, kout_base, sh.kout_group);
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFree(d_ymat));
  CUDA_CHECK(cudaFree(d_wg));
  CUDA_CHECK(cudaFree(d_col));
}

void launch_bprop_nhwc(const float* d_dy, const float* d_w, float* d_dx,
                       int n, int h, int w, int c, int r, int s, int k,
                       const Conv2DParams& p) {
  TensorNHWC x_shape(n, h, w, c);
  FilterHWIO w_shape(r, s, c / p.groups, k);
  ConvShape sh = infer_conv_shape(x_shape, w_shape, p);

  const int m = n * sh.ho * sh.wo;
  const int kdim = r * s * sh.cin_group;
  const int ncol = sh.kout_group;

  float* d_dy_mat = nullptr;
  float* d_wg = nullptr;
  float* d_dcol = nullptr;

  CUDA_CHECK(cudaMemset(d_dx, 0, static_cast<size_t>(n) * h * w * c * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&d_dy_mat, static_cast<size_t>(m) * ncol * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wg, static_cast<size_t>(kdim) * ncol * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dcol, static_cast<size_t>(m) * kdim * sizeof(float)));

  const int t = 256;

  for (int g = 0; g < p.groups; ++g) {
    const int cin_base = g * sh.cin_group;
    const int kout_base = g * sh.kout_group;

    int total_dy = m * ncol;
    int blocks_dy = (total_dy + t - 1) / t;
    pack_nhwc_group_matrix_kernel<<<blocks_dy, t>>>(d_dy, d_dy_mat, n, sh.ho, sh.wo, k, kout_base, sh.kout_group);

    int total_wg = kdim * ncol;
    int blocks_wg = (total_wg + t - 1) / t;
    pack_filter_hwio_group_kernel<<<blocks_wg, t>>>(d_w, d_wg, r, s, sh.cin_group, k, kout_base, sh.kout_group);

    launch_gemm(d_dy_mat, d_wg, d_dcol, m, kdim, ncol, false, true, 1.0f, 0.0f);

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
  CUDA_CHECK(cudaFree(d_dcol));
  CUDA_CHECK(cudaFree(d_wg));
  CUDA_CHECK(cudaFree(d_dy_mat));
}

void launch_grad_nhwc(const float* d_x, const float* d_dy, float* d_dw,
                      int n, int h, int w, int c, int r, int s, int k,
                      const Conv2DParams& p) {
  TensorNHWC x_shape(n, h, w, c);
  FilterHWIO w_shape(r, s, c / p.groups, k);
  ConvShape sh = infer_conv_shape(x_shape, w_shape, p);

  const int m = n * sh.ho * sh.wo;
  const int kdim = r * s * sh.cin_group;
  const int ncol = sh.kout_group;

  float* d_col = nullptr;
  float* d_dy_mat = nullptr;
  float* d_dwg = nullptr;

  CUDA_CHECK(cudaMemset(d_dw, 0, static_cast<size_t>(r) * s * sh.cin_group * k * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&d_col, static_cast<size_t>(m) * kdim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dy_mat, static_cast<size_t>(m) * ncol * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dwg, static_cast<size_t>(kdim) * ncol * sizeof(float)));

  const int t = 256;

  for (int g = 0; g < p.groups; ++g) {
    const int cin_base = g * sh.cin_group;
    const int kout_base = g * sh.kout_group;

    int total_col = m * kdim;
    int blocks_col = (total_col + t - 1) / t;
    im2col_nhwc_kernel<<<blocks_col, t>>>(d_x, d_col, n, h, w, c,
                                          sh.ho, sh.wo,
                                          r, s,
                                          p.pad_h, p.pad_w,
                                          p.stride_h, p.stride_w,
                                          p.dilation_h, p.dilation_w,
                                          cin_base, sh.cin_group);

    int total_dy = m * ncol;
    int blocks_dy = (total_dy + t - 1) / t;
    pack_nhwc_group_matrix_kernel<<<blocks_dy, t>>>(d_dy, d_dy_mat, n, sh.ho, sh.wo, k, kout_base, sh.kout_group);

    launch_gemm(d_col, d_dy_mat, d_dwg, kdim, ncol, m, true, false, 1.0f, 0.0f);

    int total_wg = kdim * ncol;
    int blocks_wg = (total_wg + t - 1) / t;
    unpack_filter_hwio_group_kernel<<<blocks_wg, t>>>(d_dwg, d_dw, r, s, sh.cin_group, k, kout_base, sh.kout_group);
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFree(d_dwg));
  CUDA_CHECK(cudaFree(d_dy_mat));
  CUDA_CHECK(cudaFree(d_col));
}
