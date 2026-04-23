#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Row-major batched matrix multiplication used by both the standalone BMM
 * benchmarks/tests and the convolution code paths.
 *
 * For each batch item b the function computes:
 *   C_b[M, N] = op(A_b)[M, K] @ op(B_b)[K, N]
 *
 * Memory layout conventions:
 *   trans_a == BMM_TRANSPOSE_NONE: A is stored as [batch, M, K]
 *   trans_a == BMM_TRANSPOSE_YES:  A is stored as [batch, K, M] and read as A^T
 *   trans_b == BMM_TRANSPOSE_NONE: B is stored as [batch, K, N]
 *   trans_b == BMM_TRANSPOSE_YES:  B is stored as [batch, N, K] and read as B^T
 *
 * Logical dimensions:
 *   batch - number of matrices in the batch
 *   M     - number of output rows
 *   N     - number of output columns
 *   K     - reduction dimension shared by A and B
 *
 * All pointers must point to device memory containing float32 tensors.
 */

enum BmmTranspose {
    BMM_TRANSPOSE_NONE = 0,
    BMM_TRANSPOSE_YES = 1,
};

void bmm_matmul(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    int trans_a, int trans_b,
    cudaStream_t stream = nullptr);

void bmm_matmul_accum(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    int trans_a, int trans_b,
    cudaStream_t stream = nullptr);

/*
 * Batched Matrix Multiplication layer compatibility API.
 *
 * Forward:   Y[N,W,H] = A[N,W,C] @ B[N,H,C]^T
 * Backward:  dA[N,W,C] = dY[N,W,H] @ B[N,H,C]
 *            dB[N,H,C] = dY[N,W,H]^T @ A[N,W,C]
 *
 * These entry points intentionally reuse the single `bmm_matmul` kernel
 * dispatch above instead of owning separate CUDA kernels.
 */

void bmm_fprop(
    const float* A, const float* B, float* Y,
    int N, int W, int H, int C);

void bmm_bprop_dA(
    const float* dY, const float* B, float* dA,
    int N, int W, int H, int C);

void bmm_bprop_dB(
    const float* dY, const float* A, float* dB,
    int N, int W, int H, int C);

void bmm_bprop(
    const float* A, const float* B, const float* dY,
    float* dA, float* dB,
    int N, int W, int H, int C);

#ifdef __cplusplus
void bmm_matmul_i32(
    const int32_t* A, const int32_t* B, int32_t* C,
    int batch, int M, int N, int K,
    int trans_a, int trans_b);
#endif

#ifdef __cplusplus
}
#endif
