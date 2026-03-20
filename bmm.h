#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Batched Matrix Multiplication layer.
 *
 * Forward:   Y[N,W,H] = A[N,W,C] @ B[N,H,C]^T
 * Backward:  dA[N,W,C] = dY[N,W,H] @ B[N,H,C]
 *            dB[N,H,C] = dY[N,W,H]^T @ A[N,W,C]
 *
 * All pointers must reside in device memory.
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
}
#endif
