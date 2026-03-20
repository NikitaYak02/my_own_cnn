#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Layer Normalization — NHWC layout, normalization over C.
 *
 * Sizes:
 *   x, y          — [N, H, W, C]   (N*H*W*C floats)
 *   gamma, beta   — [C]
 *   mean, inv_std — [N*H*W]
 *   dy, dx        — [N, H, W, C]
 *   dgamma, dbeta — [C]
 *
 * All pointers must reside in device memory.
 */

void layernorm_fprop(
    const float* x, const float* gamma, const float* beta,
    float* y, float* mean, float* inv_std,
    int C, int H, int W, int N, float eps);

void layernorm_bprop(
    const float* x, const float* dy,
    const float* gamma,
    const float* mean, const float* inv_std,
    float* dx, float* dgamma, float* dbeta,
    int C, int H, int W, int N);

#ifdef __cplusplus
}
#endif
