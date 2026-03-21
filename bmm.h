#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Row-major batched matrix multiplication.
 *
 * Computes, for each batch item:
 *   C[M, N] = op(A)[M, K] @ op(B)[K, N]
 *
 * Layout conventions:
 *   trans_a == 0: A is stored as [M, K]
 *   trans_a != 0: A is stored as [K, M] and read as A^T
 *   trans_b == 0: B is stored as [K, N]
 *   trans_b != 0: B is stored as [N, K] and read as B^T
 *
 * All pointers must reside in device memory.
 */

enum BmmTranspose {
    BMM_TRANSPOSE_NONE = 0,
    BMM_TRANSPOSE_YES = 1,
};

void bmm_matmul(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    int trans_a, int trans_b);

#ifdef __cplusplus
}
#endif
