#pragma once

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
    int trans_a, int trans_b);

#ifdef __cplusplus
}
#endif
