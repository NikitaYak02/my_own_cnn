#include <cuda_runtime.h>
#include <stdio.h>

#define BMM_BLOCK_M 32
#define BMM_BLOCK_N 32
#define BMM_BLOCK_K 8
#define BMM_THREAD_TILE_M 2
#define BMM_THREAD_TILE_N 2
#define BMM_PAD 1

/* ================================================================== */
/*                  REGISTER-TILED BATCHED MATMUL                     */
/* ================================================================== */
/*                                                                    */
/*  One block computes a 32x32 output tile using 16x16 = 256 threads. */
/*  Each thread accumulates a 2x2 output fragment in registers.       */
/*                                                                    */
/*  The loader is specialized at compile time for the four transpose  */
/*  combinations so the three hot paths used by the project keep      */
/*  coalesced reads for the transposed operand as well.               */
/* ------------------------------------------------------------------ */

template <bool TRANS_A, bool TRANS_B>
__global__ void bmm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int batch_idx = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int block_row = blockIdx.y * BMM_BLOCK_M;
    const int block_col = blockIdx.x * BMM_BLOCK_N;
    const int row0 = block_row + ty * BMM_THREAD_TILE_M;
    const int col0 = block_col + tx * BMM_THREAD_TILE_N;

    const float* An = A + static_cast<size_t>(batch_idx) * M * K;
    const float* Bn = B + static_cast<size_t>(batch_idx) * K * N;
    float* Cn = C + static_cast<size_t>(batch_idx) * M * N;

    __shared__ float As[BMM_BLOCK_M][BMM_BLOCK_K + BMM_PAD];
    __shared__ float Bs[BMM_BLOCK_K][BMM_BLOCK_N + BMM_PAD];

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BMM_BLOCK_K) {
        if constexpr (TRANS_A) {
            const int load_k = tid / BMM_BLOCK_M;
            const int load_m = tid % BMM_BLOCK_M;
            const int g_m = block_row + load_m;
            const int g_k = k0 + load_k;
            As[load_m][load_k] =
                (g_m < M && g_k < K) ? An[g_k * M + g_m] : 0.0f;
        } else {
            const int load_m = tid / BMM_BLOCK_K;
            const int load_k = tid % BMM_BLOCK_K;
            const int g_m = block_row + load_m;
            const int g_k = k0 + load_k;
            As[load_m][load_k] =
                (g_m < M && g_k < K) ? An[g_m * K + g_k] : 0.0f;
        }

        if constexpr (TRANS_B) {
            const int load_n = tid / BMM_BLOCK_K;
            const int load_k = tid % BMM_BLOCK_K;
            const int g_n = block_col + load_n;
            const int g_k = k0 + load_k;
            Bs[load_k][load_n] =
                (g_n < N && g_k < K) ? Bn[g_n * K + g_k] : 0.0f;
        } else {
            const int load_k = tid / BMM_BLOCK_N;
            const int load_n = tid % BMM_BLOCK_N;
            const int g_n = block_col + load_n;
            const int g_k = k0 + load_k;
            Bs[load_k][load_n] =
                (g_n < N && g_k < K) ? Bn[g_k * N + g_n] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BMM_BLOCK_K; ++kk) {
            const float a0 = As[ty * BMM_THREAD_TILE_M + 0][kk];
            const float a1 = As[ty * BMM_THREAD_TILE_M + 1][kk];
            const float b0 = Bs[kk][tx * BMM_THREAD_TILE_N + 0];
            const float b1 = Bs[kk][tx * BMM_THREAD_TILE_N + 1];

            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        __syncthreads();
    }

    if (row0 + 0 < M && col0 + 0 < N) {
        Cn[(row0 + 0) * N + (col0 + 0)] = acc00;
    }
    if (row0 + 0 < M && col0 + 1 < N) {
        Cn[(row0 + 0) * N + (col0 + 1)] = acc01;
    }
    if (row0 + 1 < M && col0 + 0 < N) {
        Cn[(row0 + 1) * N + (col0 + 0)] = acc10;
    }
    if (row0 + 1 < M && col0 + 1 < N) {
        Cn[(row0 + 1) * N + (col0 + 1)] = acc11;
    }
}

/* ================================================================== */
/*                          HOST  WRAPPERS                            */
/* ================================================================== */

extern "C" void bmm_matmul(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    int trans_a, int trans_b)
{
    dim3 block(BMM_BLOCK_N / BMM_THREAD_TILE_N,
               BMM_BLOCK_M / BMM_THREAD_TILE_M);
    dim3 grid((N + BMM_BLOCK_N - 1) / BMM_BLOCK_N,
              (M + BMM_BLOCK_M - 1) / BMM_BLOCK_M,
              batch);

    if (!trans_a && !trans_b) {
        bmm_tiled_kernel<false, false><<<grid, block>>>(A, B, C, M, N, K);
        return;
    }

    if (!trans_a && trans_b) {
        bmm_tiled_kernel<false, true><<<grid, block>>>(A, B, C, M, N, K);
        return;
    }

    if (trans_a && !trans_b) {
        bmm_tiled_kernel<true, false><<<grid, block>>>(A, B, C, M, N, K);
        return;
    }

    bmm_tiled_kernel<true, true><<<grid, block>>>(A, B, C, M, N, K);
}
