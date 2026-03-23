#include <cuda_runtime.h>
#include <stdio.h>

namespace {

// Tile geometry for the custom row-major GEMM kernel.
// A 16x16 thread block (256 threads) computes a 32x32 output tile, so each
// thread accumulates a 2x2 register tile.
constexpr int kBlockRows = 32;
constexpr int kBlockCols = 32;
constexpr int kBlockDepth = 8;
constexpr int kThreadRows = 2;
constexpr int kThreadCols = 2;
constexpr int kSharedPad = 1;

static_assert(kBlockRows % kThreadRows == 0, "row tile must divide block rows");
static_assert(kBlockCols % kThreadCols == 0, "col tile must divide block cols");

}  // namespace

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
    // blockIdx.z selects the current matrix pair within the batch.
    const int batch_idx = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int tile_row = blockIdx.y * kBlockRows;
    const int tile_col = blockIdx.x * kBlockCols;
    const int local_row = ty * kThreadRows;
    const int local_col = tx * kThreadCols;
    const int out_row0 = tile_row + local_row;
    const int out_col0 = tile_col + local_col;

    // Advance pointers to the current batch slice.
    const float* An = A + static_cast<size_t>(batch_idx) * M * K;
    const float* Bn = B + static_cast<size_t>(batch_idx) * K * N;
    float* Cn = C + static_cast<size_t>(batch_idx) * M * N;

    // Shared-memory tiles include a small pad to reduce bank conflicts.
    __shared__ float As[kBlockRows][kBlockDepth + kSharedPad];
    __shared__ float Bs[kBlockDepth][kBlockCols + kSharedPad];

    float acc[kThreadRows][kThreadCols] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    for (int k0 = 0; k0 < K; k0 += kBlockDepth) {
        // Cooperative tile load. Compile-time specialization keeps the
        // addressing formula simple inside the hot loop for each transpose
        // combination.
        if constexpr (TRANS_A) {
            const int load_k = tid / kBlockRows;
            const int load_m = tid % kBlockRows;
            const int g_m = tile_row + load_m;
            const int g_k = k0 + load_k;
            As[load_m][load_k] =
                (g_m < M && g_k < K) ? An[g_k * M + g_m] : 0.0f;
        } else {
            const int load_m = tid / kBlockDepth;
            const int load_k = tid % kBlockDepth;
            const int g_m = tile_row + load_m;
            const int g_k = k0 + load_k;
            As[load_m][load_k] =
                (g_m < M && g_k < K) ? An[g_m * K + g_k] : 0.0f;
        }

        if constexpr (TRANS_B) {
            const int load_n = tid / kBlockDepth;
            const int load_k = tid % kBlockDepth;
            const int g_n = tile_col + load_n;
            const int g_k = k0 + load_k;
            Bs[load_k][load_n] =
                (g_n < N && g_k < K) ? Bn[g_n * K + g_k] : 0.0f;
        } else {
            const int load_k = tid / kBlockCols;
            const int load_n = tid % kBlockCols;
            const int g_n = tile_col + load_n;
            const int g_k = k0 + load_k;
            Bs[load_k][load_n] =
                (g_n < N && g_k < K) ? Bn[g_k * N + g_n] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < kBlockDepth; ++kk) {
            const float a0 = As[local_row + 0][kk];
            const float a1 = As[local_row + 1][kk];
            const float b0 = Bs[kk][local_col + 0];
            const float b1 = Bs[kk][local_col + 1];

            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
        }

        __syncthreads();
    }

    // Store the thread-local 2x2 output fragment with boundary checks for
    // partially covered tiles on matrix edges.
    for (int i = 0; i < kThreadRows; ++i) {
        const int out_row = out_row0 + i;
        if (out_row >= M) {
            continue;
        }
        for (int j = 0; j < kThreadCols; ++j) {
            const int out_col = out_col0 + j;
            if (out_col < N) {
                Cn[out_row * N + out_col] = acc[i][j];
            }
        }
    }
}

template <bool TRANS_A, bool TRANS_B>
static void launch_bmm_kernel(
    const float* A,
    const float* B,
    float* C,
    int batch,
    int M,
    int N,
    int K)
{
    dim3 block(kBlockCols / kThreadCols, kBlockRows / kThreadRows);
    dim3 grid((N + kBlockCols - 1) / kBlockCols,
              (M + kBlockRows - 1) / kBlockRows,
              batch);

    bmm_tiled_kernel<TRANS_A, TRANS_B><<<grid, block>>>(A, B, C, M, N, K);
}

/* ================================================================== */
/*                          HOST  WRAPPERS                            */
/* ================================================================== */

extern "C" void bmm_matmul(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    int trans_a, int trans_b)
{
    // Dispatch once on transpose flags so the inner kernel can stay branch-free
    // with respect to data layout.
    const bool transpose_a = trans_a != 0;
    const bool transpose_b = trans_b != 0;

    if (!transpose_a && !transpose_b) {
        launch_bmm_kernel<false, false>(A, B, C, batch, M, N, K);
        return;
    }

    if (!transpose_a && transpose_b) {
        launch_bmm_kernel<false, true>(A, B, C, batch, M, N, K);
        return;
    }

    if (transpose_a && !transpose_b) {
        launch_bmm_kernel<true, false>(A, B, C, batch, M, N, K);
        return;
    }

    launch_bmm_kernel<true, true>(A, B, C, batch, M, N, K);
}
