#include "bmm.h"

#include <cstdint>
#include <cublasLt.h>
#include <cublas_v2.h>
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

constexpr int kCublasTransposeAPathMinK = 512;
constexpr size_t kCublasLtWorkspaceBytes = 64ull * 1024ull * 1024ull;

struct CublasContext {
    cublasHandle_t handle = nullptr;
    cublasLtHandle_t lt_handle = nullptr;
    void* lt_workspace = nullptr;
    size_t lt_workspace_cap = 0;
    bool init = false;

    ~CublasContext() {
        if (lt_workspace) cudaFree(lt_workspace);
        if (lt_handle) cublasLtDestroy(lt_handle);
        if (handle) cublasDestroy(handle);
    }
};

static CublasContext& get_cublas_context() {
    static CublasContext ctx;
    if (!ctx.init) {
        if (cublasCreate(&ctx.handle) != CUBLAS_STATUS_SUCCESS) {
            return ctx;
        }
        cublasSetMathMode(ctx.handle, CUBLAS_TF32_TENSOR_OP_MATH);
        if (cublasLtCreate(&ctx.lt_handle) != CUBLAS_STATUS_SUCCESS) {
            ctx.lt_handle = nullptr;
        }
        ctx.init = true;
    }
    return ctx;
}

static bool ensure_cublaslt_workspace(CublasContext& ctx, size_t bytes) {
    if (ctx.lt_workspace_cap >= bytes) return true;
    if (ctx.lt_workspace) {
        cudaFree(ctx.lt_workspace);
        ctx.lt_workspace = nullptr;
        ctx.lt_workspace_cap = 0;
    }
    if (cudaMalloc(&ctx.lt_workspace, bytes) != cudaSuccess) {
        ctx.lt_workspace = nullptr;
        ctx.lt_workspace_cap = 0;
        return false;
    }
    ctx.lt_workspace_cap = bytes;
    return true;
}

static bool launch_cublaslt_transpose_a_path(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    bool accumulate)
{
    CublasContext& ctx = get_cublas_context();
    if (ctx.lt_handle == nullptr) {
        return false;
    }

    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    int64_t batch_count = batch;
    int64_t stride_a = static_cast<int64_t>(K) * M;
    int64_t stride_b = static_cast<int64_t>(K) * N;
    int64_t stride_c = static_cast<int64_t>(M) * N;
    size_t workspace_bytes = kCublasLtWorkspaceBytes;
    int found = 0;
    bool ok = false;

    if (cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) goto cleanup;
    if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)) != CUBLAS_STATUS_SUCCESS) goto cleanup;
    if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)) != CUBLAS_STATUS_SUCCESS) goto cleanup;

    if (cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_32F, K, M, M) != CUBLAS_STATUS_SUCCESS) goto cleanup;
    if (cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_32F, K, N, N) != CUBLAS_STATUS_SUCCESS) goto cleanup;
    if (cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, M, N, N) != CUBLAS_STATUS_SUCCESS) goto cleanup;

    if (cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)) != CUBLAS_STATUS_SUCCESS) goto cleanup;
    if (cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)) != CUBLAS_STATUS_SUCCESS) goto cleanup;
    if (cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)) != CUBLAS_STATUS_SUCCESS) goto cleanup;

    if (batch > 1) {
        if (cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)) != CUBLAS_STATUS_SUCCESS) goto cleanup;
        if (cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)) != CUBLAS_STATUS_SUCCESS) goto cleanup;
        if (cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)) != CUBLAS_STATUS_SUCCESS) goto cleanup;
        if (cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)) != CUBLAS_STATUS_SUCCESS) goto cleanup;
        if (cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)) != CUBLAS_STATUS_SUCCESS) goto cleanup;
        if (cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)) != CUBLAS_STATUS_SUCCESS) goto cleanup;
    }

    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) goto cleanup;
    if (cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &workspace_bytes, sizeof(workspace_bytes)) != CUBLAS_STATUS_SUCCESS) goto cleanup;

    if (!ensure_cublaslt_workspace(ctx, workspace_bytes)) goto cleanup;

    cublasLtMatmulHeuristicResult_t heuristics[8];
    if (cublasLtMatmulAlgoGetHeuristic(ctx.lt_handle, op_desc, a_desc, b_desc, c_desc, c_desc,
                                       pref, 8, heuristics, &found) != CUBLAS_STATUS_SUCCESS) goto cleanup;
    for (int i = 0; i < found; ++i) {
        if (heuristics[i].state != CUBLAS_STATUS_SUCCESS) continue;
        const cublasStatus_t st = cublasLtMatmul(ctx.lt_handle,
                                                 op_desc,
                                                 &alpha,
                                                 A, a_desc,
                                                 B, b_desc,
                                                 &beta,
                                                 C, c_desc,
                                                 C, c_desc,
                                                 &heuristics[i].algo,
                                                 ctx.lt_workspace,
                                                 ctx.lt_workspace_cap,
                                                 nullptr);
        if (st == CUBLAS_STATUS_SUCCESS) {
            ok = true;
            break;
        }
    }

cleanup:
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
    if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
    if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
    if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    return ok;
}

static bool launch_cublas_transpose_a_path(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    bool accumulate)
{
    CublasContext& ctx = get_cublas_context();
    if (ctx.handle == nullptr) {
        return false;
    }

    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
    const cublasStatus_t st = cublasSgemmStridedBatched(
        ctx.handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        B, N, static_cast<long long>(K) * N,
        A, M, static_cast<long long>(K) * M,
        &beta,
        C, N, static_cast<long long>(M) * N,
        batch);

    return st == CUBLAS_STATUS_SUCCESS;
}

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

template <bool TRANS_A, bool TRANS_B, bool ACCUM>
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
                if constexpr (ACCUM) {
                    Cn[out_row * N + out_col] += acc[i][j];
                } else {
                    Cn[out_row * N + out_col] = acc[i][j];
                }
            }
        }
    }
}

template <bool TRANS_A, bool TRANS_B, bool ACCUM>
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

    bmm_tiled_kernel<TRANS_A, TRANS_B, ACCUM><<<grid, block>>>(A, B, C, M, N, K);
}

template <bool TRANS_A, bool TRANS_B>
__global__ void bmm_tiled_i32_kernel(
    const int32_t* __restrict__ A,
    const int32_t* __restrict__ B,
    int32_t* __restrict__ C,
    int M, int N, int K)
{
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

    const int32_t* An = A + static_cast<size_t>(batch_idx) * M * K;
    const int32_t* Bn = B + static_cast<size_t>(batch_idx) * K * N;
    int32_t* Cn = C + static_cast<size_t>(batch_idx) * M * N;

    __shared__ int32_t As[kBlockRows][kBlockDepth + kSharedPad];
    __shared__ int32_t Bs[kBlockDepth][kBlockCols + kSharedPad];

    int32_t acc[kThreadRows][kThreadCols] = {{0, 0}, {0, 0}};

    for (int k0 = 0; k0 < K; k0 += kBlockDepth) {
        if constexpr (TRANS_A) {
            const int load_k = tid / kBlockRows;
            const int load_m = tid % kBlockRows;
            const int g_m = tile_row + load_m;
            const int g_k = k0 + load_k;
            As[load_m][load_k] =
                (g_m < M && g_k < K) ? An[g_k * M + g_m] : 0;
        } else {
            const int load_m = tid / kBlockDepth;
            const int load_k = tid % kBlockDepth;
            const int g_m = tile_row + load_m;
            const int g_k = k0 + load_k;
            As[load_m][load_k] =
                (g_m < M && g_k < K) ? An[g_m * K + g_k] : 0;
        }

        if constexpr (TRANS_B) {
            const int load_n = tid / kBlockDepth;
            const int load_k = tid % kBlockDepth;
            const int g_n = tile_col + load_n;
            const int g_k = k0 + load_k;
            Bs[load_k][load_n] =
                (g_n < N && g_k < K) ? Bn[g_n * K + g_k] : 0;
        } else {
            const int load_k = tid / kBlockCols;
            const int load_n = tid % kBlockCols;
            const int g_n = tile_col + load_n;
            const int g_k = k0 + load_k;
            Bs[load_k][load_n] =
                (g_n < N && g_k < K) ? Bn[g_k * N + g_n] : 0;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < kBlockDepth; ++kk) {
            const int32_t a0 = As[local_row + 0][kk];
            const int32_t a1 = As[local_row + 1][kk];
            const int32_t b0 = Bs[kk][local_col + 0];
            const int32_t b1 = Bs[kk][local_col + 1];

            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
        }

        __syncthreads();
    }

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
static void launch_bmm_i32_kernel(
    const int32_t* A,
    const int32_t* B,
    int32_t* C,
    int batch,
    int M,
    int N,
    int K)
{
    dim3 block(kBlockCols / kThreadCols, kBlockRows / kThreadRows);
    dim3 grid((N + kBlockCols - 1) / kBlockCols,
              (M + kBlockRows - 1) / kBlockRows,
              batch);

    bmm_tiled_i32_kernel<TRANS_A, TRANS_B><<<grid, block>>>(A, B, C, M, N, K);
}

/* ================================================================== */
/*                          HOST  WRAPPERS                            */
/* ================================================================== */

static void bmm_matmul_impl(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    int trans_a, int trans_b,
    bool accumulate)
{
    // Dispatch once on transpose flags so the inner kernel can stay branch-free
    // with respect to data layout.
    const bool transpose_a = trans_a != 0;
    const bool transpose_b = trans_b != 0;

    // Prefer cuBLASLt for transpose-A weight-gradient-like shapes. It handles
    // small-output / large-reduction GEMMs much better on modern GPUs and may
    // internally choose split-K style algorithms.
    if (transpose_a && !transpose_b &&
        K >= kCublasTransposeAPathMinK &&
        launch_cublaslt_transpose_a_path(A, B, C, batch, M, N, K, accumulate)) {
        return;
    }

    if (transpose_a && !transpose_b &&
        K >= kCublasTransposeAPathMinK &&
        launch_cublas_transpose_a_path(A, B, C, batch, M, N, K, accumulate)) {
        return;
    }

    if (!transpose_a && !transpose_b) {
        if (accumulate) launch_bmm_kernel<false, false, true>(A, B, C, batch, M, N, K);
        else launch_bmm_kernel<false, false, false>(A, B, C, batch, M, N, K);
        return;
    }

    if (!transpose_a && transpose_b) {
        if (accumulate) launch_bmm_kernel<false, true, true>(A, B, C, batch, M, N, K);
        else launch_bmm_kernel<false, true, false>(A, B, C, batch, M, N, K);
        return;
    }

    if (transpose_a && !transpose_b) {
        if (accumulate) launch_bmm_kernel<true, false, true>(A, B, C, batch, M, N, K);
        else launch_bmm_kernel<true, false, false>(A, B, C, batch, M, N, K);
        return;
    }

    if (accumulate) launch_bmm_kernel<true, true, true>(A, B, C, batch, M, N, K);
    else launch_bmm_kernel<true, true, false>(A, B, C, batch, M, N, K);
}

extern "C" void bmm_matmul(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    int trans_a, int trans_b)
{
    bmm_matmul_impl(A, B, C, batch, M, N, K, trans_a, trans_b, false);
}

extern "C" void bmm_matmul_accum(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    int trans_a, int trans_b)
{
    bmm_matmul_impl(A, B, C, batch, M, N, K, trans_a, trans_b, true);
}

extern "C" void bmm_matmul_i32(
    const int32_t* A, const int32_t* B, int32_t* C,
    int batch, int M, int N, int K,
    int trans_a, int trans_b)
{
    const bool transpose_a = trans_a != 0;
    const bool transpose_b = trans_b != 0;

    if (!transpose_a && !transpose_b) {
        launch_bmm_i32_kernel<false, false>(A, B, C, batch, M, N, K);
        return;
    }

    if (!transpose_a && transpose_b) {
        launch_bmm_i32_kernel<false, true>(A, B, C, batch, M, N, K);
        return;
    }

    if (transpose_a && !transpose_b) {
        launch_bmm_i32_kernel<true, false>(A, B, C, batch, M, N, K);
        return;
    }

    launch_bmm_i32_kernel<true, true>(A, B, C, batch, M, N, K);
}

extern "C" void bmm_fprop(
    const float* A, const float* B, float* Y,
    int N, int W, int H, int C)
{
    bmm_matmul(
        A, B, Y,
        N, W, H, C,
        BMM_TRANSPOSE_NONE, BMM_TRANSPOSE_YES);
}

extern "C" void bmm_bprop_dA(
    const float* dY, const float* B, float* dA,
    int N, int W, int H, int C)
{
    bmm_matmul(
        dY, B, dA,
        N, W, C, H,
        BMM_TRANSPOSE_NONE, BMM_TRANSPOSE_NONE);
}

extern "C" void bmm_bprop_dB(
    const float* dY, const float* A, float* dB,
    int N, int W, int H, int C)
{
    bmm_matmul(
        dY, A, dB,
        N, H, C, W,
        BMM_TRANSPOSE_YES, BMM_TRANSPOSE_NONE);
}

extern "C" void bmm_bprop(
    const float* A, const float* B, const float* dY,
    float* dA, float* dB,
    int N, int W, int H, int C)
{
    bmm_bprop_dA(dY, B, dA, N, W, H, C);
    bmm_bprop_dB(dY, A, dB, N, W, H, C);
}
