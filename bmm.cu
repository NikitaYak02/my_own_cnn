#include "bmm.h"

#include <cstdint>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

namespace {

// Tile geometry for the custom row-major GEMM kernel.
// By default a 16x16 thread block (256 threads) computes a 32x32 output tile,
// so each thread accumulates a 2x2 register tile. The per-thread register tile
// is configurable at runtime through BmmTileConfig.
constexpr int kDefaultBlockRows = 32;
constexpr int kDefaultBlockCols = 32;
constexpr int kDefaultBlockDepth = 8;
constexpr int kDefaultThreadRows = 2;
constexpr int kDefaultThreadCols = 2;
constexpr int kMaxThreadRows = 8;
constexpr int kMaxThreadCols = 8;
constexpr int kSharedPad = 1;
constexpr size_t kMaxTileSharedBytes = 48ull * 1024ull;

static_assert(kDefaultBlockRows % kDefaultThreadRows == 0, "row tile must divide block rows");
static_assert(kDefaultBlockCols % kDefaultThreadCols == 0, "col tile must divide block cols");

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
    bool accumulate,
    cudaStream_t stream)
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
                                                 stream);
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
    bool accumulate,
    cudaStream_t stream)
{
    CublasContext& ctx = get_cublas_context();
    if (ctx.handle == nullptr) {
        return false;
    }

    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
    cublasSetStream(ctx.handle, stream);
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

static bool launch_cublas_transpose_b_path(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    bool accumulate,
    cudaStream_t stream)
{
    CublasContext& ctx = get_cublas_context();
    if (ctx.handle == nullptr) {
        return false;
    }

    cublasMath_t prev_math_mode = CUBLAS_DEFAULT_MATH;
    if (cublasGetMathMode(ctx.handle, &prev_math_mode) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
    cublasSetMathMode(ctx.handle, CUBLAS_DEFAULT_MATH);
    cublasSetStream(ctx.handle, stream);

    cublasStatus_t st = CUBLAS_STATUS_NOT_INITIALIZED;
    if (batch == 1) {
        st = cublasSgemm(
            ctx.handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, K,
            A, K,
            &beta,
            C, N);
    } else {
        st = cublasSgemmStridedBatched(
            ctx.handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, K, static_cast<long long>(N) * K,
            A, K, static_cast<long long>(M) * K,
            &beta,
            C, N, static_cast<long long>(M) * N,
            batch);
    }

    cublasSetMathMode(ctx.handle, prev_math_mode);
    return st == CUBLAS_STATUS_SUCCESS;
}

static BmmTileConfig default_tile_config() {
    return BmmTileConfig{
        kDefaultBlockRows,
        kDefaultBlockCols,
        kDefaultBlockDepth,
        kDefaultThreadRows,
        kDefaultThreadCols,
    };
}

static BmmTileConfig normalize_tile_config(BmmTileConfig cfg) {
    if (cfg.thread_rows == 0) {
        cfg.thread_rows = kDefaultThreadRows;
    }
    if (cfg.thread_cols == 0) {
        cfg.thread_cols = kDefaultThreadCols;
    }
    return cfg;
}

static BmmTileConfig& get_tile_config() {
    static BmmTileConfig cfg = default_tile_config();
    return cfg;
}

static BmmTileLaunchInfo& get_last_launch_info_storage() {
    static BmmTileLaunchInfo info{
        {kDefaultBlockRows, kDefaultBlockCols, kDefaultBlockDepth, kDefaultThreadRows, kDefaultThreadCols},
        kDefaultBlockCols / kDefaultThreadCols,
        kDefaultBlockRows / kDefaultThreadRows,
        static_cast<int>(((static_cast<size_t>(kDefaultBlockRows) * (kDefaultBlockDepth + kSharedPad)) +
                          (static_cast<size_t>(kDefaultBlockDepth) * (kDefaultBlockCols + kSharedPad))) * sizeof(float)),
    };
    return info;
}

static size_t tile_shared_bytes(const BmmTileConfig& cfg) {
    return (static_cast<size_t>(cfg.block_rows) * (cfg.block_depth + kSharedPad) +
            static_cast<size_t>(cfg.block_depth) * (cfg.block_cols + kSharedPad)) * sizeof(float);
}

static bool validate_tile_config(const BmmTileConfig& cfg) {
    const BmmTileConfig norm = normalize_tile_config(cfg);
    if (norm.block_rows <= 0 || norm.block_cols <= 0 || norm.block_depth <= 0) {
        return false;
    }
    if (norm.thread_rows <= 0 || norm.thread_cols <= 0) {
        return false;
    }
    if (norm.thread_rows > kMaxThreadRows || norm.thread_cols > kMaxThreadCols) {
        return false;
    }
    if ((norm.block_rows % norm.thread_rows) != 0 || (norm.block_cols % norm.thread_cols) != 0) {
        return false;
    }

    const int threads_x = norm.block_cols / norm.thread_cols;
    const int threads_y = norm.block_rows / norm.thread_rows;
    if (threads_x <= 0 || threads_y <= 0) {
        return false;
    }
    if (static_cast<size_t>(threads_x) * threads_y > 1024u) {
        return false;
    }
    if (tile_shared_bytes(norm) > kMaxTileSharedBytes) {
        return false;
    }
    return true;
}

static void record_last_launch_info(const BmmTileConfig& cfg) {
    BmmTileLaunchInfo& info = get_last_launch_info_storage();
    info.tile = cfg;
    info.threads_x = cfg.block_cols / cfg.thread_cols;
    info.threads_y = cfg.block_rows / cfg.thread_rows;
    info.shared_mem_bytes = static_cast<int>(tile_shared_bytes(cfg));
}

}  // namespace

/* ================================================================== */
/*                  REGISTER-TILED BATCHED MATMUL                     */
/* ================================================================== */
/*                                                                    */
/*  One block computes a configurable output tile.                    */
/*  Each thread accumulates a configurable register fragment.         */
/*                                                                    */
/*  The loader is specialized at compile time for the four transpose  */
/*  combinations so the three hot paths used by the project keep      */
/*  coalesced reads for the transposed operand as well.               */
/* ------------------------------------------------------------------ */

template <typename T, bool TRANS_A, bool TRANS_B, bool ACCUM>
__global__ void bmm_tiled_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int batch, int M, int N, int K,
    int tiles_m, int tiles_n,
    size_t tile_block_offset,
    int block_rows,
    int block_cols,
    int block_depth,
    int thread_rows,
    int thread_cols)
{
    // Linearize output tiles so large M or batch values do not overflow
    // gridDim.y / gridDim.z, which are capped at 65535.
    const size_t linear_tile = tile_block_offset + static_cast<size_t>(blockIdx.x);
    const size_t tile_row_batch = linear_tile / static_cast<size_t>(tiles_n);
    const size_t batch_linear = tile_row_batch / static_cast<size_t>(tiles_m);
    if (batch_linear >= static_cast<size_t>(batch)) {
        return;
    }

    const int batch_idx = static_cast<int>(batch_linear);
    const int tile_row_idx = static_cast<int>(tile_row_batch - batch_linear * static_cast<size_t>(tiles_m));
    const int tile_col_idx = static_cast<int>(linear_tile - tile_row_batch * static_cast<size_t>(tiles_n));
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int tile_row = tile_row_idx * block_rows;
    const int tile_col = tile_col_idx * block_cols;
    const int local_row = ty * thread_rows;
    const int local_col = tx * thread_cols;
    const int out_row0 = tile_row + local_row;
    const int out_col0 = tile_col + local_col;

    // Advance pointers to the current batch slice.
    const T* An = A + static_cast<size_t>(batch_idx) * M * K;
    const T* Bn = B + static_cast<size_t>(batch_idx) * K * N;
    T* Cn = C + static_cast<size_t>(batch_idx) * M * N;

    // Shared-memory tiles include a small pad to reduce bank conflicts.
    __shared__ T As[kBlockRows][kBlockDepth + kSharedPad];
    __shared__ T Bs[kBlockDepth][kBlockCols + kSharedPad];

    T acc[kThreadRows][kThreadCols] = {{static_cast<T>(0), static_cast<T>(0)},
                                       {static_cast<T>(0), static_cast<T>(0)}};

    for (int k0 = 0; k0 < K; k0 += block_depth) {
        // Cooperative tile load. Compile-time specialization keeps the
        // addressing formula simple inside the hot loop for each transpose
        // combination.
        for (int elem = tid; elem < block_rows * block_depth; elem += num_threads) {
            const int load_m = elem / block_depth;
            const int load_k = elem - load_m * block_depth;
            const int g_m = tile_row + load_m;
            const int g_k = k0 + load_k;
            As[load_m][load_k] =
                (g_m < M && g_k < K) ? An[g_k * M + g_m] : static_cast<T>(0);
        } else {
            const int load_m = tid / kBlockDepth;
            const int load_k = tid % kBlockDepth;
            const int g_m = tile_row + load_m;
            const int g_k = k0 + load_k;
            As[load_m][load_k] =
                (g_m < M && g_k < K) ? An[g_m * K + g_k] : static_cast<T>(0);
        }

        if constexpr (TRANS_B) {
            const int load_n = tid / kBlockDepth;
            const int load_k = tid % kBlockDepth;
            const int g_n = tile_col + load_n;
            const int g_k = k0 + load_k;
            Bs[load_k][load_n] =
                (g_n < N && g_k < K) ? Bn[g_n * K + g_k] : static_cast<T>(0);
        } else {
            const int load_k = tid / kBlockCols;
            const int load_n = tid % kBlockCols;
            const int g_n = tile_col + load_n;
            const int g_k = k0 + load_k;
            Bs[load_k][load_n] =
                (g_n < N && g_k < K) ? Bn[g_k * N + g_n] : static_cast<T>(0);
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < kBlockDepth; ++kk) {
            const T a0 = As[local_row + 0][kk];
            const T a1 = As[local_row + 1][kk];
            const T b0 = Bs[kk][local_col + 0];
            const T b1 = Bs[kk][local_col + 1];

            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
        }

        __syncthreads();
    }

    // Store the thread-local output fragment with boundary checks for
    // partially covered tiles on matrix edges.
    #pragma unroll
    for (int i = 0; i < kThreadRows; ++i) {
        const int out_row = out_row0 + i;
        if (out_row >= M) {
            continue;
        }
        #pragma unroll
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

template <typename T, bool TRANS_A, bool TRANS_B, bool ACCUM>
static void launch_bmm_kernel(
    const T* A,
    const T* B,
    T* C,
    int batch,
    int M,
    int N,
    int K,
    cudaStream_t stream)
{
    if (batch <= 0 || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    const BmmTileConfig cfg = get_tile_config();
    dim3 block(cfg.block_cols / cfg.thread_cols, cfg.block_rows / cfg.thread_rows);
    const int tiles_m = (M + cfg.block_rows - 1) / cfg.block_rows;
    const int tiles_n = (N + cfg.block_cols - 1) / cfg.block_cols;
    const size_t total_tiles = static_cast<size_t>(batch) * tiles_m * tiles_n;
    const size_t shared_bytes = tile_shared_bytes(cfg);
    constexpr unsigned int kMaxGridX = 0x7fffffffu;

    record_last_launch_info(cfg);

    for (size_t tile_offset = 0; tile_offset < total_tiles; ) {
        const size_t remaining_tiles = total_tiles - tile_offset;
        const unsigned int grid_x =
            remaining_tiles > static_cast<size_t>(kMaxGridX)
                ? kMaxGridX
                : static_cast<unsigned int>(remaining_tiles);
        dim3 grid(grid_x, 1, 1);
        bmm_tiled_kernel<T, TRANS_A, TRANS_B, ACCUM><<<grid, block, 0, stream>>>(
            A, B, C, batch, M, N, K, tiles_m, tiles_n, tile_offset);
        tile_offset += grid_x;
    }
}

static void bmm_matmul_impl(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    int trans_a, int trans_b,
    bool accumulate,
    cudaStream_t stream)
{
    if (batch <= 0 || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    // Dispatch once on transpose flags so the inner kernel can stay branch-free
    // with respect to data layout.
    const bool transpose_a = trans_a != 0;
    const bool transpose_b = trans_b != 0;

    // Prefer cuBLASLt for transpose-A weight-gradient-like shapes. It handles
    // small-output / large-reduction GEMMs much better on modern GPUs and may
    // internally choose split-K style algorithms.
    if (!accumulate &&
        transpose_a && !transpose_b &&
        K >= kCublasTransposeAPathMinK &&
        launch_cublaslt_transpose_a_path(A, B, C, batch, M, N, K, accumulate, stream)) {
        return;
    }

    if (!accumulate &&
        transpose_a && !transpose_b &&
        K >= kCublasTransposeAPathMinK &&
        launch_cublas_transpose_a_path(A, B, C, batch, M, N, K, accumulate, stream)) {
        return;
    }

    // Forward convolution uses row-major A[M,K] @ B[N,K]^T and was previously
    // bound by the custom transpose-B kernel. Route that shape through cuBLAS
    // so small kernels (1x1/3x3) and larger receptive fields both use a
    // vendor GEMM instead of the slower fallback.
    if (!transpose_a && transpose_b &&
        launch_cublas_transpose_b_path(A, B, C, batch, M, N, K, accumulate, stream)) {
        return;
    }

    if (!transpose_a && !transpose_b) {
        if (accumulate) launch_bmm_kernel<float, false, false, true>(A, B, C, batch, M, N, K, stream);
        else launch_bmm_kernel<float, false, false, false>(A, B, C, batch, M, N, K, stream);
        return;
    }

    if (!transpose_a && transpose_b) {
        if (accumulate) launch_bmm_kernel<float, false, true, true>(A, B, C, batch, M, N, K, stream);
        else launch_bmm_kernel<float, false, true, false>(A, B, C, batch, M, N, K, stream);
        return;
    }

    if (transpose_a && !transpose_b) {
        if (accumulate) launch_bmm_kernel<float, true, false, true>(A, B, C, batch, M, N, K, stream);
        else launch_bmm_kernel<float, true, false, false>(A, B, C, batch, M, N, K, stream);
        return;
    }

    if (accumulate) launch_bmm_kernel<float, true, true, true>(A, B, C, batch, M, N, K, stream);
    else launch_bmm_kernel<float, true, true, false>(A, B, C, batch, M, N, K, stream);
}

extern "C" void bmm_matmul(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    int trans_a, int trans_b,
    cudaStream_t stream)
{
    bmm_matmul_impl(A, B, C, batch, M, N, K, trans_a, trans_b, false, stream);
}

extern "C" void bmm_matmul_accum(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    int trans_a, int trans_b,
    cudaStream_t stream)
{
    bmm_matmul_impl(A, B, C, batch, M, N, K, trans_a, trans_b, true, stream);
}

extern "C" void bmm_matmul_i32(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    int trans_a, int trans_b)
{
    if (batch <= 0 || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    const bool transpose_a = trans_a != 0;
    const bool transpose_b = trans_b != 0;

    if (!transpose_a && !transpose_b) {
        launch_bmm_kernel<int32_t, false, false, false>(A, B, C, batch, M, N, K, nullptr);
        return;
    }

    if (!transpose_a && transpose_b) {
        launch_bmm_kernel<int32_t, false, true, false>(A, B, C, batch, M, N, K, nullptr);
        return;
    }

    if (transpose_a && !transpose_b) {
        launch_bmm_kernel<int32_t, true, false, false>(A, B, C, batch, M, N, K, nullptr);
        return;
    }

    launch_bmm_kernel<int32_t, true, true, false>(A, B, C, batch, M, N, K, nullptr);
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
