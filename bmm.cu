#include <cuda_runtime.h>
#include <stdio.h>

#define BMM_TILE 32
#define BMM_PAD  1   /* shared-memory padding to avoid bank conflicts */

/* ================================================================== */
/*                    FORWARD:  Y[N,W,H] = A[N,W,C] @ B[N,H,C]^T    */
/* ================================================================== */
/*                                                                    */
/*  One thread-block computes a TILE x TILE tile of Y for one batch.  */
/*  Grid: (ceil(H/TILE), ceil(W/TILE), N).                           */
/*  Tiles iterate over the K=C dimension in shared memory.            */
/*                                                                    */
/*  Memory access patterns (all coalesced):                           */
/*   - A loaded row-major  [W,C]:  consecutive threadIdx.x → +1 in C */
/*   - B loaded row-major  [H,C], stored transposed in smem so that  */
/*     the B^T multiply needs no strided global reads.                */
/* ------------------------------------------------------------------ */

__global__ void bmm_fprop_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ Y,
    int W, int H, int C)
{
    const int n   = blockIdx.z;
    const int row = blockIdx.y * BMM_TILE + threadIdx.y;   /* W index */
    const int col = blockIdx.x * BMM_TILE + threadIdx.x;   /* H index */

    const float* An = A + (size_t)n * W * C;
    const float* Bn = B + (size_t)n * H * C;
    float*       Yn = Y + (size_t)n * W * H;

    __shared__ float sA[BMM_TILE][BMM_TILE + BMM_PAD];
    __shared__ float sB[BMM_TILE][BMM_TILE + BMM_PAD];

    float acc = 0.f;

    for (int t = 0; t < (C + BMM_TILE - 1) / BMM_TILE; t++) {

        /* sA[ty][tx] = A[row, t*TILE + tx]  — coalesced read */
        int ac = t * BMM_TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] =
            (row < W && ac < C) ? An[row * C + ac] : 0.f;

        /* Load B coalesced, store transposed for B^T access.
         *   read  B[bh, bc]  where bh = tile_h + ty, bc = tile_c + tx
         *   store sB[tx][ty] so that sB[k][col_local] = B[col, tile_c+k]
         */
        int bh = blockIdx.x * BMM_TILE + threadIdx.y;
        int bc = t * BMM_TILE + threadIdx.x;
        float bval = (bh < H && bc < C) ? Bn[bh * C + bc] : 0.f;
        sB[threadIdx.x][threadIdx.y] = bval;          /* transpose */

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BMM_TILE; k++)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < W && col < H)
        Yn[row * H + col] = acc;
}

/* ================================================================== */
/*             BACKWARD dA:  dA[N,W,C] = dY[N,W,H] @ B[N,H,C]       */
/* ================================================================== */
/*  Standard matmul (no transpose needed).                            */
/*  Grid: (ceil(C/TILE), ceil(W/TILE), N).                           */
/* ------------------------------------------------------------------ */

__global__ void bmm_bprop_dA_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ B,
    float* __restrict__ dA,
    int W, int H, int C)
{
    const int n   = blockIdx.z;
    const int row = blockIdx.y * BMM_TILE + threadIdx.y;   /* W index */
    const int col = blockIdx.x * BMM_TILE + threadIdx.x;   /* C index */

    const float* dYn = dY + (size_t)n * W * H;
    const float* Bn  = B  + (size_t)n * H * C;
    float*       dAn = dA + (size_t)n * W * C;

    __shared__ float sdY[BMM_TILE][BMM_TILE + BMM_PAD];
    __shared__ float sB [BMM_TILE][BMM_TILE + BMM_PAD];

    float acc = 0.f;

    for (int t = 0; t < (H + BMM_TILE - 1) / BMM_TILE; t++) {

        /* sdY[ty][tx] = dY[row, t*TILE + tx]  — coalesced */
        int dh = t * BMM_TILE + threadIdx.x;
        sdY[threadIdx.y][threadIdx.x] =
            (row < W && dh < H) ? dYn[row * H + dh] : 0.f;

        /* sB[ty][tx] = B[t*TILE + ty, col]  — coalesced */
        int bh = t * BMM_TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] =
            (bh < H && col < C) ? Bn[bh * C + col] : 0.f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BMM_TILE; k++)
            acc += sdY[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < W && col < C)
        dAn[row * C + col] = acc;
}

/* ================================================================== */
/*          BACKWARD dB:  dB[N,H,C] = dY[N,W,H]^T @ A[N,W,C]        */
/* ================================================================== */
/*  dY^T loaded via coalesced read + shared-memory transpose.         */
/*  Grid: (ceil(C/TILE), ceil(H/TILE), N).                           */
/* ------------------------------------------------------------------ */

__global__ void bmm_bprop_dB_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ A,
    float* __restrict__ dB,
    int W, int H, int C)
{
    const int n   = blockIdx.z;
    const int row = blockIdx.y * BMM_TILE + threadIdx.y;   /* H index */
    const int col = blockIdx.x * BMM_TILE + threadIdx.x;   /* C index */

    const float* dYn = dY + (size_t)n * W * H;
    const float* An  = A  + (size_t)n * W * C;
    float*       dBn = dB + (size_t)n * H * C;

    __shared__ float sdYt[BMM_TILE][BMM_TILE + BMM_PAD];
    __shared__ float sA  [BMM_TILE][BMM_TILE + BMM_PAD];

    float acc = 0.f;

    for (int t = 0; t < (W + BMM_TILE - 1) / BMM_TILE; t++) {

        /* Load dY coalesced and transpose in shared memory.
         *   read  dY[w, h]   where w = tile_w + ty, h = tile_h + tx
         *   store sdYt[tx][ty]  so sdYt[h_local][w_local] = dY[w, h]
         *   then  sdYt[row_local][k] = dY[tile_w+k, row] = dY^T[row, tile_w+k]
         */
        int dw = t * BMM_TILE + threadIdx.y;
        int dh = blockIdx.y * BMM_TILE + threadIdx.x;
        float dyval = (dw < W && dh < H) ? dYn[dw * H + dh] : 0.f;
        sdYt[threadIdx.x][threadIdx.y] = dyval;           /* transpose */

        /* sA[ty][tx] = A[t*TILE + ty, col]  — coalesced */
        int aw = t * BMM_TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] =
            (aw < W && col < C) ? An[aw * C + col] : 0.f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BMM_TILE; k++)
            acc += sdYt[threadIdx.y][k] * sA[k][threadIdx.x];

        __syncthreads();
    }

    if (row < H && col < C)
        dBn[row * C + col] = acc;
}

/* ================================================================== */
/*                  GENERIC FALLBACK:  C = A^T @ B^T                  */
/* ================================================================== */
/*  This path is only used for the transpose/transpose case.          */
/* ------------------------------------------------------------------ */

template <bool TRANS_A, bool TRANS_B>
__global__ void bmm_generic_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int n   = blockIdx.z;
    const int row = blockIdx.y * BMM_TILE + threadIdx.y;
    const int col = blockIdx.x * BMM_TILE + threadIdx.x;

    const float* An = A + static_cast<size_t>(n) * M * K;
    const float* Bn = B + static_cast<size_t>(n) * K * N;
    float*       Cn = C + static_cast<size_t>(n) * M * N;

    __shared__ float sA[BMM_TILE][BMM_TILE + BMM_PAD];
    __shared__ float sB[BMM_TILE][BMM_TILE + BMM_PAD];

    float acc = 0.0f;

    for (int t = 0; t < (K + BMM_TILE - 1) / BMM_TILE; ++t) {
        const int k_a = t * BMM_TILE + threadIdx.x;
        const int k_b = t * BMM_TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] =
            (row < M && k_a < K)
                ? (TRANS_A ? An[k_a * M + row] : An[row * K + k_a])
                : 0.0f;

        sB[threadIdx.y][threadIdx.x] =
            (k_b < K && col < N)
                ? (TRANS_B ? Bn[col * K + k_b] : Bn[k_b * N + col])
                : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BMM_TILE; ++kk) {
            acc += sA[threadIdx.y][kk] * sB[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        Cn[row * N + col] = acc;
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
    dim3 block(BMM_TILE, BMM_TILE);
    dim3 grid((N + BMM_TILE - 1) / BMM_TILE,
              (M + BMM_TILE - 1) / BMM_TILE,
              batch);

    if (!trans_a && trans_b) {
        bmm_fprop_kernel<<<grid, block>>>(A, B, C, M, N, K);
        return;
    }

    if (!trans_a && !trans_b) {
        bmm_bprop_dA_kernel<<<grid, block>>>(A, B, C, M, K, N);
        return;
    }

    if (trans_a && !trans_b) {
        bmm_bprop_dB_kernel<<<grid, block>>>(A, B, C, K, M, N);
        return;
    }

    bmm_generic_kernel<true, true><<<grid, block>>>(A, B, C, M, N, K);
}
