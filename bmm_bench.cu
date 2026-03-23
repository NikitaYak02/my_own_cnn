#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bmm.h"

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call) do {                                           \
    cudaError_t e = (call);                                             \
    if (e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(e));             \
        exit(1);                                                        \
    }                                                                   \
} while (0)

#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t s = (call);                                          \
    if (s != CUBLAS_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n",                    \
                __FILE__, __LINE__, (int)s);                            \
        exit(1);                                                        \
    }                                                                   \
} while (0)

static void fill_random(float* h, size_t n) {
    for (size_t i = 0; i < n; i++)
        h[i] = (float)rand() / RAND_MAX - 0.5f;
}

static float max_abs_diff(const float* a, const float* b, size_t n) {
    float mx = 0.f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

/* ------------------------------------------------------------------ */
/*  cuBLAS reference wrappers                                         */
/* ------------------------------------------------------------------ */
/*                                                                    */
/*  Row-major trick: for row-major C = A @ B^T where A[W,C], B[H,C], */
/*  C[W,H], pass to column-major cuBLAS as:                           */
/*    C_cm = T(B_cm) @ N(A_cm)                                        */
/*  where cuBLAS interprets row-major X as column-major X^T.          */
/*                                                                    */
/*  See comments inline for each operation.                           */
/* ------------------------------------------------------------------ */

static void cublas_bmm_fprop(
    cublasHandle_t handle,
    const float* A, const float* B, float* Y,
    int N, int W, int H, int C)
{
    /*  Y[W,H] = A[W,C] @ B[H,C]^T
     *  cuBLAS column-major: Y_cm[H,W] = T(B_cm)[H,C] @ N(A_cm)[C,W]
     *  → T on first arg (B), N on second arg (A)
     *  m=H, n=W, k=C
     */
    float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        H, W, C,
        &alpha,
        B, C, (long long)H * C,
        A, C, (long long)W * C,
        &beta,
        Y, H, (long long)W * H,
        N));
}

static void cublas_bmm_bprop_dA(
    cublasHandle_t handle,
    const float* dY, const float* B, float* dA,
    int N, int W, int H, int C)
{
    /*  dA[W,C] = dY[W,H] @ B[H,C]
     *  cuBLAS: dA_cm[C,W] = N(B_cm)[C,H] @ N(dY_cm)[H,W]
     *  m=C, n=W, k=H
     */
    float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        C, W, H,
        &alpha,
        B, C, (long long)H * C,
        dY, H, (long long)W * H,
        &beta,
        dA, C, (long long)W * C,
        N));
}

static void cublas_bmm_bprop_dB(
    cublasHandle_t handle,
    const float* dY, const float* A, float* dB,
    int N, int W, int H, int C)
{
    /*  dB[H,C] = dY[W,H]^T @ A[W,C]  =  dY^T[H,W] @ A[W,C]
     *  cuBLAS: dB_cm[C,H] = N(A_cm)[C,W] @ T(dY_cm)[W,H]
     *  m=C, n=H, k=W
     */
    float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        C, H, W,
        &alpha,
        A, C, (long long)W * C,
        dY, H, (long long)W * H,
        &beta,
        dB, C, (long long)H * C,
        N));
}

/* ------------------------------------------------------------------ */
/*  Timing helper                                                     */
/* ------------------------------------------------------------------ */

typedef void (*bench_fn)(void);

static float time_kernel(bench_fn fn, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++) fn();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / iters;
}

/* ------------------------------------------------------------------ */
/*  Globals for lambdas (C function pointers used in timing)          */
/* ------------------------------------------------------------------ */

static const float *g_A, *g_B, *g_dY;
static float *g_Y, *g_dA, *g_dB;
static float *g_Y_ref, *g_dA_ref, *g_dB_ref;
static int g_N, g_W, g_H, g_C;
static cublasHandle_t g_handle;

static void run_custom_fprop(void) {
    bmm_fprop(g_A, g_B, g_Y, g_N, g_W, g_H, g_C);
}

static void run_custom_bprop(void) {
    bmm_bprop_dA(g_dY, g_B, g_dA, g_N, g_W, g_H, g_C);
    bmm_bprop_dB(g_dY, g_A, g_dB, g_N, g_W, g_H, g_C);
}

static void run_cublas_fprop(void)  { cublas_bmm_fprop(g_handle, g_A, g_B, g_Y_ref, g_N, g_W, g_H, g_C); }
static void run_cublas_bprop(void) {
    cublas_bmm_bprop_dA(g_handle, g_dY, g_B, g_dA_ref, g_N, g_W, g_H, g_C);
    cublas_bmm_bprop_dB(g_handle, g_dY, g_A, g_dB_ref, g_N, g_W, g_H, g_C);
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv)
{
    /* Default sizes */
    int N = 32, W = 64, H = 64, C = 128;
    if (argc >= 5) {
        N = atoi(argv[1]); W = atoi(argv[2]);
        H = atoi(argv[3]); C = atoi(argv[4]);
    }

    printf("BMM benchmark: N=%d  W=%d  H=%d  C=%d\n", N, W, H, C);

    size_t szA  = (size_t)N * W * C * sizeof(float);
    size_t szB  = (size_t)N * H * C * sizeof(float);
    size_t szY  = (size_t)N * W * H * sizeof(float);

    /* --- host allocations --- */
    float *hA  = (float*)malloc(szA);
    float *hB  = (float*)malloc(szB);
    float *hdY = (float*)malloc(szY);
    float *hY_custom  = (float*)malloc(szY);
    float *hY_cublas  = (float*)malloc(szY);
    float *hdA_custom = (float*)malloc(szA);
    float *hdA_cublas = (float*)malloc(szA);
    float *hdB_custom = (float*)malloc(szB);
    float *hdB_cublas = (float*)malloc(szB);

    srand(42);
    fill_random(hA, (size_t)N * W * C);
    fill_random(hB, (size_t)N * H * C);
    fill_random(hdY, (size_t)N * W * H);

    /* --- device allocations --- */
    float *dA, *dB, *dY_d, *dY_out, *ddA, *ddB;
    float *dY_ref, *ddA_ref, *ddB_ref;

    CUDA_CHECK(cudaMalloc(&dA, szA));
    CUDA_CHECK(cudaMalloc(&dB, szB));
    CUDA_CHECK(cudaMalloc(&dY_d, szY));

    CUDA_CHECK(cudaMalloc(&dY_out, szY));     /* custom Y */
    CUDA_CHECK(cudaMalloc(&ddA, szA));        /* custom dA */
    CUDA_CHECK(cudaMalloc(&ddB, szB));        /* custom dB */

    CUDA_CHECK(cudaMalloc(&dY_ref, szY));     /* cuBLAS Y */
    CUDA_CHECK(cudaMalloc(&ddA_ref, szA));    /* cuBLAS dA */
    CUDA_CHECK(cudaMalloc(&ddB_ref, szB));    /* cuBLAS dB */

    CUDA_CHECK(cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY_d, hdY, szY, cudaMemcpyHostToDevice));

    /* --- set globals --- */
    g_A = dA; g_B = dB; g_dY = dY_d;
    g_Y = dY_out; g_dA = ddA; g_dB = ddB;
    g_Y_ref = dY_ref; g_dA_ref = ddA_ref; g_dB_ref = ddB_ref;
    g_N = N; g_W = W; g_H = H; g_C = C;

    CUBLAS_CHECK(cublasCreate(&g_handle));
    CUBLAS_CHECK(cublasSetMathMode(g_handle, CUBLAS_DEFAULT_MATH));

    /* ============================================================= */
    /*  Correctness check                                            */
    /* ============================================================= */

    /* fprop */
    bmm_fprop(dA, dB, dY_out, N, W, H, C);
    cublas_bmm_fprop(g_handle, dA, dB, dY_ref, N, W, H, C);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hY_custom, dY_out, szY, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hY_cublas, dY_ref, szY, cudaMemcpyDeviceToHost));
    printf("fprop  max|diff| = %e\n",
           max_abs_diff(hY_custom, hY_cublas, (size_t)N * W * H));

    /* bprop */
    bmm_bprop_dA(dY_d, dB, ddA, N, W, H, C);
    bmm_bprop_dB(dY_d, dA, ddB, N, W, H, C);
    cublas_bmm_bprop_dA(g_handle, dY_d, dB, ddA_ref, N, W, H, C);
    cublas_bmm_bprop_dB(g_handle, dY_d, dA, ddB_ref, N, W, H, C);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hdA_custom, ddA, szA, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdA_cublas, ddA_ref, szA, cudaMemcpyDeviceToHost));
    printf("bprop dA max|diff| = %e\n",
           max_abs_diff(hdA_custom, hdA_cublas, (size_t)N * W * C));

    CUDA_CHECK(cudaMemcpy(hdB_custom, ddB, szB, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdB_cublas, ddB_ref, szB, cudaMemcpyDeviceToHost));
    printf("bprop dB max|diff| = %e\n",
           max_abs_diff(hdB_custom, hdB_cublas, (size_t)N * H * C));

    /* ============================================================= */
    /*  Performance                                                  */
    /* ============================================================= */

    int warmup = 20, iters = 100;
    printf("\nTiming (%d warmup, %d iters):\n", warmup, iters);

    float t_custom_fp = time_kernel(run_custom_fprop, warmup, iters);
    float t_cublas_fp = time_kernel(run_cublas_fprop, warmup, iters);
    printf("  fprop   custom: %.3f ms   cuBLAS: %.3f ms   ratio: %.2fx\n",
           t_custom_fp, t_cublas_fp, t_custom_fp / t_cublas_fp);

    float t_custom_bp = time_kernel(run_custom_bprop, warmup, iters);
    float t_cublas_bp = time_kernel(run_cublas_bprop, warmup, iters);
    printf("  bprop   custom: %.3f ms   cuBLAS: %.3f ms   ratio: %.2fx\n",
           t_custom_bp, t_cublas_bp, t_custom_bp / t_cublas_bp);

    /* GFLOP/s */
    double flops_fp = 2.0 * N * W * H * C;
    double flops_bp = 2.0 * flops_fp;   /* dA + dB, each same as fprop */
    printf("\n  fprop   custom: %.1f GFLOP/s   cuBLAS: %.1f GFLOP/s\n",
           flops_fp / (t_custom_fp * 1e6), flops_fp / (t_cublas_fp * 1e6));
    printf("  bprop   custom: %.1f GFLOP/s   cuBLAS: %.1f GFLOP/s\n",
           flops_bp / (t_custom_bp * 1e6), flops_bp / (t_cublas_bp * 1e6));

    /* cleanup */
    cublasDestroy(g_handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dY_d);
    cudaFree(dY_out); cudaFree(ddA); cudaFree(ddB);
    cudaFree(dY_ref); cudaFree(ddA_ref); cudaFree(ddB_ref);
    free(hA); free(hB); free(hdY);
    free(hY_custom); free(hY_cublas);
    free(hdA_custom); free(hdA_cublas);
    free(hdB_custom); free(hdB_cublas);

    return 0;
}
