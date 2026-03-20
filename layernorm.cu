#include <cuda_runtime.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

static inline int next_pow2(int v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

/* ================================================================== */
/*                        FORWARD  PASS                               */
/* ================================================================== */
/*                                                                    */
/*  Data layout: NHWC.  Normalization over the C dimension.           */
/*                                                                    */
/*  For every spatial position idx = n*H*W + h*W + w:                 */
/*    mu        = (1/C) * sum_c  x[idx, c]                            */
/*    var       = (1/C) * sum_c  x[idx, c]^2  -  mu^2                */
/*    inv_std   = rsqrt(var + eps)                                    */
/*    y[idx, c] = gamma[c] * (x[idx, c] - mu) * inv_std + beta[c]   */
/*                                                                    */
/*  Saved for backward: x, mean, inv_std (all from fprop args).       */
/*                                                                    */
/*  One block per spatial position.  Threads stride over C.           */
/* ------------------------------------------------------------------ */

__global__ void layernorm_fprop_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    float* __restrict__ mean_out,
    float* __restrict__ inv_std_out,
    int C, int NHW, float eps)
{
    const int idx = blockIdx.x;
    if (idx >= NHW) return;

    const float* x_row = x + (size_t)idx * C;
    float*       y_row = y + (size_t)idx * C;

    extern __shared__ float smem[];
    float* s_sum  = smem;                 /* blockDim.x floats */
    float* s_sum2 = smem + blockDim.x;    /* blockDim.x floats */

    /* --- partial sums for mean & variance (one pass) --- */
    float local_sum  = 0.f;
    float local_sum2 = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float v = x_row[c];
        local_sum  += v;
        local_sum2 += v * v;
    }
    s_sum [threadIdx.x] = local_sum;
    s_sum2[threadIdx.x] = local_sum2;
    __syncthreads();

    /* --- tree reduction --- */
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum [threadIdx.x] += s_sum [threadIdx.x + s];
            s_sum2[threadIdx.x] += s_sum2[threadIdx.x + s];
        }
        __syncthreads();
    }

    float mu   = s_sum[0]  / C;
    float var  = s_sum2[0] / C - mu * mu;
    float rstd = rsqrtf(var + eps);

    if (threadIdx.x == 0) {
        mean_out[idx]    = mu;
        inv_std_out[idx] = rstd;
    }

    /* --- normalize + affine transform --- */
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float x_hat = (x_row[c] - mu) * rstd;
        y_row[c] = gamma[c] * x_hat + beta[c];
    }
}

/* ================================================================== */
/*                       BACKWARD  PASS — dx                          */
/* ================================================================== */
/*                                                                    */
/*  dx_hat[c] = dy[c] * gamma[c]                                     */
/*  s1        = sum_c  dx_hat[c]                                      */
/*  s2        = sum_c  dx_hat[c] * x_hat[c]                           */
/*  dx[c]     = inv_std * (dx_hat[c] - (s1 + x_hat[c]*s2) / C)      */
/*                                                                    */
/*  One block per spatial position.  Threads stride over C.           */
/* ------------------------------------------------------------------ */

__global__ void layernorm_bprop_dx_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    float* __restrict__ dx,
    int C, int NHW)
{
    const int idx = blockIdx.x;
    if (idx >= NHW) return;

    const float* x_row  = x  + (size_t)idx * C;
    const float* dy_row = dy + (size_t)idx * C;
    float*       dx_row = dx + (size_t)idx * C;

    float mu   = mean[idx];
    float rstd = inv_std[idx];

    extern __shared__ float smem[];
    float* ss1 = smem;
    float* ss2 = smem + blockDim.x;

    /* --- accumulate s1, s2 --- */
    float ls1 = 0.f, ls2 = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float dh = dy_row[c] * gamma[c];
        float xh = (x_row[c] - mu) * rstd;
        ls1 += dh;
        ls2 += dh * xh;
    }
    ss1[threadIdx.x] = ls1;
    ss2[threadIdx.x] = ls2;
    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            ss1[threadIdx.x] += ss1[threadIdx.x + s];
            ss2[threadIdx.x] += ss2[threadIdx.x + s];
        }
        __syncthreads();
    }

    float sum1  = ss1[0];
    float sum2  = ss2[0];
    float inv_C = 1.f / C;

    /* --- write dx --- */
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float dh = dy_row[c] * gamma[c];
        float xh = (x_row[c] - mu) * rstd;
        dx_row[c] = rstd * (dh - inv_C * (sum1 + xh * sum2));
    }
}

/* ================================================================== */
/*                  BACKWARD  PASS — dgamma, dbeta                    */
/* ================================================================== */
/*                                                                    */
/*  dgamma[c] = sum_{n,h,w}  dy[n,h,w,c] * x_hat[n,h,w,c]          */
/*  dbeta [c] = sum_{n,h,w}  dy[n,h,w,c]                            */
/*                                                                    */
/*  One block per channel c.  Threads stride over NHW.                */
/* ------------------------------------------------------------------ */

__global__ void layernorm_bprop_dgamma_dbeta_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int C, int NHW)
{
    const int c = blockIdx.x;
    if (c >= C) return;

    extern __shared__ float smem[];
    float* sg = smem;
    float* sb = smem + blockDim.x;

    float lg = 0.f, lb = 0.f;
    for (int i = threadIdx.x; i < NHW; i += blockDim.x) {
        float dy_val = dy[(size_t)i * C + c];
        float xh     = (x[(size_t)i * C + c] - mean[i]) * inv_std[i];
        lg += dy_val * xh;
        lb += dy_val;
    }
    sg[threadIdx.x] = lg;
    sb[threadIdx.x] = lb;
    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sg[threadIdx.x] += sg[threadIdx.x + s];
            sb[threadIdx.x] += sb[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        dgamma[c] = sg[0];
        dbeta[c]  = sb[0];
    }
}

/* ================================================================== */
/*                        HOST  WRAPPERS                              */
/* ================================================================== */

void layernorm_fprop(
    const float* x, const float* gamma, const float* beta,
    float* y, float* mean, float* inv_std,
    int C, int H, int W, int N, float eps)
{
    int NHW = N * H * W;

    int block = next_pow2(C);
    if (block < 32)   block = 32;
    if (block > 1024) block = 1024;
    size_t smem = 2 * block * sizeof(float);

    layernorm_fprop_kernel<<<NHW, block, smem>>>(
        x, gamma, beta, y, mean, inv_std, C, NHW, eps);
}

void layernorm_bprop(
    const float* x, const float* dy,
    const float* gamma,
    const float* mean, const float* inv_std,
    float* dx, float* dgamma, float* dbeta,
    int C, int H, int W, int N)
{
    int NHW = N * H * W;

    /* --- dx: one block per spatial position --- */
    int block_dx = next_pow2(C);
    if (block_dx < 32)   block_dx = 32;
    if (block_dx > 1024) block_dx = 1024;
    size_t smem_dx = 2 * block_dx * sizeof(float);

    layernorm_bprop_dx_kernel<<<NHW, block_dx, smem_dx>>>(
        x, dy, gamma, mean, inv_std, dx, C, NHW);

    /* --- dgamma, dbeta: one block per channel --- */
    int block_gb = next_pow2(NHW);
    if (block_gb < 32)   block_gb = 32;
    if (block_gb > 1024) block_gb = 1024;
    size_t smem_gb = 2 * block_gb * sizeof(float);

    layernorm_bprop_dgamma_dbeta_kernel<<<C, block_gb, smem_gb>>>(
        x, dy, mean, inv_std, dgamma, dbeta, C, NHW);
}
