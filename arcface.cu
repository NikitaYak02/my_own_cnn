#include "arcface.h"

#include <cuda_runtime.h>

#include <cmath>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr float kGradSineEps = 1.0e-7f;
constexpr float kPi = 3.14159265358979323846f;

__device__ __forceinline__ float clamp_cosine(float cosine) {
    return fminf(1.0f, fmaxf(-1.0f, cosine));
}

__device__ __forceinline__ float target_phi(float cosine, float cos_m, float sin_m) {
    const float clamped = clamp_cosine(cosine);
    const float sine = sqrtf(fmaxf(0.0f, 1.0f - clamped * clamped));
    return clamped * cos_m - sine * sin_m;
}

__device__ __forceinline__ float target_phi_grad(float cosine, float cos_m, float sin_m) {
    const float clamped = clamp_cosine(cosine);
    const float sine_sq = fmaxf(kGradSineEps, 1.0f - clamped * clamped);
    const float sine = sqrtf(sine_sq);
    return cos_m + (clamped / sine) * sin_m;
}

__device__ __forceinline__ float apply_target_transform(
    float cosine,
    float cos_m,
    float sin_m,
    float threshold,
    float mm,
    bool easy_margin)
{
    const float phi = target_phi(cosine, cos_m, sin_m);
    if (easy_margin) {
        return cosine > 0.0f ? phi : cosine;
    }
    return cosine > threshold ? phi : (cosine - mm);
}

__device__ __forceinline__ float apply_target_grad(
    float cosine,
    float cos_m,
    float sin_m,
    float threshold,
    bool easy_margin)
{
    if (easy_margin) {
        return cosine > 0.0f ? target_phi_grad(cosine, cos_m, sin_m) : 1.0f;
    }
    return cosine > threshold ? target_phi_grad(cosine, cos_m, sin_m) : 1.0f;
}

__global__ void arcface_fprop_kernel(
    const float* __restrict__ cosine,
    const int* __restrict__ labels,
    float* __restrict__ logits,
    int batch,
    int classes,
    float cos_m,
    float sin_m,
    float threshold,
    float mm,
    float scale,
    bool easy_margin)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * classes;
    if (idx >= total) {
        return;
    }

    const int row = idx / classes;
    const int col = idx - row * classes;
    const int label = labels[row];
    const float cosine_val = cosine[idx];

    float out = cosine_val;
    if (label >= 0 && label < classes && col == label) {
        out = apply_target_transform(cosine_val, cos_m, sin_m, threshold, mm, easy_margin);
    }

    logits[idx] = scale * out;
}

__global__ void arcface_bprop_kernel(
    const float* __restrict__ cosine,
    const int* __restrict__ labels,
    const float* __restrict__ d_logits,
    float* __restrict__ d_cosine,
    int batch,
    int classes,
    float cos_m,
    float sin_m,
    float threshold,
    float scale,
    bool easy_margin)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * classes;
    if (idx >= total) {
        return;
    }

    const int row = idx / classes;
    const int col = idx - row * classes;
    const int label = labels[row];

    float jacobian = 1.0f;
    if (label >= 0 && label < classes && col == label) {
        jacobian = apply_target_grad(cosine[idx], cos_m, sin_m, threshold, easy_margin);
    }

    d_cosine[idx] = d_logits[idx] * scale * jacobian;
}

}  // namespace

extern "C" void arcface_fprop(
    const float* cosine,
    const int* labels,
    float* logits,
    int batch,
    int classes,
    float margin,
    float scale,
    int easy_margin)
{
    if (batch <= 0 || classes <= 0) {
        return;
    }

    const int total = batch * classes;
    const int blocks = (total + kThreadsPerBlock - 1) / kThreadsPerBlock;

    const float cos_m = cosf(margin);
    const float sin_m = sinf(margin);
    const float threshold = cosf(kPi - margin);
    const float mm = sinf(kPi - margin) * margin;

    arcface_fprop_kernel<<<blocks, kThreadsPerBlock>>>(
        cosine,
        labels,
        logits,
        batch,
        classes,
        cos_m,
        sin_m,
        threshold,
        mm,
        scale,
        easy_margin != 0);
}

extern "C" void arcface_bprop(
    const float* cosine,
    const int* labels,
    const float* d_logits,
    float* d_cosine,
    int batch,
    int classes,
    float margin,
    float scale,
    int easy_margin)
{
    if (batch <= 0 || classes <= 0) {
        return;
    }

    const int total = batch * classes;
    const int blocks = (total + kThreadsPerBlock - 1) / kThreadsPerBlock;

    const float cos_m = cosf(margin);
    const float sin_m = sinf(margin);
    const float threshold = cosf(kPi - margin);

    arcface_bprop_kernel<<<blocks, kThreadsPerBlock>>>(
        cosine,
        labels,
        d_logits,
        d_cosine,
        batch,
        classes,
        cos_m,
        sin_m,
        threshold,
        scale,
        easy_margin != 0);
}
