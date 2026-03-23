#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "arcface.h"

#define CUDA_CHECK(call) do {                                           \
    cudaError_t e = (call);                                             \
    if (e != cudaSuccess) {                                             \
        throw std::runtime_error(std::string("CUDA error: ") +          \
                                 cudaGetErrorString(e));                \
    }                                                                   \
} while (0)

namespace {

constexpr float kGradSineEps = 1.0e-7f;
constexpr float kPi = 3.14159265358979323846f;

struct Case {
    const char* name;
    int batch;
    int classes;
    float margin;
    float scale;
    bool easy_margin;
};

float clamp_cosine(float cosine) {
    return std::min(1.0f, std::max(-1.0f, cosine));
}

float target_phi(float cosine, float cos_m, float sin_m) {
    const float clamped = clamp_cosine(cosine);
    const float sine = std::sqrt(std::max(0.0f, 1.0f - clamped * clamped));
    return clamped * cos_m - sine * sin_m;
}

float target_phi_grad(float cosine, float cos_m, float sin_m) {
    const float clamped = clamp_cosine(cosine);
    const float sine = std::sqrt(std::max(kGradSineEps, 1.0f - clamped * clamped));
    return cos_m + (clamped / sine) * sin_m;
}

float target_transform(float cosine, float margin, bool easy_margin) {
    const float cos_m = std::cos(margin);
    const float sin_m = std::sin(margin);
    const float threshold = std::cos(kPi - margin);
    const float mm = std::sin(kPi - margin) * margin;
    const float phi = target_phi(cosine, cos_m, sin_m);
    if (easy_margin) {
        return cosine > 0.0f ? phi : cosine;
    }
    return cosine > threshold ? phi : (cosine - mm);
}

float target_grad(float cosine, float margin, bool easy_margin) {
    const float cos_m = std::cos(margin);
    const float sin_m = std::sin(margin);
    const float threshold = std::cos(kPi - margin);
    if (easy_margin) {
        return cosine > 0.0f ? target_phi_grad(cosine, cos_m, sin_m) : 1.0f;
    }
    return cosine > threshold ? target_phi_grad(cosine, cos_m, sin_m) : 1.0f;
}

void cpu_fprop(
    const std::vector<float>& cosine,
    const std::vector<int>& labels,
    std::vector<float>& logits,
    int batch,
    int classes,
    float margin,
    float scale,
    bool easy_margin)
{
    logits.resize(cosine.size());
    for (int row = 0; row < batch; ++row) {
        const int label = labels[row];
        for (int col = 0; col < classes; ++col) {
            const int idx = row * classes + col;
            float out = cosine[idx];
            if (label >= 0 && label < classes && col == label) {
                out = target_transform(cosine[idx], margin, easy_margin);
            }
            logits[idx] = scale * out;
        }
    }
}

void cpu_bprop(
    const std::vector<float>& cosine,
    const std::vector<int>& labels,
    const std::vector<float>& d_logits,
    std::vector<float>& d_cosine,
    int batch,
    int classes,
    float margin,
    float scale,
    bool easy_margin)
{
    d_cosine.resize(cosine.size());
    for (int row = 0; row < batch; ++row) {
        const int label = labels[row];
        for (int col = 0; col < classes; ++col) {
            const int idx = row * classes + col;
            float jacobian = 1.0f;
            if (label >= 0 && label < classes && col == label) {
                jacobian = target_grad(cosine[idx], margin, easy_margin);
            }
            d_cosine[idx] = d_logits[idx] * scale * jacobian;
        }
    }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        diff = std::max(diff, std::fabs(a[i] - b[i]));
    }
    return diff;
}

double objective(
    const std::vector<float>& cosine,
    const std::vector<int>& labels,
    const std::vector<float>& d_logits,
    int batch,
    int classes,
    float margin,
    float scale,
    bool easy_margin)
{
    std::vector<float> logits;
    cpu_fprop(cosine, labels, logits, batch, classes, margin, scale, easy_margin);

    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        sum += static_cast<double>(logits[i]) * static_cast<double>(d_logits[i]);
    }
    return sum;
}

std::vector<float> numeric_grad(
    const std::vector<float>& cosine,
    const std::vector<int>& labels,
    const std::vector<float>& d_logits,
    int batch,
    int classes,
    float margin,
    float scale,
    bool easy_margin)
{
    constexpr float h = 1.0e-3f;
    std::vector<float> out(cosine.size(), 0.0f);
    std::vector<float> perturbed = cosine;
    for (size_t i = 0; i < cosine.size(); ++i) {
        perturbed[i] = cosine[i] + h;
        const double pos = objective(perturbed, labels, d_logits, batch, classes, margin, scale, easy_margin);
        perturbed[i] = cosine[i] - h;
        const double neg = objective(perturbed, labels, d_logits, batch, classes, margin, scale, easy_margin);
        perturbed[i] = cosine[i];
        out[i] = static_cast<float>((pos - neg) / (2.0 * static_cast<double>(h)));
    }
    return out;
}

void fill_random(std::vector<float>& values, std::mt19937& gen, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (float& v : values) {
        v = dist(gen);
    }
}

void populate_case_inputs(
    const Case& tc,
    std::vector<float>& cosine,
    std::vector<int>& labels,
    std::vector<float>& d_logits)
{
    const size_t total = static_cast<size_t>(tc.batch) * tc.classes;
    cosine.resize(total);
    labels.resize(tc.batch);
    d_logits.resize(total);

    std::mt19937 gen(1234 + tc.batch * 17 + tc.classes);
    fill_random(cosine, gen, -0.85f, 0.85f);
    fill_random(d_logits, gen, -0.7f, 0.7f);

    for (int row = 0; row < tc.batch; ++row) {
        labels[row] = (row * 3 + 1) % tc.classes;
    }

    if (std::string(tc.name) == "standard_fallback") {
        for (int row = 0; row < tc.batch; ++row) {
            cosine[row * tc.classes + labels[row]] = -0.97f + 0.01f * row;
        }
    }

    if (std::string(tc.name) == "easy_margin") {
        cosine[0 * tc.classes + labels[0]] = 0.65f;
        cosine[1 * tc.classes + labels[1]] = -0.35f;
        cosine[2 * tc.classes + labels[2]] = 0.20f;
        cosine[3 * tc.classes + labels[3]] = -0.60f;
    }
}

void run_case(const Case& tc, float fprop_eps, float bprop_eps) {
    std::vector<float> h_cosine;
    std::vector<int> h_labels;
    std::vector<float> h_d_logits;
    populate_case_inputs(tc, h_cosine, h_labels, h_d_logits);

    std::vector<float> h_ref_logits;
    std::vector<float> h_ref_d_cosine;
    cpu_fprop(h_cosine, h_labels, h_ref_logits, tc.batch, tc.classes, tc.margin, tc.scale, tc.easy_margin);
    cpu_bprop(h_cosine, h_labels, h_d_logits, h_ref_d_cosine, tc.batch, tc.classes, tc.margin, tc.scale, tc.easy_margin);
    const std::vector<float> h_num_d_cosine =
        numeric_grad(h_cosine, h_labels, h_d_logits, tc.batch, tc.classes, tc.margin, tc.scale, tc.easy_margin);

    float* d_cosine = nullptr;
    int* d_labels = nullptr;
    float* d_logits = nullptr;
    float* d_d_logits = nullptr;
    float* d_d_cosine = nullptr;

    const size_t total = static_cast<size_t>(tc.batch) * tc.classes;
    CUDA_CHECK(cudaMalloc(&d_cosine, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, tc.batch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_logits, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_logits, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_cosine, total * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_cosine, h_cosine.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), tc.batch * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d_logits, h_d_logits.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    arcface_fprop(d_cosine, d_labels, d_logits, tc.batch, tc.classes, tc.margin, tc.scale, tc.easy_margin ? 1 : 0);
    arcface_bprop(d_cosine, d_labels, d_d_logits, d_d_cosine, tc.batch, tc.classes, tc.margin, tc.scale, tc.easy_margin ? 1 : 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_logits(total);
    std::vector<float> h_d_cosine(total);
    CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits, total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_d_cosine.data(), d_d_cosine, total * sizeof(float), cudaMemcpyDeviceToHost));

    const float fprop_diff = max_abs_diff(h_logits, h_ref_logits);
    const float bprop_diff = max_abs_diff(h_d_cosine, h_ref_d_cosine);
    const float numeric_diff = max_abs_diff(h_d_cosine, h_num_d_cosine);

    CUDA_CHECK(cudaFree(d_d_cosine));
    CUDA_CHECK(cudaFree(d_d_logits));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_cosine));

    if (fprop_diff > fprop_eps || bprop_diff > bprop_eps || numeric_diff > 5.0f * bprop_eps) {
        std::ostringstream oss;
        oss << "case " << tc.name << " failed:"
            << " fprop_diff=" << fprop_diff
            << " bprop_diff=" << bprop_diff
            << " numeric_diff=" << numeric_diff;
        throw std::runtime_error(oss.str());
    }

    std::cout << std::fixed << std::setprecision(6)
              << "[PASS] " << tc.name
              << " batch=" << tc.batch
              << " classes=" << tc.classes
              << " margin=" << tc.margin
              << " scale=" << tc.scale
              << " easy_margin=" << (tc.easy_margin ? "true" : "false")
              << " | diff(fprop/bprop/numeric)="
              << fprop_diff << "/" << bprop_diff << "/" << numeric_diff
              << "\n";
}

}  // namespace

int main() {
    try {
        const std::vector<Case> cases = {
            {"standard_random", 8, 11, 0.50f, 64.0f, false},
            {"standard_fallback", 4, 7, 0.50f, 32.0f, false},
            {"easy_margin", 4, 7, 0.35f, 48.0f, true},
        };

        for (const Case& tc : cases) {
            run_case(tc, 1.0e-5f, 3.0e-4f);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
