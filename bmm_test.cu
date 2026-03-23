#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bmm.h"

#define CUDA_CHECK(call) do {                                           \
    cudaError_t e = (call);                                             \
    if (e != cudaSuccess) {                                             \
        throw std::runtime_error(std::string("CUDA error: ") +          \
                                 cudaGetErrorString(e));                \
    }                                                                   \
} while (0)

#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t s = (call);                                          \
    if (s != CUBLAS_STATUS_SUCCESS) {                                   \
        throw std::runtime_error("cuBLAS call failed");                \
    }                                                                   \
} while (0)

namespace {

// Test dimensions reuse the project naming convention:
//   N - batch, W - left rows, H - right rows before transpose, C - reduction dim.
struct Case {
    const char* name;
    int N;
    int W;
    int H;
    int C;
};

struct Timing {
    float custom_ms = 0.0f;
    float cublas_ms = 0.0f;
};

struct CaseResult {
    float fprop_diff = 0.0f;
    float dA_diff = 0.0f;
    float dB_diff = 0.0f;
    Timing fprop;
    Timing bprop_dA;
    Timing bprop_dB;
};

// Cover both tiny edge cases and the production-like shapes used in the repo.
std::vector<Case> default_cases() {
    return {
        {"tiny_1x1x1x1", 1, 1, 1, 1},
        {"irregular_17x19x13", 1, 17, 19, 13},
        {"shape_128x32x16", 1, 128, 32, 16},
        {"default_64x64x128", 32, 64, 64, 128},
        {"wide_128x128x256", 32, 128, 128, 256},
        {"tall_256x64x256", 8, 256, 64, 256},
    };
}

void fill_random(std::vector<float>& v, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (float& x : v) {
        x = dist(gen);
    }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float mx = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        mx = std::max(mx, std::fabs(a[i] - b[i]));
    }
    return mx;
}

/*
 * cuBLAS expects column-major matrices, while the project stores everything in
 * row-major form. These wrappers encode the standard row-major-to-column-major
 * mapping so the rest of the test can compare logical operations directly.
 */
static void cublas_bmm_fprop(
    cublasHandle_t handle,
    const float* A, const float* B, float* Y,
    int N, int W, int H, int C)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        H, W, C,
        &alpha,
        B, C, static_cast<long long>(H) * C,
        A, C, static_cast<long long>(W) * C,
        &beta,
        Y, H, static_cast<long long>(W) * H,
        N));
}

static void cublas_bmm_bprop_dA(
    cublasHandle_t handle,
    const float* dY, const float* B, float* dA,
    int N, int W, int H, int C)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        C, W, H,
        &alpha,
        B, C, static_cast<long long>(H) * C,
        dY, H, static_cast<long long>(W) * H,
        &beta,
        dA, C, static_cast<long long>(W) * C,
        N));
}

static void cublas_bmm_bprop_dB(
    cublasHandle_t handle,
    const float* dY, const float* A, float* dB,
    int N, int W, int H, int C)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        C, H, W,
        &alpha,
        A, C, static_cast<long long>(W) * C,
        dY, H, static_cast<long long>(W) * H,
        &beta,
        dB, C, static_cast<long long>(H) * C,
        N));
}

// Time a device operation after an explicit warmup and synchronization phase.
template <typename Fn>
float time_op(Fn&& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / static_cast<float>(iters);
}

// Run one logical BMM case end-to-end: allocate buffers, compare against
// cuBLAS for correctness, then measure custom and library timings.
CaseResult run_case(const Case& tc, int warmup, int iters, float eps) {
    const size_t a_elems = static_cast<size_t>(tc.N) * tc.W * tc.C;
    const size_t b_elems = static_cast<size_t>(tc.N) * tc.H * tc.C;
    const size_t y_elems = static_cast<size_t>(tc.N) * tc.W * tc.H;

    std::vector<float> hA(a_elems);
    std::vector<float> hB(b_elems);
    std::vector<float> hDY(y_elems);
    std::vector<float> hCustomY(y_elems);
    std::vector<float> hCublasY(y_elems);
    std::vector<float> hCustomDA(a_elems);
    std::vector<float> hCublasDA(a_elems);
    std::vector<float> hCustomDB(b_elems);
    std::vector<float> hCublasDB(b_elems);

    std::mt19937 gen(42);
    fill_random(hA, gen);
    fill_random(hB, gen);
    fill_random(hDY, gen);

    float* dA = nullptr;
    float* dB = nullptr;
    float* dDY = nullptr;
    float* dCustomY = nullptr;
    float* dCublasY = nullptr;
    float* dCustomDA = nullptr;
    float* dCublasDA = nullptr;
    float* dCustomDB = nullptr;
    float* dCublasDB = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, a_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDY, y_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCustomY, y_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCublasY, y_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCustomDA, a_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCublasDA, a_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCustomDB, b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCublasDB, b_elems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), a_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), b_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDY, hDY.data(), y_elems * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    // These lambdas mirror the three logical operations used by the project:
    // forward output, gradient w.r.t. the left operand, and gradient w.r.t.
    // the right operand.
    auto custom_fprop = [&]() {
        bmm_fprop(dA, dB, dCustomY, tc.N, tc.W, tc.H, tc.C);
    };
    auto ref_fprop = [&]() {
        cublas_bmm_fprop(handle, dA, dB, dCublasY, tc.N, tc.W, tc.H, tc.C);
    };
    auto custom_dA = [&]() {
        bmm_bprop_dA(dDY, dB, dCustomDA, tc.N, tc.W, tc.H, tc.C);
    };
    auto ref_dA = [&]() {
        cublas_bmm_bprop_dA(handle, dDY, dB, dCublasDA, tc.N, tc.W, tc.H, tc.C);
    };
    auto custom_dB = [&]() {
        bmm_bprop_dB(dDY, dA, dCustomDB, tc.N, tc.W, tc.H, tc.C);
    };
    auto ref_dB = [&]() {
        cublas_bmm_bprop_dB(handle, dDY, dA, dCublasDB, tc.N, tc.W, tc.H, tc.C);
    };

    custom_fprop();
    ref_fprop();
    custom_dA();
    ref_dA();
    custom_dB();
    ref_dB();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hCustomY.data(), dCustomY, y_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hCublasY.data(), dCublasY, y_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hCustomDA.data(), dCustomDA, a_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hCublasDA.data(), dCublasDA, a_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hCustomDB.data(), dCustomDB, b_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hCublasDB.data(), dCublasDB, b_elems * sizeof(float), cudaMemcpyDeviceToHost));

    CaseResult result;
    result.fprop_diff = max_abs_diff(hCustomY, hCublasY);
    result.dA_diff = max_abs_diff(hCustomDA, hCublasDA);
    result.dB_diff = max_abs_diff(hCustomDB, hCublasDB);

    if (result.fprop_diff > eps || result.dA_diff > eps || result.dB_diff > eps) {
        std::ostringstream oss;
        oss << "case " << tc.name << " failed:"
            << " fprop_diff=" << result.fprop_diff
            << " dA_diff=" << result.dA_diff
            << " dB_diff=" << result.dB_diff;
        throw std::runtime_error(oss.str());
    }

    // Timings are collected only after correctness passes so the benchmark
    // output always corresponds to a validated implementation.
    result.fprop.custom_ms = time_op(custom_fprop, warmup, iters);
    result.fprop.cublas_ms = time_op(ref_fprop, warmup, iters);
    result.bprop_dA.custom_ms = time_op(custom_dA, warmup, iters);
    result.bprop_dA.cublas_ms = time_op(ref_dA, warmup, iters);
    result.bprop_dB.custom_ms = time_op(custom_dB, warmup, iters);
    result.bprop_dB.cublas_ms = time_op(ref_dB, warmup, iters);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dCublasDB));
    CUDA_CHECK(cudaFree(dCustomDB));
    CUDA_CHECK(cudaFree(dCublasDA));
    CUDA_CHECK(cudaFree(dCustomDA));
    CUDA_CHECK(cudaFree(dCublasY));
    CUDA_CHECK(cudaFree(dCustomY));
    CUDA_CHECK(cudaFree(dDY));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));

    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string only_case;
        int warmup = 5;
        int iters = 20;
        float eps = 1.0e-4f;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--case" && i + 1 < argc) {
                only_case = argv[++i];
            } else if (arg == "--warmup" && i + 1 < argc) {
                warmup = std::stoi(argv[++i]);
            } else if (arg == "--iters" && i + 1 < argc) {
                iters = std::stoi(argv[++i]);
            } else if (arg == "--eps" && i + 1 < argc) {
                eps = std::stof(argv[++i]);
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        const std::vector<Case> cases = default_cases();
        bool matched = only_case.empty();

        // Keep output machine-readable enough for CI logs while still showing
        // the key correctness and timing numbers per case.
        std::cout << std::fixed << std::setprecision(3);
        for (const Case& tc : cases) {
            if (!only_case.empty() && only_case != tc.name) {
                continue;
            }
            matched = true;

            const CaseResult r = run_case(tc, warmup, iters, eps);
            std::cout
                << "[PASS] " << tc.name
                << " N=" << tc.N << " W=" << tc.W << " H=" << tc.H << " C=" << tc.C
                << " | diff(fprop/dA/dB)="
                << r.fprop_diff << "/" << r.dA_diff << "/" << r.dB_diff
                << " | fprop " << r.fprop.custom_ms << " ms vs " << r.fprop.cublas_ms << " ms"
                << " | dA " << r.bprop_dA.custom_ms << " ms vs " << r.bprop_dA.cublas_ms << " ms"
                << " | dB " << r.bprop_dB.custom_ms << " ms vs " << r.bprop_dB.cublas_ms << " ms"
                << "\n";
        }

        if (!matched) {
            throw std::runtime_error("requested case was not found");
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
