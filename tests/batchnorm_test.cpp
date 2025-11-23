#include <gtest/gtest.h>

#include "batchnorm.h"

using namespace mycnn;

TEST(BatchNormTest, ForwardCentering) {
    Tensor input({1, 1, 2, 2});
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 1, 0) = 3.0f;
    input(0, 0, 0, 1) = 5.0f;
    input(0, 0, 1, 1) = 7.0f;

    std::vector<float> gamma{1.0f, 1.0f};
    std::vector<float> beta{0.0f, 0.0f};
    BatchNormCache cache;
    BatchNormParams params;
    Tensor output = batchnorm_forward_cpu(input, gamma, beta, cache, params);

    EXPECT_NEAR(cache.mean[0], 2.0f, 1e-5);
    EXPECT_NEAR(cache.mean[1], 6.0f, 1e-5);
    EXPECT_NEAR(output(0, 0, 0, 0), (1.0f - 2.0f) / std::sqrt(cache.variance[0] + params.epsilon), 1e-5);
}

TEST(BatchNormTest, BackwardComputesGradients) {
    Tensor input({1, 1, 1, 1});
    input(0, 0, 0, 0) = 2.0f;
    std::vector<float> gamma{1.0f};
    std::vector<float> beta{0.0f};
    BatchNormCache cache;
    BatchNormParams params;
    Tensor output = batchnorm_forward_cpu(input, gamma, beta, cache, params);

    Tensor grad_out({1, 1, 1, 1});
    grad_out(0, 0, 0, 0) = 1.0f;
    std::vector<float> grad_gamma, grad_beta;
    Tensor grad_in = batchnorm_backward_cpu(grad_out, input, gamma, cache, grad_gamma, grad_beta, params);

    EXPECT_NEAR(grad_gamma[0], output(0, 0, 0, 0), 1e-5);
    EXPECT_NEAR(grad_beta[0], 1.0f, 1e-5);
    EXPECT_NEAR(grad_in(0, 0, 0, 0), 1.0f * gamma[0] / std::sqrt(cache.variance[0] + params.epsilon), 1e-5);
}

TEST(BatchNormTest, CpuCudaParity) {
    Tensor input({2, 1, 1, 2});
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 0, 1) = 2.0f;
    input(1, 0, 0, 0) = 3.0f;
    input(1, 0, 0, 1) = 4.0f;

    std::vector<float> gamma{0.5f, 1.5f};
    std::vector<float> beta{0.1f, -0.2f};
    BatchNormParams params;
    BatchNormCache cpu_cache;
    BatchNormCache cuda_cache;
    Tensor cpu_out = batchnorm_forward_cpu(input, gamma, beta, cpu_cache, params);
    Tensor cuda_out = batchnorm_forward_cuda(input, gamma, beta, cuda_cache, params);
    for (int idx = 0; idx < cpu_out.num_elements(); ++idx) {
        EXPECT_NEAR(cpu_out.data()[idx], cuda_out.data()[idx], 1e-6);
    }

    Tensor grad_out({2, 1, 1, 2});
    grad_out(0, 0, 0, 0) = 0.4f;
    grad_out(0, 0, 0, 1) = -0.6f;
    grad_out(1, 0, 0, 0) = 0.8f;
    grad_out(1, 0, 0, 1) = -1.0f;
    std::vector<float> cpu_grad_gamma, cpu_grad_beta;
    std::vector<float> cuda_grad_gamma, cuda_grad_beta;
    Tensor cpu_grad = batchnorm_backward_cpu(grad_out, input, gamma, cpu_cache, cpu_grad_gamma, cpu_grad_beta, params);
    Tensor cuda_grad = batchnorm_backward_cuda(grad_out, input, gamma, cuda_cache, cuda_grad_gamma, cuda_grad_beta, params);
    for (int idx = 0; idx < cpu_grad.num_elements(); ++idx) {
        EXPECT_NEAR(cpu_grad.data()[idx], cuda_grad.data()[idx], 1e-5);
    }
    for (size_t idx = 0; idx < cpu_grad_gamma.size(); ++idx) {
        EXPECT_NEAR(cpu_grad_gamma[idx], cuda_grad_gamma[idx], 1e-5);
        EXPECT_NEAR(cpu_grad_beta[idx], cuda_grad_beta[idx], 1e-5);
    }
}

