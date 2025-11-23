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

