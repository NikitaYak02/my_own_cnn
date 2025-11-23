#include <gtest/gtest.h>

#include "softmax.h"

using namespace mycnn;

TEST(SoftmaxTest, ForwardSumsToOne) {
    Tensor input({1, 1, 1, 3});
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 0, 1) = 2.0f;
    input(0, 0, 0, 2) = 3.0f;
    Tensor output = softmax_forward_cpu(input);
    float sum = output(0, 0, 0, 0) + output(0, 0, 0, 1) + output(0, 0, 0, 2);
    EXPECT_NEAR(sum, 1.0f, 1e-5);
}

TEST(SoftmaxTest, BackwardMatchesAnalytic) {
    Tensor input({1, 1, 1, 2});
    input(0, 0, 0, 0) = 0.0f;
    input(0, 0, 0, 1) = 0.0f;
    Tensor output = softmax_forward_cpu(input);
    Tensor grad_out({1, 1, 1, 2});
    grad_out(0, 0, 0, 0) = 1.0f;
    grad_out(0, 0, 0, 1) = -1.0f;
    Tensor grad_in = softmax_backward_cpu(grad_out, output);
    EXPECT_NEAR(grad_in(0, 0, 0, 0), 0.25f, 1e-5);
    EXPECT_NEAR(grad_in(0, 0, 0, 1), -0.25f, 1e-5);
}

TEST(SoftmaxTest, CpuAndCudaAgree) {
    Tensor input({1, 1, 1, 3});
    input(0, 0, 0, 0) = -1.0f;
    input(0, 0, 0, 1) = 0.0f;
    input(0, 0, 0, 2) = 1.0f;

    Tensor cpu_out = softmax_forward_cpu(input);
    Tensor cuda_out = softmax_forward_cuda(input);
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(cpu_out(0, 0, 0, c), cuda_out(0, 0, 0, c), 1e-6);
    }

    Tensor grad_out({1, 1, 1, 3});
    grad_out(0, 0, 0, 0) = 0.2f;
    grad_out(0, 0, 0, 1) = -0.1f;
    grad_out(0, 0, 0, 2) = 0.3f;
    Tensor cpu_grad = softmax_backward_cpu(grad_out, cpu_out);
    Tensor cuda_grad = softmax_backward_cuda(grad_out, cuda_out);
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(cpu_grad(0, 0, 0, c), cuda_grad(0, 0, 0, c), 1e-6);
    }
}

