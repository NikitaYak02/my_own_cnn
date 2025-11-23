#include <gtest/gtest.h>

#include "cross_entropy.h"

using namespace mycnn;

TEST(CrossEntropyTest, MatchesExpectedLoss) {
    Tensor logits({2, 1, 1, 2});
    logits(0, 0, 0, 0) = 0.0f;
    logits(0, 0, 0, 1) = 0.0f;
    logits(1, 0, 0, 0) = 2.0f;
    logits(1, 0, 0, 1) = -2.0f;
    std::vector<int> labels{0, 1};
    float loss = cross_entropy_with_logits_cpu(logits, labels);
    EXPECT_GT(loss, 0.0f);
}

TEST(CrossEntropyTest, BackwardShapes) {
    Tensor logits({1, 1, 1, 3});
    logits(0, 0, 0, 0) = 0.5f;
    logits(0, 0, 0, 1) = -0.5f;
    logits(0, 0, 0, 2) = 1.0f;
    std::vector<int> labels{2};
    Tensor grad = cross_entropy_with_logits_backward_cpu(logits, labels);
    EXPECT_EQ(grad.shape().c, 3);
}

TEST(CrossEntropyTest, CpuCudaParity) {
    Tensor logits({2, 1, 1, 3});
    logits(0, 0, 0, 0) = 0.1f;
    logits(0, 0, 0, 1) = 0.2f;
    logits(0, 0, 0, 2) = 0.3f;
    logits(1, 0, 0, 0) = -0.4f;
    logits(1, 0, 0, 1) = 0.6f;
    logits(1, 0, 0, 2) = 1.2f;
    std::vector<int> labels{2, 1};

    float cpu_loss = cross_entropy_with_logits_cpu(logits, labels);
    float cuda_loss = cross_entropy_with_logits_cuda(logits, labels);
    EXPECT_NEAR(cpu_loss, cuda_loss, 1e-6);

    Tensor cpu_grad = cross_entropy_with_logits_backward_cpu(logits, labels);
    Tensor cuda_grad = cross_entropy_with_logits_backward_cuda(logits, labels);
    for (int idx = 0; idx < cpu_grad.num_elements(); ++idx) {
        EXPECT_NEAR(cpu_grad.data()[idx], cuda_grad.data()[idx], 1e-6);
    }
}

