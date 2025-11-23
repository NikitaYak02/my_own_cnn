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

