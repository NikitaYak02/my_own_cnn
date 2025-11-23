#include <gtest/gtest.h>

#include "pooling.h"

using namespace mycnn;

TEST(PoolingTest, MaxForwardBackward) {
    Tensor input({1, 2, 2, 1});
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 1, 0) = 3.0f;
    input(0, 1, 0, 0) = -2.0f;
    input(0, 1, 1, 0) = 4.0f;

    PoolingParams params{2, 2, 2, 2, 0, 0};
    Tensor output = pooling_forward_cpu(input, PoolingType::Max, params);
    ASSERT_FLOAT_EQ(output(0, 0, 0, 0), 4.0f);

    Tensor grad_out({1, 1, 1, 1});
    grad_out(0, 0, 0, 0) = 1.0f;
    Tensor grad_in = pooling_backward_cpu(input, grad_out, PoolingType::Max, params);
    EXPECT_FLOAT_EQ(grad_in(0, 1, 1, 0), 1.0f);
    EXPECT_FLOAT_EQ(grad_in(0, 0, 0, 0), 0.0f);
}

TEST(PoolingTest, AverageForwardBackward) {
    Tensor input({1, 2, 2, 1});
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 1, 0) = 3.0f;
    input(0, 1, 0, 0) = 5.0f;
    input(0, 1, 1, 0) = 7.0f;

    PoolingParams params{2, 2, 2, 2, 0, 0};
    Tensor output = pooling_forward_cpu(input, PoolingType::Average, params);
    ASSERT_FLOAT_EQ(output(0, 0, 0, 0), 4.0f);

    Tensor grad_out({1, 1, 1, 1});
    grad_out(0, 0, 0, 0) = 1.0f;
    Tensor grad_in = pooling_backward_cpu(input, grad_out, PoolingType::Average, params);
    EXPECT_NEAR(grad_in(0, 0, 0, 0), 0.25f, 1e-6);
    EXPECT_NEAR(grad_in(0, 1, 1, 0), 0.25f, 1e-6);
}

TEST(PoolingTest, GroupedChannels) {
    Tensor input({1, 1, 1, 2});
    input(0, 0, 0, 0) = 5.0f;
    input(0, 0, 0, 1) = 7.0f;

    PoolingParams params{1, 1, 1, 1, 0, 0, 2};
    Tensor output = pooling_forward_cpu(input, PoolingType::Max, params);
    ASSERT_EQ(output.shape().c, 2);
    EXPECT_FLOAT_EQ(output(0, 0, 0, 0), 5.0f);
    EXPECT_FLOAT_EQ(output(0, 0, 0, 1), 7.0f);

    Tensor grad_out({1, 1, 1, 2});
    grad_out(0, 0, 0, 0) = 1.0f;
    grad_out(0, 0, 0, 1) = 1.0f;
    Tensor grad_in = pooling_backward_cpu(input, grad_out, PoolingType::Max, params);
    EXPECT_FLOAT_EQ(grad_in(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(grad_in(0, 0, 0, 1), 1.0f);
}

