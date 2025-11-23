#include <gtest/gtest.h>

#include "convolution.h"

using namespace mycnn;

TEST(ConvolutionTest, ForwardSimple) {
    Tensor input({1, 2, 2, 1});
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 1, 0) = 2.0f;
    input(0, 1, 0, 0) = 3.0f;
    input(0, 1, 1, 0) = 4.0f;

    Tensor weights({1, 2, 2, 1});
    for (int i = 0; i < 4; ++i) {
        weights.data()[i] = 1.0f;
    }
    std::vector<float> bias{0.0f};
    Conv2DParams params{2, 2, 1, 1, 0, 0};
    Tensor output = conv2d_forward_cpu(input, weights, bias, params);
    ASSERT_EQ(output.shape().h, 1);
    ASSERT_EQ(output.shape().w, 1);
    EXPECT_FLOAT_EQ(output(0, 0, 0, 0), 10.0f);
}

TEST(ConvolutionTest, BackwardInputAndWeights) {
    Tensor input({1, 2, 2, 1});
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 1, 0) = 2.0f;
    input(0, 1, 0, 0) = 3.0f;
    input(0, 1, 1, 0) = 4.0f;

    Tensor weights({1, 2, 2, 1});
    for (int i = 0; i < 4; ++i) weights.data()[i] = 1.0f;
    std::vector<float> bias{0.0f};
    Conv2DParams params{2, 2, 1, 1, 0, 0};
    Tensor grad_output({1, 1, 1, 1});
    grad_output(0, 0, 0, 0) = 1.0f;
    Tensor grad_input = conv2d_backward_input_cpu(grad_output, weights, params, input.shape());
    Tensor grad_weights = conv2d_backward_weights_cpu(grad_output, input, params, weights.shape());
    EXPECT_FLOAT_EQ(grad_input(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(grad_weights(0, 0, 0, 0), 1.0f);
}

TEST(ConvolutionTest, GroupedForward) {
    Tensor input({1, 1, 2, 2});
    // Channel 0
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 1, 0) = 2.0f;
    // Channel 1
    input(0, 0, 0, 1) = 3.0f;
    input(0, 0, 1, 1) = 4.0f;

    Tensor weights({2, 1, 1, 1});
    // Group 0 weight (channel 0)
    weights(0, 0, 0, 0) = 1.0f;
    // Group 1 weight (channel 1)
    weights(1, 0, 0, 0) = 2.0f;
    std::vector<float> bias{0.0f, 0.0f};

    Conv2DParams params{1, 1, 1, 1, 0, 0, 2};
    Tensor output = conv2d_forward_cpu(input, weights, bias, params);
    ASSERT_EQ(output.shape().c, 2);
    EXPECT_FLOAT_EQ(output(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(output(0, 0, 1, 0), 2.0f);
    EXPECT_FLOAT_EQ(output(0, 0, 0, 1), 6.0f);
    EXPECT_FLOAT_EQ(output(0, 0, 1, 1), 8.0f);
}

