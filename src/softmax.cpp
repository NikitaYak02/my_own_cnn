#include "softmax.h"

#include <cmath>

namespace mycnn {

Tensor softmax_forward_cpu(const Tensor &input) {
    Shape shape = input.shape();
    Tensor output(shape);

    for (int n = 0; n < shape.n; ++n) {
        for (int h = 0; h < shape.h; ++h) {
            for (int w = 0; w < shape.w; ++w) {
                float max_val = -1e30f;
                for (int c = 0; c < shape.c; ++c) {
                    max_val = std::max(max_val, input(n, h, w, c));
                }
                float sum = 0.0f;
                for (int c = 0; c < shape.c; ++c) {
                    float exp_val = std::exp(input(n, h, w, c) - max_val);
                    output(n, h, w, c) = exp_val;
                    sum += exp_val;
                }
                for (int c = 0; c < shape.c; ++c) {
                    output(n, h, w, c) /= sum;
                }
            }
        }
    }
    return output;
}

Tensor softmax_backward_cpu(const Tensor &grad_output, const Tensor &softmax_output) {
    Shape shape = grad_output.shape();
    Tensor grad_input(shape);

    for (int n = 0; n < shape.n; ++n) {
        for (int h = 0; h < shape.h; ++h) {
            for (int w = 0; w < shape.w; ++w) {
                float dot = 0.0f;
                for (int c = 0; c < shape.c; ++c) {
                    dot += grad_output(n, h, w, c) * softmax_output(n, h, w, c);
                }
                for (int c = 0; c < shape.c; ++c) {
                    grad_input(n, h, w, c) = softmax_output(n, h, w, c) * (grad_output(n, h, w, c) - dot);
                }
            }
        }
    }
    return grad_input;
}

} // namespace mycnn
