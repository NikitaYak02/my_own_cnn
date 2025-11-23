#include "convolution.h"

#include <algorithm>
#include <stdexcept>

namespace mycnn {

namespace {
int output_size(int input, int kernel, int stride, int padding) {
    return (input + 2 * padding - kernel) / stride + 1;
}
}

Tensor conv2d_forward_cpu(const Tensor &input, const Tensor &weights, const std::vector<float> &bias,
                          const Conv2DParams &params) {
    Shape in_shape = input.shape();
    Shape w_shape = weights.shape();
    if (params.groups <= 0) {
        throw std::runtime_error("groups must be positive");
    }
    if (in_shape.c % params.groups != 0) {
        throw std::runtime_error("Input channels must be divisible by groups");
    }
    if (w_shape.n % params.groups != 0) {
        throw std::runtime_error("Output channels must be divisible by groups");
    }
    int in_c_per_group = in_shape.c / params.groups;
    if (w_shape.c != in_c_per_group) {
        throw std::runtime_error("Weight channels must equal input channels per group");
    }
    int out_c_per_group = w_shape.n / params.groups;
    Shape out_shape{in_shape.n,
                    output_size(in_shape.h, params.kernel_h, params.stride_h, params.padding_h),
                    output_size(in_shape.w, params.kernel_w, params.stride_w, params.padding_w),
                    w_shape.n};
    Tensor output(out_shape);
    for (int n = 0; n < out_shape.n; ++n) {
        for (int oh = 0; oh < out_shape.h; ++oh) {
            for (int ow = 0; ow < out_shape.w; ++ow) {
                for (int oc = 0; oc < out_shape.c; ++oc) {
                    float value = bias[oc];
                    int group = oc / out_c_per_group;
                    int input_c_offset = group * in_c_per_group;
                    for (int kh = 0; kh < params.kernel_h; ++kh) {
                        int ih = oh * params.stride_h + kh - params.padding_h;
                        if (ih < 0 || ih >= in_shape.h) continue;
                        for (int kw = 0; kw < params.kernel_w; ++kw) {
                            int iw = ow * params.stride_w + kw - params.padding_w;
                            if (iw < 0 || iw >= in_shape.w) continue;
                            for (int ic = 0; ic < in_c_per_group; ++ic) {
                                int input_c = input_c_offset + ic;
                                value += input(n, ih, iw, input_c) * weights(oc, kh, kw, ic);
                            }
                        }
                    }
                    output(n, oh, ow, oc) = value;
                }
            }
        }
    }
    return output;
}

Tensor conv2d_backward_input_cpu(const Tensor &grad_output, const Tensor &weights, const Conv2DParams &params, Shape input_shape) {
    Shape w_shape = weights.shape();
    Tensor grad_input(input_shape);
    grad_input.fill(0.0f);

    if (params.groups <= 0) {
        throw std::runtime_error("groups must be positive");
    }
    if (input_shape.c % params.groups != 0) {
        throw std::runtime_error("Input channels must be divisible by groups");
    }
    if (w_shape.n % params.groups != 0) {
        throw std::runtime_error("Output channels must be divisible by groups");
    }
    int in_c_per_group = input_shape.c / params.groups;
    int out_c_per_group = w_shape.n / params.groups;
    if (w_shape.c != in_c_per_group) {
        throw std::runtime_error("Weight channels must equal input channels per group");
    }

    for (int n = 0; n < grad_output.shape().n; ++n) {
        for (int ih = 0; ih < input_shape.h; ++ih) {
            for (int iw = 0; iw < input_shape.w; ++iw) {
                for (int ic = 0; ic < input_shape.c; ++ic) {
                    float grad = 0.0f;
                    int group = ic / in_c_per_group;
                    int out_c_start = group * out_c_per_group;
                    for (int kh = 0; kh < params.kernel_h; ++kh) {
                        int oh = (ih + params.padding_h - kh);
                        if (oh % params.stride_h != 0) continue;
                        oh /= params.stride_h;
                        if (oh < 0 || oh >= grad_output.shape().h) continue;
                        for (int kw = 0; kw < params.kernel_w; ++kw) {
                            int ow = (iw + params.padding_w - kw);
                            if (ow % params.stride_w != 0) continue;
                            ow /= params.stride_w;
                            if (ow < 0 || ow >= grad_output.shape().w) continue;
                            for (int oc = out_c_start; oc < out_c_start + out_c_per_group; ++oc) {
                                int local_ic = ic - group * in_c_per_group;
                                grad += grad_output(n, oh, ow, oc) * weights(oc, kh, kw, local_ic);
                            }
                        }
                    }
                    grad_input(n, ih, iw, ic) = grad;
                }
            }
        }
    }
    return grad_input;
}

Tensor conv2d_backward_weights_cpu(const Tensor &grad_output, const Tensor &input, const Conv2DParams &params, Shape weight_shape) {
    Tensor grad_weights(weight_shape);
    grad_weights.fill(0.0f);

    if (params.groups <= 0) {
        throw std::runtime_error("groups must be positive");
    }
    if (weight_shape.n % params.groups != 0) {
        throw std::runtime_error("Output channels must be divisible by groups");
    }
    if (input.shape().c % params.groups != 0) {
        throw std::runtime_error("Input channels must be divisible by groups");
    }
    if (weight_shape.c * params.groups != input.shape().c) {
        throw std::runtime_error("Weight channels do not match grouped input channels");
    }
    int in_c_per_group = weight_shape.c;
    int out_c_per_group = weight_shape.n / params.groups;

    for (int oc = 0; oc < weight_shape.n; ++oc) {
        int group = oc / out_c_per_group;
        int input_c_offset = group * in_c_per_group;
        for (int kh = 0; kh < weight_shape.h; ++kh) {
            for (int kw = 0; kw < weight_shape.w; ++kw) {
                for (int ic = 0; ic < weight_shape.c; ++ic) {
                    float grad = 0.0f;
                    for (int n = 0; n < input.shape().n; ++n) {
                        for (int oh = 0; oh < grad_output.shape().h; ++oh) {
                            for (int ow = 0; ow < grad_output.shape().w; ++ow) {
                                int ih = oh * params.stride_h + kh - params.padding_h;
                                int iw = ow * params.stride_w + kw - params.padding_w;
                                if (ih < 0 || ih >= input.shape().h || iw < 0 || iw >= input.shape().w) continue;
                                int input_c = input_c_offset + ic;
                                grad += grad_output(n, oh, ow, oc) * input(n, ih, iw, input_c);
                            }
                        }
                    }
                    grad_weights(oc, kh, kw, ic) = grad;
                }
            }
        }
    }
    return grad_weights;
}

} // namespace mycnn
