#include "pooling.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mycnn {

namespace {
int output_size(int input, int kernel, int stride, int padding) {
    return (input + 2 * padding - kernel) / stride + 1;
}
}

Tensor pooling_forward_cpu(const Tensor &input, PoolingType type, const PoolingParams &params) {
    Shape in_shape = input.shape();
    if (params.groups <= 0) {
        throw std::runtime_error("groups must be positive");
    }
    if (in_shape.c % params.groups != 0) {
        throw std::runtime_error("Input channels must be divisible by groups");
    }
    int channels_per_group = in_shape.c / params.groups;
    Shape out_shape{in_shape.n,
                    output_size(in_shape.h, params.kernel_h, params.stride_h, params.padding_h),
                    output_size(in_shape.w, params.kernel_w, params.stride_w, params.padding_w),
                    in_shape.c};
    Tensor output(out_shape);

    for (int n = 0; n < out_shape.n; ++n) {
        for (int oh = 0; oh < out_shape.h; ++oh) {
            for (int ow = 0; ow < out_shape.w; ++ow) {
                for (int c = 0; c < out_shape.c; ++c) {
                    int group = c / channels_per_group;
                    (void)group; // grouping keeps channel partitioned but does not mix values
                    std::vector<float> window;
                    window.reserve(params.kernel_h * params.kernel_w);
                    for (int kh = 0; kh < params.kernel_h; ++kh) {
                        int ih = oh * params.stride_h + kh - params.padding_h;
                        if (ih < 0 || ih >= in_shape.h) continue;
                        for (int kw = 0; kw < params.kernel_w; ++kw) {
                            int iw = ow * params.stride_w + kw - params.padding_w;
                            if (iw < 0 || iw >= in_shape.w) continue;
                            window.push_back(input(n, ih, iw, c));
                        }
                    }
                    float value = 0.0f;
                    if (window.empty()) {
                        output(n, oh, ow, c) = 0.0f;
                        continue;
                    }
                    switch (type) {
                    case PoolingType::Max:
                        value = *std::max_element(window.begin(), window.end());
                        break;
                    case PoolingType::Average:
                        value = std::accumulate(window.begin(), window.end(), 0.0f) / window.size();
                        break;
                    case PoolingType::Median:
                        std::nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
                        value = window[window.size() / 2];
                        break;
                    case PoolingType::Min:
                        value = *std::min_element(window.begin(), window.end());
                        break;
                    }
                    output(n, oh, ow, c) = value;
                }
            }
        }
    }
    return output;
}

Tensor pooling_backward_cpu(const Tensor &input, const Tensor &output_grad, PoolingType type, const PoolingParams &params) {
    Shape in_shape = input.shape();
    Shape out_shape = output_grad.shape();
    if (params.groups <= 0) {
        throw std::runtime_error("groups must be positive");
    }
    if (in_shape.c % params.groups != 0 || out_shape.c % params.groups != 0) {
        throw std::runtime_error("Channels must be divisible by groups");
    }
    int channels_per_group = in_shape.c / params.groups;
    Tensor grad_input(in_shape);
    grad_input.fill(0.0f);

    for (int n = 0; n < out_shape.n; ++n) {
        for (int oh = 0; oh < out_shape.h; ++oh) {
            for (int ow = 0; ow < out_shape.w; ++ow) {
                for (int c = 0; c < out_shape.c; ++c) {
                    int group = c / channels_per_group;
                    (void)group;
                    float grad = output_grad(n, oh, ow, c);
                    std::vector<std::pair<float, std::pair<int, int>>> window;
                    for (int kh = 0; kh < params.kernel_h; ++kh) {
                        int ih = oh * params.stride_h + kh - params.padding_h;
                        if (ih < 0 || ih >= in_shape.h) continue;
                        for (int kw = 0; kw < params.kernel_w; ++kw) {
                            int iw = ow * params.stride_w + kw - params.padding_w;
                            if (iw < 0 || iw >= in_shape.w) continue;
                            window.push_back({input(n, ih, iw, c), {ih, iw}});
                        }
                    }
                    if (window.empty()) continue;
                    if (type == PoolingType::Average) {
                        float share = grad / static_cast<float>(window.size());
                        for (auto &entry : window) {
                            grad_input(n, entry.second.first, entry.second.second, c) += share;
                        }
                    } else {
                        auto cmp = [](const auto &a, const auto &b) { return a.first < b.first; };
                        std::pair<float, std::pair<int, int>> target = *std::min_element(window.begin(), window.end(), cmp);
                        if (type == PoolingType::Max) {
                            target = *std::max_element(window.begin(), window.end(), cmp);
                        } else if (type == PoolingType::Median) {
                            std::nth_element(window.begin(), window.begin() + window.size() / 2, window.end(), cmp);
                            target = window[window.size() / 2];
                        }
                        grad_input(n, target.second.first, target.second.second, c) += grad;
                    }
                }
            }
        }
    }
    return grad_input;
}

} // namespace mycnn
