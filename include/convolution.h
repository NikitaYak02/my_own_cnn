#pragma once

#include "tensor.h"
#include <vector>

namespace mycnn {

struct Conv2DParams {
    int kernel_h{3};
    int kernel_w{3};
    int stride_h{1};
    int stride_w{1};
    int padding_h{1};
    int padding_w{1};
    int groups{1};
};

Tensor conv2d_forward_cpu(const Tensor &input, const Tensor &weights, const std::vector<float> &bias,
                          const Conv2DParams &params);
Tensor conv2d_backward_input_cpu(const Tensor &grad_output, const Tensor &weights, const Conv2DParams &params, Shape input_shape);
Tensor conv2d_backward_weights_cpu(const Tensor &grad_output, const Tensor &input, const Conv2DParams &params, Shape weight_shape);

#ifdef __CUDACC__
Tensor conv2d_forward_cuda(const Tensor &input, const Tensor &weights, const std::vector<float> &bias,
                           const Conv2DParams &params);
Tensor conv2d_backward_input_cuda(const Tensor &grad_output, const Tensor &weights, const Conv2DParams &params, Shape input_shape);
Tensor conv2d_backward_weights_cuda(const Tensor &grad_output, const Tensor &input, const Conv2DParams &params, Shape weight_shape);
#endif

Tensor conv2d_forward_cudnn(const Tensor &input, const Tensor &weights, const std::vector<float> &bias,
                            const Conv2DParams &params);

} // namespace mycnn
