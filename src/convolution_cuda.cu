#include "convolution.h"

namespace mycnn {

Tensor conv2d_forward_cuda(const Tensor &input, const Tensor &weights, const std::vector<float> &bias,
                           const Conv2DParams &params) {
    // Baseline uses CPU path; CUDA kernels can be specialized for NHWC.
    return conv2d_forward_cpu(input, weights, bias, params);
}

Tensor conv2d_backward_input_cuda(const Tensor &grad_output, const Tensor &weights, const Conv2DParams &params, Shape input_shape) {
    return conv2d_backward_input_cpu(grad_output, weights, params, input_shape);
}

Tensor conv2d_backward_weights_cuda(const Tensor &grad_output, const Tensor &input, const Conv2DParams &params, Shape weight_shape) {
    return conv2d_backward_weights_cpu(grad_output, input, params, weight_shape);
}

} // namespace mycnn
