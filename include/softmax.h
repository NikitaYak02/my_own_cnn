#pragma once

#include "tensor.h"

namespace mycnn {

Tensor softmax_forward_cpu(const Tensor &input);
Tensor softmax_backward_cpu(const Tensor &grad_output, const Tensor &softmax_output);

Tensor softmax_forward_cuda(const Tensor &input);
Tensor softmax_backward_cuda(const Tensor &grad_output, const Tensor &softmax_output);

} // namespace mycnn
