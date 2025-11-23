#include "softmax.h"

namespace mycnn {

Tensor softmax_forward_cuda(const Tensor &input) { return softmax_forward_cpu(input); }

Tensor softmax_backward_cuda(const Tensor &grad_output, const Tensor &softmax_output) {
    return softmax_backward_cpu(grad_output, softmax_output);
}

} // namespace mycnn
