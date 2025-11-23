#include "pooling.h"

namespace mycnn {

Tensor pooling_forward_cuda(const Tensor &input, PoolingType type, const PoolingParams &params) {
    // Placeholder calls CPU implementation for correctness-first baseline.
    return pooling_forward_cpu(input, type, params);
}

Tensor pooling_backward_cuda(const Tensor &input, const Tensor &output_grad, PoolingType type, const PoolingParams &params) {
    return pooling_backward_cpu(input, output_grad, type, params);
}

} // namespace mycnn
