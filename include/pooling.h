#pragma once

#include "tensor.h"
#include <string>

namespace mycnn {

enum class PoolingType { Max, Average, Median, Min };

struct PoolingParams {
    int kernel_h{2};
    int kernel_w{2};
    int stride_h{2};
    int stride_w{2};
    int padding_h{0};
    int padding_w{0};
    int groups{1};
};

Tensor pooling_forward_cpu(const Tensor &input, PoolingType type, const PoolingParams &params);
Tensor pooling_backward_cpu(const Tensor &input, const Tensor &output_grad, PoolingType type, const PoolingParams &params);

Tensor pooling_forward_cuda(const Tensor &input, PoolingType type, const PoolingParams &params);
Tensor pooling_backward_cuda(const Tensor &input, const Tensor &output_grad, PoolingType type, const PoolingParams &params);

} // namespace mycnn
