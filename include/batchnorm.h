#pragma once

#include "tensor.h"
#include <vector>

namespace mycnn {

struct BatchNormCache {
    std::vector<float> mean;
    std::vector<float> variance;
};

struct BatchNormParams {
    float epsilon{1e-5f};
    float momentum{0.1f};
};

Tensor batchnorm_forward_cpu(const Tensor &input, const std::vector<float> &gamma, const std::vector<float> &beta,
                              BatchNormCache &cache, const BatchNormParams &params);

Tensor batchnorm_backward_cpu(const Tensor &grad_output, const Tensor &input, const std::vector<float> &gamma,
                               const BatchNormCache &cache, std::vector<float> &grad_gamma, std::vector<float> &grad_beta,
                               const BatchNormParams &params);

Tensor batchnorm_forward_cuda(const Tensor &input, const std::vector<float> &gamma, const std::vector<float> &beta,
                              BatchNormCache &cache, const BatchNormParams &params);

Tensor batchnorm_backward_cuda(const Tensor &grad_output, const Tensor &input, const std::vector<float> &gamma,
                               const BatchNormCache &cache, std::vector<float> &grad_gamma, std::vector<float> &grad_beta,
                               const BatchNormParams &params);

} // namespace mycnn
