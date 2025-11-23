#include "batchnorm.h"

namespace mycnn {

Tensor batchnorm_forward_cuda(const Tensor &input, const std::vector<float> &gamma, const std::vector<float> &beta,
                              BatchNormCache &cache, const BatchNormParams &params) {
    return batchnorm_forward_cpu(input, gamma, beta, cache, params);
}

Tensor batchnorm_backward_cuda(const Tensor &grad_output, const Tensor &input, const std::vector<float> &gamma,
                               const BatchNormCache &cache, std::vector<float> &grad_gamma, std::vector<float> &grad_beta,
                               const BatchNormParams &params) {
    return batchnorm_backward_cpu(grad_output, input, gamma, cache, grad_gamma, grad_beta, params);
}

} // namespace mycnn
