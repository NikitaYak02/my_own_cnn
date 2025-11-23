#include "convolution.h"

#ifdef USE_CUDNN
#include <cudnn.h>
#include <stdexcept>
#endif

namespace mycnn {

Tensor conv2d_forward_cudnn(const Tensor &input, const Tensor &weights, const std::vector<float> &bias,
                            const Conv2DParams &params) {
#ifdef USE_CUDNN
    // Minimal placeholder demonstrating cuDNN dispatch. Full workspace/reformatting omitted for brevity.
    // In practice, weights should be rearranged to NHWC and descriptors configured accordingly.
    (void)params;
    return conv2d_forward_cuda(input, weights, bias, params);
#else
    return conv2d_forward_cpu(input, weights, bias, params);
#endif
}

} // namespace mycnn
