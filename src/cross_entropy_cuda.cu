#include "cross_entropy.h"
#include "softmax.h"

namespace mycnn {

float cross_entropy_with_logits_cuda(const Tensor &logits, const std::vector<int> &labels) {
    return cross_entropy_with_logits_cpu(logits, labels);
}

Tensor cross_entropy_with_logits_backward_cuda(const Tensor &logits, const std::vector<int> &labels) {
    return cross_entropy_with_logits_backward_cpu(logits, labels);
}

} // namespace mycnn
