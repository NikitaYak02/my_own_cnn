#pragma once

#include "tensor.h"
#include <vector>

namespace mycnn {

float cross_entropy_with_logits_cpu(const Tensor &logits, const std::vector<int> &labels);
Tensor cross_entropy_with_logits_backward_cpu(const Tensor &logits, const std::vector<int> &labels);

float cross_entropy_with_logits_cuda(const Tensor &logits, const std::vector<int> &labels);
Tensor cross_entropy_with_logits_backward_cuda(const Tensor &logits, const std::vector<int> &labels);

} // namespace mycnn
