#include "cross_entropy.h"
#include "softmax.h"

#include <cmath>
#include <stdexcept>

namespace mycnn {

float cross_entropy_with_logits_cpu(const Tensor &logits, const std::vector<int> &labels) {
    Shape shape = logits.shape();
    if (static_cast<int>(labels.size()) != shape.n) {
        throw std::runtime_error("Label count must match batch size");
    }
    Tensor probs = softmax_forward_cpu(logits);
    double loss = 0.0;
    for (int n = 0; n < shape.n; ++n) {
        int label = labels[n];
        int h = 0;
        int w = 0;
        if (label < 0 || label >= shape.c) {
            throw std::runtime_error("Label index out of range");
        }
        loss -= std::log(std::max(probs(n, h, w, label), 1e-12f));
    }
    return static_cast<float>(loss / shape.n);
}

Tensor cross_entropy_with_logits_backward_cpu(const Tensor &logits, const std::vector<int> &labels) {
    Shape shape = logits.shape();
    Tensor probs = softmax_forward_cpu(logits);
    Tensor grad(shape);

    for (int n = 0; n < shape.n; ++n) {
        int label = labels[n];
        for (int c = 0; c < shape.c; ++c) {
            float target = (c == label) ? 1.0f : 0.0f;
            grad(n, 0, 0, c) = (probs(n, 0, 0, c) - target) / shape.n;
        }
    }
    return grad;
}

} // namespace mycnn
