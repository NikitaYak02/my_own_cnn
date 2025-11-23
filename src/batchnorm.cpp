#include "batchnorm.h"

#include <cmath>
#include <numeric>

namespace mycnn {

Tensor batchnorm_forward_cpu(const Tensor &input, const std::vector<float> &gamma, const std::vector<float> &beta,
                              BatchNormCache &cache, const BatchNormParams &params) {
    Shape shape = input.shape();
    Tensor output(shape);
    cache.mean.assign(shape.c, 0.0f);
    cache.variance.assign(shape.c, 0.0f);

    for (int c = 0; c < shape.c; ++c) {
        double sum = 0.0;
        double sq_sum = 0.0;
        for (int n = 0; n < shape.n; ++n) {
            for (int h = 0; h < shape.h; ++h) {
                for (int w = 0; w < shape.w; ++w) {
                    float val = input(n, h, w, c);
                    sum += val;
                    sq_sum += val * val;
                }
            }
        }
        double count = static_cast<double>(shape.n * shape.h * shape.w);
        double mean = sum / count;
        double var = sq_sum / count - mean * mean;
        cache.mean[c] = static_cast<float>(mean);
        cache.variance[c] = static_cast<float>(var);
    }

    for (int n = 0; n < shape.n; ++n) {
        for (int h = 0; h < shape.h; ++h) {
            for (int w = 0; w < shape.w; ++w) {
                for (int c = 0; c < shape.c; ++c) {
                    float normalized = (input(n, h, w, c) - cache.mean[c]) / std::sqrt(cache.variance[c] + params.epsilon);
                    output(n, h, w, c) = gamma[c] * normalized + beta[c];
                }
            }
        }
    }
    return output;
}

Tensor batchnorm_backward_cpu(const Tensor &grad_output, const Tensor &input, const std::vector<float> &gamma,
                               const BatchNormCache &cache, std::vector<float> &grad_gamma, std::vector<float> &grad_beta,
                               const BatchNormParams &params) {
    Shape shape = input.shape();
    grad_gamma.assign(shape.c, 0.0f);
    grad_beta.assign(shape.c, 0.0f);
    Tensor grad_input(shape);
    grad_input.fill(0.0f);

    for (int c = 0; c < shape.c; ++c) {
        double dgamma = 0.0;
        double dbeta = 0.0;
        for (int n = 0; n < shape.n; ++n) {
            for (int h = 0; h < shape.h; ++h) {
                for (int w = 0; w < shape.w; ++w) {
                    float normalized = (input(n, h, w, c) - cache.mean[c]) / std::sqrt(cache.variance[c] + params.epsilon);
                    dgamma += grad_output(n, h, w, c) * normalized;
                    dbeta += grad_output(n, h, w, c);
                }
            }
        }
        grad_gamma[c] = static_cast<float>(dgamma);
        grad_beta[c] = static_cast<float>(dbeta);
    }

    for (int n = 0; n < shape.n; ++n) {
        for (int h = 0; h < shape.h; ++h) {
            for (int w = 0; w < shape.w; ++w) {
                for (int c = 0; c < shape.c; ++c) {
                    float std_inv = 1.0f / std::sqrt(cache.variance[c] + params.epsilon);
                    float x_mu = input(n, h, w, c) - cache.mean[c];
                    float normalized = x_mu * std_inv;
                    float dy = grad_output(n, h, w, c);
                    float count = static_cast<float>(shape.n * shape.h * shape.w);

                    float dvar = -0.5f * dy * gamma[c] * x_mu * std::pow(cache.variance[c] + params.epsilon, -1.5f);
                    float dmean = -dy * gamma[c] * std_inv;

                    grad_input(n, h, w, c) += dy * gamma[c] * std_inv + dvar * 2.0f * x_mu / count + dmean / count;
                }
            }
        }
    }
    return grad_input;
}

} // namespace mycnn
