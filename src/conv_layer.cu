#include "conv_layer.h"

#include <stdexcept>
#include <string>

#include "cuda_utils.h"

namespace {

__global__ void sgd_update_kernel(float* w, const float* dw, size_t count, float learning_rate) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  w[idx] -= learning_rate * dw[idx];
}

}  // namespace

Conv2DLayer::Conv2DLayer(int n, int h, int w, int c,
                         const FilterKRSC& weights,
                         const Conv2DParams& params,
                         GradKernelAlgo grad_algo)
    : blocked_(false),
      conv_params_(params),
      grad_algo_(grad_algo),
      n_(n),
      h_(h),
      w_(w),
      c_(c),
      r_(weights.r),
      s_(weights.s),
      k_(weights.k),
      cin_per_group_(weights.cin_per_group),
      ay_(weights.ay),
      ax_(weights.ax),
      weight_elements_(weights.elements()) {
  TensorNHWC x_shape(n_, h_, w_, c_);
  const ConvShape sh = infer_conv_shape(x_shape, weights, conv_params_);
  out_h_ = sh.ho;
  out_w_ = sh.wo;
  allocate_buffers();
  copy_weights_from(weights);
}

Conv2DLayer::Conv2DLayer(int n, int h, int w, int c,
                         const BlockFilterKByBxRSC& weights,
                         const BlockConv2DParams& params,
                         GradKernelAlgo grad_algo)
    : blocked_(true),
      block_params_(params),
      grad_algo_(grad_algo),
      n_(n),
      h_(h),
      w_(w),
      c_(c),
      r_(weights.r),
      s_(weights.s),
      k_(weights.k),
      cin_per_group_(weights.cin_per_group),
      block_by_(weights.by),
      block_bx_(weights.bx),
      ay_(weights.ay),
      ax_(weights.ax),
      weight_elements_(weights.elements()) {
  TensorNHWC x_shape(n_, h_, w_, c_);
  const BlockConvShape sh = infer_block_conv_shape(x_shape, weights, block_params_);
  out_h_ = sh.base.ho;
  out_w_ = sh.base.wo;
  allocate_buffers();
  copy_weights_from(weights);
}

Conv2DLayer::~Conv2DLayer() {
  cudaFree(d_w_);
  cudaFree(d_dw_);
}

size_t Conv2DLayer::input_elements() const {
  return static_cast<size_t>(n_) * h_ * w_ * c_;
}

size_t Conv2DLayer::output_elements() const {
  return static_cast<size_t>(n_) * out_h_ * out_w_ * k_;
}

void Conv2DLayer::copy_weights_from(const FilterKRSC& weights) {
  if (blocked_) {
    throw std::runtime_error("copy_weights_from(FilterKRSC) called for blocked Conv2DLayer");
  }
  validate_regular_filter_shape(weights);
  copy_weights_from_host(weights.ptr(), weights.elements());
}

void Conv2DLayer::copy_weights_from(const BlockFilterKByBxRSC& weights) {
  if (!blocked_) {
    throw std::runtime_error("copy_weights_from(BlockFilterKByBxRSC) called for regular Conv2DLayer");
  }
  validate_block_filter_shape(weights);
  copy_weights_from_host(weights.ptr(), weights.elements());
}

void Conv2DLayer::copy_weights_to(FilterKRSC& weights) const {
  if (blocked_) {
    throw std::runtime_error("copy_weights_to(FilterKRSC) called for blocked Conv2DLayer");
  }
  weights = FilterKRSC(r_, s_, cin_per_group_, k_, ay_, ax_);
  copy_device_array_to_host(weights.ptr(), weight_elements_, d_w_);
}

void Conv2DLayer::copy_weights_to(BlockFilterKByBxRSC& weights) const {
  if (!blocked_) {
    throw std::runtime_error("copy_weights_to(BlockFilterKByBxRSC) called for regular Conv2DLayer");
  }
  weights = BlockFilterKByBxRSC(k_, block_by_, block_bx_, r_, s_, cin_per_group_, ay_, ax_);
  copy_device_array_to_host(weights.ptr(), weight_elements_, d_w_);
}

void Conv2DLayer::copy_grad_to(FilterKRSC& grad) const {
  if (blocked_) {
    throw std::runtime_error("copy_grad_to(FilterKRSC) called for blocked Conv2DLayer");
  }
  grad = FilterKRSC(r_, s_, cin_per_group_, k_, ay_, ax_);
  copy_device_array_to_host(grad.ptr(), weight_elements_, d_dw_);
}

void Conv2DLayer::copy_grad_to(BlockFilterKByBxRSC& grad) const {
  if (!blocked_) {
    throw std::runtime_error("copy_grad_to(BlockFilterKByBxRSC) called for regular Conv2DLayer");
  }
  grad = BlockFilterKByBxRSC(k_, block_by_, block_bx_, r_, s_, cin_per_group_, ay_, ax_);
  copy_device_array_to_host(grad.ptr(), weight_elements_, d_dw_);
}

void Conv2DLayer::forward(const float* d_x, float* d_y) const {
  require_non_null(d_x, "d_x");
  require_non_null(d_y, "d_y");
  if (blocked_) {
    launch_block_fprop_nhwc(d_x, d_w_, d_y, n_, h_, w_, c_, r_, s_, k_, block_params_);
  } else {
    launch_fprop_nhwc(d_x, d_w_, d_y, n_, h_, w_, c_, r_, s_, k_, conv_params_);
  }
}

void Conv2DLayer::backward_input(const float* d_dy, float* d_dx) const {
  require_non_null(d_dy, "d_dy");
  require_non_null(d_dx, "d_dx");
  if (blocked_) {
    launch_block_bprop_nhwc(d_dy, d_w_, d_dx, n_, h_, w_, c_, r_, s_, k_, block_params_);
  } else {
    launch_bprop_nhwc(d_dy, d_w_, d_dx, n_, h_, w_, c_, r_, s_, k_, conv_params_);
  }
}

void Conv2DLayer::backward_filter(const float* d_x, const float* d_dy) {
  require_non_null(d_x, "d_x");
  require_non_null(d_dy, "d_dy");
  if (blocked_) {
    launch_block_grad_nhwc(d_x, d_dy, d_dw_, n_, h_, w_, c_, r_, s_, k_, block_params_, grad_algo_);
  } else {
    launch_grad_nhwc(d_x, d_dy, d_dw_, n_, h_, w_, c_, r_, s_, k_, conv_params_, grad_algo_);
  }
}

void Conv2DLayer::backward(const float* d_x, const float* d_dy, float* d_dx) {
  backward_input(d_dy, d_dx);
  backward_filter(d_x, d_dy);
}

void Conv2DLayer::zero_grad() {
  CUDA_CHECK(cudaMemset(d_dw_, 0, weight_elements_ * sizeof(float)));
}

void Conv2DLayer::step(float learning_rate, bool zero_grad_after) {
  const int threads = 256;
  const int blocks = static_cast<int>((weight_elements_ + threads - 1) / threads);
  sgd_update_kernel<<<blocks, threads>>>(d_w_, d_dw_, weight_elements_, learning_rate);
  CUDA_CHECK(cudaGetLastError());
  invalidate_cached_weights();
  if (zero_grad_after) {
    zero_grad();
  }
}

void Conv2DLayer::allocate_buffers() {
  CUDA_CHECK(cudaMalloc(&d_w_, weight_elements_ * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dw_, weight_elements_ * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dw_, 0, weight_elements_ * sizeof(float)));
}

void Conv2DLayer::copy_weights_from_host(const float* src, size_t count) {
  require_non_null(src, "src_weights");
  if (count != weight_elements_) {
    throw std::runtime_error("weight element count mismatch");
  }
  CUDA_CHECK(cudaMemcpy(d_w_, src, count * sizeof(float), cudaMemcpyHostToDevice));
  invalidate_cached_weights();
}

void Conv2DLayer::copy_device_array_to_host(float* dst, size_t count, const float* src) const {
  require_non_null(dst, "dst");
  require_non_null(src, "src");
  CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToHost));
}

void Conv2DLayer::validate_regular_filter_shape(const FilterKRSC& weights) const {
  if (weights.r != r_ || weights.s != s_ || weights.cin_per_group != cin_per_group_ ||
      weights.k != k_ || weights.ay != ay_ || weights.ax != ax_) {
    throw std::runtime_error("regular filter shape mismatch");
  }
}

void Conv2DLayer::validate_block_filter_shape(const BlockFilterKByBxRSC& weights) const {
  if (weights.k != k_ || weights.by != block_by_ || weights.bx != block_bx_ ||
      weights.r != r_ || weights.s != s_ || weights.cin_per_group != cin_per_group_ ||
      weights.ay != ay_ || weights.ax != ax_) {
    throw std::runtime_error("blocked filter shape mismatch");
  }
}

void Conv2DLayer::invalidate_cached_weights() const {
  if (blocked_) {
    invalidate_block_conv_weight_cache(d_w_);
  } else {
    invalidate_conv_weight_cache(d_w_);
  }
}

void Conv2DLayer::require_non_null(const void* ptr, const char* name) {
  if (!ptr) {
    throw std::runtime_error(std::string(name) + " must not be null");
  }
}
