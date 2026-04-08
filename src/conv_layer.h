#pragma once

#include <cstddef>

#include "conv_types.h"

class Conv2DLayer {
 public:
  Conv2DLayer(int n, int h, int w, int c,
              const FilterKRSC& weights,
              const Conv2DParams& params,
              GradKernelAlgo grad_algo = GradKernelAlgo::GemmIm2Col);

  Conv2DLayer(int n, int h, int w, int c,
              const BlockFilterKByBxRSC& weights,
              const BlockConv2DParams& params,
              GradKernelAlgo grad_algo = GradKernelAlgo::GemmIm2Col);

  ~Conv2DLayer();

  Conv2DLayer(const Conv2DLayer&) = delete;
  Conv2DLayer& operator=(const Conv2DLayer&) = delete;
  Conv2DLayer(Conv2DLayer&&) = delete;
  Conv2DLayer& operator=(Conv2DLayer&&) = delete;

  bool blocked() const { return blocked_; }
  int batch() const { return n_; }
  int input_h() const { return h_; }
  int input_w() const { return w_; }
  int input_c() const { return c_; }
  int output_h() const { return out_h_; }
  int output_w() const { return out_w_; }
  int output_c() const { return k_; }
  int filter_h() const { return r_; }
  int filter_w() const { return s_; }
  size_t input_elements() const;
  size_t output_elements() const;
  size_t weight_elements() const { return weight_elements_; }

  const float* device_weights() const { return d_w_; }
  const float* device_grad() const { return d_dw_; }
  const Conv2DRuntimeConfig* regular_config() const { return blocked_ ? nullptr : &conv_config_; }
  const BlockConv2DRuntimeConfig* blocked_config() const { return blocked_ ? &block_config_ : nullptr; }

  void copy_weights_from(const FilterKRSC& weights);
  void copy_weights_from(const BlockFilterKByBxRSC& weights);
  void copy_weights_to(FilterKRSC& weights) const;
  void copy_weights_to(BlockFilterKByBxRSC& weights) const;
  void copy_grad_to(FilterKRSC& grad) const;
  void copy_grad_to(BlockFilterKByBxRSC& grad) const;

  void forward(const float* d_x, float* d_y) const;
  void backward_input(const float* d_dy, float* d_dx) const;
  void backward_filter(const float* d_x, const float* d_dy);
  void backward(const float* d_x, const float* d_dy, float* d_dx);
  void zero_grad();
  void step(float learning_rate, bool zero_grad_after = true);

 private:
  void allocate_buffers();
  void copy_weights_from_host(const float* src, size_t count);
  void copy_device_array_to_host(float* dst, size_t count, const float* src) const;
  void validate_regular_filter_shape(const FilterKRSC& weights) const;
  void validate_block_filter_shape(const BlockFilterKByBxRSC& weights) const;
  void invalidate_cached_weights() const;
  static void require_non_null(const void* ptr, const char* name);

  bool blocked_ = false;
  Conv2DParams conv_params_;
  BlockConv2DParams block_params_;
  Conv2DRuntimeConfig conv_config_;
  BlockConv2DRuntimeConfig block_config_;
  GradKernelAlgo grad_algo_ = GradKernelAlgo::GemmIm2Col;
  int n_ = 0;
  int h_ = 0;
  int w_ = 0;
  int c_ = 0;
  int out_h_ = 0;
  int out_w_ = 0;
  int r_ = 0;
  int s_ = 0;
  int k_ = 0;
  int cin_per_group_ = 0;
  int block_by_ = 1;
  int block_bx_ = 1;
  int ay_ = 1;
  int ax_ = 1;
  size_t weight_elements_ = 0;
  float* d_w_ = nullptr;
  float* d_dw_ = nullptr;
};
