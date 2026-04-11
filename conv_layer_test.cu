#include "src/conv_layer.h"
#include "src/cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;

  ~DeviceBuffer() {
    cudaFree(ptr);
  }

  void allocate(size_t count) {
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
  }
};

void fill_pattern(std::vector<float>& data, int salt) {
  for (size_t i = 0; i < data.size(); ++i) {
    const int code = static_cast<int>((i * 97 + static_cast<size_t>(salt) * 31) % 521);
    data[i] = static_cast<float>(code - 260) / 128.0f;
  }
}

void expect_close(const std::vector<float>& ref,
                  const std::vector<float>& got,
                  float abs_eps,
                  float rel_eps,
                  const std::string& label) {
  const VerifyResult vr = verify_tensors(ref, got, abs_eps, rel_eps);
  if (vr.passed) return;

  std::ostringstream oss;
  oss << label
      << " max_abs=" << vr.max_abs_err
      << " max_rel=" << vr.max_rel_err
      << " abs_idx=" << vr.max_abs_idx
      << " rel_idx=" << vr.max_rel_idx;
  throw std::runtime_error(oss.str());
}

std::vector<float> copy_from_device(const float* d_ptr, size_t count) {
  std::vector<float> out(count);
  CUDA_CHECK(cudaMemcpy(out.data(), d_ptr, count * sizeof(float), cudaMemcpyDeviceToHost));
  return out;
}

void apply_sgd(FilterKRSC& weights, const FilterKRSC& grad, float lr) {
  for (size_t i = 0; i < weights.elements(); ++i) {
    weights.data[i] -= lr * grad.data[i];
  }
}

void apply_sgd(BlockFilterKByBxRSC& weights, const BlockFilterKByBxRSC& grad, float lr) {
  for (size_t i = 0; i < weights.elements(); ++i) {
    weights.data[i] -= lr * grad.data[i];
  }
}

void run_regular_case() {
  const int n = 2;
  const int h = 7;
  const int w = 6;
  const int c = 4;

  Conv2DParams p;
  p.pad_h = 1;
  p.pad_w = 0;
  p.stride_h = 1;
  p.stride_w = 2;
  p.dilation_h = 1;
  p.dilation_w = 1;
  p.groups = 1;

  TensorNHWC x(n, h, w, c);
  FilterKRSC weights(3, 2, c, 5);
  fill_pattern(x.data, 7);
  fill_pattern(weights.data, 101);

  const ConvShape sh = infer_conv_shape(x, weights, p);
  TensorNHWC dy(n, sh.ho, sh.wo, weights.k);
  fill_pattern(dy.data, 211);

  TensorNHWC y_ref;
  TensorNHWC dx_ref(n, h, w, c);
  FilterKRSC dw_ref(weights.r, weights.s, weights.cin_per_group, weights.k, weights.ay, weights.ax);
  cpu_fprop_nhwc(x, weights, p, y_ref);
  cpu_bprop_nhwc(dy, weights, p, dx_ref);
  cpu_grad_nhwc(x, dy, p, dw_ref);

  Conv2DLayer layer(n, h, w, c, weights, p);
  if (!layer.regular_config() || layer.blocked_config()) {
    throw std::runtime_error("regular layer config exposure is invalid");
  }
  if (layer.regular_config()->shape.ho != y_ref.h ||
      layer.regular_config()->shape.wo != y_ref.w ||
      layer.regular_config()->output_elements != y_ref.elements()) {
    throw std::runtime_error("regular runtime config mismatch");
  }

  DeviceBuffer<float> d_x;
  DeviceBuffer<float> d_dy;
  DeviceBuffer<float> d_y;
  DeviceBuffer<float> d_dx;
  d_x.allocate(x.elements());
  d_dy.allocate(dy.elements());
  d_y.allocate(y_ref.elements());
  d_dx.allocate(x.elements());

  CUDA_CHECK(cudaMemcpy(d_x.ptr, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dy.ptr, dy.ptr(), dy.elements() * sizeof(float), cudaMemcpyHostToDevice));

  layer.forward(d_x.ptr, d_y.ptr);
  expect_close(y_ref.data, copy_from_device(d_y.ptr, y_ref.elements()), 1e-4f, 1e-3f, "regular fprop");

  layer.backward(d_x.ptr, d_dy.ptr, d_dx.ptr);
  expect_close(dx_ref.data, copy_from_device(d_dx.ptr, dx_ref.elements()), 1e-4f, 1e-3f, "regular bprop");

  FilterKRSC dw_gpu;
  layer.copy_grad_to(dw_gpu);
  expect_close(dw_ref.data, dw_gpu.data, 1e-4f, 1e-3f, "regular grad");

  const float lr = 0.05f;
  FilterKRSC updated_weights = weights;
  apply_sgd(updated_weights, dw_ref, lr);

  layer.step(lr);
  FilterKRSC updated_gpu_weights;
  layer.copy_weights_to(updated_gpu_weights);
  expect_close(updated_weights.data, updated_gpu_weights.data, 1e-6f, 1e-6f, "regular step");
  FilterKRSC zero_grad(weights.r, weights.s, weights.cin_per_group, weights.k, weights.ay, weights.ax);
  layer.copy_grad_to(zero_grad);
  expect_close(std::vector<float>(zero_grad.elements(), 0.0f), zero_grad.data, 1e-7f, 1e-7f, "regular zero grad after step");

  TensorNHWC updated_y_ref;
  TensorNHWC updated_dx_ref(n, h, w, c);
  cpu_fprop_nhwc(x, updated_weights, p, updated_y_ref);
  cpu_bprop_nhwc(dy, updated_weights, p, updated_dx_ref);

  layer.forward(d_x.ptr, d_y.ptr);
  expect_close(updated_y_ref.data, copy_from_device(d_y.ptr, updated_y_ref.elements()), 1e-4f, 1e-3f, "regular fprop after step");

  layer.backward_input(d_dy.ptr, d_dx.ptr);
  expect_close(updated_dx_ref.data, copy_from_device(d_dx.ptr, updated_dx_ref.elements()), 1e-4f, 1e-3f, "regular bprop after step");
}

void run_regular_ayax_case() {
  const int n = 2;
  const int h = 7;
  const int w = 6;
  const int c = 4;

  Conv2DParams p;
  p.pad_h = 1;
  p.pad_w = 0;
  p.stride_h = 1;
  p.stride_w = 2;
  p.dilation_h = 1;
  p.dilation_w = 1;
  p.groups = 1;
  p.ay = 2;
  p.ax = 3;

  TensorNHWC x(n, h, w, c);
  FilterKRSC weights(3, 2, c, 5, p.ay, p.ax);
  fill_pattern(x.data, 23);
  fill_pattern(weights.data, 509);

  const ConvShape sh = infer_conv_shape(x, weights, p);
  TensorNHWC dy(n, sh.ho, sh.wo, weights.k);
  fill_pattern(dy.data, 601);

  TensorNHWC y_ref;
  TensorNHWC dx_ref(n, h, w, c);
  FilterKRSC dw_ref(weights.r, weights.s, weights.cin_per_group, weights.k, weights.ay, weights.ax);
  cpu_fprop_nhwc(x, weights, p, y_ref);
  cpu_bprop_nhwc(dy, weights, p, dx_ref);
  cpu_grad_nhwc(x, dy, p, dw_ref);

  Conv2DLayer layer(n, h, w, c, weights, p);
  if (!layer.regular_config() || layer.blocked_config()) {
    throw std::runtime_error("regular ay/ax layer config exposure is invalid");
  }
  if (layer.regular_config()->shape.ho != y_ref.h ||
      layer.regular_config()->shape.wo != y_ref.w ||
      layer.regular_config()->shape.ay != p.ay ||
      layer.regular_config()->shape.ax != p.ax ||
      layer.regular_config()->output_elements != y_ref.elements()) {
    throw std::runtime_error("regular ay/ax runtime config mismatch");
  }

  DeviceBuffer<float> d_x;
  DeviceBuffer<float> d_dy;
  DeviceBuffer<float> d_y;
  DeviceBuffer<float> d_dx;
  d_x.allocate(x.elements());
  d_dy.allocate(dy.elements());
  d_y.allocate(y_ref.elements());
  d_dx.allocate(x.elements());

  CUDA_CHECK(cudaMemcpy(d_x.ptr, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dy.ptr, dy.ptr(), dy.elements() * sizeof(float), cudaMemcpyHostToDevice));

  layer.forward(d_x.ptr, d_y.ptr);
  expect_close(y_ref.data, copy_from_device(d_y.ptr, y_ref.elements()), 1e-4f, 1e-3f, "regular ay/ax fprop");

  layer.backward(d_x.ptr, d_dy.ptr, d_dx.ptr);
  expect_close(dx_ref.data, copy_from_device(d_dx.ptr, dx_ref.elements()), 1e-4f, 1e-3f, "regular ay/ax bprop");

  FilterKRSC dw_gpu;
  layer.copy_grad_to(dw_gpu);
  expect_close(dw_ref.data, dw_gpu.data, 1e-4f, 1e-3f, "regular ay/ax grad");

  const float lr = 0.04f;
  FilterKRSC updated_weights = weights;
  apply_sgd(updated_weights, dw_ref, lr);

  layer.step(lr);
  FilterKRSC updated_gpu_weights;
  layer.copy_weights_to(updated_gpu_weights);
  expect_close(updated_weights.data, updated_gpu_weights.data, 1e-6f, 1e-6f, "regular ay/ax step");

  TensorNHWC updated_y_ref;
  TensorNHWC updated_dx_ref(n, h, w, c);
  cpu_fprop_nhwc(x, updated_weights, p, updated_y_ref);
  cpu_bprop_nhwc(dy, updated_weights, p, updated_dx_ref);

  layer.forward(d_x.ptr, d_y.ptr);
  expect_close(updated_y_ref.data, copy_from_device(d_y.ptr, updated_y_ref.elements()), 1e-4f, 1e-3f, "regular ay/ax fprop after step");

  layer.backward_input(d_dy.ptr, d_dx.ptr);
  expect_close(updated_dx_ref.data, copy_from_device(d_dx.ptr, updated_dx_ref.elements()), 1e-4f, 1e-3f, "regular ay/ax bprop after step");
}

void run_blocked_case() {
  const int n = 2;
  const int h = 8;
  const int w = 8;
  const int c = 3;

  BlockConv2DParams p;
  p.conv.pad_h = 1;
  p.conv.pad_w = 1;
  p.conv.stride_h = 1;
  p.conv.stride_w = 1;
  p.conv.dilation_h = 1;
  p.conv.dilation_w = 1;
  p.conv.groups = 1;
  p.block_by = 2;
  p.block_bx = 2;

  TensorNHWC x(n, h, w, c);
  BlockFilterKByBxRSC weights(4, p.block_by, p.block_bx, 3, 3, c);
  fill_pattern(x.data, 17);
  fill_pattern(weights.data, 313);

  const BlockConvShape sh = infer_block_conv_shape(x, weights, p);
  TensorNHWC dy(n, sh.base.ho, sh.base.wo, weights.k);
  fill_pattern(dy.data, 419);

  TensorNHWC y_ref;
  TensorNHWC dx_ref(n, h, w, c);
  BlockFilterKByBxRSC dw_ref(weights.k, weights.by, weights.bx, weights.r, weights.s,
                             weights.cin_per_group, weights.ay, weights.ax);
  cpu_block_fprop_nhwc(x, weights, p, y_ref);
  cpu_block_bprop_nhwc(dy, weights, p, dx_ref);
  cpu_block_grad_nhwc(x, dy, p, dw_ref);

  Conv2DLayer layer(n, h, w, c, weights, p);
  if (!layer.blocked_config() || layer.regular_config()) {
    throw std::runtime_error("blocked layer config exposure is invalid");
  }
  if (layer.blocked_config()->shape.base.ho != y_ref.h ||
      layer.blocked_config()->shape.base.wo != y_ref.w ||
      layer.blocked_config()->output_elements != y_ref.elements()) {
    throw std::runtime_error("blocked runtime config mismatch");
  }

  DeviceBuffer<float> d_x;
  DeviceBuffer<float> d_dy;
  DeviceBuffer<float> d_y;
  DeviceBuffer<float> d_dx;
  d_x.allocate(x.elements());
  d_dy.allocate(dy.elements());
  d_y.allocate(y_ref.elements());
  d_dx.allocate(x.elements());

  CUDA_CHECK(cudaMemcpy(d_x.ptr, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dy.ptr, dy.ptr(), dy.elements() * sizeof(float), cudaMemcpyHostToDevice));

  layer.forward(d_x.ptr, d_y.ptr);
  expect_close(y_ref.data, copy_from_device(d_y.ptr, y_ref.elements()), 1e-4f, 1e-3f, "blocked fprop");

  layer.backward(d_x.ptr, d_dy.ptr, d_dx.ptr);
  expect_close(dx_ref.data, copy_from_device(d_dx.ptr, dx_ref.elements()), 1e-4f, 1e-3f, "blocked bprop");

  BlockFilterKByBxRSC dw_gpu;
  layer.copy_grad_to(dw_gpu);
  expect_close(dw_ref.data, dw_gpu.data, 1e-4f, 1e-3f, "blocked grad");

  const float lr = 0.03f;
  BlockFilterKByBxRSC updated_weights = weights;
  apply_sgd(updated_weights, dw_ref, lr);

  layer.step(lr);
  BlockFilterKByBxRSC updated_gpu_weights;
  layer.copy_weights_to(updated_gpu_weights);
  expect_close(updated_weights.data, updated_gpu_weights.data, 1e-6f, 1e-6f, "blocked step");
  BlockFilterKByBxRSC zero_grad(weights.k, weights.by, weights.bx, weights.r, weights.s,
                                weights.cin_per_group, weights.ay, weights.ax);
  layer.copy_grad_to(zero_grad);
  expect_close(std::vector<float>(zero_grad.elements(), 0.0f), zero_grad.data, 1e-7f, 1e-7f, "blocked zero grad after step");

  TensorNHWC updated_y_ref;
  TensorNHWC updated_dx_ref(n, h, w, c);
  cpu_block_fprop_nhwc(x, updated_weights, p, updated_y_ref);
  cpu_block_bprop_nhwc(dy, updated_weights, p, updated_dx_ref);

  layer.forward(d_x.ptr, d_y.ptr);
  expect_close(updated_y_ref.data, copy_from_device(d_y.ptr, updated_y_ref.elements()), 1e-4f, 1e-3f, "blocked fprop after step");

  layer.backward_input(d_dy.ptr, d_dx.ptr);
  expect_close(updated_dx_ref.data, copy_from_device(d_dx.ptr, updated_dx_ref.elements()), 1e-4f, 1e-3f, "blocked bprop after step");
}

void run_blocked_ayax_case() {
  const int n = 2;
  const int h = 6;
  const int w = 6;
  const int c = 3;

  BlockConv2DParams p;
  p.conv.pad_h = 1;
  p.conv.pad_w = 1;
  p.conv.stride_h = 1;
  p.conv.stride_w = 1;
  p.conv.dilation_h = 1;
  p.conv.dilation_w = 1;
  p.conv.groups = 1;
  p.conv.ay = 2;
  p.conv.ax = 2;
  p.block_by = 2;
  p.block_bx = 3;

  TensorNHWC x(n, h, w, c);
  BlockFilterKByBxRSC weights(4, p.block_by, p.block_bx, 3, 3, c, p.conv.ay, p.conv.ax);
  fill_pattern(x.data, 71);
  fill_pattern(weights.data, 907);

  const BlockConvShape sh = infer_block_conv_shape(x, weights, p);
  TensorNHWC dy(n, sh.base.ho, sh.base.wo, weights.k);
  fill_pattern(dy.data, 1013);

  TensorNHWC y_ref;
  TensorNHWC dx_ref(n, h, w, c);
  BlockFilterKByBxRSC dw_ref(weights.k, weights.by, weights.bx, weights.r, weights.s,
                             weights.cin_per_group, weights.ay, weights.ax);
  cpu_block_fprop_nhwc(x, weights, p, y_ref);
  cpu_block_bprop_nhwc(dy, weights, p, dx_ref);
  cpu_block_grad_nhwc(x, dy, p, dw_ref);

  Conv2DLayer layer(n, h, w, c, weights, p);
  if (!layer.blocked_config() || layer.regular_config()) {
    throw std::runtime_error("blocked ay/ax layer config exposure is invalid");
  }
  if (layer.blocked_config()->shape.base.ho != y_ref.h ||
      layer.blocked_config()->shape.base.wo != y_ref.w ||
      layer.blocked_config()->shape.base.ay != p.conv.ay ||
      layer.blocked_config()->shape.base.ax != p.conv.ax ||
      layer.blocked_config()->output_elements != y_ref.elements()) {
    throw std::runtime_error("blocked ay/ax runtime config mismatch");
  }

  DeviceBuffer<float> d_x;
  DeviceBuffer<float> d_dy;
  DeviceBuffer<float> d_y;
  DeviceBuffer<float> d_dx;
  d_x.allocate(x.elements());
  d_dy.allocate(dy.elements());
  d_y.allocate(y_ref.elements());
  d_dx.allocate(x.elements());

  CUDA_CHECK(cudaMemcpy(d_x.ptr, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dy.ptr, dy.ptr(), dy.elements() * sizeof(float), cudaMemcpyHostToDevice));

  layer.forward(d_x.ptr, d_y.ptr);
  expect_close(y_ref.data, copy_from_device(d_y.ptr, y_ref.elements()), 1e-4f, 1e-3f, "blocked ay/ax fprop");

  layer.backward(d_x.ptr, d_dy.ptr, d_dx.ptr);
  expect_close(dx_ref.data, copy_from_device(d_dx.ptr, dx_ref.elements()), 1e-4f, 1e-3f, "blocked ay/ax bprop");

  BlockFilterKByBxRSC dw_gpu;
  layer.copy_grad_to(dw_gpu);
  expect_close(dw_ref.data, dw_gpu.data, 1e-4f, 1e-3f, "blocked ay/ax grad");

  const float lr = 0.025f;
  BlockFilterKByBxRSC updated_weights = weights;
  apply_sgd(updated_weights, dw_ref, lr);

  layer.step(lr);
  BlockFilterKByBxRSC updated_gpu_weights;
  layer.copy_weights_to(updated_gpu_weights);
  expect_close(updated_weights.data, updated_gpu_weights.data, 1e-6f, 1e-6f, "blocked ay/ax step");

  TensorNHWC updated_y_ref;
  TensorNHWC updated_dx_ref(n, h, w, c);
  cpu_block_fprop_nhwc(x, updated_weights, p, updated_y_ref);
  cpu_block_bprop_nhwc(dy, updated_weights, p, updated_dx_ref);

  layer.forward(d_x.ptr, d_y.ptr);
  expect_close(updated_y_ref.data, copy_from_device(d_y.ptr, updated_y_ref.elements()), 1e-4f, 1e-3f, "blocked ay/ax fprop after step");

  layer.backward_input(d_dy.ptr, d_dx.ptr);
  expect_close(updated_dx_ref.data, copy_from_device(d_dx.ptr, updated_dx_ref.elements()), 1e-4f, 1e-3f, "blocked ay/ax bprop after step");
}

}  // namespace

int main() {
  try {
    invalidate_all_conv_workspace_caches();
    run_regular_case();
    run_regular_ayax_case();
    run_blocked_case();
    run_blocked_ayax_case();
    std::cout << "conv_layer_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "conv_layer_test failed: " << e.what() << "\n";
    return 1;
  }
}
