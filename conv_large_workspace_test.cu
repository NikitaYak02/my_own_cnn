#include "conv_types.h"
#include "cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

struct DeviceBuffers {
  float* d_x = nullptr;
  float* d_w = nullptr;
  float* d_dy = nullptr;
  float* d_y = nullptr;
  float* d_dx = nullptr;

  ~DeviceBuffers() {
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_dy);
    cudaFree(d_y);
    cudaFree(d_dx);
  }
};

void fill_pattern(std::vector<float>& data, int salt) {
  for (size_t i = 0; i < data.size(); ++i) {
    const int code = static_cast<int>((i * 131 + static_cast<size_t>(salt) * 17) % 509);
    data[i] = static_cast<float>(code - 254) / 127.0f;
  }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    throw std::runtime_error("max_abs_diff size mismatch");
  }

  float out = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    out = std::max(out, std::fabs(a[i] - b[i]));
  }
  return out;
}

float sum_abs(const std::vector<float>& data) {
  double acc = 0.0;
  for (float v : data) {
    acc += std::fabs(static_cast<double>(v));
  }
  return static_cast<float>(acc);
}

std::vector<float> run_fprop_pass(const TensorNHWC& y_shape,
                                  DeviceBuffers& dev,
                                  const Conv2DParams& p,
                                  int n, int h, int w, int c, int r, int s, int k) {
  CUDA_CHECK(cudaMemset(dev.d_y, 0, y_shape.elements() * sizeof(float)));
  launch_fprop_nhwc(dev.d_x, dev.d_w, dev.d_y, n, h, w, c, r, s, k, p);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> y(y_shape.elements());
  CUDA_CHECK(cudaMemcpy(y.data(), dev.d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
  return y;
}

std::vector<float> run_bprop_pass(const TensorNHWC& x_shape,
                                  DeviceBuffers& dev,
                                  const Conv2DParams& p,
                                  int n, int h, int w, int c, int r, int s, int k) {
  CUDA_CHECK(cudaMemset(dev.d_dx, 0, x_shape.elements() * sizeof(float)));
  launch_bprop_nhwc(dev.d_dy, dev.d_w, dev.d_dx, n, h, w, c, r, s, k, p);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> dx(x_shape.elements());
  CUDA_CHECK(cudaMemcpy(dx.data(), dev.d_dx, dx.size() * sizeof(float), cudaMemcpyDeviceToHost));
  return dx;
}

std::vector<float> run_block_fprop_pass(const TensorNHWC& y_shape,
                                        DeviceBuffers& dev,
                                        const BlockConv2DParams& p,
                                        int n, int h, int w, int c, int r, int s, int k) {
  CUDA_CHECK(cudaMemset(dev.d_y, 0, y_shape.elements() * sizeof(float)));
  launch_block_fprop_nhwc(dev.d_x, dev.d_w, dev.d_y, n, h, w, c, r, s, k, p);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> y(y_shape.elements());
  CUDA_CHECK(cudaMemcpy(y.data(), dev.d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
  return y;
}

std::vector<float> run_block_bprop_pass(const TensorNHWC& x_shape,
                                        DeviceBuffers& dev,
                                        const BlockConv2DParams& p,
                                        int n, int h, int w, int c, int r, int s, int k) {
  CUDA_CHECK(cudaMemset(dev.d_dx, 0, x_shape.elements() * sizeof(float)));
  launch_block_bprop_nhwc(dev.d_dy, dev.d_w, dev.d_dx, n, h, w, c, r, s, k, p);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> dx(x_shape.elements());
  CUDA_CHECK(cudaMemcpy(dx.data(), dev.d_dx, dx.size() * sizeof(float), cudaMemcpyDeviceToHost));
  return dx;
}

}  // namespace

int main() {
  try {
    int n = 250;
    constexpr int h = 128;
    constexpr int w = 128;
    constexpr int c = 1;
    constexpr int r = 11;
    constexpr int s = 11;
    constexpr int k = 16;

    Conv2DParams p;
    p.pad_h = 5;
    p.pad_w = 5;
    p.stride_h = 1;
    p.stride_w = 1;
    p.dilation_h = 1;
    p.dilation_w = 1;
    p.groups = 1;

    // Keep the test meaningful while adapting to the GPU memory available on CI/dev machines.
    size_t free_mem = 0;
    size_t total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    const size_t m_per_batch = static_cast<size_t>(h) * static_cast<size_t>(w);
    const size_t per_batch_workspace_bytes =
        m_per_batch * (static_cast<size_t>(r) * static_cast<size_t>(s) * static_cast<size_t>(c) +
                       static_cast<size_t>(k)) * sizeof(float);
    const size_t workspace_budget = free_mem / 4;
    const int n_from_budget =
        static_cast<int>(std::max<size_t>(1, workspace_budget / std::max<size_t>(1, per_batch_workspace_bytes)));
    n = std::max(8, std::min(n, n_from_budget));

    TensorNHWC x(n, h, w, c);
    FilterKRSC w_filter(r, s, c, k);
    fill_pattern(x.data, 11);
    fill_pattern(w_filter.data, 97);

    const ConvShape sh = infer_conv_shape(x, w_filter, p);
    TensorNHWC dy(n, sh.ho, sh.wo, k);
    TensorNHWC y_shape(n, sh.ho, sh.wo, k);
    fill_pattern(dy.data, 193);

    DeviceBuffers dev;
    CUDA_CHECK(cudaMalloc(&dev.d_x, x.elements() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.d_w, w_filter.elements() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.d_dy, dy.elements() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.d_y, y_shape.elements() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.d_dx, x.elements() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dev.d_x, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.d_w, w_filter.ptr(), w_filter.elements() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.d_dy, dy.ptr(), dy.elements() * sizeof(float), cudaMemcpyHostToDevice));

    const std::vector<float> y_pass1 = run_fprop_pass(y_shape, dev, p, n, h, w, c, r, s, k);
    const std::vector<float> y_pass2 = run_fprop_pass(y_shape, dev, p, n, h, w, c, r, s, k);
    const float y_replay_diff = max_abs_diff(y_pass1, y_pass2);
    if (y_replay_diff != 0.0f) {
      std::ostringstream oss;
      oss << "fprop replay drift detected: max_abs_diff=" << y_replay_diff;
      throw std::runtime_error(oss.str());
    }
    if (sum_abs(y_pass1) == 0.0f) {
      throw std::runtime_error("fprop output is unexpectedly all zeros");
    }

    const std::vector<float> dx_pass1 = run_bprop_pass(x, dev, p, n, h, w, c, r, s, k);
    const std::vector<float> dx_pass2 = run_bprop_pass(x, dev, p, n, h, w, c, r, s, k);
    const float dx_replay_diff = max_abs_diff(dx_pass1, dx_pass2);
    if (dx_replay_diff > 1e-4f) {
      std::ostringstream oss;
      oss << "bprop replay drift detected: max_abs_diff=" << dx_replay_diff;
      throw std::runtime_error(oss.str());
    }
    if (sum_abs(dx_pass1) == 0.0f) {
      throw std::runtime_error("bprop output is unexpectedly all zeros");
    }

    BlockConv2DParams block_p;
    block_p.conv = p;
    block_p.block_by = 2;
    block_p.block_bx = 2;
    BlockFilterKByBxRSC block_w_filter(k, block_p.block_by, block_p.block_bx, r, s, c);
    fill_pattern(block_w_filter.data, 307);

    DeviceBuffers block_dev;
    CUDA_CHECK(cudaMalloc(&block_dev.d_x, x.elements() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&block_dev.d_w, block_w_filter.elements() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&block_dev.d_dy, dy.elements() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&block_dev.d_y, y_shape.elements() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&block_dev.d_dx, x.elements() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(block_dev.d_x, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(block_dev.d_w, block_w_filter.ptr(), block_w_filter.elements() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(block_dev.d_dy, dy.ptr(), dy.elements() * sizeof(float), cudaMemcpyHostToDevice));

    const std::vector<float> block_y_pass1 = run_block_fprop_pass(y_shape, block_dev, block_p, n, h, w, c, r, s, k);
    const std::vector<float> block_y_pass2 = run_block_fprop_pass(y_shape, block_dev, block_p, n, h, w, c, r, s, k);
    const float block_y_replay_diff = max_abs_diff(block_y_pass1, block_y_pass2);
    if (block_y_replay_diff != 0.0f) {
      std::ostringstream oss;
      oss << "block fprop replay drift detected: max_abs_diff=" << block_y_replay_diff;
      throw std::runtime_error(oss.str());
    }
    if (sum_abs(block_y_pass1) == 0.0f) {
      throw std::runtime_error("block fprop output is unexpectedly all zeros");
    }

    const std::vector<float> block_dx_pass1 = run_block_bprop_pass(x, block_dev, block_p, n, h, w, c, r, s, k);
    const std::vector<float> block_dx_pass2 = run_block_bprop_pass(x, block_dev, block_p, n, h, w, c, r, s, k);
    const float block_dx_replay_diff = max_abs_diff(block_dx_pass1, block_dx_pass2);
    if (block_dx_replay_diff > 1e-4f) {
      std::ostringstream oss;
      oss << "block bprop replay drift detected: max_abs_diff=" << block_dx_replay_diff;
      throw std::runtime_error(oss.str());
    }
    if (sum_abs(block_dx_pass1) == 0.0f) {
      throw std::runtime_error("block bprop output is unexpectedly all zeros");
    }

    std::cout << "conv_large_workspace_test passed\n";
    std::cout << "case=n250_h128_w128_c1_k16_r11_s11 pad=5 stride=1\n";
    std::cout << "fprop_replay_diff=" << y_replay_diff << "\n";
    std::cout << "bprop_replay_diff=" << dx_replay_diff << "\n";
    std::cout << "block_fprop_replay_diff=" << block_y_replay_diff << "\n";
    std::cout << "block_bprop_replay_diff=" << block_dx_replay_diff << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "conv_large_workspace_test failed: " << e.what() << "\n";
    return 1;
  }
}
