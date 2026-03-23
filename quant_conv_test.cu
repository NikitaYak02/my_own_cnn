#include "src/conv_types.h"
#include "src/cuda_utils.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <nnalgebra::DataType Tin>
using QParams = nnalgebra::QuantizationParameters<Tin>;

template <nnalgebra::DataType Tin>
int quant_code(size_t idx, int salt);

template <>
int quant_code<nnalgebra::DataType::LinQuantU8>(size_t idx, int salt) {
  return static_cast<int>((idx * 37 + static_cast<size_t>(salt) * 19) % 251);
}

template <>
int quant_code<nnalgebra::DataType::LinQuantS5>(size_t idx, int salt) {
  return static_cast<int>((idx * 17 + static_cast<size_t>(salt) * 11) % 31) - 15;
}

template <nnalgebra::DataType Tin>
void fill_quantized(std::vector<float>& data, int salt) {
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>(quant_code<Tin>(i, salt));
  }
}

template <nnalgebra::DataType Tin>
std::vector<QParams<Tin>> make_input_qparams(int n);

template <>
std::vector<QParams<nnalgebra::DataType::LinQuantU8>> make_input_qparams(int n) {
  std::vector<QParams<nnalgebra::DataType::LinQuantU8>> qp(n);
  for (int i = 0; i < n; ++i) {
    qp[i].zero_point = static_cast<uint8_t>(37 + i * 29);
    qp[i].scale = 0.25f + 0.1f * static_cast<float>(i);
    qp[i].symmetric = false;
  }
  return qp;
}

template <>
std::vector<QParams<nnalgebra::DataType::LinQuantS5>> make_input_qparams(int n) {
  std::vector<QParams<nnalgebra::DataType::LinQuantS5>> qp(n);
  for (int i = 0; i < n; ++i) {
    qp[i].zero_point = static_cast<int8_t>(-5 + i * 7);
    qp[i].n_bins = 31;
    qp[i].scale = 0.5f + 0.125f * static_cast<float>(i);
  }
  return qp;
}

template <nnalgebra::DataType Tin>
QParams<Tin> make_filter_qp();

template <>
QParams<nnalgebra::DataType::LinQuantU8> make_filter_qp() {
  QParams<nnalgebra::DataType::LinQuantU8> qp;
  qp.zero_point = 91;
  qp.scale = 0.75f;
  qp.symmetric = false;
  return qp;
}

template <>
QParams<nnalgebra::DataType::LinQuantS5> make_filter_qp() {
  QParams<nnalgebra::DataType::LinQuantS5> qp;
  qp.zero_point = 3;
  qp.n_bins = 31;
  qp.scale = 0.875f;
  return qp;
}

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

void expect_equal(const std::vector<int32_t>& ref,
                  const std::vector<int32_t>& got,
                  const std::string& label) {
  if (ref.size() != got.size()) {
    throw std::runtime_error(label + ": size mismatch");
  }

  for (size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) {
      std::ostringstream oss;
      oss << label << ": mismatch at " << i
          << " ref=" << ref[i]
          << " got=" << got[i];
      throw std::runtime_error(oss.str());
    }
  }
}

template <nnalgebra::DataType Tin>
void run_regular_case(const std::string& label) {
  Conv2DParams p;
  p.pad_h = 1;
  p.pad_w = 0;
  p.stride_h = 1;
  p.stride_w = 1;
  p.dilation_h = 1;
  p.dilation_w = 1;
  p.groups = 2;

  TensorNHWC x(2, 5, 6, 4);
  FilterKRSC w(3, 2, x.c / p.groups, 6);
  fill_quantized<Tin>(x.data, 1);
  fill_quantized<Tin>(w.data, 7);

  const auto in_qp = make_input_qparams<Tin>(x.n);
  const auto f_qp = make_filter_qp<Tin>();

  TensorNHWCI32 y_cpu;
  cpu_fprop_nhwc_qi32<Tin>(x, w, p, y_cpu, in_qp.data(), &f_qp);

  DeviceBuffer<float> d_x;
  DeviceBuffer<float> d_w;
  DeviceBuffer<int32_t> d_y;
  DeviceBuffer<QParams<Tin>> d_in_qp;
  DeviceBuffer<QParams<Tin>> d_f_qp;

  d_x.allocate(x.elements());
  d_w.allocate(w.elements());
  d_y.allocate(y_cpu.elements());
  d_in_qp.allocate(in_qp.size());
  d_f_qp.allocate(1);

  CUDA_CHECK(cudaMemcpy(d_x.ptr, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w.ptr, w.ptr(), w.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_in_qp.ptr, in_qp.data(), in_qp.size() * sizeof(QParams<Tin>), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_f_qp.ptr, &f_qp, sizeof(QParams<Tin>), cudaMemcpyHostToDevice));

  launch_fprop_nhwc_qi32<Tin>(d_x.ptr, d_w.ptr, d_y.ptr,
                              x.n, x.h, x.w, x.c, w.r, w.s, w.k,
                              p, d_in_qp.ptr, d_f_qp.ptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int32_t> y_cuda(y_cpu.elements());
  CUDA_CHECK(cudaMemcpy(y_cuda.data(), d_y.ptr, y_cuda.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  expect_equal(y_cpu.data, y_cuda, label);
}

template <nnalgebra::DataType Tin>
void run_blocked_case(const std::string& label) {
  BlockConv2DParams p;
  p.conv.pad_h = 1;
  p.conv.pad_w = 1;
  p.conv.stride_h = 1;
  p.conv.stride_w = 1;
  p.conv.dilation_h = 1;
  p.conv.dilation_w = 1;
  p.conv.groups = 1;
  p.block_by = 2;
  p.block_bx = 3;

  TensorNHWC x(2, 6, 6, 4);
  BlockFilterKByBxRSC w(5, p.block_by, p.block_bx, 3, 3, x.c / p.conv.groups);
  fill_quantized<Tin>(x.data, 3);
  fill_quantized<Tin>(w.data, 9);

  const auto in_qp = make_input_qparams<Tin>(x.n);
  const auto f_qp = make_filter_qp<Tin>();

  TensorNHWCI32 y_cpu;
  cpu_block_fprop_nhwc_qi32<Tin>(x, w, p, y_cpu, in_qp.data(), &f_qp);

  DeviceBuffer<float> d_x;
  DeviceBuffer<float> d_w;
  DeviceBuffer<int32_t> d_y;
  DeviceBuffer<QParams<Tin>> d_in_qp;
  DeviceBuffer<QParams<Tin>> d_f_qp;

  d_x.allocate(x.elements());
  d_w.allocate(w.elements());
  d_y.allocate(y_cpu.elements());
  d_in_qp.allocate(in_qp.size());
  d_f_qp.allocate(1);

  CUDA_CHECK(cudaMemcpy(d_x.ptr, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w.ptr, w.ptr(), w.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_in_qp.ptr, in_qp.data(), in_qp.size() * sizeof(QParams<Tin>), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_f_qp.ptr, &f_qp, sizeof(QParams<Tin>), cudaMemcpyHostToDevice));

  launch_block_fprop_nhwc_qi32<Tin>(d_x.ptr, d_w.ptr, d_y.ptr,
                                    x.n, x.h, x.w, x.c, w.r, w.s, w.k,
                                    p, d_in_qp.ptr, d_f_qp.ptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int32_t> y_cuda(y_cpu.elements());
  CUDA_CHECK(cudaMemcpy(y_cuda.data(), d_y.ptr, y_cuda.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  expect_equal(y_cpu.data, y_cuda, label);
}

}  // namespace

int main() {
  try {
    run_regular_case<nnalgebra::DataType::LinQuantU8>("regular_u8");
    run_regular_case<nnalgebra::DataType::LinQuantS5>("regular_s5");
    run_blocked_case<nnalgebra::DataType::LinQuantU8>("blocked_u8");
    run_blocked_case<nnalgebra::DataType::LinQuantS5>("blocked_s5");
    std::cout << "quant_conv_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "quant_conv_test failed: " << e.what() << "\n";
    return 1;
  }
}
