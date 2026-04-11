#include "src/conv_types.h"
#include "src/cuda_utils.h"
#include "bmm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <nnalgebra::DataType Tin>
using QParams = nnalgebra::QuantizationParameters<Tin>;
using OutQParams = nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>;

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

void expect_equal(const std::vector<QuantizedAccumStorage>& ref,
                  const std::vector<QuantizedAccumStorage>& got,
                  const std::string& label) {
  if (ref.size() != got.size()) {
    throw std::runtime_error(label + ": size mismatch");
  }

  for (size_t i = 0; i < ref.size(); ++i) {
    if (std::fabs(ref[i] - got[i]) > 1e-5f) {
      std::ostringstream oss;
      oss << label << ": mismatch at " << i
          << " ref=" << ref[i]
          << " got=" << got[i];
      throw std::runtime_error(oss.str());
    }
  }
}

void expect_output_qparams_equal(const std::vector<OutQParams>& ref,
                                 const std::vector<OutQParams>& got,
                                 const std::string& label) {
  if (ref.size() != got.size()) {
    throw std::runtime_error(label + ": qparam size mismatch");
  }

  for (size_t i = 0; i < ref.size(); ++i) {
    const float diff = std::fabs(ref[i].scale - got[i].scale);
    if (diff > 1e-6f) {
      std::ostringstream oss;
      oss << label << ": qparam mismatch at " << i
          << " ref_scale=" << ref[i].scale
          << " got_scale=" << got[i].scale;
      throw std::runtime_error(oss.str());
    }
  }
}

template <nnalgebra::DataType Tin>
std::vector<OutQParams> make_output_qparams(const std::vector<QParams<Tin>>& in_qp,
                                            const QParams<Tin>& f_qp) {
  std::vector<OutQParams> out(in_qp.size());
  for (size_t i = 0; i < in_qp.size(); ++i) {
    out[i].scale = nnalgebra::getScale(in_qp[i]) * nnalgebra::getScale(f_qp);
  }
  return out;
}

template <nnalgebra::DataType Tin>
std::vector<QuantizedAccumStorage> manual_regular_expected(const TensorNHWC& x,
                                                           const FilterKRSC& w,
                                                           const std::vector<QParams<Tin>>& in_qp,
                                                           const QParams<Tin>& f_qp) {
  std::vector<QuantizedAccumStorage> out(x.n);
  const int32_t wv =
      static_cast<int32_t>(w.data[idx_krsc(0, 0, 0, 0, w.r, w.s, w.cin_per_group)]) -
      nnalgebra::getZeroPoint(f_qp);
  for (int n = 0; n < x.n; ++n) {
    const int32_t xv =
        static_cast<int32_t>(x.data[idx_nhwc(n, 0, 0, 0, x.h, x.w, x.c)]) -
        nnalgebra::getZeroPoint(in_qp[n]);
    out[static_cast<size_t>(n)] = static_cast<QuantizedAccumStorage>(xv * wv);
  }
  return out;
}

template <nnalgebra::DataType Tin>
std::vector<QuantizedAccumStorage> manual_blocked_expected(const TensorNHWC& x,
                                                           const BlockFilterKByBxRSC& w,
                                                           const std::vector<QParams<Tin>>& in_qp,
                                                           const QParams<Tin>& f_qp) {
  std::vector<QuantizedAccumStorage> out(x.elements());
  for (int n = 0; n < x.n; ++n) {
    for (int h = 0; h < x.h; ++h) {
      for (int wi = 0; wi < x.w; ++wi) {
        const int32_t xv =
            static_cast<int32_t>(x.data[idx_nhwc(n, h, wi, 0, x.h, x.w, x.c)]) -
            nnalgebra::getZeroPoint(in_qp[n]);
        const int32_t wv =
            static_cast<int32_t>(w.data[idx_kbybxrsc(0, h, wi, 0, 0, 0,
                                                     w.by, w.bx, w.r, w.s, w.cin_per_group)]) -
            nnalgebra::getZeroPoint(f_qp);
        out[idx_nhwc(n, h, wi, 0, x.h, x.w, x.c)] = static_cast<QuantizedAccumStorage>(xv * wv);
      }
    }
  }
  return out;
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
  p.ay = 2;
  p.ax = 2;

  TensorNHWC x(2, 5, 6, 4);
  FilterKRSC w(3, 2, x.c / p.groups, 6, p.ay, p.ax);
  fill_quantized<Tin>(x.data, 1);
  fill_quantized<Tin>(w.data, 7);

  const auto in_qp = make_input_qparams<Tin>(x.n);
  const auto f_qp = make_filter_qp<Tin>();
  const auto expected_out_qp = make_output_qparams(in_qp, f_qp);
  std::vector<OutQParams> y_cpu_qp(x.n);

  TensorNHWCI32 y_cpu;
  cpu_fprop_nhwc_qi32<Tin>(x, w, p, y_cpu, in_qp.data(), &f_qp, y_cpu_qp.data());
  expect_output_qparams_equal(expected_out_qp, y_cpu_qp, label + "_cpu_qp");

  DeviceBuffer<float> d_x;
  DeviceBuffer<float> d_w;
  DeviceBuffer<QuantizedAccumStorage> d_y;
  DeviceBuffer<QParams<Tin>> d_in_qp;
  DeviceBuffer<QParams<Tin>> d_f_qp;
  DeviceBuffer<OutQParams> d_out_qp;

  d_x.allocate(x.elements());
  d_w.allocate(w.elements());
  d_y.allocate(y_cpu.elements());
  d_in_qp.allocate(in_qp.size());
  d_f_qp.allocate(1);
  d_out_qp.allocate(expected_out_qp.size());

  CUDA_CHECK(cudaMemcpy(d_x.ptr, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w.ptr, w.ptr(), w.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_in_qp.ptr, in_qp.data(), in_qp.size() * sizeof(QParams<Tin>), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_f_qp.ptr, &f_qp, sizeof(QParams<Tin>), cudaMemcpyHostToDevice));

  launch_fprop_nhwc_qi32<Tin>(d_x.ptr, d_w.ptr, d_y.ptr,
                              x.n, x.h, x.w, x.c, w.r, w.s, w.k,
                              p, d_in_qp.ptr, d_f_qp.ptr, d_out_qp.ptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<QuantizedAccumStorage> y_cuda(y_cpu.elements());
  std::vector<OutQParams> y_cuda_qp(expected_out_qp.size());
  CUDA_CHECK(cudaMemcpy(y_cuda.data(), d_y.ptr, y_cuda.size() * sizeof(QuantizedAccumStorage), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(y_cuda_qp.data(), d_out_qp.ptr, y_cuda_qp.size() * sizeof(OutQParams), cudaMemcpyDeviceToHost));
  expect_equal(y_cpu.data, y_cuda, label);
  expect_output_qparams_equal(expected_out_qp, y_cuda_qp, label + "_cuda_qp");
}

template <nnalgebra::DataType Tin>
void run_regular_ayax_case(const std::string& label) {
  Conv2DParams p;
  p.pad_h = 1;
  p.pad_w = 0;
  p.stride_h = 1;
  p.stride_w = 1;
  p.dilation_h = 1;
  p.dilation_w = 1;
  p.groups = 2;
  p.ay = 2;
  p.ax = 3;

  TensorNHWC x(2, 5, 6, 4);
  FilterKRSC w(3, 2, x.c / p.groups, 6, p.ay, p.ax);
  fill_quantized<Tin>(x.data, 5);
  fill_quantized<Tin>(w.data, 13);

  const auto in_qp = make_input_qparams<Tin>(x.n);
  const auto f_qp = make_filter_qp<Tin>();
  const auto expected_out_qp = make_output_qparams(in_qp, f_qp);
  std::vector<OutQParams> y_cpu_qp(x.n);

  TensorNHWCI32 y_cpu;
  cpu_fprop_nhwc_qi32<Tin>(x, w, p, y_cpu, in_qp.data(), &f_qp, y_cpu_qp.data());
  expect_output_qparams_equal(expected_out_qp, y_cpu_qp, label + "_cpu_qp");

  DeviceBuffer<float> d_x;
  DeviceBuffer<float> d_w;
  DeviceBuffer<QuantizedAccumStorage> d_y;
  DeviceBuffer<QParams<Tin>> d_in_qp;
  DeviceBuffer<QParams<Tin>> d_f_qp;
  DeviceBuffer<OutQParams> d_out_qp;

  d_x.allocate(x.elements());
  d_w.allocate(w.elements());
  d_y.allocate(y_cpu.elements());
  d_in_qp.allocate(in_qp.size());
  d_f_qp.allocate(1);
  d_out_qp.allocate(expected_out_qp.size());

  CUDA_CHECK(cudaMemcpy(d_x.ptr, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w.ptr, w.ptr(), w.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_in_qp.ptr, in_qp.data(), in_qp.size() * sizeof(QParams<Tin>), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_f_qp.ptr, &f_qp, sizeof(QParams<Tin>), cudaMemcpyHostToDevice));

  launch_fprop_nhwc_qi32<Tin>(d_x.ptr, d_w.ptr, d_y.ptr,
                              x.n, x.h, x.w, x.c, w.r, w.s, w.k,
                              p, d_in_qp.ptr, d_f_qp.ptr, d_out_qp.ptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<QuantizedAccumStorage> y_cuda(y_cpu.elements());
  std::vector<OutQParams> y_cuda_qp(expected_out_qp.size());
  CUDA_CHECK(cudaMemcpy(y_cuda.data(), d_y.ptr, y_cuda.size() * sizeof(QuantizedAccumStorage), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(y_cuda_qp.data(), d_out_qp.ptr, y_cuda_qp.size() * sizeof(OutQParams), cudaMemcpyDeviceToHost));
  expect_equal(y_cpu.data, y_cuda, label);
  expect_output_qparams_equal(expected_out_qp, y_cuda_qp, label + "_cuda_qp");
}

template <nnalgebra::DataType Tin>
void run_qparam_contract_regular_case(const std::string& label) {
  Conv2DParams p;
  p.pad_h = 0;
  p.pad_w = 0;
  p.stride_h = 1;
  p.stride_w = 1;
  p.dilation_h = 1;
  p.dilation_w = 1;
  p.groups = 1;

  TensorNHWC x(3, 1, 1, 1);
  FilterKRSC w(1, 1, 1, 1);

  if constexpr (Tin == nnalgebra::DataType::LinQuantU8) {
    x.data = {12.0f, 30.0f, 200.0f};
    w.data = {77.0f};
  } else {
    x.data = {-9.0f, 5.0f, 14.0f};
    w.data = {6.0f};
  }

  const auto in_qp = make_input_qparams<Tin>(x.n);
  const auto f_qp = make_filter_qp<Tin>();
  const auto expected = manual_regular_expected(x, w, in_qp, f_qp);
  const auto expected_out_qp = make_output_qparams(in_qp, f_qp);
  std::vector<OutQParams> y_cpu_qp(x.n);

  TensorNHWCI32 y_cpu;
  cpu_fprop_nhwc_qi32<Tin>(x, w, p, y_cpu, in_qp.data(), &f_qp, y_cpu_qp.data());
  expect_equal(expected, y_cpu.data, label + "_cpu_manual");
  expect_output_qparams_equal(expected_out_qp, y_cpu_qp, label + "_cpu_qp");

  DeviceBuffer<float> d_x;
  DeviceBuffer<float> d_w;
  DeviceBuffer<QuantizedAccumStorage> d_y;
  DeviceBuffer<QParams<Tin>> d_in_qp;
  DeviceBuffer<QParams<Tin>> d_f_qp;
  DeviceBuffer<OutQParams> d_out_qp;

  d_x.allocate(x.elements());
  d_w.allocate(w.elements());
  d_y.allocate(expected.size());
  d_in_qp.allocate(in_qp.size());
  d_f_qp.allocate(1);
  d_out_qp.allocate(expected_out_qp.size());

  CUDA_CHECK(cudaMemcpy(d_x.ptr, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w.ptr, w.ptr(), w.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_in_qp.ptr, in_qp.data(), in_qp.size() * sizeof(QParams<Tin>), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_f_qp.ptr, &f_qp, sizeof(QParams<Tin>), cudaMemcpyHostToDevice));

  launch_fprop_nhwc_qi32<Tin>(d_x.ptr, d_w.ptr, d_y.ptr,
                              x.n, x.h, x.w, x.c, w.r, w.s, w.k,
                              p, d_in_qp.ptr, d_f_qp.ptr, d_out_qp.ptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<QuantizedAccumStorage> y_cuda(expected.size());
  std::vector<OutQParams> y_cuda_qp(expected_out_qp.size());
  CUDA_CHECK(cudaMemcpy(y_cuda.data(), d_y.ptr, y_cuda.size() * sizeof(QuantizedAccumStorage), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(y_cuda_qp.data(), d_out_qp.ptr, y_cuda_qp.size() * sizeof(OutQParams), cudaMemcpyDeviceToHost));
  expect_equal(expected, y_cuda, label + "_cuda_manual");
  expect_output_qparams_equal(expected_out_qp, y_cuda_qp, label + "_cuda_qp");
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
  p.conv.ay = 2;
  p.conv.ax = 2;
  p.block_by = 2;
  p.block_bx = 3;

  TensorNHWC x(2, 6, 6, 4);
  BlockFilterKByBxRSC w(5, p.block_by, p.block_bx, 3, 3, x.c / p.conv.groups, p.conv.ay, p.conv.ax);
  fill_quantized<Tin>(x.data, 3);
  fill_quantized<Tin>(w.data, 9);

  const auto in_qp = make_input_qparams<Tin>(x.n);
  const auto f_qp = make_filter_qp<Tin>();
  const auto expected_out_qp = make_output_qparams(in_qp, f_qp);
  std::vector<OutQParams> y_cpu_qp(x.n);

  TensorNHWCI32 y_cpu;
  cpu_block_fprop_nhwc_qi32<Tin>(x, w, p, y_cpu, in_qp.data(), &f_qp, y_cpu_qp.data());
  expect_output_qparams_equal(expected_out_qp, y_cpu_qp, label + "_cpu_qp");

  DeviceBuffer<float> d_x;
  DeviceBuffer<float> d_w;
  DeviceBuffer<QuantizedAccumStorage> d_y;
  DeviceBuffer<QParams<Tin>> d_in_qp;
  DeviceBuffer<QParams<Tin>> d_f_qp;
  DeviceBuffer<OutQParams> d_out_qp;

  d_x.allocate(x.elements());
  d_w.allocate(w.elements());
  d_y.allocate(y_cpu.elements());
  d_in_qp.allocate(in_qp.size());
  d_f_qp.allocate(1);
  d_out_qp.allocate(expected_out_qp.size());

  CUDA_CHECK(cudaMemcpy(d_x.ptr, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w.ptr, w.ptr(), w.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_in_qp.ptr, in_qp.data(), in_qp.size() * sizeof(QParams<Tin>), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_f_qp.ptr, &f_qp, sizeof(QParams<Tin>), cudaMemcpyHostToDevice));

  launch_block_fprop_nhwc_qi32<Tin>(d_x.ptr, d_w.ptr, d_y.ptr,
                                    x.n, x.h, x.w, x.c, w.r, w.s, w.k,
                                    p, d_in_qp.ptr, d_f_qp.ptr, d_out_qp.ptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<QuantizedAccumStorage> y_cuda(y_cpu.elements());
  std::vector<OutQParams> y_cuda_qp(expected_out_qp.size());
  CUDA_CHECK(cudaMemcpy(y_cuda.data(), d_y.ptr, y_cuda.size() * sizeof(QuantizedAccumStorage), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(y_cuda_qp.data(), d_out_qp.ptr, y_cuda_qp.size() * sizeof(OutQParams), cudaMemcpyDeviceToHost));
  expect_equal(y_cpu.data, y_cuda, label);
  expect_output_qparams_equal(expected_out_qp, y_cuda_qp, label + "_cuda_qp");
}

template <nnalgebra::DataType Tin>
void run_blocked_ayax_case(const std::string& label) {
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

  TensorNHWC x(2, 6, 6, 4);
  BlockFilterKByBxRSC w(5, p.block_by, p.block_bx, 3, 3, x.c / p.conv.groups, p.conv.ay, p.conv.ax);
  fill_quantized<Tin>(x.data, 17);
  fill_quantized<Tin>(w.data, 23);

  const auto in_qp = make_input_qparams<Tin>(x.n);
  const auto f_qp = make_filter_qp<Tin>();
  const auto expected_out_qp = make_output_qparams(in_qp, f_qp);
  std::vector<OutQParams> y_cpu_qp(x.n);

  TensorNHWCI32 y_cpu;
  cpu_block_fprop_nhwc_qi32<Tin>(x, w, p, y_cpu, in_qp.data(), &f_qp, y_cpu_qp.data());
  expect_output_qparams_equal(expected_out_qp, y_cpu_qp, label + "_cpu_qp");

  DeviceBuffer<float> d_x;
  DeviceBuffer<float> d_w;
  DeviceBuffer<QuantizedAccumStorage> d_y;
  DeviceBuffer<QParams<Tin>> d_in_qp;
  DeviceBuffer<QParams<Tin>> d_f_qp;
  DeviceBuffer<OutQParams> d_out_qp;

  d_x.allocate(x.elements());
  d_w.allocate(w.elements());
  d_y.allocate(y_cpu.elements());
  d_in_qp.allocate(in_qp.size());
  d_f_qp.allocate(1);
  d_out_qp.allocate(expected_out_qp.size());

  CUDA_CHECK(cudaMemcpy(d_x.ptr, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w.ptr, w.ptr(), w.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_in_qp.ptr, in_qp.data(), in_qp.size() * sizeof(QParams<Tin>), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_f_qp.ptr, &f_qp, sizeof(QParams<Tin>), cudaMemcpyHostToDevice));

  launch_block_fprop_nhwc_qi32<Tin>(d_x.ptr, d_w.ptr, d_y.ptr,
                                    x.n, x.h, x.w, x.c, w.r, w.s, w.k,
                                    p, d_in_qp.ptr, d_f_qp.ptr, d_out_qp.ptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<QuantizedAccumStorage> y_cuda(y_cpu.elements());
  std::vector<OutQParams> y_cuda_qp(expected_out_qp.size());
  CUDA_CHECK(cudaMemcpy(y_cuda.data(), d_y.ptr, y_cuda.size() * sizeof(QuantizedAccumStorage), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(y_cuda_qp.data(), d_out_qp.ptr, y_cuda_qp.size() * sizeof(OutQParams), cudaMemcpyDeviceToHost));
  expect_equal(y_cpu.data, y_cuda, label);
  expect_output_qparams_equal(expected_out_qp, y_cuda_qp, label + "_cuda_qp");
}

template <nnalgebra::DataType Tin>
void run_qparam_contract_blocked_case(const std::string& label) {
  BlockConv2DParams p;
  p.conv.pad_h = 0;
  p.conv.pad_w = 0;
  p.conv.stride_h = 1;
  p.conv.stride_w = 1;
  p.conv.dilation_h = 1;
  p.conv.dilation_w = 1;
  p.conv.groups = 1;
  p.block_by = 2;
  p.block_bx = 2;

  TensorNHWC x(3, 2, 2, 1);
  BlockFilterKByBxRSC w(1, p.block_by, p.block_bx, 1, 1, 1);

  if constexpr (Tin == nnalgebra::DataType::LinQuantU8) {
    x.data = {
        12.0f, 15.0f,
        18.0f, 21.0f,
        60.0f, 63.0f,
        66.0f, 69.0f,
        120.0f, 123.0f,
        126.0f, 129.0f,
    };
    w.data = {80.0f, 81.0f, 82.0f, 83.0f};
  } else {
    x.data = {
        -9.0f, -6.0f,
        -3.0f, 0.0f,
        2.0f, 5.0f,
        8.0f, 11.0f,
        12.0f, 13.0f,
        14.0f, 15.0f,
    };
    w.data = {-2.0f, 1.0f, 4.0f, 7.0f};
  }

  const auto in_qp = make_input_qparams<Tin>(x.n);
  const auto f_qp = make_filter_qp<Tin>();
  const auto expected = manual_blocked_expected(x, w, in_qp, f_qp);
  const auto expected_out_qp = make_output_qparams(in_qp, f_qp);
  std::vector<OutQParams> y_cpu_qp(x.n);

  TensorNHWCI32 y_cpu;
  cpu_block_fprop_nhwc_qi32<Tin>(x, w, p, y_cpu, in_qp.data(), &f_qp, y_cpu_qp.data());
  expect_equal(expected, y_cpu.data, label + "_cpu_manual");
  expect_output_qparams_equal(expected_out_qp, y_cpu_qp, label + "_cpu_qp");

  DeviceBuffer<float> d_x;
  DeviceBuffer<float> d_w;
  DeviceBuffer<QuantizedAccumStorage> d_y;
  DeviceBuffer<QParams<Tin>> d_in_qp;
  DeviceBuffer<QParams<Tin>> d_f_qp;
  DeviceBuffer<OutQParams> d_out_qp;

  d_x.allocate(x.elements());
  d_w.allocate(w.elements());
  d_y.allocate(expected.size());
  d_in_qp.allocate(in_qp.size());
  d_f_qp.allocate(1);
  d_out_qp.allocate(expected_out_qp.size());

  CUDA_CHECK(cudaMemcpy(d_x.ptr, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w.ptr, w.ptr(), w.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_in_qp.ptr, in_qp.data(), in_qp.size() * sizeof(QParams<Tin>), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_f_qp.ptr, &f_qp, sizeof(QParams<Tin>), cudaMemcpyHostToDevice));

  launch_block_fprop_nhwc_qi32<Tin>(d_x.ptr, d_w.ptr, d_y.ptr,
                                    x.n, x.h, x.w, x.c, w.r, w.s, w.k,
                                    p, d_in_qp.ptr, d_f_qp.ptr, d_out_qp.ptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<QuantizedAccumStorage> y_cuda(expected.size());
  std::vector<OutQParams> y_cuda_qp(expected_out_qp.size());
  CUDA_CHECK(cudaMemcpy(y_cuda.data(), d_y.ptr, y_cuda.size() * sizeof(QuantizedAccumStorage), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(y_cuda_qp.data(), d_out_qp.ptr, y_cuda_qp.size() * sizeof(OutQParams), cudaMemcpyDeviceToHost));
  expect_equal(expected, y_cuda, label + "_cuda_manual");
  expect_output_qparams_equal(expected_out_qp, y_cuda_qp, label + "_cuda_qp");
}

struct TileConfigGuard {
  BmmTileConfig saved{};

  TileConfigGuard() {
    bmm_get_tile_config(&saved);
  }

  ~TileConfigGuard() {
    (void)bmm_set_tile_config(&saved);
  }
};

constexpr int kDefaultThreadRows = 2;
constexpr int kDefaultThreadCols = 2;
constexpr int kSharedPad = 1;

BmmTileConfig normalize_tile_config(BmmTileConfig cfg) {
  if (cfg.thread_rows == 0) {
    cfg.thread_rows = kDefaultThreadRows;
  }
  if (cfg.thread_cols == 0) {
    cfg.thread_cols = kDefaultThreadCols;
  }
  return cfg;
}

void expect_tile_config_equal(BmmTileConfig got,
                              BmmTileConfig expected,
                              const std::string& label) {
  got = normalize_tile_config(got);
  expected = normalize_tile_config(expected);
  if (got.block_rows != expected.block_rows ||
      got.block_cols != expected.block_cols ||
      got.block_depth != expected.block_depth ||
      got.thread_rows != expected.thread_rows ||
      got.thread_cols != expected.thread_cols) {
    std::ostringstream oss;
    oss << label
        << ": unexpected tile config"
        << " got=" << got.block_rows << "x" << got.block_cols << "x" << got.block_depth
        << " threads=" << got.thread_rows << "x" << got.thread_cols
        << " expected=" << expected.block_rows << "x" << expected.block_cols << "x" << expected.block_depth
        << " threads=" << expected.thread_rows << "x" << expected.thread_cols;
    throw std::runtime_error(oss.str());
  }
}

void expect_launch_info(const BmmTileLaunchInfo& info,
                        BmmTileConfig expected,
                        const std::string& label) {
  expected = normalize_tile_config(expected);
  const int expected_threads_x = expected.block_cols / expected.thread_cols;
  const int expected_threads_y = expected.block_rows / expected.thread_rows;
  const int expected_shared_bytes = static_cast<int>(
      (static_cast<size_t>(expected.block_rows) * (expected.block_depth + kSharedPad) +
       static_cast<size_t>(expected.block_depth) * (expected.block_cols + kSharedPad)) * sizeof(float));

  if (info.tile.block_rows != expected.block_rows ||
      info.tile.block_cols != expected.block_cols ||
      info.tile.block_depth != expected.block_depth ||
      info.tile.thread_rows != expected.thread_rows ||
      info.tile.thread_cols != expected.thread_cols ||
      info.threads_x != expected_threads_x ||
      info.threads_y != expected_threads_y ||
      info.shared_mem_bytes != expected_shared_bytes) {
    std::ostringstream oss;
    oss << label
        << ": unexpected launch info"
        << " tile=" << info.tile.block_rows << "x" << info.tile.block_cols << "x" << info.tile.block_depth
        << " thread_tile=" << info.tile.thread_rows << "x" << info.tile.thread_cols
        << " threads=" << info.threads_x << "x" << info.threads_y
        << " smem=" << info.shared_mem_bytes;
    throw std::runtime_error(oss.str());
  }
}

void run_tile_config_switch_case() {
  TileConfigGuard guard;

  const BmmTileConfig cfg_a{16, 16, 4, 2, 2};
  const BmmTileConfig cfg_b{32, 16, 8, 2, 2};
  if (!bmm_set_tile_config(&cfg_a)) {
    throw std::runtime_error("failed to set tile config A");
  }
  run_qparam_contract_blocked_case<nnalgebra::DataType::LinQuantU8>("tile_switch_a");
  BmmTileLaunchInfo info_a{};
  bmm_get_last_launch_info(&info_a);
  expect_launch_info(info_a, cfg_a, "tile_switch_a");

  if (!bmm_set_tile_config(&cfg_b)) {
    throw std::runtime_error("failed to set tile config B");
  }
  run_qparam_contract_blocked_case<nnalgebra::DataType::LinQuantU8>("tile_switch_b");
  BmmTileLaunchInfo info_b{};
  bmm_get_last_launch_info(&info_b);
  expect_launch_info(info_b, cfg_b, "tile_switch_b");

  if (info_a.tile.block_rows == info_b.tile.block_rows &&
      info_a.tile.block_cols == info_b.tile.block_cols &&
      info_a.tile.block_depth == info_b.tile.block_depth &&
      info_a.threads_x == info_b.threads_x &&
      info_a.threads_y == info_b.threads_y &&
      info_a.shared_mem_bytes == info_b.shared_mem_bytes) {
    throw std::runtime_error("tile launch info did not change after switching configs");
  }
}

void run_thread_tile_switch_case() {
  TileConfigGuard guard;

  const BmmTileConfig cfg_a{16, 16, 4, 2, 2};
  const BmmTileConfig cfg_b{16, 16, 4, 1, 4};
  if (!bmm_set_tile_config(&cfg_a)) {
    throw std::runtime_error("failed to set thread-tile config A");
  }
  run_qparam_contract_blocked_case<nnalgebra::DataType::LinQuantU8>("thread_tile_switch_a");
  BmmTileLaunchInfo info_a{};
  bmm_get_last_launch_info(&info_a);
  expect_launch_info(info_a, cfg_a, "thread_tile_switch_a");

  if (!bmm_set_tile_config(&cfg_b)) {
    throw std::runtime_error("failed to set thread-tile config B");
  }
  run_qparam_contract_blocked_case<nnalgebra::DataType::LinQuantU8>("thread_tile_switch_b");
  BmmTileLaunchInfo info_b{};
  bmm_get_last_launch_info(&info_b);
  expect_launch_info(info_b, cfg_b, "thread_tile_switch_b");

  if (info_a.tile.thread_rows == info_b.tile.thread_rows &&
      info_a.tile.thread_cols == info_b.tile.thread_cols &&
      info_a.threads_x == info_b.threads_x &&
      info_a.threads_y == info_b.threads_y) {
    throw std::runtime_error("thread tile launch info did not change after switching configs");
  }
}

void run_invalid_thread_tile_case() {
  TileConfigGuard guard;

  const BmmTileConfig valid_cfg{16, 16, 4, 2, 2};
  if (!bmm_set_tile_config(&valid_cfg)) {
    throw std::runtime_error("failed to set baseline thread-tile config");
  }

  const BmmTileConfig invalid_divisible{16, 16, 4, 3, 2};
  if (bmm_set_tile_config(&invalid_divisible)) {
    throw std::runtime_error("invalid divisible thread-tile config was accepted");
  }

  const BmmTileConfig invalid_limit{16, 16, 4, 9, 1};
  if (bmm_set_tile_config(&invalid_limit)) {
    throw std::runtime_error("invalid limit thread-tile config was accepted");
  }

  BmmTileConfig current{};
  bmm_get_tile_config(&current);
  expect_tile_config_equal(current, valid_cfg, "invalid_thread_tile_preserves_previous");
}

}  // namespace

int main() {
  try {
    run_regular_case<nnalgebra::DataType::LinQuantU8>("regular_u8");
    run_regular_case<nnalgebra::DataType::LinQuantS5>("regular_s5");
    run_regular_ayax_case<nnalgebra::DataType::LinQuantU8>("regular_ayax_u8");
    run_regular_ayax_case<nnalgebra::DataType::LinQuantS5>("regular_ayax_s5");
    run_blocked_case<nnalgebra::DataType::LinQuantU8>("blocked_u8");
    run_blocked_case<nnalgebra::DataType::LinQuantS5>("blocked_s5");
    run_blocked_ayax_case<nnalgebra::DataType::LinQuantU8>("blocked_ayax_u8");
    run_blocked_ayax_case<nnalgebra::DataType::LinQuantS5>("blocked_ayax_s5");
    run_qparam_contract_regular_case<nnalgebra::DataType::LinQuantU8>("qparam_contract_regular_u8");
    run_qparam_contract_regular_case<nnalgebra::DataType::LinQuantS5>("qparam_contract_regular_s5");
    run_qparam_contract_blocked_case<nnalgebra::DataType::LinQuantU8>("qparam_contract_blocked_u8");
    run_qparam_contract_blocked_case<nnalgebra::DataType::LinQuantS5>("qparam_contract_blocked_s5");
    run_tile_config_switch_case();
    run_thread_tile_switch_case();
    run_invalid_thread_tile_case();
    std::cout << "quant_conv_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "quant_conv_test failed: " << e.what() << "\n";
    return 1;
  }
}
