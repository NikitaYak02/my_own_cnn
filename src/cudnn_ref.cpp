#include "conv_types.h"
#include "cuda_utils.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef USE_CUDNN
#include <cudnn.h>
#include <cudnn_cnn.h>

namespace {

void check_cudnn(cudnnStatus_t s, const char* stmt, const char* file, int line) {
  if (s != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuDNN error at ") + file + ":" + std::to_string(line) + " in " + stmt + " -> " + cudnnGetErrorString(s));
  }
}

#define CUDNN_CHECK(stmt) check_cudnn((stmt), #stmt, __FILE__, __LINE__)

struct CudnnHandle {
  cudnnHandle_t h = nullptr;
  CudnnHandle() { CUDNN_CHECK(cudnnCreate(&h)); }
  ~CudnnHandle() { cudnnDestroy(h); }
};

struct DescPack {
  cudnnTensorDescriptor_t x = nullptr;
  cudnnTensorDescriptor_t y = nullptr;
  cudnnTensorDescriptor_t dx = nullptr;
  cudnnTensorDescriptor_t dy = nullptr;
  cudnnFilterDescriptor_t w = nullptr;
  cudnnFilterDescriptor_t dw = nullptr;
  cudnnConvolutionDescriptor_t conv = nullptr;

  DescPack() {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dx));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&dw));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv));
  }
  ~DescPack() {
    cudnnDestroyTensorDescriptor(x);
    cudnnDestroyTensorDescriptor(y);
    cudnnDestroyTensorDescriptor(dx);
    cudnnDestroyTensorDescriptor(dy);
    cudnnDestroyFilterDescriptor(w);
    cudnnDestroyFilterDescriptor(dw);
    cudnnDestroyConvolutionDescriptor(conv);
  }
};

void setup_descs(DescPack& d, int n, int h, int w, int c, int r, int s, int k, const Conv2DParams& p, int ho, int wo) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(d.x, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, n, c, h, w));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(d.dx, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, n, c, h, w));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(d.y, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, n, k, ho, wo));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(d.dy, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, n, k, ho, wo));

  CUDNN_CHECK(cudnnSetFilter4dDescriptor(d.w, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, k, c / p.groups, r, s));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(d.dw, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, k, c / p.groups, r, s));

  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(d.conv,
                                               p.pad_h, p.pad_w,
                                               p.stride_h, p.stride_w,
                                               p.dilation_h, p.dilation_w,
                                               CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetConvolutionGroupCount(d.conv, p.groups));
}

template <typename Fn>
BenchResult run_bench(size_t flops, int warmup, int iters, Fn&& fn) {
  return benchmark_cuda_op("cudnn", warmup, iters, flops, fn);
}

const char* fwd_algo_to_string(cudnnConvolutionFwdAlgo_t algo) {
  switch (algo) {
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_GEMM: return "CUDNN_CONVOLUTION_FWD_ALGO_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: return "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT: return "CUDNN_CONVOLUTION_FWD_ALGO_FFT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: return "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED";
    case CUDNN_CONVOLUTION_FWD_ALGO_COUNT: return "CUDNN_CONVOLUTION_FWD_ALGO_COUNT";
    default: return "CUDNN_CONVOLUTION_FWD_ALGO_UNKNOWN";
  }
}

const char* math_type_to_string(cudnnMathType_t math_type) {
  switch (math_type) {
    case CUDNN_DEFAULT_MATH: return "CUDNN_DEFAULT_MATH";
    case CUDNN_TENSOR_OP_MATH: return "CUDNN_TENSOR_OP_MATH";
    case CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION: return "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION";
    case CUDNN_FMA_MATH: return "CUDNN_FMA_MATH";
    default: return "CUDNN_MATH_UNKNOWN";
  }
}

void hwcn_to_krsc_host(const std::vector<float>& hwcn, std::vector<float>& krsc, int r, int s, int cin_group, int k_total) {
  krsc.resize(static_cast<size_t>(k_total) * r * s * cin_group);
  for (int kk = 0; kk < k_total; ++kk) {
    for (int rr = 0; rr < r; ++rr) {
      for (int ss = 0; ss < s; ++ss) {
        for (int ci = 0; ci < cin_group; ++ci) {
          const size_t src = idx_hwcn(rr, ss, ci, kk, s, cin_group, k_total);
          const size_t dst = ((static_cast<size_t>(kk) * r + rr) * s + ss) * cin_group + ci;
          krsc[dst] = hwcn[src];
        }
      }
    }
  }
}

void krsc_to_hwcn_host(const std::vector<float>& krsc, std::vector<float>& hwcn, int r, int s, int cin_group, int k_total) {
  hwcn.resize(static_cast<size_t>(k_total) * r * s * cin_group);
  for (int kk = 0; kk < k_total; ++kk) {
    for (int rr = 0; rr < r; ++rr) {
      for (int ss = 0; ss < s; ++ss) {
        for (int ci = 0; ci < cin_group; ++ci) {
          const size_t src = ((static_cast<size_t>(kk) * r + rr) * s + ss) * cin_group + ci;
          const size_t dst = idx_hwcn(rr, ss, ci, kk, s, cin_group, k_total);
          hwcn[dst] = krsc[src];
        }
      }
    }
  }
}

}  // namespace

bool cudnn_is_available() { return true; }

BenchResult cudnn_fprop_bench(const float* d_x, const float* d_w, float* d_y,
                              int n, int h, int w, int c, int r, int s, int k,
                              const Conv2DParams& p,
                              int warmup, int iters) {
  TensorNHWC xt(n, h, w, c);
  FilterHWCN wt(r, s, c / p.groups, k);
  ConvShape sh = infer_conv_shape(xt, wt, p);

  CudnnHandle handle;
  DescPack d;
  setup_descs(d, n, h, w, c, r, s, k, p, sh.ho, sh.wo);

  const int cin_group = c / p.groups;
  const int total_w = k * r * s * cin_group;
  float* d_w_krsc = nullptr;
  CUDA_CHECK(cudaMalloc(&d_w_krsc, static_cast<size_t>(total_w) * sizeof(float)));
  std::vector<float> h_w_hwcn(static_cast<size_t>(total_w));
  std::vector<float> h_w_krsc;
  CUDA_CHECK(cudaMemcpy(h_w_hwcn.data(), d_w, static_cast<size_t>(total_w) * sizeof(float), cudaMemcpyDeviceToHost));
  hwcn_to_krsc_host(h_w_hwcn, h_w_krsc, r, s, cin_group, k);
  CUDA_CHECK(cudaMemcpy(d_w_krsc, h_w_krsc.data(), static_cast<size_t>(total_w) * sizeof(float), cudaMemcpyHostToDevice));

  int returned = 0;
  cudnnConvolutionFwdAlgoPerf_t perf{};
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(handle.h, d.x, d.w, d.conv, d.y, 1, &returned, &perf));
  if (returned <= 0 || perf.status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error("cuDNN failed to select forward algo");
  }
  std::cout << "cudnn fprop algo=" << fwd_algo_to_string(perf.algo)
            << " algo_id=" << static_cast<int>(perf.algo)
            << " est_time_ms=" << perf.time
            << " workspace_bytes=" << perf.memory
            << " math_type=" << math_type_to_string(perf.mathType)
            << " math_type_id=" << static_cast<int>(perf.mathType)
            << "\n";
  cudnnConvolutionFwdAlgo_t algo = perf.algo;

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle.h, d.x, d.w, d.conv, d.y, algo, &ws_size));
  void* ws = nullptr;
  if (ws_size) CUDA_CHECK(cudaMalloc(&ws, ws_size));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto fn = [&]() {
    CUDNN_CHECK(cudnnConvolutionForward(handle.h, &alpha, d.x, d_x, d.w, d_w_krsc, d.conv, algo, ws, ws_size, &beta, d.y, d_y));
  };

  const size_t flops = static_cast<size_t>(2ULL) * n * sh.ho * sh.wo * r * s * (c / p.groups) * k;
  BenchResult out = run_bench(flops, warmup, iters, fn);

  if (ws) CUDA_CHECK(cudaFree(ws));
  CUDA_CHECK(cudaFree(d_w_krsc));
  return out;
}

BenchResult cudnn_bprop_bench(const float* d_dy, const float* d_w, float* d_dx,
                              int n, int h, int w, int c, int r, int s, int k,
                              const Conv2DParams& p,
                              int warmup, int iters) {
  TensorNHWC xt(n, h, w, c);
  FilterHWCN wt(r, s, c / p.groups, k);
  ConvShape sh = infer_conv_shape(xt, wt, p);

  CudnnHandle handle;
  DescPack d;
  setup_descs(d, n, h, w, c, r, s, k, p, sh.ho, sh.wo);

  const int cin_group = c / p.groups;
  const int total_w = k * r * s * cin_group;
  float* d_w_krsc = nullptr;
  CUDA_CHECK(cudaMalloc(&d_w_krsc, static_cast<size_t>(total_w) * sizeof(float)));
  std::vector<float> h_w_hwcn(static_cast<size_t>(total_w));
  std::vector<float> h_w_krsc;
  CUDA_CHECK(cudaMemcpy(h_w_hwcn.data(), d_w, static_cast<size_t>(total_w) * sizeof(float), cudaMemcpyDeviceToHost));
  hwcn_to_krsc_host(h_w_hwcn, h_w_krsc, r, s, cin_group, k);
  CUDA_CHECK(cudaMemcpy(d_w_krsc, h_w_krsc.data(), static_cast<size_t>(total_w) * sizeof(float), cudaMemcpyHostToDevice));

  int returned = 0;
  cudnnConvolutionBwdDataAlgoPerf_t perf{};
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(handle.h, d.w, d.dy, d.conv, d.dx, 1, &returned, &perf));
  if (returned <= 0 || perf.status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error("cuDNN failed to select backward-data algo");
  }
  cudnnConvolutionBwdDataAlgo_t algo = perf.algo;

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle.h, d.w, d.dy, d.conv, d.dx, algo, &ws_size));
  void* ws = nullptr;
  if (ws_size) CUDA_CHECK(cudaMalloc(&ws, ws_size));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto fn = [&]() {
    CUDNN_CHECK(cudnnConvolutionBackwardData(handle.h, &alpha, d.w, d_w_krsc, d.dy, d_dy, d.conv, algo, ws, ws_size, &beta, d.dx, d_dx));
  };

  const size_t flops = static_cast<size_t>(2ULL) * n * sh.ho * sh.wo * r * s * (c / p.groups) * k;
  BenchResult out = run_bench(flops, warmup, iters, fn);

  if (ws) CUDA_CHECK(cudaFree(ws));
  CUDA_CHECK(cudaFree(d_w_krsc));
  return out;
}

BenchResult cudnn_grad_bench(const float* d_x, const float* d_dy, float* d_dw,
                             int n, int h, int w, int c, int r, int s, int k,
                             const Conv2DParams& p,
                             int warmup, int iters) {
  TensorNHWC xt(n, h, w, c);
  FilterHWCN wt(r, s, c / p.groups, k);
  ConvShape sh = infer_conv_shape(xt, wt, p);

  CudnnHandle handle;
  DescPack d;
  setup_descs(d, n, h, w, c, r, s, k, p, sh.ho, sh.wo);

  const int cin_group = c / p.groups;
  const int total_w = k * r * s * cin_group;
  float* d_dw_krsc = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dw_krsc, static_cast<size_t>(total_w) * sizeof(float)));

  int returned = 0;
  cudnnConvolutionBwdFilterAlgoPerf_t perf{};
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle.h, d.x, d.dy, d.conv, d.dw, 1, &returned, &perf));
  if (returned <= 0 || perf.status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error("cuDNN failed to select backward-filter algo");
  }
  cudnnConvolutionBwdFilterAlgo_t algo = perf.algo;

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle.h, d.x, d.dy, d.conv, d.dw, algo, &ws_size));
  void* ws = nullptr;
  if (ws_size) CUDA_CHECK(cudaMalloc(&ws, ws_size));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto fn = [&]() {
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle.h, &alpha, d.x, d_x, d.dy, d_dy, d.conv, algo, ws, ws_size, &beta, d.dw, d_dw_krsc));
  };

  const size_t flops = static_cast<size_t>(2ULL) * n * sh.ho * sh.wo * r * s * (c / p.groups) * k;
  BenchResult out = run_bench(flops, warmup, iters, fn);

  std::vector<float> h_dw_krsc(static_cast<size_t>(total_w));
  std::vector<float> h_dw_hwcn;
  CUDA_CHECK(cudaMemcpy(h_dw_krsc.data(), d_dw_krsc, static_cast<size_t>(total_w) * sizeof(float), cudaMemcpyDeviceToHost));
  krsc_to_hwcn_host(h_dw_krsc, h_dw_hwcn, r, s, cin_group, k);
  CUDA_CHECK(cudaMemcpy(d_dw, h_dw_hwcn.data(), static_cast<size_t>(total_w) * sizeof(float), cudaMemcpyHostToDevice));

  if (ws) CUDA_CHECK(cudaFree(ws));
  CUDA_CHECK(cudaFree(d_dw_krsc));
  return out;
}

#else

bool cudnn_is_available() { return false; }

BenchResult cudnn_fprop_bench(const float*, const float*, float*,
                              int, int, int, int, int, int, int,
                              const Conv2DParams&,
                              int, int) {
  throw std::runtime_error("cuDNN support is not compiled in");
}

BenchResult cudnn_bprop_bench(const float*, const float*, float*,
                              int, int, int, int, int, int, int,
                              const Conv2DParams&,
                              int, int) {
  throw std::runtime_error("cuDNN support is not compiled in");
}

BenchResult cudnn_grad_bench(const float*, const float*, float*,
                             int, int, int, int, int, int, int,
                             const Conv2DParams&,
                             int, int) {
  throw std::runtime_error("cuDNN support is not compiled in");
}

#endif
