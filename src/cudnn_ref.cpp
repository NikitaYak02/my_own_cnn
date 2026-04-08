#include "conv_types.h"
#include "cuda_utils.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef USE_CUDNN
#include <cudnn.h>
#include <cudnn_cnn.h>

void gather_input_block_nhwc(const float* d_src, float* d_dst,
                             int n, int src_h, int src_w, int c,
                             int hi_start, int wi_start,
                             int local_h, int local_w);
void gather_output_block_nhwc(const float* d_src, float* d_dst,
                              int n, int src_h, int src_w, int c,
                              int ho_start, int wo_start,
                              int block_ho, int block_wo);
void scatter_output_block_nhwc(const float* d_src, float* d_dst,
                               int n, int dst_h, int dst_w, int c,
                               int ho_start, int wo_start,
                               int block_ho, int block_wo);
void scatter_add_input_block_nhwc(const float* d_src, float* d_dst,
                                  int n, int dst_h, int dst_w, int c,
                                  int hi_start, int wi_start,
                                  int local_h, int local_w);
void gather_block_filter_to_krsc(const float* d_src, float* d_dst,
                                 int by_count, int bx_count,
                                 int r, int s, int cin_group, int k,
                                 int block_y, int block_x);
void scatter_block_filter_from_krsc(const float* d_src, float* d_dst,
                                    int by_count, int bx_count,
                                    int r, int s, int cin_group, int k,
                                    int block_y, int block_x);

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
  if (p.ay != 1 || p.ax != 1) {
    throw std::runtime_error("cuDNN reference does not support ay/ax != 1");
  }
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

const char* bwd_data_algo_to_string(cudnnConvolutionBwdDataAlgo_t algo) {
  switch (algo) {
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT";
    default: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_UNKNOWN";
  }
}

const char* bwd_filter_algo_to_string(cudnnConvolutionBwdFilterAlgo_t algo) {
  switch (algo) {
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT";
    default: return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_UNKNOWN";
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

struct BlockWindow {
  int by = 0;
  int bx = 0;
  int ho_start = 0;
  int wo_start = 0;
  int hi_start = 0;
  int wi_start = 0;
};

struct BlockBuffers {
  float* d_x = nullptr;
  float* d_w = nullptr;
  float* d_y = nullptr;
  float* d_dy = nullptr;
  float* d_dx = nullptr;
  float* d_dw = nullptr;
};

void free_block_buffers(std::vector<BlockBuffers>& blocks) {
  for (BlockBuffers& block : blocks) {
    cudaFree(block.d_x);
    cudaFree(block.d_w);
    cudaFree(block.d_y);
    cudaFree(block.d_dy);
    cudaFree(block.d_dx);
    cudaFree(block.d_dw);
    block = {};
  }
}

std::vector<BlockWindow> enumerate_block_windows(const BlockConvShape& shape,
                                                 const BlockConv2DParams& p,
                                                 int r, int s) {
  std::vector<BlockWindow> blocks;
  blocks.reserve(static_cast<size_t>(p.block_by) * p.block_bx);
  for (int by = 0; by < p.block_by; ++by) {
    const int ho_start = by * shape.block_ho;
    const int hi_start = ho_start * p.conv.stride_h - p.conv.pad_h;
    for (int bx = 0; bx < p.block_bx; ++bx) {
      const int wo_start = bx * shape.block_wo;
      const int wi_start = wo_start * p.conv.stride_w - p.conv.pad_w;
      blocks.push_back(BlockWindow{by, bx, ho_start, wo_start, hi_start, wi_start});
    }
  }
  return blocks;
}

size_t blocked_conv_flops(int n, int block_ho, int block_wo,
                          int by_count, int bx_count,
                          int r, int s, int cin_group, int k) {
  return static_cast<size_t>(2ULL) * n * block_ho * block_wo * by_count * bx_count * r * s * cin_group * k;
}

}  // namespace

bool cudnn_is_available() { return true; }

BenchResult cudnn_fprop_bench(const float* d_x, const float* d_w, float* d_y,
                              int n, int h, int w, int c, int r, int s, int k,
                              const Conv2DParams& p,
                              int warmup, int iters) {
  TensorNHWC xt(n, h, w, c);
  FilterKRSC wt(r, s, c / p.groups, k);
  ConvShape sh = infer_conv_shape(xt, wt, p);

  CudnnHandle handle;
  DescPack d;
  setup_descs(d, n, h, w, c, r, s, k, p, sh.ho, sh.wo);

  const int cin_group = c / p.groups;
  const int total_w = k * r * s * cin_group;
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
    CUDNN_CHECK(cudnnConvolutionForward(handle.h, &alpha, d.x, d_x, d.w, d_w, d.conv, algo, ws, ws_size, &beta, d.y, d_y));
  };

  const size_t flops = static_cast<size_t>(2ULL) * n * sh.ho * sh.wo * r * s * (c / p.groups) * k;
  BenchResult out = run_bench(flops, warmup, iters, fn);

  if (ws) CUDA_CHECK(cudaFree(ws));
  return out;
}

BenchResult cudnn_bprop_bench(const float* d_dy, const float* d_w, float* d_dx,
                              int n, int h, int w, int c, int r, int s, int k,
                              const Conv2DParams& p,
                              int warmup, int iters) {
  TensorNHWC xt(n, h, w, c);
  FilterKRSC wt(r, s, c / p.groups, k);
  ConvShape sh = infer_conv_shape(xt, wt, p);

  CudnnHandle handle;
  DescPack d;
  setup_descs(d, n, h, w, c, r, s, k, p, sh.ho, sh.wo);

  const int cin_group = c / p.groups;
  const int total_w = k * r * s * cin_group;
  int returned = 0;
  cudnnConvolutionBwdDataAlgoPerf_t perf{};
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(handle.h, d.w, d.dy, d.conv, d.dx, 1, &returned, &perf));
  if (returned <= 0 || perf.status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error("cuDNN failed to select backward-data algo");
  }
  std::cout << "cudnn bprop algo=" << bwd_data_algo_to_string(perf.algo)
            << " algo_id=" << static_cast<int>(perf.algo)
            << " est_time_ms=" << perf.time
            << " workspace_bytes=" << perf.memory
            << " math_type=" << math_type_to_string(perf.mathType)
            << " math_type_id=" << static_cast<int>(perf.mathType)
            << "\n";
  cudnnConvolutionBwdDataAlgo_t algo = perf.algo;

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle.h, d.w, d.dy, d.conv, d.dx, algo, &ws_size));
  void* ws = nullptr;
  if (ws_size) CUDA_CHECK(cudaMalloc(&ws, ws_size));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto fn = [&]() {
    CUDNN_CHECK(cudnnConvolutionBackwardData(handle.h, &alpha, d.w, d_w, d.dy, d_dy, d.conv, algo, ws, ws_size, &beta, d.dx, d_dx));
  };

  const size_t flops = static_cast<size_t>(2ULL) * n * sh.ho * sh.wo * r * s * (c / p.groups) * k;
  BenchResult out = run_bench(flops, warmup, iters, fn);

  if (ws) CUDA_CHECK(cudaFree(ws));
  return out;
}

BenchResult cudnn_grad_bench(const float* d_x, const float* d_dy, float* d_dw,
                             int n, int h, int w, int c, int r, int s, int k,
                             const Conv2DParams& p,
                             int warmup, int iters) {
  TensorNHWC xt(n, h, w, c);
  FilterKRSC wt(r, s, c / p.groups, k);
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
  std::cout << "cudnn grad algo=" << bwd_filter_algo_to_string(perf.algo)
            << " algo_id=" << static_cast<int>(perf.algo)
            << " est_time_ms=" << perf.time
            << " workspace_bytes=" << perf.memory
            << " math_type=" << math_type_to_string(perf.mathType)
            << " math_type_id=" << static_cast<int>(perf.mathType)
            << "\n";
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

  CUDA_CHECK(cudaMemcpy(d_dw, d_dw_krsc, static_cast<size_t>(total_w) * sizeof(float), cudaMemcpyDeviceToDevice));

  if (ws) CUDA_CHECK(cudaFree(ws));
  CUDA_CHECK(cudaFree(d_dw_krsc));
  return out;
}

BenchResult cudnn_block_fprop_bench(const float* d_x, const float* d_w, float* d_y,
                                    int n, int h, int w, int c, int r, int s, int k,
                                    const BlockConv2DParams& p,
                                    int warmup, int iters) {
  TensorNHWC xt(n, h, w, c);
  BlockFilterKByBxRSC wt(k, p.block_by, p.block_bx, r, s, c / p.conv.groups);
  const BlockConvShape sh = infer_block_conv_shape(xt, wt, p);
  const int cin_group = c / p.conv.groups;
  const int local_h = (sh.block_ho - 1) * p.conv.stride_h + (r - 1) * p.conv.dilation_h + 1;
  const int local_w = (sh.block_wo - 1) * p.conv.stride_w + (s - 1) * p.conv.dilation_w + 1;
  const int total_w = k * r * s * cin_group;
  const size_t x_elems = static_cast<size_t>(n) * local_h * local_w * c;
  const size_t y_elems = static_cast<size_t>(n) * sh.block_ho * sh.block_wo * k;
  const std::vector<BlockWindow> windows = enumerate_block_windows(sh, p, r, s);

  Conv2DParams local_p = p.conv;
  local_p.pad_h = 0;
  local_p.pad_w = 0;
  TensorNHWC local_x_shape(n, local_h, local_w, c);
  FilterKRSC local_w_shape(r, s, cin_group, k);
  const ConvShape local_sh = infer_conv_shape(local_x_shape, local_w_shape, local_p);
  if (local_sh.ho != sh.block_ho || local_sh.wo != sh.block_wo) {
    throw std::runtime_error("blocked cuDNN fprop local shape mismatch");
  }

  CudnnHandle handle;
  DescPack d;
  setup_descs(d, n, local_h, local_w, c, r, s, k, local_p, sh.block_ho, sh.block_wo);

  int returned = 0;
  cudnnConvolutionFwdAlgoPerf_t perf{};
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(handle.h, d.x, d.w, d.conv, d.y, 1, &returned, &perf));
  if (returned <= 0 || perf.status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error("cuDNN failed to select blocked forward algo");
  }
  cudnnConvolutionFwdAlgo_t algo = perf.algo;
  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle.h, d.x, d.w, d.conv, d.y, algo, &ws_size));
  void* ws = nullptr;
  if (ws_size) CUDA_CHECK(cudaMalloc(&ws, ws_size));

  std::vector<BlockBuffers> blocks(windows.size());
  for (size_t i = 0; i < windows.size(); ++i) {
    const BlockWindow& window = windows[i];
    CUDA_CHECK(cudaMalloc(&blocks[i].d_x, x_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&blocks[i].d_w, static_cast<size_t>(total_w) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&blocks[i].d_y, y_elems * sizeof(float)));

    gather_input_block_nhwc(d_x, blocks[i].d_x, n, h, w, c,
                            window.hi_start, window.wi_start,
                            local_h, local_w);
    gather_block_filter_to_krsc(d_w, blocks[i].d_w,
                                p.block_by, p.block_bx,
                                r, s, cin_group, k,
                                window.by, window.bx);
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto fn = [&]() {
    for (const BlockBuffers& block : blocks) {
      CUDNN_CHECK(cudnnConvolutionForward(handle.h, &alpha,
                                          d.x, block.d_x,
                                          d.w, block.d_w,
                                          d.conv, algo,
                                          ws, ws_size,
                                          &beta,
                                          d.y, block.d_y));
    }
  };

  BenchResult out = run_bench(blocked_conv_flops(n, sh.block_ho, sh.block_wo, p.block_by, p.block_bx, r, s, cin_group, k),
                              warmup, iters, fn);

  for (size_t i = 0; i < windows.size(); ++i) {
    const BlockWindow& window = windows[i];
    scatter_output_block_nhwc(blocks[i].d_y, d_y,
                              n, sh.base.ho, sh.base.wo, k,
                              window.ho_start, window.wo_start,
                              sh.block_ho, sh.block_wo);
  }

  free_block_buffers(blocks);
  if (ws) CUDA_CHECK(cudaFree(ws));
  return out;
}

BenchResult cudnn_block_bprop_bench(const float* d_dy, const float* d_w, float* d_dx,
                                    int n, int h, int w, int c, int r, int s, int k,
                                    const BlockConv2DParams& p,
                                    int warmup, int iters) {
  TensorNHWC xt(n, h, w, c);
  BlockFilterKByBxRSC wt(k, p.block_by, p.block_bx, r, s, c / p.conv.groups);
  const BlockConvShape sh = infer_block_conv_shape(xt, wt, p);
  const int cin_group = c / p.conv.groups;
  const int local_h = (sh.block_ho - 1) * p.conv.stride_h + (r - 1) * p.conv.dilation_h + 1;
  const int local_w = (sh.block_wo - 1) * p.conv.stride_w + (s - 1) * p.conv.dilation_w + 1;
  const int total_w = k * r * s * cin_group;
  const size_t dy_elems = static_cast<size_t>(n) * sh.block_ho * sh.block_wo * k;
  const size_t dx_elems = static_cast<size_t>(n) * local_h * local_w * c;
  const std::vector<BlockWindow> windows = enumerate_block_windows(sh, p, r, s);

  Conv2DParams local_p = p.conv;
  local_p.pad_h = 0;
  local_p.pad_w = 0;
  TensorNHWC local_x_shape(n, local_h, local_w, c);
  FilterKRSC local_w_shape(r, s, cin_group, k);
  const ConvShape local_sh = infer_conv_shape(local_x_shape, local_w_shape, local_p);
  if (local_sh.ho != sh.block_ho || local_sh.wo != sh.block_wo) {
    throw std::runtime_error("blocked cuDNN bprop local shape mismatch");
  }

  CudnnHandle handle;
  DescPack d;
  setup_descs(d, n, local_h, local_w, c, r, s, k, local_p, sh.block_ho, sh.block_wo);

  int returned = 0;
  cudnnConvolutionBwdDataAlgoPerf_t perf{};
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(handle.h, d.w, d.dy, d.conv, d.dx, 1, &returned, &perf));
  if (returned <= 0 || perf.status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error("cuDNN failed to select blocked backward-data algo");
  }
  std::cout << "cudnn bprop algo=" << bwd_data_algo_to_string(perf.algo)
            << " algo_id=" << static_cast<int>(perf.algo)
            << " est_time_ms=" << perf.time
            << " workspace_bytes=" << perf.memory
            << " math_type=" << math_type_to_string(perf.mathType)
            << " math_type_id=" << static_cast<int>(perf.mathType)
            << "\n";
  cudnnConvolutionBwdDataAlgo_t algo = perf.algo;
  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle.h, d.w, d.dy, d.conv, d.dx, algo, &ws_size));
  void* ws = nullptr;
  if (ws_size) CUDA_CHECK(cudaMalloc(&ws, ws_size));

  std::vector<BlockBuffers> blocks(windows.size());
  for (size_t i = 0; i < windows.size(); ++i) {
    const BlockWindow& window = windows[i];
    CUDA_CHECK(cudaMalloc(&blocks[i].d_w, static_cast<size_t>(total_w) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&blocks[i].d_dy, dy_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&blocks[i].d_dx, dx_elems * sizeof(float)));

    gather_output_block_nhwc(d_dy, blocks[i].d_dy,
                             n, sh.base.ho, sh.base.wo, k,
                             window.ho_start, window.wo_start,
                             sh.block_ho, sh.block_wo);
    gather_block_filter_to_krsc(d_w, blocks[i].d_w,
                                p.block_by, p.block_bx,
                                r, s, cin_group, k,
                                window.by, window.bx);
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto fn = [&]() {
    for (const BlockBuffers& block : blocks) {
      CUDNN_CHECK(cudnnConvolutionBackwardData(handle.h, &alpha,
                                               d.w, block.d_w,
                                               d.dy, block.d_dy,
                                               d.conv, algo,
                                               ws, ws_size,
                                               &beta,
                                               d.dx, block.d_dx));
    }
  };

  BenchResult out = run_bench(blocked_conv_flops(n, sh.block_ho, sh.block_wo, p.block_by, p.block_bx, r, s, cin_group, k),
                              warmup, iters, fn);

  CUDA_CHECK(cudaMemset(d_dx, 0, static_cast<size_t>(n) * h * w * c * sizeof(float)));
  for (size_t i = 0; i < windows.size(); ++i) {
    const BlockWindow& window = windows[i];
    scatter_add_input_block_nhwc(blocks[i].d_dx, d_dx,
                                 n, h, w, c,
                                 window.hi_start, window.wi_start,
                                 local_h, local_w);
  }

  free_block_buffers(blocks);
  if (ws) CUDA_CHECK(cudaFree(ws));
  return out;
}

BenchResult cudnn_block_grad_bench(const float* d_x, const float* d_dy, float* d_dw,
                                   int n, int h, int w, int c, int r, int s, int k,
                                   const BlockConv2DParams& p,
                                   int warmup, int iters) {
  TensorNHWC xt(n, h, w, c);
  BlockFilterKByBxRSC wt(k, p.block_by, p.block_bx, r, s, c / p.conv.groups);
  const BlockConvShape sh = infer_block_conv_shape(xt, wt, p);
  const int cin_group = c / p.conv.groups;
  const int local_h = (sh.block_ho - 1) * p.conv.stride_h + (r - 1) * p.conv.dilation_h + 1;
  const int local_w = (sh.block_wo - 1) * p.conv.stride_w + (s - 1) * p.conv.dilation_w + 1;
  const int total_w = k * r * s * cin_group;
  const size_t x_elems = static_cast<size_t>(n) * local_h * local_w * c;
  const size_t dy_elems = static_cast<size_t>(n) * sh.block_ho * sh.block_wo * k;
  const std::vector<BlockWindow> windows = enumerate_block_windows(sh, p, r, s);

  Conv2DParams local_p = p.conv;
  local_p.pad_h = 0;
  local_p.pad_w = 0;
  TensorNHWC local_x_shape(n, local_h, local_w, c);
  FilterKRSC local_w_shape(r, s, cin_group, k);
  const ConvShape local_sh = infer_conv_shape(local_x_shape, local_w_shape, local_p);
  if (local_sh.ho != sh.block_ho || local_sh.wo != sh.block_wo) {
    throw std::runtime_error("blocked cuDNN grad local shape mismatch");
  }

  CudnnHandle handle;
  DescPack d;
  setup_descs(d, n, local_h, local_w, c, r, s, k, local_p, sh.block_ho, sh.block_wo);

  int returned = 0;
  cudnnConvolutionBwdFilterAlgoPerf_t perf{};
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle.h, d.x, d.dy, d.conv, d.dw, 1, &returned, &perf));
  if (returned <= 0 || perf.status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error("cuDNN failed to select blocked backward-filter algo");
  }
  std::cout << "cudnn grad algo=" << bwd_filter_algo_to_string(perf.algo)
            << " algo_id=" << static_cast<int>(perf.algo)
            << " est_time_ms=" << perf.time
            << " workspace_bytes=" << perf.memory
            << " math_type=" << math_type_to_string(perf.mathType)
            << " math_type_id=" << static_cast<int>(perf.mathType)
            << "\n";
  cudnnConvolutionBwdFilterAlgo_t algo = perf.algo;
  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle.h, d.x, d.dy, d.conv, d.dw, algo, &ws_size));
  void* ws = nullptr;
  if (ws_size) CUDA_CHECK(cudaMalloc(&ws, ws_size));

  std::vector<BlockBuffers> blocks(windows.size());
  for (size_t i = 0; i < windows.size(); ++i) {
    const BlockWindow& window = windows[i];
    CUDA_CHECK(cudaMalloc(&blocks[i].d_x, x_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&blocks[i].d_dy, dy_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&blocks[i].d_dw, static_cast<size_t>(total_w) * sizeof(float)));

    gather_input_block_nhwc(d_x, blocks[i].d_x, n, h, w, c,
                            window.hi_start, window.wi_start,
                            local_h, local_w);
    gather_output_block_nhwc(d_dy, blocks[i].d_dy,
                             n, sh.base.ho, sh.base.wo, k,
                             window.ho_start, window.wo_start,
                             sh.block_ho, sh.block_wo);
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto fn = [&]() {
    for (BlockBuffers& block : blocks) {
      CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle.h, &alpha,
                                                 d.x, block.d_x,
                                                 d.dy, block.d_dy,
                                                 d.conv, algo,
                                                 ws, ws_size,
                                                 &beta,
                                                 d.dw, block.d_dw));
    }
  };

  BenchResult out = run_bench(blocked_conv_flops(n, sh.block_ho, sh.block_wo, p.block_by, p.block_bx, r, s, cin_group, k),
                              warmup, iters, fn);

  CUDA_CHECK(cudaMemset(d_dw, 0, static_cast<size_t>(k) * p.block_by * p.block_bx * r * s * cin_group * sizeof(float)));
  for (size_t i = 0; i < windows.size(); ++i) {
    const BlockWindow& window = windows[i];
    scatter_block_filter_from_krsc(blocks[i].d_dw, d_dw,
                                   p.block_by, p.block_bx,
                                   r, s, cin_group, k,
                                   window.by, window.bx);
  }

  free_block_buffers(blocks);
  if (ws) CUDA_CHECK(cudaFree(ws));
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

BenchResult cudnn_block_fprop_bench(const float*, const float*, float*,
                                    int, int, int, int, int, int, int,
                                    const BlockConv2DParams&,
                                    int, int) {
  throw std::runtime_error("cuDNN support is not compiled in");
}

BenchResult cudnn_block_bprop_bench(const float*, const float*, float*,
                                    int, int, int, int, int, int, int,
                                    const BlockConv2DParams&,
                                    int, int) {
  throw std::runtime_error("cuDNN support is not compiled in");
}

BenchResult cudnn_block_grad_bench(const float*, const float*, float*,
                                   int, int, int, int, int, int, int,
                                   const BlockConv2DParams&,
                                   int, int) {
  throw std::runtime_error("cuDNN support is not compiled in");
}

#endif
