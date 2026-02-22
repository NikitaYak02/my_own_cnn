#include "cuda_utils.h"

CudaEventTimer::CudaEventTimer() {
  CUDA_CHECK(cudaEventCreate(&start_));
  CUDA_CHECK(cudaEventCreate(&stop_));
}

CudaEventTimer::~CudaEventTimer() {
  cudaEventDestroy(start_);
  cudaEventDestroy(stop_);
}

void CudaEventTimer::start() {
  CUDA_CHECK(cudaEventRecord(start_));
}

float CudaEventTimer::stop_ms() {
  CUDA_CHECK(cudaEventRecord(stop_));
  CUDA_CHECK(cudaEventSynchronize(stop_));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
  return ms;
}
