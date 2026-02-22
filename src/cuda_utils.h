#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t err, const char* stmt, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error at ") + file + ":" + std::to_string(line) + " in " + stmt + " -> " + cudaGetErrorString(err));
  }
}

#define CUDA_CHECK(stmt) cuda_check((stmt), #stmt, __FILE__, __LINE__)

class CudaEventTimer {
 public:
  CudaEventTimer();
  ~CudaEventTimer();
  void start();
  float stop_ms();

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};
