#include "conv_types.h"
#include "cuda_utils.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

BenchResult benchmark_cuda_op(const std::string& op_name,
                              int warmup,
                              int iters,
                              size_t flops,
                              const std::function<void()>& fn) {
  (void)op_name;
  if (warmup < 0 || iters <= 0) {
    throw std::runtime_error("invalid benchmark warmup/iters");
  }

  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> samples;
  samples.reserve(static_cast<size_t>(iters));

  CudaEventTimer timer;
  for (int i = 0; i < iters; ++i) {
    timer.start();
    fn();
    const float ms = timer.stop_ms();
    samples.push_back(ms);
  }

  std::sort(samples.begin(), samples.end());
  const size_t mid = samples.size() / 2;
  const size_t p90_idx = static_cast<size_t>(0.9f * static_cast<float>(samples.size() - 1));

  BenchResult out;
  out.median_ms = samples[mid];
  out.p90_ms = samples[p90_idx];
  out.gflops = (out.median_ms > 0.0f)
      ? static_cast<float>(static_cast<double>(flops) / (static_cast<double>(out.median_ms) * 1.0e6))
      : 0.0f;
  return out;
}
