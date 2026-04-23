#include "conv_types.h"
#include "cuda_utils.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct LayerSpec {
  const char* name = "";
  int n = 1;
  int h = 0;
  int w = 0;
  int c = 0;
  int k = 0;
  int r = 0;
  int s = 0;
  int pad_h = 0;
  int pad_w = 0;
  int stride_h = 1;
  int stride_w = 1;
  int out_h = 0;
  int out_w = 0;
};

// Regression chain derived from the user's 128x128 network trace.
// It covers every numbered conv layer present in the trace plus merge4_rc,
// yielding 19 successive convolution launches with varying shapes.
std::vector<LayerSpec> make_layer_specs() {
  return {
      {"conv1", 1, 128, 128, 1, 12, 3, 3, 1, 1, 1, 1, 128, 128},
      {"conv2", 1, 128, 128, 12, 16, 3, 3, 1, 1, 2, 2, 64, 64},
      {"conv3", 1, 64, 64, 16, 32, 3, 3, 1, 1, 1, 1, 64, 64},
      {"conv4", 1, 64, 64, 32, 32, 3, 3, 1, 1, 2, 2, 32, 32},
      {"conv5", 1, 32, 32, 32, 32, 3, 3, 1, 1, 1, 1, 32, 32},
      {"conv6", 1, 32, 32, 32, 64, 3, 3, 1, 1, 2, 2, 16, 16},
      {"conv7", 1, 16, 16, 64, 64, 3, 3, 1, 1, 1, 1, 16, 16},
      {"conv8", 1, 16, 16, 64, 64, 3, 3, 1, 1, 2, 2, 8, 8},
      {"conv9", 1, 8, 8, 64, 64, 3, 3, 1, 1, 1, 1, 8, 8},
      {"conv11", 1, 16, 16, 64, 64, 3, 3, 1, 1, 1, 1, 16, 16},
      {"conv12", 1, 16, 16, 64, 32, 3, 3, 1, 1, 1, 1, 16, 16},
      {"conv13", 1, 32, 32, 64, 32, 3, 3, 1, 1, 1, 1, 32, 32},
      {"conv14", 1, 32, 32, 32, 32, 3, 3, 1, 1, 1, 1, 32, 32},
      {"conv15", 1, 64, 64, 64, 16, 3, 3, 1, 1, 1, 1, 64, 64},
      {"conv16", 1, 64, 64, 16, 12, 3, 3, 1, 1, 1, 1, 64, 64},
      {"merge4_rc", 1, 128, 128, 24, 64, 1, 1, 0, 0, 1, 1, 128, 128},
      {"conv17", 1, 128, 128, 64, 12, 3, 3, 1, 1, 1, 1, 128, 128},
      {"conv18", 1, 128, 128, 12, 12, 3, 3, 1, 1, 1, 1, 128, 128},
      {"conv19", 1, 128, 128, 12, 12, 1, 1, 0, 0, 1, 1, 128, 128},
  };
}

void fill_pattern(std::vector<float>& data, int salt) {
  for (size_t i = 0; i < data.size(); ++i) {
    const int code = static_cast<int>((i * 37 + static_cast<size_t>(salt) * 17) % 251);
    data[i] = static_cast<float>(code - 125) / 64.0f;
  }
}

struct PreparedCase {
  LayerSpec spec;
  Conv2DParams p;
  TensorNHWC x;
  FilterKRSC w;
  TensorNHWC y_ref;
  std::vector<float> first_pass_output;
  float* d_x = nullptr;
  float* d_w = nullptr;
  float* d_y = nullptr;

  ~PreparedCase() {
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_y);
  }
};

struct StageTimesMs {
  double memset_ms = 0.0;
  double launch_ms = 0.0;
  double memcpy_ms = 0.0;
  double verify_ms = 0.0;
  int samples = 0;
};

double measure_cuda_stage_ms(const std::function<void()>& fn) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  fn();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return static_cast<double>(ms);
}

std::string describe_case(const PreparedCase& tc) {
  std::ostringstream oss;
  oss << tc.spec.name
      << " n=" << tc.spec.n
      << " h=" << tc.spec.h
      << " w=" << tc.spec.w
      << " c=" << tc.spec.c
      << " k=" << tc.spec.k
      << " r=" << tc.spec.r
      << " s=" << tc.spec.s
      << " pad=" << tc.spec.pad_h << "x" << tc.spec.pad_w
      << " stride=" << tc.spec.stride_h << "x" << tc.spec.stride_w;
  return oss.str();
}

template <typename Fn>
void with_case_context(const PreparedCase& tc, int pass_idx, Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "pass " << (pass_idx + 1) << " " << describe_case(tc) << ": " << e.what();
    throw std::runtime_error(oss.str());
  }
}

void expect_passed(const VerifyResult& vr,
                   const PreparedCase& tc,
                   int pass_idx,
                   const char* label) {
  if (vr.passed) return;

  std::ostringstream oss;
  oss << label << " failed for pass " << (pass_idx + 1)
      << " " << describe_case(tc)
      << " max_abs=" << vr.max_abs_err
      << " max_rel=" << vr.max_rel_err
      << " abs_idx=" << vr.max_abs_idx
      << " rel_idx=" << vr.max_rel_idx;
  throw std::runtime_error(oss.str());
}

std::unique_ptr<PreparedCase> prepare_case(const LayerSpec& spec, int salt) {
  auto tc = std::make_unique<PreparedCase>();
  tc->spec = spec;
  tc->p.pad_h = spec.pad_h;
  tc->p.pad_w = spec.pad_w;
  tc->p.stride_h = spec.stride_h;
  tc->p.stride_w = spec.stride_w;
  tc->p.dilation_h = 1;
  tc->p.dilation_w = 1;
  tc->p.groups = 1;

  tc->x = TensorNHWC(spec.n, spec.h, spec.w, spec.c);
  tc->w = FilterKRSC(spec.r, spec.s, spec.c, spec.k);
  fill_pattern(tc->x.data, salt);
  fill_pattern(tc->w.data, salt + 1000);

  const ConvShape shape = infer_conv_shape(tc->x, tc->w, tc->p);
  if (shape.ho != spec.out_h || shape.wo != spec.out_w) {
    std::ostringstream oss;
    oss << "shape mismatch for " << spec.name
        << ": expected " << spec.out_h << "x" << spec.out_w
        << " got " << shape.ho << "x" << shape.wo;
    throw std::runtime_error(oss.str());
  }

  cpu_fprop_nhwc(tc->x, tc->w, tc->p, tc->y_ref);

  CUDA_CHECK(cudaMalloc(&tc->d_x, tc->x.elements() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&tc->d_w, tc->w.elements() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&tc->d_y, tc->y_ref.elements() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(tc->d_x, tc->x.ptr(), tc->x.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tc->d_w, tc->w.ptr(), tc->w.elements() * sizeof(float), cudaMemcpyHostToDevice));
  return tc;
}

void run_pass(std::vector<std::unique_ptr<PreparedCase>>& cases,
              int pass_idx,
              std::unordered_map<std::string, StageTimesMs>& timing_by_layer) {
  for (auto& tc_ptr : cases) {
    PreparedCase& tc = *tc_ptr;
    with_case_context(tc, pass_idx, [&]() {
      StageTimesMs& t = timing_by_layer[tc.spec.name];

      t.memset_ms += measure_cuda_stage_ms([&]() {
        CUDA_CHECK(cudaMemset(tc.d_y, 0, tc.y_ref.elements() * sizeof(float)));
      });

      t.launch_ms += measure_cuda_stage_ms([&]() {
        launch_fprop_nhwc(tc.d_x, tc.d_w, tc.d_y,
                          tc.spec.n, tc.spec.h, tc.spec.w, tc.spec.c,
                          tc.spec.r, tc.spec.s, tc.spec.k, tc.p);
        CUDA_CHECK(cudaPeekAtLastError());
      });

      std::vector<float> y_cuda(tc.y_ref.elements());
      t.memcpy_ms += measure_cuda_stage_ms([&]() {
        CUDA_CHECK(cudaMemcpy(y_cuda.data(), tc.d_y, y_cuda.size() * sizeof(float), cudaMemcpyDeviceToHost));
      });

      const auto verify_start = std::chrono::steady_clock::now();

      const VerifyResult ref_vr = verify_tensors(tc.y_ref.data, y_cuda, 1e-4f, 1e-3f);
      expect_passed(ref_vr, tc, pass_idx, "custom_vs_cpu");

      if (pass_idx == 0) {
        tc.first_pass_output = std::move(y_cuda);
      } else {
        const VerifyResult replay_vr = verify_tensors(tc.first_pass_output, y_cuda, 0.0f, 0.0f);
        expect_passed(replay_vr, tc, pass_idx, "replay_consistency");
      }
      const auto verify_end = std::chrono::steady_clock::now();
      t.verify_ms += std::chrono::duration<double, std::milli>(verify_end - verify_start).count();
      t.samples += 1;
    });
  }
}

}  // namespace

int main() {
  try {
    std::unordered_map<std::string, StageTimesMs> timing_by_layer;
    std::vector<std::unique_ptr<PreparedCase>> cases;
    const std::vector<LayerSpec> specs = make_layer_specs();
    cases.reserve(specs.size());
    for (size_t i = 0; i < specs.size(); ++i) {
      cases.push_back(prepare_case(specs[i], static_cast<int>(i) * 13 + 7));
    }

    run_pass(cases, 0, timing_by_layer);
    run_pass(cases, 1, timing_by_layer);

    std::cout << "conv_chain_test passed\n";
    std::cout << "cases=" << cases.size() << " passes=2\n";
    std::cout << "timing_breakdown_ms_per_layer:\n";
    for (const auto& tc_ptr : cases) {
      const auto it = timing_by_layer.find(tc_ptr->spec.name);
      if (it == timing_by_layer.end() || it->second.samples == 0) continue;
      const StageTimesMs& t = it->second;
      const double inv = 1.0 / static_cast<double>(t.samples);
      std::cout << "  " << tc_ptr->spec.name
                << " memset=" << (t.memset_ms * inv)
                << " launch=" << (t.launch_ms * inv)
                << " memcpy=" << (t.memcpy_ms * inv)
                << " verify=" << (t.verify_ms * inv)
                << " samples=" << t.samples
                << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "conv_chain_test failed: " << e.what() << "\n";
    return 1;
  }
}
