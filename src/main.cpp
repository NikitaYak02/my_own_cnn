#include "conv_types.h"
#include "cuda_utils.h"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Options {
  std::string op = "all";
  std::string custom_mode = "explicit";
  int n = 32;
  int h = 56;
  int w = 56;
  int c = 64;
  int k = 64;
  int r = 3;
  int s = 3;
  int pad_h = 1;
  int pad_w = 1;
  int stride_h = 1;
  int stride_w = 1;
  int dilation_h = 1;
  int dilation_w = 1;
  int groups = 1;
  int warmup = 10;
  int iters = 50;
  bool check = false;
  bool with_cudnn = false;
  bool bench_suite = false;
  uint32_t seed = 42;
};

int parse_int(const std::string& v, const char* name) {
  try {
    return std::stoi(v);
  } catch (...) {
    throw std::runtime_error(std::string("invalid integer for ") + name + ": " + v);
  }
}

Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need_val = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return std::string(argv[++i]);
    };

    if (a == "--op") o.op = need_val("--op");
    else if (a == "--custom_mode") o.custom_mode = need_val("--custom_mode");
    else if (a == "--n") o.n = parse_int(need_val("--n"), "--n");
    else if (a == "--h") o.h = parse_int(need_val("--h"), "--h");
    else if (a == "--w") o.w = parse_int(need_val("--w"), "--w");
    else if (a == "--c") o.c = parse_int(need_val("--c"), "--c");
    else if (a == "--k") o.k = parse_int(need_val("--k"), "--k");
    else if (a == "--r") o.r = parse_int(need_val("--r"), "--r");
    else if (a == "--s") o.s = parse_int(need_val("--s"), "--s");
    else if (a == "--pad_h") o.pad_h = parse_int(need_val("--pad_h"), "--pad_h");
    else if (a == "--pad_w") o.pad_w = parse_int(need_val("--pad_w"), "--pad_w");
    else if (a == "--stride_h") o.stride_h = parse_int(need_val("--stride_h"), "--stride_h");
    else if (a == "--stride_w") o.stride_w = parse_int(need_val("--stride_w"), "--stride_w");
    else if (a == "--dilation_h") o.dilation_h = parse_int(need_val("--dilation_h"), "--dilation_h");
    else if (a == "--dilation_w") o.dilation_w = parse_int(need_val("--dilation_w"), "--dilation_w");
    else if (a == "--groups") o.groups = parse_int(need_val("--groups"), "--groups");
    else if (a == "--warmup") o.warmup = parse_int(need_val("--warmup"), "--warmup");
    else if (a == "--iters") o.iters = parse_int(need_val("--iters"), "--iters");
    else if (a == "--seed") o.seed = static_cast<uint32_t>(parse_int(need_val("--seed"), "--seed"));
    else if (a == "--check") o.check = true;
    else if (a == "--with_cudnn") o.with_cudnn = true;
    else if (a == "--bench_suite") o.bench_suite = true;
    else if (a == "--help") {
      std::cout
          << "conv_bench options:\n"
          << "  --op fprop|bprop|grad|all\n"
          << "  --custom_mode explicit|implicit_precomp\n"
          << "  --n --h --w --c --k --r --s\n"
          << "  --pad_h --pad_w --stride_h --stride_w --dilation_h --dilation_w --groups\n"
          << "  --warmup --iters --seed\n"
          << "  --check --with_cudnn --bench_suite\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + a);
    }
  }
  return o;
}

void fill_random(std::vector<float>& v, std::mt19937& gen, float lo = -1.0f, float hi = 1.0f) {
  std::uniform_real_distribution<float> dist(lo, hi);
  for (float& x : v) x = dist(gen);
}

void print_bench(const std::string& tag, const BenchResult& b) {
  std::cout << std::fixed << std::setprecision(3)
            << tag << " median=" << b.median_ms << " ms"
            << " p90=" << b.p90_ms << " ms"
            << " gflops=" << b.gflops << "\n";
}

void ensure(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error(msg);
}

size_t conv_flops(int n, int ho, int wo, int r, int s, int cin_group, int k) {
  return static_cast<size_t>(2ULL) * n * ho * wo * r * s * cin_group * k;
}

struct BenchCase {
  std::string name;
  int n;
  int h;
  int w;
  int c;
  int k;
  int r;
  int s;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int groups;
};

struct CaseRatios {
  float fprop = -1.0f;
  float bprop = -1.0f;
  float grad = -1.0f;
};

float vec_min(std::vector<float> v) {
  if (v.empty()) return -1.0f;
  return *std::min_element(v.begin(), v.end());
}

float vec_median(std::vector<float> v) {
  if (v.empty()) return -1.0f;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

float vec_max(std::vector<float> v) {
  if (v.empty()) return -1.0f;
  return *std::max_element(v.begin(), v.end());
}

std::vector<BenchCase> default_bench_suite() {
  return {
      {"resnet_like_56x56", 32, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1},
      {"resnet_like_28x28", 32, 28, 28, 128, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1},
      {"downsample_stride2", 32, 112, 112, 32, 32, 3, 3, 1, 1, 2, 2, 1, 1, 1},
      {"depthwise_3x3", 32, 56, 56, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 128},
      {"grouped_g4", 32, 56, 56, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 4},
      {"pointwise_1x1", 32, 56, 56, 128, 256, 1, 1, 0, 0, 1, 1, 1, 1, 1},
      {"dilated_3x3", 16, 64, 64, 64, 64, 3, 3, 2, 2, 1, 1, 2, 2, 1},
      {"small_batch", 1, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1},
  };
}

CaseRatios run_case(const Options& o, const BenchCase& bc) {
  CaseRatios ratios;
  Conv2DParams p;
  p.pad_h = bc.pad_h;
  p.pad_w = bc.pad_w;
  p.stride_h = bc.stride_h;
  p.stride_w = bc.stride_w;
  p.dilation_h = bc.dilation_h;
  p.dilation_w = bc.dilation_w;
  p.groups = bc.groups;

  TensorNHWC x(bc.n, bc.h, bc.w, bc.c);
  FilterHWCN w(bc.r, bc.s, bc.c / bc.groups, bc.k);
  ConvShape sh = infer_conv_shape(x, w, p);

  TensorNHWC y(bc.n, sh.ho, sh.wo, bc.k);
  TensorNHWC dy(bc.n, sh.ho, sh.wo, bc.k);
  TensorNHWC dx(bc.n, bc.h, bc.w, bc.c);
  FilterHWCN dw(bc.r, bc.s, bc.c / bc.groups, bc.k);

  std::mt19937 gen(o.seed);
  fill_random(x.data, gen);
  fill_random(w.data, gen);
  fill_random(dy.data, gen);

  float *d_x = nullptr, *d_w = nullptr, *d_y = nullptr, *d_dy = nullptr, *d_dx = nullptr, *d_dw = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, x.elements() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_w, w.elements() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, y.elements() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dy, dy.elements() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dx, dx.elements() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dw, dw.elements() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_x, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, w.ptr(), w.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dy, dy.ptr(), dy.elements() * sizeof(float), cudaMemcpyHostToDevice));

  const bool do_fprop = (o.op == "fprop" || o.op == "all");
  const bool do_bprop = (o.op == "bprop" || o.op == "all");
  const bool do_grad = (o.op == "grad" || o.op == "all");
  ensure(do_fprop || do_bprop || do_grad, "--op must be fprop|bprop|grad|all");
  const bool use_implicit_precomp = (o.custom_mode == "implicit_precomp");
  ensure(use_implicit_precomp || o.custom_mode == "explicit", "--custom_mode must be explicit|implicit_precomp");

  const size_t flops = conv_flops(bc.n, sh.ho, sh.wo, bc.r, bc.s, sh.cin_group, bc.k);
  std::cout << "\n[case] " << bc.name
            << " n=" << bc.n << " h=" << bc.h << " w=" << bc.w
            << " c=" << bc.c << " k=" << bc.k
            << " r=" << bc.r << " s=" << bc.s
            << " stride=" << bc.stride_h << "x" << bc.stride_w
            << " pad=" << bc.pad_h << "x" << bc.pad_w
            << " dilation=" << bc.dilation_h << "x" << bc.dilation_w
            << " groups=" << bc.groups << "\n";

  if (do_fprop) {
    BenchResult b = benchmark_cuda_op("fprop", o.warmup, o.iters, flops, [&]() {
      launch_fprop_nhwc(d_x, d_w, d_y, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, p, use_implicit_precomp);
    });
    print_bench("custom fprop", b);
    if (o.with_cudnn) {
      ensure(cudnn_is_available(), "--with_cudnn requested, but cuDNN is unavailable");
      BenchResult cb = cudnn_fprop_bench(d_x, d_w, d_y, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, p, o.warmup, o.iters);
      print_bench("cudnn fprop", cb);
      ratios.fprop = b.median_ms / std::max(cb.median_ms, 1e-6f);
      std::cout << "ratio fprop custom/cudnn=" << std::fixed << std::setprecision(3) << ratios.fprop << "x\n";
    }
  }

  if (do_bprop) {
    BenchResult b = benchmark_cuda_op("bprop", o.warmup, o.iters, flops, [&]() {
      launch_bprop_nhwc(d_dy, d_w, d_dx, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, p, use_implicit_precomp);
    });
    print_bench("custom bprop", b);
    if (o.with_cudnn) {
      ensure(cudnn_is_available(), "--with_cudnn requested, but cuDNN is unavailable");
      BenchResult cb = cudnn_bprop_bench(d_dy, d_w, d_dx, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, p, o.warmup, o.iters);
      print_bench("cudnn bprop", cb);
      ratios.bprop = b.median_ms / std::max(cb.median_ms, 1e-6f);
      std::cout << "ratio bprop custom/cudnn=" << std::fixed << std::setprecision(3) << ratios.bprop << "x\n";
    }
  }

  if (do_grad) {
    BenchResult b = benchmark_cuda_op("grad", o.warmup, o.iters, flops, [&]() {
      launch_grad_nhwc(d_x, d_dy, d_dw, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, p, use_implicit_precomp);
    });
    print_bench("custom grad", b);
    if (o.with_cudnn) {
      ensure(cudnn_is_available(), "--with_cudnn requested, but cuDNN is unavailable");
      BenchResult cb = cudnn_grad_bench(d_x, d_dy, d_dw, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, p, o.warmup, o.iters);
      print_bench("cudnn grad", cb);
      ratios.grad = b.median_ms / std::max(cb.median_ms, 1e-6f);
      std::cout << "ratio grad custom/cudnn=" << std::fixed << std::setprecision(3) << ratios.grad << "x\n";
    }
  }

  if (o.check) {
    if (do_fprop) {
      TensorNHWC y_cpu;
      cpu_fprop_nhwc(x, w, p, y_cpu);
      CUDA_CHECK(cudaMemcpy(y.ptr(), d_y, y.elements() * sizeof(float), cudaMemcpyDeviceToHost));
      VerifyResult vr = verify_tensors(y_cpu.data, y.data, 1e-4f, 1e-3f);
      std::cout << "check fprop pass=" << (vr.passed ? "yes" : "no")
                << " max_abs=" << vr.max_abs_err << " max_rel=" << vr.max_rel_err << "\n";
    }
    if (do_bprop) {
      TensorNHWC dx_cpu(bc.n, bc.h, bc.w, bc.c);
      cpu_bprop_nhwc(dy, w, p, dx_cpu);
      CUDA_CHECK(cudaMemcpy(dx.ptr(), d_dx, dx.elements() * sizeof(float), cudaMemcpyDeviceToHost));
      VerifyResult vr = verify_tensors(dx_cpu.data, dx.data, 1e-4f, 1e-3f);
      std::cout << "check bprop pass=" << (vr.passed ? "yes" : "no")
                << " max_abs=" << vr.max_abs_err << " max_rel=" << vr.max_rel_err << "\n";
    }
    if (do_grad) {
      FilterHWCN dw_cpu(bc.r, bc.s, bc.c / bc.groups, bc.k);
      cpu_grad_nhwc(x, dy, p, dw_cpu);
      CUDA_CHECK(cudaMemcpy(dw.ptr(), d_dw, dw.elements() * sizeof(float), cudaMemcpyDeviceToHost));
      VerifyResult vr = verify_tensors(dw_cpu.data, dw.data, 1e-4f, 1e-3f);
      std::cout << "check grad pass=" << (vr.passed ? "yes" : "no")
                << " max_abs=" << vr.max_abs_err << " max_rel=" << vr.max_rel_err << "\n";
    }
  }

  CUDA_CHECK(cudaFree(d_dw));
  CUDA_CHECK(cudaFree(d_dx));
  CUDA_CHECK(cudaFree(d_dy));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_x));
  return ratios;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options o = parse_args(argc, argv);
    if (o.bench_suite) {
      const std::vector<BenchCase> suite = default_bench_suite();
      std::cout << "Running benchmark suite (" << suite.size() << " cases)\n";
      std::vector<float> fprop_ratios;
      std::vector<float> bprop_ratios;
      std::vector<float> grad_ratios;
      for (const BenchCase& bc : suite) {
        CaseRatios cr = run_case(o, bc);
        if (cr.fprop > 0.0f) fprop_ratios.push_back(cr.fprop);
        if (cr.bprop > 0.0f) bprop_ratios.push_back(cr.bprop);
        if (cr.grad > 0.0f) grad_ratios.push_back(cr.grad);
      }
      if (o.with_cudnn) {
        std::cout << "\n[summary] custom/cudnn ratio (lower is better)\n";
        if (!fprop_ratios.empty()) {
          std::cout << "fprop min=" << std::fixed << std::setprecision(3) << vec_min(fprop_ratios)
                    << "x median=" << vec_median(fprop_ratios)
                    << "x max=" << vec_max(fprop_ratios) << "x\n";
        }
        if (!bprop_ratios.empty()) {
          std::cout << "bprop min=" << std::fixed << std::setprecision(3) << vec_min(bprop_ratios)
                    << "x median=" << vec_median(bprop_ratios)
                    << "x max=" << vec_max(bprop_ratios) << "x\n";
        }
        if (!grad_ratios.empty()) {
          std::cout << "grad min=" << std::fixed << std::setprecision(3) << vec_min(grad_ratios)
                    << "x median=" << vec_median(grad_ratios)
                    << "x max=" << vec_max(grad_ratios) << "x\n";
        }
      }
    } else {
      BenchCase single{
          "single",
          o.n, o.h, o.w, o.c, o.k, o.r, o.s,
          o.pad_h, o.pad_w,
          o.stride_h, o.stride_w,
          o.dilation_h, o.dilation_w,
          o.groups};
      (void)run_case(o, single);
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
