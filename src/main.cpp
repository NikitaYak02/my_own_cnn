#include "conv_types.h"
#include "cuda_utils.h"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string op = "all";
  std::string custom_mode = "explicit";
  std::string grad_algo = "gemm";
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
  int ay = 1;
  int ax = 1;
  int block_by = 1;
  int block_bx = 1;
  int warmup = 10;
  int iters = 50;
  bool check = false;
  bool with_cudnn = false;
  bool bench_suite = false;
  uint32_t seed = 42;
};

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
  int ay;
  int ax;
  int block_by;
  int block_bx;
};

struct CaseRatios {
  float fprop = -1.0f;
  float bprop = -1.0f;
  float grad = -1.0f;
};

int parse_int(const std::string& v, const char* name) {
  try {
    return std::stoi(v);
  } catch (...) {
    throw std::runtime_error(std::string("invalid integer for ") + name + ": " + v);
  }
}

GradKernelAlgo parse_grad_algo(const std::string& value) {
  if (value == "gemm") return GradKernelAlgo::GemmIm2Col;
  if (value == "algo0") return GradKernelAlgo::Algo0Atomic;
  if (value == "algo1") return GradKernelAlgo::Algo1Deterministic;
  if (value == "algo2" || value == "tile") return GradKernelAlgo::Algo2TiledAtomic;
  throw std::runtime_error("invalid value for --grad_algo: " + value + " (expected gemm|algo0|algo1|algo2|tile|all)");
}

const char* grad_algo_to_string(GradKernelAlgo algo) {
  switch (algo) {
    case GradKernelAlgo::GemmIm2Col: return "gemm";
    case GradKernelAlgo::Algo0Atomic: return "algo0";
    case GradKernelAlgo::Algo1Deterministic: return "algo1";
    case GradKernelAlgo::Algo2TiledAtomic: return "algo2";
    default: return "unknown";
  }
}

std::vector<std::string> expand_grad_algo_names(const std::string& value) {
  if (value == "all") {
    return {"gemm", "algo0", "algo1", "algo2"};
  }
  (void)parse_grad_algo(value);
  if (value == "tile") return {"algo2"};
  return {value};
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
    else if (a == "--grad_algo") o.grad_algo = need_val("--grad_algo");
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
    else if (a == "--ay") o.ay = parse_int(need_val("--ay"), "--ay");
    else if (a == "--ax") o.ax = parse_int(need_val("--ax"), "--ax");
    else if (a == "--block_by") o.block_by = parse_int(need_val("--block_by"), "--block_by");
    else if (a == "--block_bx") o.block_bx = parse_int(need_val("--block_bx"), "--block_bx");
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
          << "  --custom_mode explicit|blocked\n"
          << "  --grad_algo gemm|algo0|algo1|algo2|tile|all\n"
          << "  --n --h --w --c --k --r --s\n"
          << "  --pad_h --pad_w --stride_h --stride_w --dilation_h --dilation_w --groups --ay --ax\n"
          << "  --block_by --block_bx\n"
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

void push_ratio(std::vector<float>& out, float ratio) {
  if (ratio > 0.0f) out.push_back(ratio);
}

void print_ratio_summary_line(const char* label, const std::vector<float>& ratios) {
  if (ratios.empty()) return;
  std::cout << label
            << " min=" << std::fixed << std::setprecision(3) << vec_min(ratios)
            << "x median=" << vec_median(ratios)
            << "x max=" << vec_max(ratios) << "x\n";
}

std::vector<BenchCase> default_bench_suite() {
  return {
      {"resnet_like_56x56", 32, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {"resnet_like_28x28", 32, 28, 28, 128, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {"downsample_stride2", 32, 112, 112, 32, 32, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1},
      {"depthwise_3x3", 32, 56, 56, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 128, 1, 1, 1, 1},
      {"grouped_g4", 32, 56, 56, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1},
      {"pointwise_1x1", 32, 56, 56, 128, 256, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {"dilated_3x3", 16, 64, 64, 64, 64, 3, 3, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1},
      {"small_batch", 1, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {"large_batch_128x128_n16_c64_k128", 16, 128, 128, 64, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {"large_batch_128x128_n8_c128_k128", 8, 128, 128, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {"pointwise_1x1_large_56x56_n32_c256_k256", 32, 56, 56, 256, 256, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {"pointwise_1x1_large_112x112_n16_c128_k256", 16, 112, 112, 128, 256, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {"pointwise_1x1_grouped_g4_56x56_n32_c256_k256", 32, 56, 56, 256, 256, 1, 1, 0, 0, 1, 1, 1, 1, 4, 1, 1, 1, 1},
  };
}

std::vector<BenchCase> default_blocked_bench_suite() {
  return {
      {"blocked_resnet_56x56_b2x2", 32, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2},
      {"blocked_resnet_56x56_b4x4", 32, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4},
      {"blocked_resnet_28x28_b2x4", 32, 28, 28, 128, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4},
      {"blocked_grouped_g4_b2x2", 32, 56, 56, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 4, 1, 1, 2, 2},
      {"blocked_pointwise_1x1_b4x4", 32, 56, 56, 128, 256, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 4, 4},
      {"blocked_dilated_3x3_b2x2", 16, 64, 64, 64, 64, 3, 3, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2},
      {"blocked_small_batch_b4x4", 1, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4},
      {"blocked_pointwise_1x1_large_56x56_n32_c256_k256_b4x4", 32, 56, 56, 256, 256, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 4, 4},
      {"blocked_pointwise_1x1_large_112x112_n16_c128_k256_b4x4", 16, 112, 112, 128, 256, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 4, 4},
      {"blocked_pointwise_1x1_grouped_g4_56x56_n32_c256_k256_b4x4", 32, 56, 56, 256, 256, 1, 1, 0, 0, 1, 1, 1, 1, 4, 1, 1, 4, 4},
  };
}

void print_verify(const std::string& tag, const VerifyResult& vr) {
  std::cout << tag
            << " pass=" << (vr.passed ? "yes" : "no")
            << " max_abs=" << vr.max_abs_err
            << " max_rel=" << vr.max_rel_err << "\n";
}

CaseRatios run_case(const Options& o, const BenchCase& bc) {
  const bool blocked = (o.custom_mode == "blocked");
  ensure(blocked || o.custom_mode == "explicit", "--custom_mode must be explicit|blocked");
  const GradKernelAlgo grad_algo = parse_grad_algo(o.grad_algo);
  ensure(bc.ay > 0 && bc.ax > 0, "ay and ax must be > 0");
  if ((bc.ay != 1 || bc.ax != 1) && grad_algo != GradKernelAlgo::GemmIm2Col) {
    throw std::runtime_error("ay/ax currently support only grad_algo=gemm");
  }
  if ((bc.ay != 1 || bc.ax != 1) && o.with_cudnn) {
    throw std::runtime_error("cuDNN path does not support ay/ax != 1");
  }

  CaseRatios ratios;
  Conv2DParams p;
  p.pad_h = bc.pad_h;
  p.pad_w = bc.pad_w;
  p.stride_h = bc.stride_h;
  p.stride_w = bc.stride_w;
  p.dilation_h = bc.dilation_h;
  p.dilation_w = bc.dilation_w;
  p.groups = bc.groups;
  p.ay = bc.ay;
  p.ax = bc.ax;

  BlockConv2DParams bp;
  bp.conv = p;
  bp.block_by = bc.block_by;
  bp.block_bx = bc.block_bx;

  TensorNHWC x(bc.n, bc.h, bc.w, bc.c);
  std::mt19937 gen(o.seed);
  fill_random(x.data, gen);

  int ho = 0;
  int wo = 0;
  int cin_group = 0;
  FilterKRSC w_exp;
  FilterKRSC dw_exp;
  BlockFilterKByBxRSC w_blk;
  BlockFilterKByBxRSC dw_blk;

  if (blocked) {
    w_blk = BlockFilterKByBxRSC(bc.k, bp.block_by, bp.block_bx, bc.r, bc.s, bc.c / bc.groups, bc.ay, bc.ax);
    const BlockConvShape sh = infer_block_conv_shape(x, w_blk, bp);
    ho = sh.base.ho;
    wo = sh.base.wo;
    cin_group = sh.base.cin_group;
    dw_blk = BlockFilterKByBxRSC(bc.k, bp.block_by, bp.block_bx, bc.r, bc.s, bc.c / bc.groups, bc.ay, bc.ax);
    fill_random(w_blk.data, gen);
  } else {
    w_exp = FilterKRSC(bc.r, bc.s, bc.c / bc.groups, bc.k, bc.ay, bc.ax);
    const ConvShape sh = infer_conv_shape(x, w_exp, p);
    ho = sh.ho;
    wo = sh.wo;
    cin_group = sh.cin_group;
    dw_exp = FilterKRSC(bc.r, bc.s, bc.c / bc.groups, bc.k, bc.ay, bc.ax);
    fill_random(w_exp.data, gen);
  }

  TensorNHWC y(bc.n, ho, wo, bc.k);
  TensorNHWC dy(bc.n, ho, wo, bc.k);
  TensorNHWC dx(bc.n, bc.h, bc.w, bc.c);
  fill_random(dy.data, gen);

  const size_t w_elems = blocked ? w_blk.elements() : w_exp.elements();
  const size_t dw_elems = blocked ? dw_blk.elements() : dw_exp.elements();

  float* d_x = nullptr;
  float* d_w = nullptr;
  float* d_y = nullptr;
  float* d_dy = nullptr;
  float* d_dx = nullptr;
  float* d_dw = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, x.elements() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_w, w_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, y.elements() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dy, dy.elements() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dx, dx.elements() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dw, dw_elems * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_x, x.ptr(), x.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, blocked ? w_blk.ptr() : w_exp.ptr(), w_elems * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dy, dy.ptr(), dy.elements() * sizeof(float), cudaMemcpyHostToDevice));

  const bool do_fprop = (o.op == "fprop" || o.op == "all");
  const bool do_bprop = (o.op == "bprop" || o.op == "all");
  const bool do_grad = (o.op == "grad" || o.op == "all");
  ensure(do_fprop || do_bprop || do_grad, "--op must be fprop|bprop|grad|all");

  const size_t flops = conv_flops(bc.n, ho, wo, bc.r, bc.s, cin_group, bc.k);
  std::cout << "\n[case] " << bc.name
            << " mode=" << o.custom_mode
            << " n=" << bc.n << " h=" << bc.h << " w=" << bc.w
            << " c=" << bc.c << " k=" << bc.k
            << " r=" << bc.r << " s=" << bc.s
            << " stride=" << bc.stride_h << "x" << bc.stride_w
            << " pad=" << bc.pad_h << "x" << bc.pad_w
            << " dilation=" << bc.dilation_h << "x" << bc.dilation_w
            << " groups=" << bc.groups
            << " ay=" << bc.ay << " ax=" << bc.ax
            << " grad_algo=" << grad_algo_to_string(grad_algo);
  if (blocked) {
    std::cout << " blocks=" << bp.block_by << "x" << bp.block_bx;
  }
  std::cout << "\n";
  if (blocked && o.with_cudnn) {
    std::cout << "note: blocked cuDNN timings are kernel-only; staging/scatter are excluded\n";
  }

  if (do_fprop) {
    BenchResult b = benchmark_cuda_op("fprop", o.warmup, o.iters, flops, [&]() {
      if (blocked) {
        launch_block_fprop_nhwc(d_x, d_w, d_y, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, bp);
      } else {
        launch_fprop_nhwc(d_x, d_w, d_y, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, p);
      }
    });
    print_bench("custom fprop", b);

    std::vector<float> custom_y;
    std::vector<float> cpu_y;
    if (o.check) {
      CUDA_CHECK(cudaMemcpy(y.ptr(), d_y, y.elements() * sizeof(float), cudaMemcpyDeviceToHost));
      if (blocked) {
        TensorNHWC y_cpu;
        cpu_block_fprop_nhwc(x, w_blk, bp, y_cpu);
        VerifyResult vr = verify_tensors(y_cpu.data, y.data, 1e-4f, 1e-3f);
        print_verify("check fprop custom_vs_cpu", vr);
        cpu_y = y_cpu.data;
      } else {
        TensorNHWC y_cpu;
        cpu_fprop_nhwc(x, w_exp, p, y_cpu);
        VerifyResult vr = verify_tensors(y_cpu.data, y.data, 1e-4f, 1e-3f);
        print_verify("check fprop custom_vs_cpu", vr);
        cpu_y = y_cpu.data;
      }
      if (o.with_cudnn) custom_y = y.data;
    }

    if (o.with_cudnn) {
      ensure(cudnn_is_available(), "--with_cudnn requested, but cuDNN is unavailable");
      BenchResult cb = blocked
          ? cudnn_block_fprop_bench(d_x, d_w, d_y, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, bp, o.warmup, o.iters)
          : cudnn_fprop_bench(d_x, d_w, d_y, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, p, o.warmup, o.iters);
      print_bench(blocked ? "cudnn fprop (kernel-only)" : "cudnn fprop", cb);
      ratios.fprop = b.median_ms / std::max(cb.median_ms, 1e-6f);
      std::cout << "ratio fprop custom/cudnn=" << std::fixed << std::setprecision(3) << ratios.fprop << "x\n";

      if (o.check) {
        CUDA_CHECK(cudaMemcpy(y.ptr(), d_y, y.elements() * sizeof(float), cudaMemcpyDeviceToHost));
        VerifyResult vr = verify_tensors(custom_y, y.data, 1e-4f, 1e-3f);
        print_verify("check fprop custom_vs_cudnn", vr);
        VerifyResult cudnn_cpu_vr = verify_tensors(cpu_y, y.data, 1e-4f, 1e-3f);
        print_verify("check fprop cudnn_vs_cpu", cudnn_cpu_vr);
      }
    }
  }

  if (do_bprop) {
    BenchResult b = benchmark_cuda_op("bprop", o.warmup, o.iters, flops, [&]() {
      if (blocked) {
        launch_block_bprop_nhwc(d_dy, d_w, d_dx, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, bp);
      } else {
        launch_bprop_nhwc(d_dy, d_w, d_dx, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, p);
      }
    });
    print_bench("custom bprop", b);

    std::vector<float> custom_dx;
    std::vector<float> cpu_dx;
    if (o.check) {
      CUDA_CHECK(cudaMemcpy(dx.ptr(), d_dx, dx.elements() * sizeof(float), cudaMemcpyDeviceToHost));
      if (blocked) {
        TensorNHWC dx_cpu(bc.n, bc.h, bc.w, bc.c);
        cpu_block_bprop_nhwc(dy, w_blk, bp, dx_cpu);
        VerifyResult vr = verify_tensors(dx_cpu.data, dx.data, 1e-4f, 1e-3f);
        print_verify("check bprop custom_vs_cpu", vr);
        cpu_dx = dx_cpu.data;
      } else {
        TensorNHWC dx_cpu(bc.n, bc.h, bc.w, bc.c);
        cpu_bprop_nhwc(dy, w_exp, p, dx_cpu);
        VerifyResult vr = verify_tensors(dx_cpu.data, dx.data, 1e-4f, 1e-3f);
        print_verify("check bprop custom_vs_cpu", vr);
        cpu_dx = dx_cpu.data;
      }
      if (o.with_cudnn) custom_dx = dx.data;
    }

    if (o.with_cudnn) {
      ensure(cudnn_is_available(), "--with_cudnn requested, but cuDNN is unavailable");
      BenchResult cb = blocked
          ? cudnn_block_bprop_bench(d_dy, d_w, d_dx, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, bp, o.warmup, o.iters)
          : cudnn_bprop_bench(d_dy, d_w, d_dx, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, p, o.warmup, o.iters);
      print_bench(blocked ? "cudnn bprop (kernel-only)" : "cudnn bprop", cb);
      ratios.bprop = b.median_ms / std::max(cb.median_ms, 1e-6f);
      std::cout << "ratio bprop custom/cudnn=" << std::fixed << std::setprecision(3) << ratios.bprop << "x\n";

      if (o.check) {
        CUDA_CHECK(cudaMemcpy(dx.ptr(), d_dx, dx.elements() * sizeof(float), cudaMemcpyDeviceToHost));
        VerifyResult vr = verify_tensors(custom_dx, dx.data, 1e-4f, 1e-3f);
        print_verify("check bprop custom_vs_cudnn", vr);
        VerifyResult cudnn_cpu_vr = verify_tensors(cpu_dx, dx.data, 1e-4f, 1e-3f);
        print_verify("check bprop cudnn_vs_cpu", cudnn_cpu_vr);
      }
    }
  }

  if (do_grad) {
    BenchResult b = benchmark_cuda_op("grad", o.warmup, o.iters, flops, [&]() {
      if (blocked) {
        launch_block_grad_nhwc(d_x, d_dy, d_dw, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, bp, grad_algo);
      } else {
        launch_grad_nhwc(d_x, d_dy, d_dw, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, p, grad_algo);
      }
    });
    print_bench("custom grad", b);

    std::vector<float> custom_dw;
    std::vector<float> cpu_dw;
    const bool gemm_grad = (grad_algo == GradKernelAlgo::GemmIm2Col);
    const float grad_abs_eps = gemm_grad ? 1e-2f : 1e-4f;
    const float grad_rel_eps = gemm_grad ? 1e-2f : 1e-3f;
    if (o.check) {
      if (blocked) {
        CUDA_CHECK(cudaMemcpy(dw_blk.ptr(), d_dw, dw_blk.elements() * sizeof(float), cudaMemcpyDeviceToHost));
        BlockFilterKByBxRSC dw_cpu(bc.k, bp.block_by, bp.block_bx, bc.r, bc.s, bc.c / bc.groups, bc.ay, bc.ax);
        cpu_block_grad_nhwc(x, dy, bp, dw_cpu);
        VerifyResult vr = verify_tensors(dw_cpu.data, dw_blk.data, grad_abs_eps, grad_rel_eps);
        print_verify("check grad custom_vs_cpu", vr);
        cpu_dw = dw_cpu.data;
        if (o.with_cudnn) custom_dw = dw_blk.data;
      } else {
        CUDA_CHECK(cudaMemcpy(dw_exp.ptr(), d_dw, dw_exp.elements() * sizeof(float), cudaMemcpyDeviceToHost));
        FilterKRSC dw_cpu(bc.r, bc.s, bc.c / bc.groups, bc.k, bc.ay, bc.ax);
        cpu_grad_nhwc(x, dy, p, dw_cpu);
        VerifyResult vr = verify_tensors(dw_cpu.data, dw_exp.data, grad_abs_eps, grad_rel_eps);
        print_verify("check grad custom_vs_cpu", vr);
        cpu_dw = dw_cpu.data;
        if (o.with_cudnn) custom_dw = dw_exp.data;
      }
    }

    if (o.with_cudnn) {
      ensure(cudnn_is_available(), "--with_cudnn requested, but cuDNN is unavailable");
      BenchResult cb = blocked
          ? cudnn_block_grad_bench(d_x, d_dy, d_dw, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, bp, o.warmup, o.iters)
          : cudnn_grad_bench(d_x, d_dy, d_dw, bc.n, bc.h, bc.w, bc.c, bc.r, bc.s, bc.k, p, o.warmup, o.iters);
      print_bench(blocked ? "cudnn grad (kernel-only)" : "cudnn grad", cb);
      ratios.grad = b.median_ms / std::max(cb.median_ms, 1e-6f);
      std::cout << "ratio grad custom/cudnn=" << std::fixed << std::setprecision(3) << ratios.grad << "x\n";

      if (o.check) {
        if (blocked) {
          CUDA_CHECK(cudaMemcpy(dw_blk.ptr(), d_dw, dw_blk.elements() * sizeof(float), cudaMemcpyDeviceToHost));
          VerifyResult vr = verify_tensors(custom_dw, dw_blk.data, 1e-4f, 1e-3f);
          print_verify("check grad custom_vs_cudnn", vr);
          VerifyResult cudnn_cpu_vr = verify_tensors(cpu_dw, dw_blk.data, grad_abs_eps, grad_rel_eps);
          print_verify("check grad cudnn_vs_cpu", cudnn_cpu_vr);
        } else {
          CUDA_CHECK(cudaMemcpy(dw_exp.ptr(), d_dw, dw_exp.elements() * sizeof(float), cudaMemcpyDeviceToHost));
          VerifyResult vr = verify_tensors(custom_dw, dw_exp.data, 1e-4f, 1e-3f);
          print_verify("check grad custom_vs_cudnn", vr);
          VerifyResult cudnn_cpu_vr = verify_tensors(cpu_dw, dw_exp.data, grad_abs_eps, grad_rel_eps);
          print_verify("check grad cudnn_vs_cpu", cudnn_cpu_vr);
        }
      }
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
    // Default to strict FP32 numerics unless the user explicitly overrides it
    // in the environment.
    if (std::getenv("NVIDIA_TF32_OVERRIDE") == nullptr) {
#ifdef _WIN32
      _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
      setenv("NVIDIA_TF32_OVERRIDE", "0", 0);
#endif
    }

    const Options o = parse_args(argc, argv);
    auto execute_cases = [&](const std::vector<BenchCase>& cases) {
      const std::vector<std::string> grad_algo_names = expand_grad_algo_names(o.grad_algo);
      const bool run_all_grad_algos = (grad_algo_names.size() > 1);

      std::vector<float> fprop_ratios;
      std::vector<float> bprop_ratios;
      std::vector<float> grad_ratios_gemm;
      std::vector<float> grad_ratios_algo0;
      std::vector<float> grad_ratios_algo1;
      std::vector<float> grad_ratios_algo2;

      auto append_grad_ratio = [&](const std::string& algo_name, float ratio) {
        if (algo_name == "gemm") push_ratio(grad_ratios_gemm, ratio);
        else if (algo_name == "algo0") push_ratio(grad_ratios_algo0, ratio);
        else if (algo_name == "algo1") push_ratio(grad_ratios_algo1, ratio);
        else if (algo_name == "algo2") push_ratio(grad_ratios_algo2, ratio);
      };

      for (const BenchCase& bc : cases) {
        if (run_all_grad_algos && o.op == "all") {
          Options fprop_o = o;
          fprop_o.op = "fprop";
          fprop_o.grad_algo = "gemm";
          push_ratio(fprop_ratios, run_case(fprop_o, bc).fprop);

          Options bprop_o = o;
          bprop_o.op = "bprop";
          bprop_o.grad_algo = "gemm";
          push_ratio(bprop_ratios, run_case(bprop_o, bc).bprop);

          for (const std::string& algo_name : grad_algo_names) {
            Options grad_o = o;
            grad_o.op = "grad";
            grad_o.grad_algo = algo_name;
            append_grad_ratio(algo_name, run_case(grad_o, bc).grad);
          }
        } else if (run_all_grad_algos && o.op == "grad") {
          for (const std::string& algo_name : grad_algo_names) {
            Options grad_o = o;
            grad_o.grad_algo = algo_name;
            append_grad_ratio(algo_name, run_case(grad_o, bc).grad);
          }
        } else {
          CaseRatios cr = run_case(o, bc);
          push_ratio(fprop_ratios, cr.fprop);
          push_ratio(bprop_ratios, cr.bprop);
          if (!grad_algo_names.empty()) {
            append_grad_ratio(grad_algo_names.front(), cr.grad);
          }
        }
      }

      if (o.with_cudnn) {
        std::cout << "\n[summary] custom/cudnn ratio (lower is better)\n";
        print_ratio_summary_line("fprop", fprop_ratios);
        print_ratio_summary_line("bprop", bprop_ratios);
        if (run_all_grad_algos) {
          print_ratio_summary_line("grad[gemm]", grad_ratios_gemm);
          print_ratio_summary_line("grad[algo0]", grad_ratios_algo0);
          print_ratio_summary_line("grad[algo1]", grad_ratios_algo1);
          print_ratio_summary_line("grad[algo2]", grad_ratios_algo2);
        } else if (!grad_algo_names.empty()) {
          const std::string label = std::string("grad[") + grad_algo_names.front() + "]";
          if (grad_algo_names.front() == "gemm") print_ratio_summary_line(label.c_str(), grad_ratios_gemm);
          else if (grad_algo_names.front() == "algo0") print_ratio_summary_line(label.c_str(), grad_ratios_algo0);
          else if (grad_algo_names.front() == "algo1") print_ratio_summary_line(label.c_str(), grad_ratios_algo1);
          else if (grad_algo_names.front() == "algo2") print_ratio_summary_line(label.c_str(), grad_ratios_algo2);
        }
      }
    };

    if (o.bench_suite) {
      const bool blocked = (o.custom_mode == "blocked");
      const std::vector<BenchCase> suite = blocked ? default_blocked_bench_suite() : default_bench_suite();
      const std::vector<std::string> grad_algo_names = expand_grad_algo_names(o.grad_algo);
      std::cout << "Running benchmark suite (" << suite.size() << " cases";
      if (grad_algo_names.size() > 1 && (o.op == "grad" || o.op == "all")) {
        std::cout << ", grad algos=" << grad_algo_names.size();
      }
      std::cout << ")\n";
      execute_cases(suite);
    } else {
      BenchCase single{
          "single",
          o.n, o.h, o.w, o.c, o.k, o.r, o.s,
          o.pad_h, o.pad_w,
          o.stride_h, o.stride_w,
          o.dilation_h, o.dilation_w,
          o.groups,
          o.ay, o.ax,
          o.block_by, o.block_bx};
      execute_cases({single});
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
