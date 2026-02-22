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

namespace {

struct Options {
  std::string op = "all";
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
    else if (a == "--help") {
      std::cout
          << "conv_bench options:\n"
          << "  --op fprop|bprop|grad|all\n"
          << "  --n --h --w --c --k --r --s\n"
          << "  --pad_h --pad_w --stride_h --stride_w --dilation_h --dilation_w --groups\n"
          << "  --warmup --iters --seed\n"
          << "  --check --with_cudnn\n";
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options o = parse_args(argc, argv);

    Conv2DParams p;
    p.pad_h = o.pad_h;
    p.pad_w = o.pad_w;
    p.stride_h = o.stride_h;
    p.stride_w = o.stride_w;
    p.dilation_h = o.dilation_h;
    p.dilation_w = o.dilation_w;
    p.groups = o.groups;

    TensorNHWC x(o.n, o.h, o.w, o.c);
    FilterHWIO w(o.r, o.s, o.c / o.groups, o.k);
    ConvShape sh = infer_conv_shape(x, w, p);

    TensorNHWC y(o.n, sh.ho, sh.wo, o.k);
    TensorNHWC dy(o.n, sh.ho, sh.wo, o.k);
    TensorNHWC dx(o.n, o.h, o.w, o.c);
    FilterHWIO dw(o.r, o.s, o.c / o.groups, o.k);

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

    const size_t flops = conv_flops(o.n, sh.ho, sh.wo, o.r, o.s, sh.cin_group, o.k);

    if (do_fprop) {
      BenchResult b = benchmark_cuda_op("fprop", o.warmup, o.iters, flops, [&]() {
        launch_fprop_nhwc(d_x, d_w, d_y, o.n, o.h, o.w, o.c, o.r, o.s, o.k, p);
      });
      print_bench("custom fprop", b);

      if (o.check) {
        TensorNHWC y_cpu;
        cpu_fprop_nhwc(x, w, p, y_cpu);
        CUDA_CHECK(cudaMemcpy(y.ptr(), d_y, y.elements() * sizeof(float), cudaMemcpyDeviceToHost));
        VerifyResult vr = verify_tensors(y_cpu.data, y.data, 1e-4f, 1e-3f);
        std::cout << "check fprop pass=" << (vr.passed ? "yes" : "no")
                  << " max_abs=" << vr.max_abs_err
                  << " max_rel=" << vr.max_rel_err << "\n";
      }

      if (o.with_cudnn) {
        ensure(cudnn_is_available(), "--with_cudnn requested, but cuDNN is unavailable");
        BenchResult cb = cudnn_fprop_bench(d_x, d_w, d_y, o.n, o.h, o.w, o.c, o.r, o.s, o.k, p, o.warmup, o.iters);
        print_bench("cudnn fprop", cb);
        if (o.check) {
          TensorNHWC y_cpu;
          cpu_fprop_nhwc(x, w, p, y_cpu);
          CUDA_CHECK(cudaMemcpy(y.ptr(), d_y, y.elements() * sizeof(float), cudaMemcpyDeviceToHost));
          VerifyResult vr = verify_tensors(y_cpu.data, y.data, 1e-4f, 1e-3f);
          std::cout << "check cudnn fprop vs cpu pass=" << (vr.passed ? "yes" : "no")
                    << " max_abs=" << vr.max_abs_err
                    << " max_rel=" << vr.max_rel_err << "\n";
        }
      }
    }

    if (do_bprop) {
      BenchResult b = benchmark_cuda_op("bprop", o.warmup, o.iters, flops, [&]() {
        launch_bprop_nhwc(d_dy, d_w, d_dx, o.n, o.h, o.w, o.c, o.r, o.s, o.k, p);
      });
      print_bench("custom bprop", b);

      if (o.check) {
        TensorNHWC dx_cpu(o.n, o.h, o.w, o.c);
        cpu_bprop_nhwc(dy, w, p, dx_cpu);
        CUDA_CHECK(cudaMemcpy(dx.ptr(), d_dx, dx.elements() * sizeof(float), cudaMemcpyDeviceToHost));
        VerifyResult vr = verify_tensors(dx_cpu.data, dx.data, 1e-4f, 1e-3f);
        std::cout << "check bprop pass=" << (vr.passed ? "yes" : "no")
                  << " max_abs=" << vr.max_abs_err
                  << " max_rel=" << vr.max_rel_err << "\n";
      }

      if (o.with_cudnn) {
        ensure(cudnn_is_available(), "--with_cudnn requested, but cuDNN is unavailable");
        BenchResult cb = cudnn_bprop_bench(d_dy, d_w, d_dx, o.n, o.h, o.w, o.c, o.r, o.s, o.k, p, o.warmup, o.iters);
        print_bench("cudnn bprop", cb);
        if (o.check) {
          TensorNHWC dx_cpu(o.n, o.h, o.w, o.c);
          cpu_bprop_nhwc(dy, w, p, dx_cpu);
          CUDA_CHECK(cudaMemcpy(dx.ptr(), d_dx, dx.elements() * sizeof(float), cudaMemcpyDeviceToHost));
          VerifyResult vr = verify_tensors(dx_cpu.data, dx.data, 1e-4f, 1e-3f);
          std::cout << "check cudnn bprop vs cpu pass=" << (vr.passed ? "yes" : "no")
                    << " max_abs=" << vr.max_abs_err
                    << " max_rel=" << vr.max_rel_err << "\n";
        }
      }
    }

    if (do_grad) {
      BenchResult b = benchmark_cuda_op("grad", o.warmup, o.iters, flops, [&]() {
        launch_grad_nhwc(d_x, d_dy, d_dw, o.n, o.h, o.w, o.c, o.r, o.s, o.k, p);
      });
      print_bench("custom grad", b);

      if (o.check) {
        FilterHWIO dw_cpu(o.r, o.s, o.c / o.groups, o.k);
        cpu_grad_nhwc(x, dy, p, dw_cpu);
        CUDA_CHECK(cudaMemcpy(dw.ptr(), d_dw, dw.elements() * sizeof(float), cudaMemcpyDeviceToHost));
        VerifyResult vr = verify_tensors(dw_cpu.data, dw.data, 1e-4f, 1e-3f);
        std::cout << "check grad pass=" << (vr.passed ? "yes" : "no")
                  << " max_abs=" << vr.max_abs_err
                  << " max_rel=" << vr.max_rel_err << "\n";
      }

      if (o.with_cudnn) {
        ensure(cudnn_is_available(), "--with_cudnn requested, but cuDNN is unavailable");
        BenchResult cb = cudnn_grad_bench(d_x, d_dy, d_dw, o.n, o.h, o.w, o.c, o.r, o.s, o.k, p, o.warmup, o.iters);
        print_bench("cudnn grad", cb);
        if (o.check) {
          FilterHWIO dw_cpu(o.r, o.s, o.c / o.groups, o.k);
          cpu_grad_nhwc(x, dy, p, dw_cpu);
          CUDA_CHECK(cudaMemcpy(dw.ptr(), d_dw, dw.elements() * sizeof(float), cudaMemcpyDeviceToHost));
          VerifyResult vr = verify_tensors(dw_cpu.data, dw.data, 1e-4f, 1e-3f);
          std::cout << "check cudnn grad vs cpu pass=" << (vr.passed ? "yes" : "no")
                    << " max_abs=" << vr.max_abs_err
                    << " max_rel=" << vr.max_rel_err << "\n";
        }
      }
    }

    CUDA_CHECK(cudaFree(d_dw));
    CUDA_CHECK(cudaFree(d_dx));
    CUDA_CHECK(cudaFree(d_dy));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_x));

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
