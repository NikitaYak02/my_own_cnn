#include "conv_types.h"

#include <algorithm>

void cpu_fprop_nhwc(const TensorNHWC& x, const FilterHWCN& w, const Conv2DParams& p, TensorNHWC& y) {
  const ConvShape shape = infer_conv_shape(x, w, p);
  y = TensorNHWC(x.n, shape.ho, shape.wo, w.k);
  std::fill(y.data.begin(), y.data.end(), 0.0f);

  for (int n = 0; n < x.n; ++n) {
    for (int ho = 0; ho < shape.ho; ++ho) {
      for (int wo = 0; wo < shape.wo; ++wo) {
        for (int g = 0; g < p.groups; ++g) {
          const int cin_base = g * shape.cin_group;
          const int kout_base = g * shape.kout_group;
          for (int ko = 0; ko < shape.kout_group; ++ko) {
            float acc = 0.0f;
            for (int rr = 0; rr < w.r; ++rr) {
              const int hi = ho * p.stride_h - p.pad_h + rr * p.dilation_h;
              if (hi < 0 || hi >= x.h) continue;
              for (int ss = 0; ss < w.s; ++ss) {
                const int wi = wo * p.stride_w - p.pad_w + ss * p.dilation_w;
                if (wi < 0 || wi >= x.w) continue;
                for (int ci = 0; ci < shape.cin_group; ++ci) {
                  const float xv = x.data[idx_nhwc(n, hi, wi, cin_base + ci, x.h, x.w, x.c)];
                  const float wv = w.data[idx_hwcn(rr, ss, ci, kout_base + ko, w.s, w.cin_per_group, w.k)];
                  acc += xv * wv;
                }
              }
            }
            y.data[idx_nhwc(n, ho, wo, kout_base + ko, shape.ho, shape.wo, w.k)] = acc;
          }
        }
      }
    }
  }
}

void cpu_bprop_nhwc(const TensorNHWC& dy, const FilterHWCN& w, const Conv2DParams& p, TensorNHWC& dx) {
  if (p.groups <= 0 || dx.c % p.groups != 0 || w.k % p.groups != 0) {
    throw std::runtime_error("invalid group configuration for bprop");
  }

  const int cin_group = dx.c / p.groups;
  const int kout_group = w.k / p.groups;
  std::fill(dx.data.begin(), dx.data.end(), 0.0f);

  for (int n = 0; n < dx.n; ++n) {
    for (int hi = 0; hi < dx.h; ++hi) {
      for (int wi = 0; wi < dx.w; ++wi) {
        for (int g = 0; g < p.groups; ++g) {
          const int cin_base = g * cin_group;
          const int kout_base = g * kout_group;
          for (int ci = 0; ci < cin_group; ++ci) {
            float acc = 0.0f;
            for (int rr = 0; rr < w.r; ++rr) {
              const int ho_nom = hi + p.pad_h - rr * p.dilation_h;
              if (ho_nom < 0 || (ho_nom % p.stride_h) != 0) continue;
              const int ho = ho_nom / p.stride_h;
              if (ho < 0 || ho >= dy.h) continue;

              for (int ss = 0; ss < w.s; ++ss) {
                const int wo_nom = wi + p.pad_w - ss * p.dilation_w;
                if (wo_nom < 0 || (wo_nom % p.stride_w) != 0) continue;
                const int wo = wo_nom / p.stride_w;
                if (wo < 0 || wo >= dy.w) continue;

                for (int ko = 0; ko < kout_group; ++ko) {
                  const float dyv = dy.data[idx_nhwc(n, ho, wo, kout_base + ko, dy.h, dy.w, dy.c)];
                  const float wv = w.data[idx_hwcn(rr, ss, ci, kout_base + ko, w.s, w.cin_per_group, w.k)];
                  acc += dyv * wv;
                }
              }
            }
            dx.data[idx_nhwc(n, hi, wi, cin_base + ci, dx.h, dx.w, dx.c)] = acc;
          }
        }
      }
    }
  }
}

void cpu_grad_nhwc(const TensorNHWC& x, const TensorNHWC& dy, const Conv2DParams& p, FilterHWCN& dw) {
  if (p.groups <= 0 || x.c % p.groups != 0 || dy.c % p.groups != 0) {
    throw std::runtime_error("invalid group configuration for grad");
  }

  const int cin_group = x.c / p.groups;
  const int kout_group = dy.c / p.groups;
  std::fill(dw.data.begin(), dw.data.end(), 0.0f);

  for (int rr = 0; rr < dw.r; ++rr) {
    for (int ss = 0; ss < dw.s; ++ss) {
      for (int g = 0; g < p.groups; ++g) {
        const int cin_base = g * cin_group;
        const int kout_base = g * kout_group;
        for (int ci = 0; ci < cin_group; ++ci) {
          for (int ko = 0; ko < kout_group; ++ko) {
            float acc = 0.0f;
            for (int n = 0; n < x.n; ++n) {
              for (int ho = 0; ho < dy.h; ++ho) {
                const int hi = ho * p.stride_h - p.pad_h + rr * p.dilation_h;
                if (hi < 0 || hi >= x.h) continue;
                for (int wo = 0; wo < dy.w; ++wo) {
                  const int wi = wo * p.stride_w - p.pad_w + ss * p.dilation_w;
                  if (wi < 0 || wi >= x.w) continue;
                  const float xv = x.data[idx_nhwc(n, hi, wi, cin_base + ci, x.h, x.w, x.c)];
                  const float dyv = dy.data[idx_nhwc(n, ho, wo, kout_base + ko, dy.h, dy.w, dy.c)];
                  acc += xv * dyv;
                }
              }
            }
            dw.data[idx_hwcn(rr, ss, ci, kout_base + ko, dw.s, dw.cin_per_group, dw.k)] = acc;
          }
        }
      }
    }
  }
}
