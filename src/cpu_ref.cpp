#include "conv_types.h"

#include <algorithm>

namespace {

inline int output_block_index(int coord, int block_extent) {
  return coord / block_extent;
}

inline void decode_output_app(int out_h, int out_w, int ay, int ax,
                              int& base_h, int& base_w,
                              int& ay_idx, int& ax_idx) {
  base_h = out_h / ay;
  base_w = out_w / ax;
  ay_idx = out_h - base_h * ay;
  ax_idx = out_w - base_w * ax;
}

template <nnalgebra::DataType Tin>
inline int32_t centered_input_value(const TensorNHWC& x,
                                    int n, int h, int w, int c,
                                    const nnalgebra::QuantizationParameters<Tin>* in_qp) {
  return static_cast<int32_t>(x.data[idx_nhwc(n, h, w, c, x.h, x.w, x.c)]) -
         nnalgebra::getZeroPoint(in_qp[n]);
}

template <nnalgebra::DataType Tin>
inline int32_t centered_weight_value(const FilterKRSC& w,
                                     int k, int r, int s, int c,
                                     const nnalgebra::QuantizationParameters<Tin>* f_qp) {
  return static_cast<int32_t>(w.data[idx_krsc(k, r, s, c, w.r, w.s, w.cin_per_group)]) -
         nnalgebra::getZeroPoint(*f_qp);
}

template <nnalgebra::DataType Tin>
inline int32_t centered_block_weight_value(const BlockFilterKByBxRSC& w,
                                           int k, int by, int bx, int r, int s, int c,
                                           const nnalgebra::QuantizationParameters<Tin>* f_qp) {
  return static_cast<int32_t>(w.data[idx_kbybxrsc(k, by, bx, r, s, c,
                                                  w.by, w.bx, w.r, w.s, w.cin_per_group)]) -
         nnalgebra::getZeroPoint(*f_qp);
}

template <nnalgebra::DataType Tin>
void populate_output_qparams(int n,
                             const nnalgebra::QuantizationParameters<Tin>* in_qp,
                             const nnalgebra::QuantizationParameters<Tin>* f_qp,
                             nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp) {
  if (!out_qp) return;
  const float f_scale = nnalgebra::getScale(*f_qp);
  for (int i = 0; i < n; ++i) {
    out_qp[i].scale = nnalgebra::getScale(in_qp[i]) * f_scale;
  }
}

template <nnalgebra::DataType Tin>
void cpu_fprop_nhwc_qi32_impl(const TensorNHWC& x, const FilterKRSC& w, const Conv2DParams& p,
                              TensorNHWCI32& y,
                              const nnalgebra::QuantizationParameters<Tin>* in_qp,
                              const nnalgebra::QuantizationParameters<Tin>* f_qp,
                              nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp) {
  if (p.ay != 1 || p.ax != 1) {
    throw std::runtime_error("quantized CPU fprop does not support ay/ax != 1");
  }
  const ConvShape shape = infer_conv_shape(x, w, p);
  y = TensorNHWCI32(x.n, shape.ho, shape.wo, w.k);
  std::fill(y.data.begin(), y.data.end(), 0);
  populate_output_qparams(x.n, in_qp, f_qp, out_qp);

  for (int n = 0; n < x.n; ++n) {
    for (int ho = 0; ho < shape.ho; ++ho) {
      for (int wo = 0; wo < shape.wo; ++wo) {
        for (int g = 0; g < p.groups; ++g) {
          const int cin_base = g * shape.cin_group;
          const int kout_base = g * shape.kout_group;
          for (int ko = 0; ko < shape.kout_group; ++ko) {
            int32_t acc = 0;
            for (int rr = 0; rr < w.r; ++rr) {
              const int hi = ho * p.stride_h - p.pad_h + rr * p.dilation_h;
              if (hi < 0 || hi >= x.h) continue;
              for (int ss = 0; ss < w.s; ++ss) {
                const int wi = wo * p.stride_w - p.pad_w + ss * p.dilation_w;
                if (wi < 0 || wi >= x.w) continue;
                for (int ci = 0; ci < shape.cin_group; ++ci) {
                  const int32_t xv = centered_input_value(x, n, hi, wi, cin_base + ci, in_qp);
                  const int32_t wv = centered_weight_value(w, kout_base + ko, rr, ss, ci, f_qp);
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

template <nnalgebra::DataType Tin>
void cpu_block_fprop_nhwc_qi32_impl(const TensorNHWC& x, const BlockFilterKByBxRSC& w, const BlockConv2DParams& p,
                                    TensorNHWCI32& y,
                                    const nnalgebra::QuantizationParameters<Tin>* in_qp,
                                    const nnalgebra::QuantizationParameters<Tin>* f_qp,
                                    nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp) {
  if (p.conv.ay != 1 || p.conv.ax != 1) {
    throw std::runtime_error("quantized CPU blocked fprop does not support ay/ax != 1");
  }
  const BlockConvShape shape = infer_block_conv_shape(x, w, p);
  y = TensorNHWCI32(x.n, shape.base.ho, shape.base.wo, w.k);
  std::fill(y.data.begin(), y.data.end(), 0);
  populate_output_qparams(x.n, in_qp, f_qp, out_qp);

  for (int n = 0; n < x.n; ++n) {
    for (int ho = 0; ho < shape.base.ho; ++ho) {
      const int by = output_block_index(ho, shape.block_ho);
      for (int wo = 0; wo < shape.base.wo; ++wo) {
        const int bx = output_block_index(wo, shape.block_wo);
        for (int g = 0; g < p.conv.groups; ++g) {
          const int cin_base = g * shape.base.cin_group;
          const int kout_base = g * shape.base.kout_group;
          for (int ko = 0; ko < shape.base.kout_group; ++ko) {
            int32_t acc = 0;
            for (int rr = 0; rr < w.r; ++rr) {
              const int hi = ho * p.conv.stride_h - p.conv.pad_h + rr * p.conv.dilation_h;
              if (hi < 0 || hi >= x.h) continue;
              for (int ss = 0; ss < w.s; ++ss) {
                const int wi = wo * p.conv.stride_w - p.conv.pad_w + ss * p.conv.dilation_w;
                if (wi < 0 || wi >= x.w) continue;
                for (int ci = 0; ci < shape.base.cin_group; ++ci) {
                  const int32_t xv = centered_input_value(x, n, hi, wi, cin_base + ci, in_qp);
                  const int32_t wv = centered_block_weight_value(w, kout_base + ko, by, bx, rr, ss, ci, f_qp);
                  acc += xv * wv;
                }
              }
            }
            y.data[idx_nhwc(n, ho, wo, kout_base + ko, shape.base.ho, shape.base.wo, w.k)] = acc;
          }
        }
      }
    }
  }
}

}  // namespace

void cpu_fprop_nhwc(const TensorNHWC& x, const FilterKRSC& w, const Conv2DParams& p, TensorNHWC& y) {
  const ConvShape shape = infer_conv_shape(x, w, p);
  y = TensorNHWC(x.n, shape.ho, shape.wo, w.k);
  std::fill(y.data.begin(), y.data.end(), 0.0f);

  for (int n = 0; n < x.n; ++n) {
    for (int ho = 0; ho < shape.ho; ++ho) {
      for (int wo = 0; wo < shape.wo; ++wo) {
        int base_ho = 0;
        int base_wo = 0;
        int ay_idx = 0;
        int ax_idx = 0;
        decode_output_app(ho, wo, p.ay, p.ax, base_ho, base_wo, ay_idx, ax_idx);
        for (int g = 0; g < p.groups; ++g) {
          const int cin_base = g * shape.cin_group;
          const int kout_base = g * shape.kout_group;
          for (int ko = 0; ko < shape.kout_group; ++ko) {
            float acc = 0.0f;
            for (int rr = 0; rr < w.r; ++rr) {
              const int hi = base_ho * p.stride_h - p.pad_h + rr * p.dilation_h;
              if (hi < 0 || hi >= x.h) continue;
              for (int ss = 0; ss < w.s; ++ss) {
                const int wi = base_wo * p.stride_w - p.pad_w + ss * p.dilation_w;
                if (wi < 0 || wi >= x.w) continue;
                for (int ci = 0; ci < shape.cin_group; ++ci) {
                  const float xv = x.data[idx_nhwc(n, hi, wi, cin_base + ci, x.h, x.w, x.c)];
                  const float wv = w.data[idx_krsc(kout_base + ko, rr, ss, ci, ay_idx, ax_idx,
                                                   w.r, w.s, w.cin_per_group, w.ay, w.ax)];
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

void cpu_bprop_nhwc(const TensorNHWC& dy, const FilterKRSC& w, const Conv2DParams& p, TensorNHWC& dx) {
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
              const int base_ho_nom = hi + p.pad_h - rr * p.dilation_h;
              if (base_ho_nom < 0 || (base_ho_nom % p.stride_h) != 0) continue;
              const int base_ho = base_ho_nom / p.stride_h;
              if (base_ho < 0 || base_ho >= (dy.h / p.ay)) continue;

              for (int ss = 0; ss < w.s; ++ss) {
                for (int ko = 0; ko < kout_group; ++ko) {
                  const int base_wo_nom = wi + p.pad_w - ss * p.dilation_w;
                  if (base_wo_nom < 0 || (base_wo_nom % p.stride_w) != 0) continue;
                  const int base_wo = base_wo_nom / p.stride_w;
                  if (base_wo < 0 || base_wo >= (dy.w / p.ax)) continue;

                  for (int ay_idx = 0; ay_idx < p.ay; ++ay_idx) {
                    const int ho = base_ho * p.ay + ay_idx;
                    for (int ax_idx = 0; ax_idx < p.ax; ++ax_idx) {
                      const int wo = base_wo * p.ax + ax_idx;
                      const float dyv = dy.data[idx_nhwc(n, ho, wo, kout_base + ko, dy.h, dy.w, dy.c)];
                      const float wv = w.data[idx_krsc(kout_base + ko, rr, ss, ci, ay_idx, ax_idx,
                                                       w.r, w.s, w.cin_per_group, w.ay, w.ax)];
                      acc += dyv * wv;
                    }
                  }
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

void cpu_grad_nhwc(const TensorNHWC& x, const TensorNHWC& dy, const Conv2DParams& p, FilterKRSC& dw) {
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
            for (int ay_idx = 0; ay_idx < dw.ay; ++ay_idx) {
              for (int ax_idx = 0; ax_idx < dw.ax; ++ax_idx) {
                float acc = 0.0f;
                for (int n = 0; n < x.n; ++n) {
                  for (int base_ho = 0; base_ho < (dy.h / p.ay); ++base_ho) {
                    const int hi = base_ho * p.stride_h - p.pad_h + rr * p.dilation_h;
                    if (hi < 0 || hi >= x.h) continue;
                    const int ho = base_ho * p.ay + ay_idx;
                    for (int base_wo = 0; base_wo < (dy.w / p.ax); ++base_wo) {
                      const int wi = base_wo * p.stride_w - p.pad_w + ss * p.dilation_w;
                      if (wi < 0 || wi >= x.w) continue;
                      const int wo = base_wo * p.ax + ax_idx;
                      const float xv = x.data[idx_nhwc(n, hi, wi, cin_base + ci, x.h, x.w, x.c)];
                      const float dyv = dy.data[idx_nhwc(n, ho, wo, kout_base + ko, dy.h, dy.w, dy.c)];
                      acc += xv * dyv;
                    }
                  }
                }
                dw.data[idx_krsc(kout_base + ko, rr, ss, ci, ay_idx, ax_idx,
                                 dw.r, dw.s, dw.cin_per_group, dw.ay, dw.ax)] = acc;
              }
            }
          }
        }
      }
    }
  }
}

void cpu_block_fprop_nhwc(const TensorNHWC& x, const BlockFilterKByBxRSC& w, const BlockConv2DParams& p, TensorNHWC& y) {
  const BlockConvShape shape = infer_block_conv_shape(x, w, p);
  y = TensorNHWC(x.n, shape.base.ho, shape.base.wo, w.k);
  std::fill(y.data.begin(), y.data.end(), 0.0f);

  for (int n = 0; n < x.n; ++n) {
    for (int ho = 0; ho < shape.base.ho; ++ho) {
      const int base_ho = ho / p.conv.ay;
      const int ay_idx = ho - base_ho * p.conv.ay;
      const int by = output_block_index(base_ho, shape.block_ho);
      for (int wo = 0; wo < shape.base.wo; ++wo) {
        const int base_wo = wo / p.conv.ax;
        const int ax_idx = wo - base_wo * p.conv.ax;
        const int bx = output_block_index(base_wo, shape.block_wo);
        for (int g = 0; g < p.conv.groups; ++g) {
          const int cin_base = g * shape.base.cin_group;
          const int kout_base = g * shape.base.kout_group;
          for (int ko = 0; ko < shape.base.kout_group; ++ko) {
            float acc = 0.0f;
            for (int rr = 0; rr < w.r; ++rr) {
              const int hi = base_ho * p.conv.stride_h - p.conv.pad_h + rr * p.conv.dilation_h;
              if (hi < 0 || hi >= x.h) continue;
              for (int ss = 0; ss < w.s; ++ss) {
                const int wi = base_wo * p.conv.stride_w - p.conv.pad_w + ss * p.conv.dilation_w;
                if (wi < 0 || wi >= x.w) continue;
                for (int ci = 0; ci < shape.base.cin_group; ++ci) {
                  const float xv = x.data[idx_nhwc(n, hi, wi, cin_base + ci, x.h, x.w, x.c)];
                  const float wv = w.data[idx_kbybxrsc(kout_base + ko, by, bx, rr, ss, ci, ay_idx, ax_idx,
                                                       w.by, w.bx, w.r, w.s, w.cin_per_group, w.ay, w.ax)];
                  acc += xv * wv;
                }
              }
            }
            y.data[idx_nhwc(n, ho, wo, kout_base + ko, shape.base.ho, shape.base.wo, w.k)] = acc;
          }
        }
      }
    }
  }
}

void cpu_block_bprop_nhwc(const TensorNHWC& dy, const BlockFilterKByBxRSC& w, const BlockConv2DParams& p, TensorNHWC& dx) {
  const BlockConvShape shape = infer_block_conv_shape(dx, w, p);
  if (dy.h != shape.base.ho || dy.w != shape.base.wo || dy.c != w.k || dy.n != dx.n) {
    throw std::runtime_error("blocked bprop dy shape mismatch");
  }

  std::fill(dx.data.begin(), dx.data.end(), 0.0f);
  for (int n = 0; n < dx.n; ++n) {
    for (int hi = 0; hi < dx.h; ++hi) {
      for (int wi = 0; wi < dx.w; ++wi) {
        for (int g = 0; g < p.conv.groups; ++g) {
          const int cin_base = g * shape.base.cin_group;
          const int kout_base = g * shape.base.kout_group;
          for (int ci = 0; ci < shape.base.cin_group; ++ci) {
            float acc = 0.0f;
            for (int rr = 0; rr < w.r; ++rr) {
              const int base_ho_nom = hi + p.conv.pad_h - rr * p.conv.dilation_h;
              if (base_ho_nom < 0 || (base_ho_nom % p.conv.stride_h) != 0) continue;
              const int base_ho = base_ho_nom / p.conv.stride_h;
              if (base_ho < 0 || base_ho >= shape.base.base_ho) continue;
              const int by = output_block_index(base_ho, shape.block_ho);

              for (int ss = 0; ss < w.s; ++ss) {
                for (int ko = 0; ko < shape.base.kout_group; ++ko) {
                  const int base_wo_nom = wi + p.conv.pad_w - ss * p.conv.dilation_w;
                  if (base_wo_nom < 0 || (base_wo_nom % p.conv.stride_w) != 0) continue;
                  const int base_wo = base_wo_nom / p.conv.stride_w;
                  if (base_wo < 0 || base_wo >= shape.base.base_wo) continue;
                  const int bx = output_block_index(base_wo, shape.block_wo);
                  for (int ay_idx = 0; ay_idx < p.conv.ay; ++ay_idx) {
                    const int ho = base_ho * p.conv.ay + ay_idx;
                    for (int ax_idx = 0; ax_idx < p.conv.ax; ++ax_idx) {
                      const int wo = base_wo * p.conv.ax + ax_idx;
                      const float dyv = dy.data[idx_nhwc(n, ho, wo, kout_base + ko, dy.h, dy.w, dy.c)];
                      const float wv = w.data[idx_kbybxrsc(kout_base + ko, by, bx, rr, ss, ci, ay_idx, ax_idx,
                                                           w.by, w.bx, w.r, w.s, w.cin_per_group, w.ay, w.ax)];
                      acc += dyv * wv;
                    }
                  }
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

void cpu_block_grad_nhwc(const TensorNHWC& x, const TensorNHWC& dy, const BlockConv2DParams& p, BlockFilterKByBxRSC& dw) {
  const BlockConvShape shape = infer_block_conv_shape(x, dw, p);
  if (dy.h != shape.base.ho || dy.w != shape.base.wo || dy.c != dw.k || dy.n != x.n) {
    throw std::runtime_error("blocked grad dy shape mismatch");
  }

  std::fill(dw.data.begin(), dw.data.end(), 0.0f);
  for (int by = 0; by < dw.by; ++by) {
    const int ho0 = by * shape.block_ho;
    for (int bx = 0; bx < dw.bx; ++bx) {
      const int wo0 = bx * shape.block_wo;
      for (int rr = 0; rr < dw.r; ++rr) {
        for (int ss = 0; ss < dw.s; ++ss) {
          for (int g = 0; g < p.conv.groups; ++g) {
            const int cin_base = g * shape.base.cin_group;
            const int kout_base = g * shape.base.kout_group;
            for (int ci = 0; ci < shape.base.cin_group; ++ci) {
              for (int ko = 0; ko < shape.base.kout_group; ++ko) {
                for (int ay_idx = 0; ay_idx < dw.ay; ++ay_idx) {
                  for (int ax_idx = 0; ax_idx < dw.ax; ++ax_idx) {
                    float acc = 0.0f;
                    for (int n = 0; n < x.n; ++n) {
                      for (int lho = 0; lho < shape.block_ho; ++lho) {
                        const int base_ho = ho0 + lho;
                        const int hi = base_ho * p.conv.stride_h - p.conv.pad_h + rr * p.conv.dilation_h;
                        if (hi < 0 || hi >= x.h) continue;
                        const int ho = base_ho * p.conv.ay + ay_idx;
                        for (int lwo = 0; lwo < shape.block_wo; ++lwo) {
                          const int base_wo = wo0 + lwo;
                          const int wi = base_wo * p.conv.stride_w - p.conv.pad_w + ss * p.conv.dilation_w;
                          if (wi < 0 || wi >= x.w) continue;
                          const int wo = base_wo * p.conv.ax + ax_idx;
                          const float xv = x.data[idx_nhwc(n, hi, wi, cin_base + ci, x.h, x.w, x.c)];
                          const float dyv = dy.data[idx_nhwc(n, ho, wo, kout_base + ko, dy.h, dy.w, dy.c)];
                          acc += xv * dyv;
                        }
                      }
                    }
                    dw.data[idx_kbybxrsc(kout_base + ko, by, bx, rr, ss, ci, ay_idx, ax_idx,
                                         dw.by, dw.bx, dw.r, dw.s, dw.cin_per_group, dw.ay, dw.ax)] = acc;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

namespace conv_quant_detail {

void cpu_fprop_nhwc_qi32_u8(const TensorNHWC& x, const FilterKRSC& w, const Conv2DParams& p,
                            TensorNHWCI32& y,
                            const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* in_qp,
                            const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* f_qp,
                            nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp) {
  cpu_fprop_nhwc_qi32_impl(x, w, p, y, in_qp, f_qp, out_qp);
}

void cpu_fprop_nhwc_qi32_s5(const TensorNHWC& x, const FilterKRSC& w, const Conv2DParams& p,
                            TensorNHWCI32& y,
                            const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* in_qp,
                            const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* f_qp,
                            nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp) {
  cpu_fprop_nhwc_qi32_impl(x, w, p, y, in_qp, f_qp, out_qp);
}

void cpu_block_fprop_nhwc_qi32_u8(const TensorNHWC& x, const BlockFilterKByBxRSC& w, const BlockConv2DParams& p,
                                  TensorNHWCI32& y,
                                  const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* in_qp,
                                  const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantU8>* f_qp,
                                  nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp) {
  cpu_block_fprop_nhwc_qi32_impl(x, w, p, y, in_qp, f_qp, out_qp);
}

void cpu_block_fprop_nhwc_qi32_s5(const TensorNHWC& x, const BlockFilterKByBxRSC& w, const BlockConv2DParams& p,
                                  TensorNHWCI32& y,
                                  const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* in_qp,
                                  const nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantS5>* f_qp,
                                  nnalgebra::QuantizationParameters<nnalgebra::DataType::LinQuantI32>* out_qp) {
  cpu_block_fprop_nhwc_qi32_impl(x, w, p, y, in_qp, f_qp, out_qp);
}

}  // namespace conv_quant_detail
