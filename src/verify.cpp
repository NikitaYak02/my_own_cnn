#include "conv_types.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

VerifyResult verify_tensors(const std::vector<float>& ref, const std::vector<float>& got, float abs_eps, float rel_eps) {
  if (ref.size() != got.size()) {
    throw std::runtime_error("verify_tensors size mismatch");
  }

  VerifyResult out;
  bool ok = true;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float a = ref[i];
    const float b = got[i];
    const float abs_err = std::fabs(a - b);
    const float denom = std::max(std::fabs(a), 1e-6f);
    const float rel_err = abs_err / denom;
    if (abs_err > out.max_abs_err) {
      out.max_abs_err = abs_err;
      out.max_abs_idx = i;
    }
    if (rel_err > out.max_rel_err) {
      out.max_rel_err = rel_err;
      out.max_rel_idx = i;
    }

    const bool rel_relevant = std::fabs(a) > 1e-6f;
    const bool fail = (abs_err > abs_eps) && (!rel_relevant || rel_err > rel_eps);
    if (fail) ok = false;
  }

  out.passed = ok;
  return out;
}
