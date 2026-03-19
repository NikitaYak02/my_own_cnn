

__global__
void update_ideals_ker(float *lbls, int *cntr_new, double *idls_new, double *ideals_pow_2,
                       float *v, const int ni, const int nc, const int alphabet_size, int i_lbl)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int c = threadIdx.y + blockDim.y * blockIdx.y;

  if (i < ni && c < nc) {
    const int nn = int(lbls[i + i_lbl * ni]);
    if (c == 0) {
      atomicAdd(&cntr_new[nn], 1);
    }
    atomicAdd(&idls_new[nn * nc + c], static_cast<double>(v[c * ni + i]));
    atomicAdd(&ideals_pow_2[nn * nc + c], static_cast<double>(v[c * ni + i] * v[c * ni + i]));
  }
}

void UpdateIdeals(double *ideals, double *ideals_pow_2, int *counter, IMatrixFloat &v, IMatrixFloat &labels, int i_lbl, const int alphabet_size) {
  const int ni = v.getLastDimShape();
  const int nc = v.getFirstDimsShape();

  dim3 ths(MIN(ni, 32), MIN(nc, 4), 1);
  dim3 bls(DIVUP(ni, ths.x), DIVUP(nc, ths.y), 1);

  update_ideals_ker<<<bls, ths>>>(labels.getDataAsType<real32_t>(), counter, ideals, ideals_pow_2, v.getDataAsType<real32_t>(),
                                  ni, nc, alphabet_size, i_lbl);
  cutilCheckMsg("UpdateIdeals: Kernel execution failed", bls, ths);
}



template<typename T> __global__
void to_ker(T *ideals, float *to, float *lbl, const int ni, const int ch) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < ni) {
    int c = blockIdx.y;
    int cur_lbl = lbl[i];

    to[c * ni + i] = ideals[cur_lbl * ch + c];
  }
}


template<typename T> __global__
    void to_ker_countered(T *ideals, float *to, float *lbl, const int ni, const int ch, const int *counter) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < ni) {
    int c = blockIdx.y;
    int cur_lbl = lbl[i];
    if (counter[cur_lbl] > 0) {
      to[c * ni + i] = ideals[cur_lbl * ch + c];
    }
  }
}

void fillTo(double *ideals, GPUMatrixFloat &to, GPUMatrixFloat &lbl) {
  const int ni = to.getLastDimShape();
  const int ch = to.getFirstDimsShape();

  dim3 ths(std::min(ni, 32), 1, 1);
  dim3 bls(DIVUP(ni, ths.x), ch, 1);

  to_ker<double><<<bls, ths>>>(ideals, to.getDataAsType<real32_t>(), lbl.getDataAsType<real32_t>(), ni, ch);
  cutilCheckMsg("fillTo<double>: Kernel execution failed", bls, ths);
}

void fillTo(float *ideals, GPUMatrixFloat &to, GPUMatrixFloat &lbl) {
  const int ni = to.getLastDimShape();
  const int ch = to.getFirstDimsShape();

  dim3 ths(std::min(ni, 32), 1, 1);
  dim3 bls(DIVUP(ni, ths.x), ch, 1);
  to_ker<float><<<bls, ths>>>(ideals, to.getDataAsType<real32_t>(), lbl.getDataAsType<real32_t>(), ni, ch);
  cutilCheckMsg("fillTo<float>: Kernel execution failed", bls, ths);
}

void fillToCountered(float *ideals, GPUMatrixFloat &to, GPUMatrixFloat &lbl, int *counter) {
  const int ni = to.getLastDimShape();
  const int ch = to.getFirstDimsShape();

  dim3 ths(std::min(ni, 32), 1, 1);
  dim3 bls(DIVUP(ni, ths.x), ch, 1);
  to_ker_countered<float><<<bls, ths>>>(ideals, to.getDataAsType<real32_t>(), lbl.getDataAsType<real32_t>(), ni, ch, counter);
  cutilCheckMsg("fillTo<float>: Kernel execution failed", bls, ths);
}

__global__
void to_dists(float *d, double *d_vec, float *lbl, int ni) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < ni) {
    int cur_lbl = lbl[i];
    atomicAdd(&d_vec[cur_lbl], static_cast<double>(d[i]));
  }
}

void fillToDists(GPUMatrixFloat &dists, double *dists_vec, GPUMatrixFloat &lbl) {
  const int ni = dists.getLastDimShape();

  dim3 ths(MIN(ni, 512), 1);
  dim3 bls(DIVUP(ni, ths.x), 1, 1);
  my_assert(bls.x <= NUM_BLOCKS_MAX_X);
  to_dists<<<bls, ths>>>(dists.getDataAsType<real32_t>(), dists_vec, lbl.getDataAsType<real32_t>(), ni);
  cutilCheckMsg("fillTo<double>: Kernel execution failed", bls, ths);
}

__global__
void shift_dists_ker(float *v, float *l, float *ad, float *al, const int ni, const int ch, const float pow) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j_d = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < ni && j_d < ni - 1) {
    int j_v = (j_d + i + 1) % ni;

    float s = 0.f;
    float t;
    for (int c = 0; c < ch; ++c) {
      t = fabsf(v[c * ni + i] - v[c * ni + j_v]);
      s += powf(t, pow);
    }

    ad[j_d * ni + i] = powf(s, 1.f / pow);
    al[j_d * ni + i] = l[i] != l[j_v];
  }
}

__global__
void shift_dists_ker(float *v, float *l, float *ad, float *al, const int ni, const int ch) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j_d = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < ni && j_d < ni - 1) {
    int j_v = (j_d + i + 1) % ni;

    float s = 0.f;
    float t;
    for (int c = 0; c < ch; ++c) {
      t = v[c * ni + i] - v[c * ni + j_v];
      s += t * t;
    }

    ad[j_d * ni + i] = __fsqrt_rz(s);
    al[j_d * ni + i] = l[i] != l[j_v];
  }
}

void getDistsWithShifts(GPUMatrixFloat &v, GPUMatrixFloat &l, GPUMatrixFloat &all_dists, GPUMatrixFloat &all_lbls, const float pow) {
  const int ni = v.getLastDimShape();
  const int ch = v.getFirstDimsShape();
  dim3 ths(std::min(32, ni), std::min(8, ni - 1), 1);
  dim3 bls(DIVUP(ni, ths.x), DIVUP(ni - 1, ths.y), 1);
  shift_dists_ker<<<bls, ths>>>(v.getDataAsType<real32_t>(), l.getDataAsType<real32_t>(), all_dists.getDataAsType<real32_t>(), all_lbls.getDataAsType<real32_t>(), ni, ch, pow);
}

void getDistsWithShifts(GPUMatrixFloat &v, GPUMatrixFloat &l, GPUMatrixFloat &all_dists, GPUMatrixFloat &all_lbls) {
  const int ni = v.getLastDimShape();
  const int ch = v.getFirstDimsShape();
  dim3 ths(std::min(32, ni), std::min(8, ni - 1), 1);
  dim3 bls(DIVUP(ni, ths.x), DIVUP(ni - 1, ths.y), 1);
  shift_dists_ker<<<bls, ths>>>(v.getDataAsType<real32_t>(), l.getDataAsType<real32_t>(), all_dists.getDataAsType<real32_t>(),
                                all_lbls.getDataAsType<real32_t>(), ni, ch);
}

__global__
void shift_grads_ker(float *v, float *agd, float *ags, float *ad, float *al,
                     const int ni, const int ch, const float alpha, const float pow, const bool ideal) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j_d = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z;

  if (i < ni && j_d < ni - 1) {
    int j_v = (j_d + i + 1) % ni;

    if (ideal) {
      agd[j_d * ch * ni + c * ni + i] = al[j_d * ni + i] * (ad[j_d * ni + i] < alpha) * (-pow * powf(v[c * ni + i] - v[c * ni + j_v], pow - 1.f));
    } else {
      agd[j_d * ch * ni + c * ni + i] = al[j_d * ni + i] * (ad[j_d * ni + i] < alpha) * (-pow * (powf(alpha - ad[j_d * ni + i], pow - 1) / (ad[j_d * ni + i] + EPSILON)) * (v[c * ni + i] - v[c * ni + j_v]));
      ags[j_d * ch * ni + c * ni + i] = (1 - al[j_d * ni + i]) * pow * powf(ad[j_d * ni + i], pow - 1) * (v[c * ni + i] - v[c * ni + j_v]);
    }
  }
}

__global__
void shift_grads_ker(float *v, float *ag, float *ad, float *al,
                     const int ni, const int ch, const float alpha, const float pow) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j_d = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z;

  if (i < ni && j_d < ni - 1) {
    int j_v = (j_d + i + 1) % ni;

    ag[j_d * ch * ni + c * ni + i] = al[j_d * ni + i] *
            (-pow) * powf((v[c * ni + i] - v[c * ni + j_v]), pow - 1.f);
  }
}

void getGradsWithShifts(GPUMatrixFloat &v, GPUMatrixFloat &all_grads,
                        GPUMatrixFloat &all_dists, GPUMatrixFloat &all_lbls, const float alpha, const float pow) {
  const int ni = v.getLastDimShape();
  const int ch = v.getFirstDimsShape();
  dim3 ths(std::min(32, ni), std::min(8, ni - 1), 1);
  dim3 bls(DIVUP(ni, ths.x), DIVUP(ni - 1, ths.y), ch);
  shift_grads_ker<<<bls, ths>>>(v.getDataAsType<real32_t>(), all_grads.getDataAsType<real32_t>(), all_dists.getDataAsType<real32_t>(), all_lbls.getDataAsType<real32_t>(),
                                ni, ch, alpha, pow);
}

void getGradsWithShifts(GPUMatrixFloat &v, GPUMatrixFloat &all_grads_diff, GPUMatrixFloat &all_grads_same,
                        GPUMatrixFloat &all_dists, GPUMatrixFloat &all_lbls, const float alpha, const float pow, const bool ideal) {
  const int ni = v.getLastDimShape();
  const int ch = v.getFirstDimsShape();
  dim3 ths(std::min(32, ni), std::min(8, ni - 1), 1);
  dim3 bls(DIVUP(ni, ths.x), DIVUP(ni - 1, ths.y), ch);
  shift_grads_ker<<<bls, ths>>>(v.getDataAsType<real32_t>(), all_grads_diff.getDataAsType<real32_t>(), all_grads_same.getDataAsType<real32_t>(), all_dists.getDataAsType<real32_t>(), all_lbls.getDataAsType<real32_t>(),
                                ni, ch, alpha, pow, ideal);
}
