#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ArcFace loss pre-processing layer operating on cosine logits.
 *
 * Input/output layout:
 *   cosine, logits, d_logits, d_cosine - [batch, classes] row-major
 *   labels                             - [batch]
 *
 * Forward:
 *   logits[i, j] = scale * cosine[i, j]                        for j != labels[i]
 *   logits[i, y] = scale * cos(acos(cosine[i, y]) + margin)    for y = labels[i]
 *
 * When easy_margin is disabled, the standard ArcFace fallback
 *   cos(theta + margin) -> cosine - sin(pi - margin) * margin
 * is used below the monotonicity threshold.
 *
 * Backward consumes the gradient produced by the downstream cross-entropy
 * layer and returns the gradient with respect to the input cosine logits.
 *
 * All pointers must refer to device memory. Labels are int32 indices.
 */

void arcface_fprop(
    const float* cosine,
    const int* labels,
    float* logits,
    int batch,
    int classes,
    float margin,
    float scale,
    int easy_margin);

void arcface_bprop(
    const float* cosine,
    const int* labels,
    const float* d_logits,
    float* d_cosine,
    int batch,
    int classes,
    float margin,
    float scale,
    int easy_margin);

#ifdef __cplusplus
}
#endif
