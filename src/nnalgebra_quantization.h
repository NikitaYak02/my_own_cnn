#pragma once

#include <cstdint>
#include <type_traits>

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

using real32_t = float;

namespace nnalgebra {

enum class DataType : int {
  Real32,
  LinQuantU8,
  LinQuantS5,
  LinQuantI32,
};

template <DataType T>
struct QuantizationParameters;

template <>
struct QuantizationParameters<DataType::LinQuantU8> {
  uint8_t zero_point = 0;
  real32_t scale = 1.f;
  bool symmetric = false;
};

template <>
struct QuantizationParameters<DataType::LinQuantS5> {
  int8_t zero_point = 0;
  uint8_t n_bins = 0;
  real32_t scale = 1.f;
};

template <>
struct QuantizationParameters<DataType::LinQuantI32> {
  static constexpr int32_t zero_point = 0;
  real32_t scale = 1.f;
};

template <>
struct QuantizationParameters<DataType::Real32> {
  uint16_t zero_point = 0;
  real32_t scale = 1.f;
  bool symmetric = false;
};

template <DataType T>
__host__ __device__ inline int getZeroPoint(const QuantizationParameters<T>& qp);

template <DataType T>
__host__ __device__ inline float getScale(const QuantizationParameters<T>& qp);

template <>
__host__ __device__ inline int getZeroPoint<DataType::LinQuantU8>(
    const QuantizationParameters<DataType::LinQuantU8>& qp) {
  return static_cast<int>(qp.zero_point);
}

template <>
__host__ __device__ inline int getZeroPoint<DataType::LinQuantS5>(
    const QuantizationParameters<DataType::LinQuantS5>& qp) {
  return static_cast<int>(qp.zero_point);
}

template <>
__host__ __device__ inline int getZeroPoint<DataType::LinQuantI32>(
    const QuantizationParameters<DataType::LinQuantI32>&) {
  return 0;
}

template <>
__host__ __device__ inline int getZeroPoint<DataType::Real32>(
    const QuantizationParameters<DataType::Real32>& qp) {
  return static_cast<int>(qp.zero_point);
}

template <>
__host__ __device__ inline float getScale<DataType::LinQuantU8>(
    const QuantizationParameters<DataType::LinQuantU8>& qp) {
  return static_cast<float>(qp.scale);
}

template <>
__host__ __device__ inline float getScale<DataType::LinQuantS5>(
    const QuantizationParameters<DataType::LinQuantS5>& qp) {
  return static_cast<float>(qp.scale);
}

template <>
__host__ __device__ inline float getScale<DataType::LinQuantI32>(
    const QuantizationParameters<DataType::LinQuantI32>& qp) {
  return static_cast<float>(qp.scale);
}

template <>
__host__ __device__ inline float getScale<DataType::Real32>(
    const QuantizationParameters<DataType::Real32>& qp) {
  return static_cast<float>(qp.scale);
}

template <DataType T>
struct IsSupportedQuantizedInput : std::false_type {};

template <>
struct IsSupportedQuantizedInput<DataType::LinQuantU8> : std::true_type {};

template <>
struct IsSupportedQuantizedInput<DataType::LinQuantS5> : std::true_type {};

template <DataType T>
inline constexpr bool kIsSupportedQuantizedInput = IsSupportedQuantizedInput<T>::value;

}  // namespace nnalgebra
