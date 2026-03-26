cmake_minimum_required(VERSION 3.24)
project(conv_bench LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "Semicolon-separated CUDA SM targets")
endif()

message(STATUS "Configuring CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
  message(WARNING "CUDA architecture 52 is a legacy target. Use a fresh build directory or one of the presets (sm86, sm89) for Ampere/Ada GPUs.")
endif()

option(ENABLE_CUDNN "Enable cuDNN benchmark/reference" ON)

find_package(CUDAToolkit REQUIRED)

function(configure_cuda_target target_name)
  target_include_directories(${target_name} PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${target_name} PRIVATE CUDA::cudart CUDA::cublas)
  set_target_properties(${target_name} PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    CUDA_SEPARABLE_COMPILATION ON
  )
endfunction()

add_executable(conv_bench
  bmm.cu
  src/main.cpp
  src/cpu_ref.cpp
  src/cuda_kernels.cu
  src/cuda_utils.cpp
  src/verify.cpp
  src/bench.cpp
  src/cudnn_ref.cpp
)

configure_cuda_target(conv_bench)

add_executable(bmm_bench
  bmm.cu
  bmm_bench.cu
)

configure_cuda_target(bmm_bench)

add_executable(bmm_test
  bmm.cu
  bmm_test.cu
)

configure_cuda_target(bmm_test)

add_executable(arcface_test
  arcface.cu
  arcface_test.cu
)

configure_cuda_target(arcface_test)

add_executable(quant_conv_test
  bmm.cu
  quant_conv_test.cu
  src/cpu_ref.cpp
  src/cuda_kernels.cu
  src/cuda_utils.cpp
)

configure_cuda_target(quant_conv_test)

add_executable(conv_grad_check
  conv_grad_check.cpp
  src/cpu_ref.cpp
)

target_include_directories(conv_grad_check PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(conv_grad_check PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

if(ENABLE_CUDNN)
  set(CUDNN_ROOT_HINTS
    $ENV{CUDNN_ROOT}
    "C:/Program Files/NVIDIA/CUDNN"
    "C:/Program Files/NVIDIA/CUDNN/v9.19"
  )

  find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS
      ${CUDNN_ROOT_HINTS}
      ${CUDAToolkit_INCLUDE_DIRS}
      /usr/include
      /usr/local/cuda/include
    PATH_SUFFIXES
      include
      include/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}
      include/12.9
      include/13.1
  )

  find_library(CUDNN_LIBRARY
    NAMES cudnn cudnn64_9
    HINTS
      ${CUDNN_ROOT_HINTS}
      /usr/lib
      /usr/local/cuda/lib64
    PATH_SUFFIXES
      lib
      lib/x64
      lib/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}/x64
      lib/12.9/x64
      lib/13.1/x64
  )

  if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    target_compile_definitions(conv_bench PRIVATE USE_CUDNN=1)
    target_include_directories(conv_bench PRIVATE ${CUDNN_INCLUDE_DIR})
    target_link_libraries(conv_bench PRIVATE ${CUDNN_LIBRARY})
    message(STATUS "cuDNN found: ${CUDNN_LIBRARY}")
  else()
    message(STATUS "cuDNN not found; build without cuDNN support")
  endif()
endif()

if(BUILD_TESTING)
  add_test(NAME bmm_test_tiny_1x1x1x1
    COMMAND bmm_test --case tiny_1x1x1x1)
  add_test(NAME bmm_test_irregular_17x19x13
    COMMAND bmm_test --case irregular_17x19x13)
  add_test(NAME bmm_test_shape_128x32x16
    COMMAND bmm_test --case shape_128x32x16)
  add_test(NAME bmm_test_default_64x64x128
    COMMAND bmm_test --case default_64x64x128)
  add_test(NAME bmm_test_wide_128x128x256
    COMMAND bmm_test --case wide_128x128x256)
  add_test(NAME bmm_test_tall_256x64x256
    COMMAND bmm_test --case tall_256x64x256)

  set_tests_properties(
    bmm_test_tiny_1x1x1x1
    bmm_test_irregular_17x19x13
    bmm_test_shape_128x32x16
    bmm_test_default_64x64x128
    bmm_test_wide_128x128x256
    bmm_test_tall_256x64x256
    PROPERTIES LABELS "bmm"
  )

  add_test(NAME arcface_test COMMAND arcface_test)
  set_tests_properties(arcface_test PROPERTIES LABELS "arcface")

  add_test(NAME quant_conv_test COMMAND quant_conv_test)
  set_tests_properties(quant_conv_test PROPERTIES LABELS "quant_conv")

  add_test(NAME conv_grad_check_128x128 COMMAND conv_grad_check)
  set_tests_properties(conv_grad_check_128x128 PROPERTIES LABELS "conv;grad")
endif()
