cmake_minimum_required(VERSION 3.24)
project(conv_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

option(ENABLE_CUDNN "Enable cuDNN benchmark/reference" ON)

find_package(CUDAToolkit REQUIRED)

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

target_include_directories(conv_bench PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(conv_bench PRIVATE CUDA::cudart)

if(ENABLE_CUDNN)
  find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS
      $ENV{CUDNN_ROOT}
      ${CUDAToolkit_INCLUDE_DIRS}
      /usr/include
      /usr/local/cuda/include
  )

  find_library(CUDNN_LIBRARY cudnn
    HINTS
      $ENV{CUDNN_ROOT}
      /usr/lib
      /usr/local/cuda/lib64
      "C:/Program Files/NVIDIA/CUDNN"
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

set_target_properties(conv_bench PROPERTIES
  CUDA_ARCHITECTURES 75
  CUDA_SEPARABLE_COMPILATION ON
)
