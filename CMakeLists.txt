cmake_minimum_required(VERSION 3.18)
project(my_own_cnn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(ENABLE_CUDNN "Enable cuDNN acceleration" ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_library(mycnn
    src/tensor.cpp
    src/pooling.cpp
    src/batchnorm.cpp
    src/softmax.cpp
    src/cross_entropy.cpp
    src/convolution.cpp
    src/convolution_cuda.cu
    src/convolution_cudnn.cpp
    src/pooling_cuda.cu
    src/softmax_cuda.cu
    src/batchnorm_cuda.cu
    src/cross_entropy_cuda.cu)

target_include_directories(mycnn PUBLIC include)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(mycnn PUBLIC CUDA::cudart)

if(ENABLE_CUDNN)
  find_path(CUDNN_INCLUDE_DIR cudnn.h)
  find_library(CUDNN_LIBRARY cudnn)
  if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    target_compile_definitions(mycnn PUBLIC USE_CUDNN)
    target_include_directories(mycnn PUBLIC ${CUDNN_INCLUDE_DIR})
    target_link_libraries(mycnn PUBLIC ${CUDNN_LIBRARY})
  else()
    message(WARNING "cuDNN not found, building without cuDNN support")
  endif()
endif()

enable_testing()

add_executable(mycnn_tests
    tests/pooling_test.cpp
    tests/softmax_test.cpp
    tests/batchnorm_test.cpp
    tests/cross_entropy_test.cpp
    tests/convolution_test.cpp)

target_link_libraries(mycnn_tests PRIVATE mycnn GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(mycnn_tests)
