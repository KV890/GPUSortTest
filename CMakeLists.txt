cmake_minimum_required(VERSION 3.26)
project(GPUSortTest CXX C CUDA ASM)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(CMAKE_CXX_STANDARD 17)

add_executable(GPUSortTest main.cpp sort/gpu_sort.cu
        util/properties.cc
        sort/sorting_factory.cc
        producer/client.cc
        producer/workload.cc
        producer/Workload2.cc
        sort/cpu_sort.cc
        util/random.cc)

set_target_properties(GPUSortTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_ARCHITECTURES "50;60;61;72;75"
        CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)
