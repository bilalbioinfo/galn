cmake_minimum_required(VERSION 3.18)
project(galn LANGUAGES C)

set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O2")

find_package(ZLIB REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS C)
find_package(Threads REQUIRED)

# bwa_cal_maxdiff must round exactly like upstream: no FMA contraction (aarch64 GCC contracts by default)
set_source_files_properties(src/aln_host.c PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "No CUDA toolkit found: G-ALN needs CUDA 11.8 or newer (nvcc).")
endif()
enable_language(CUDA)
# Default: Volta to Hopper (V100=70, T4=75, A100=80, A40/A10/RTX30=86, L40/RTX40=89, H100/GH200=90).
# Pin one to build faster, e.g. -DCMAKE_CUDA_ARCHITECTURES=90.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

set(GALN_INDEX_SRC src/index/galn_index.c src/index/gpu_index.cu src/index/cpu_ref.c src/index/idxcommon.c
    src/index/bwa_upstream/bntseq.c src/index/bwa_upstream/utils.c
    third_party/libsais/src/libsais.c third_party/libsais/src/libsais64.c third_party/libcubwt/libcubwt.cu)
set_source_files_properties(src/index/gpu_index.cu PROPERTIES COMPILE_DEFINITIONS "main=galn_index_gpu_main;LIBSAIS_OPENMP")
set_source_files_properties(src/index/cpu_ref.c PROPERTIES COMPILE_DEFINITIONS "main=galn_index_cpu_main;LIBSAIS_OPENMP")
set_source_files_properties(third_party/libsais/src/libsais.c third_party/libsais/src/libsais64.c
    PROPERTIES COMPILE_DEFINITIONS LIBSAIS_OPENMP)

add_executable(galn src/galn.cu src/host_io.c src/aln_host.c src/aln_il.c ${GALN_INDEX_SRC})
target_include_directories(galn PRIVATE src src/index third_party/libsais/include third_party/libcubwt)
target_compile_options(galn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
                                    $<$<COMPILE_LANGUAGE:C>:${OpenMP_C_FLAGS}>)
target_link_libraries(galn PRIVATE ZLIB::ZLIB gomp m Threads::Threads)
set_target_properties(galn PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 17)
