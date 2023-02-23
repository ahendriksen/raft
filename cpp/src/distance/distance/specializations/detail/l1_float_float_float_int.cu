/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <raft/distance/detail/distance.cuh>

/*
 * Configure with:
 * cmake -S cpp -B cpp/build \
   -DCMAKE_INSTALL_PREFIX= \
   -DCMAKE_CUDA_ARCHITECTURES=70 \
   -DCMAKE_BUILD_TYPE=Release \
   -DRAFT_COMPILE_LIBRARIES=OFF \
   -DRAFT_ENABLE_NN_DEPENDENCIES=OFF \
   -DRAFT_NVTX=ON \
   -DDISABLE_DEPRECATION_WARNINGS=ON \
   -DBUILD_TESTS=ON \
   -DBUILD_BENCH=ON \
   -DCMAKE_MESSAGE_LOG_LEVEL=  \
   -DRAFT_COMPILE_NN_LIBRARY=OFF \
   -DRAFT_COMPILE_DIST_LIBRARY=ON \
   -DRAFT_USE_FAISS_STATIC=OFF \
   -DRAFT_ENABLE_thrust_DEPENDENCY=ON \
   -DFIND_RAFT_CPP=ON \
   -DCUDA_LOG_COMPILE_TIMES=ON \
   -G Ninja

 * Compile with:
 *
 * time nvcc \
   --time=../../nvcc_compile_full.csv \
   -forward-unknown-to-host-compiler -DCUTLASS_NAMESPACE=raft_cutlass \
   -DFMT_HEADER_ONLY=1 -DNVTX_ENABLED -DRAFT_SYSTEM_LITTLE_ENDIAN=1 -DSPDLOG_FMT_EXTERNAL \
   -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP \
   -Draft_distance_lib_EXPORTS \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/include \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/rmm-src/include \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/thrust-src \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/thrust-src/dependencies/cub \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/fmt-src/include \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/spdlog-src/include \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/cuco-src/include \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/libcudacxx-src/lib/cmake/libcudacxx/../../../include \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/nvidiacutlass-src/include \
   -I/home/ahendriksen/projects/raft-spdlog-issue/cpp/build/_deps/nvidiacutlass-build/include \
   -isystem=/nix/store/sivnyagc6j0skwfgwhy2a4hlwy972kic-mycudatoolkit/include \
   -O3 -DNDEBUG --generate-code=arch=compute_70,code=[compute_70,sm_70] \
   -Xcompiler=-fPIC -Xcompiler=-Wno-deprecated-declarations \
   --expt-extended-lambda --expt-relaxed-constexpr \
   -DCUDA_API_PER_THREAD_DEFAULT_STREAM \
   -Xfatbin=-compress-all  -Werror=all-warnings -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations \
   -Xcompiler=-fopenmp -std=c++17 \
   -MD -MT CMakeFiles/raft_distance_lib.dir/src/distance/distance/specializations/detail/l1_float_float_float_int.cu.o \
   -MF CMakeFiles/raft_distance_lib.dir/src/distance/distance/specializations/detail/l1_float_float_float_int.cu.o.d \
   -x cu \
   -c /home/ahendriksen/projects/raft-spdlog-issue/cpp/src/distance/distance/specializations/detail/l1_float_float_float_int.cu \
   -o CMakeFiles/raft_distance_lib.dir/src/distance/distance/specializations/detail/l1_float_float_float_int.cu.o
 *
 * real    0m20.184s
 * user    0m18.713s
 * sys     0m1.441s
 *
 * python -c 'import pandas as pd; print(pd.read_csv("../../nvcc_compile_full.csv").rename(columns=str.strip)[["phase name", "metric", "unit"]].sort_values("metric"))'
 *
 *                 phase name     metric unit
 * 4               fatbinary     35.7960   ms
 * 7           nvcc (driver)     57.6406   ms
 * 1   gcc (preprocessing 4)    657.9370   ms
 * 0   gcc (preprocessing 1)    672.9990   ms
 * 3                   ptxas    767.4330   ms
 * 5                cudafe++   3136.6760   ms
 * 6         gcc (compiling)   6988.9419   ms
 * 2                    cicc   7786.4458   ms
 */

namespace raft {
namespace distance {
namespace detail {
template void distance<raft::distance::DistanceType::L1, float, float, float, int>(
  raft::resources const& handle,
  const float* x,
  const float* y,
  float* dist,
  int m,
  int n,
  int k,
  void* workspace,
  std::size_t worksize,
  bool isRowMajor,
  float metric_arg);

}  // namespace detail
}  // namespace distance
}  // namespace raft
