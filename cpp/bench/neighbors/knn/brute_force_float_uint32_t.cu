/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "../knn.cuh"

namespace raft::spatial::knn {

#define RAFT_INST(idx_t, value_t, value_int)                     \
  extern template void brute_force_knn<uint32_t, float, size_t>( \
    raft::device_resources const& handle,                        \
    std::vector<value_t*>& input,                                \
    std::vector<value_int>& sizes,                               \
    value_int D,                                                 \
    value_t* search_items,                                       \
    value_int n,                                                 \
    idx_t* res_I,                                                \
    value_t* res_D,                                              \
    value_int k,                                                 \
    bool rowMajorIndex,                                          \
    bool rowMajorQuery,                                          \
    std::vector<idx_t>* translations,                            \
    distance::DistanceType metric,                               \
    float metric_arg);

RAFT_INST(uint32_t, float, size_t);
#undef RAFT_INST
}  // namespace raft::spatial::knn

namespace raft::bench::spatial {

#define KNN_REGISTER(ValT, IdxT, ImplT, inputs, strats, scope)                   \
  namespace BENCHMARK_PRIVATE_NAME(knn)                                          \
  {                                                                              \
    using KNN = knn<ValT, IdxT, ImplT<ValT, IdxT>>;                              \
    RAFT_BENCH_REGISTER(KNN, #ValT "/" #IdxT "/" #ImplT, inputs, strats, scope); \
  }

using ValT = float;
using IdxT = uint32_t;
using KNN  = knn<ValT, IdxT, brute_force_knn<ValT, IdxT>>;

inline const std::vector<params> inputs{
  // n_samples, n_dims, n_queries, k
  // {1<<10, 32, 1<<10, 10},
  // {1<<10, 32, 1<<10, 64},
  // {1<<10, 32, 1<<10, 128},

  {1 << 21, 512, 1 << 15, 10},
  {1 << 21, 512, 1 << 15, 64},
  {1 << 21, 512, 1 << 15, 128},
};

RAFT_BENCH_REGISTER(KNN, "float/uint32_t/brute_force", inputs, kNoCopyOnly, kScopeFull);

}  // namespace raft::bench::spatial
