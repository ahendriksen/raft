/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#pragma once
#include <raft/util/cuda_utils.cuh>

namespace raft::distance::detail::ops {

// Describes the computation the hellinger distance
//
// Fill in the TODO items.

struct hellinger_distance_op {
  // Load norms of input data
  static constexpr bool use_norms = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy, typename DataT>
  constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize;
  }

  template <typename AccT, typename DataT>
  DI void core(AccT& acc, DataT& x, DataT& y) const
  {
    // This is sqrt(x) * sqrt(y).
    const auto product = x * y;
    acc += product;
  };

  template <typename Policy, typename AccT, typename DataT, typename IdxT>
  DI void epilog(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 DataT* regxn,
                 DataT* regyn,
                 IdxT gridStrideX,
                 IdxT gridStrideY) const
  {
#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < Policy::AccColsPerTh; ++j) {
        // Adjust to replace NaN in sqrt with 0 if input to sqrt is negative
        const auto finalVal  = (1 - acc[i][j]);
        const auto rectifier = (!signbit(finalVal));
        acc[i][j]            = raft::sqrt(rectifier * finalVal);
      }
    }
  }
};

}  // namespace raft::distance::detail::ops
