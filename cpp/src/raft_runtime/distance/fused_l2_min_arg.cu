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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/kvp.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/distance/fused_l2_nn.cuh>
#include <raft/distance/specializations.cuh>
#include <raft/linalg/norm.cuh>
#include <thrust/for_each.h>
#include <thrust/tuple.h>

namespace raft::runtime::distance {

template <typename IndexT, typename DataT>
struct KeyValueIndexOp {
  __host__ __device__ __forceinline__ IndexT
  operator()(const raft::KeyValuePair<IndexT, DataT>& a) const
  {
    return a.key;
  }
};

template <typename value_t, typename idx_t>
void compute_fused_l2_nn_min_arg(raft::device_resources const& handle,
                                 idx_t* min,
                                 const value_t* x,
                                 const value_t* y,
                                 idx_t m,
                                 idx_t n,
                                 idx_t k,
                                 bool sqrt)
{
  rmm::device_uvector<int> workspace(m, handle.get_stream());
  auto kvp = raft::make_device_vector<raft::KeyValuePair<idx_t, value_t>>(handle, m);

  rmm::device_uvector<value_t> x_norms(m, handle.get_stream());
  rmm::device_uvector<value_t> y_norms(n, handle.get_stream());
  raft::linalg::rowNorm(x_norms.data(), x, k, m, raft::linalg::L2Norm, true, handle.get_stream());
  raft::linalg::rowNorm(y_norms.data(), y, k, n, raft::linalg::L2Norm, true, handle.get_stream());

  raft::distance::fusedL2NNMinReduce(kvp.data_handle(),
                                     x,
                                     y,
                                     x_norms.data(),
                                     y_norms.data(),
                                     m,
                                     n,
                                     k,
                                     (void*)workspace.data(),
                                     sqrt,
                                     true,
                                     handle.get_stream());

  KeyValueIndexOp<idx_t, value_t> conversion_op;
  thrust::transform(
    handle.get_thrust_policy(), kvp.data_handle(), kvp.data_handle() + m, min, conversion_op);
  handle.sync_stream();
}

void fused_l2_nn_min_arg(raft::device_resources const& handle,
                         int* min,
                         const float* x,
                         const float* y,
                         int m,
                         int n,
                         int k,
                         bool sqrt)
{
  compute_fused_l2_nn_min_arg<float, int>(handle, min, x, y, m, n, k, sqrt);
}

void fused_l2_nn_min_arg(raft::device_resources const& handle,
                         int* min,
                         const double* x,
                         const double* y,
                         int m,
                         int n,
                         int k,
                         bool sqrt)
{
  compute_fused_l2_nn_min_arg<double, int>(handle, min, x, y, m, n, k, sqrt);
}

}  // end namespace raft::runtime::distance
