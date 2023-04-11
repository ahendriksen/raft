#!/usr/bin/env python3

# NOTE: this template is not perfectly formatted. Use pre-commit to get
# everything in shape again.
header = """/*
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

#include <raft/core/operators.hpp> // raft::identity_op
#include <raft/distance/detail/distance_ops/all_ops.cuh>  // ops::*
#include <raft/distance/detail/pairwise_matrix/dispatch-inl.cuh> // dispatch
"""


macro = """
#define instantiate_raft_distance_detail_pairwise_matrix_dispatch(                     \\
  OpT, DataT, AccT, OutT, FinOpT, IdxT)                                                \\
  template void raft::distance::detail::                                               \\
    pairwise_matrix_dispatch<OpT<DataT, AccT, IdxT>, DataT, AccT, OutT, FinOpT, IdxT>( \\
      OpT<DataT, AccT, IdxT> distance_op,                                              \\
      IdxT m,                                                                          \\
      IdxT n,                                                                          \\
      IdxT k,                                                                          \\
      const DataT* x,                                                                  \\
      const DataT* y,                                                                  \\
      const DataT* x_norm,                                                             \\
      const DataT* y_norm,                                                             \\
      OutT* out,                                                                       \\
      FinOpT fin_op,                                                                   \\
      cudaStream_t stream,                                                             \\
      bool is_row_major)
"""

data_type_instances = [
    dict(
        DataT="float",
        AccT="float",
        OutT="float",
        IdxT="int",
    ),
    dict(
        DataT="double",
        AccT="double",
        OutT="double",
        IdxT="int",
    ),
]

op_instances = [
    dict(
        path_prefix="canberra",
        OpT="raft::distance::detail::ops::canberra_distance_op",
        archs = [60],
    ),
    dict(
        path_prefix="correlation",
        OpT="raft::distance::detail::ops::correlation_distance_op",
        archs = [60],
    ),
    dict(
        path_prefix="cosine",
        OpT="raft::distance::detail::ops::cosine_distance_op",
        archs = [60, 80],
    ),
    dict(
        path_prefix="hamming_unexpanded",
        OpT="raft::distance::detail::ops::hamming_distance_op",
        archs = [60],
    ),
    dict(
        path_prefix="hellinger_expanded",
        OpT="raft::distance::detail::ops::hellinger_distance_op",
        archs = [60],
    ),
    # inner product is handled by cublas.
    dict(
        path_prefix="jensen_shannon",
        OpT="raft::distance::detail::ops::jensen_shannon_distance_op",
        archs = [60],
    ),
    dict(
        path_prefix="kl_divergence",
        OpT="raft::distance::detail::ops::kl_divergence_op",
        archs = [60],
    ),
    dict(
        path_prefix="l1",
        OpT="raft::distance::detail::ops::l1_distance_op",
        archs = [60],
    ),
    dict(
        path_prefix="l2_expanded",
        OpT="raft::distance::detail::ops::l2_exp_distance_op",
        archs = [60, 80],
    ),
    dict(
        path_prefix="l2_unexpanded",
        OpT="raft::distance::detail::ops::l2_unexp_distance_op",
        archs = [60],
    ),
    dict(
        path_prefix="l_inf",
        OpT="raft::distance::detail::ops::l_inf_distance_op",
        archs = [60],
    ),
    dict(
        path_prefix="lp_unexpanded",
        OpT="raft::distance::detail::ops::lp_unexp_distance_op",
        archs = [60],
    ),
    dict(
        path_prefix="russel_rao",
        OpT="raft::distance::detail::ops::russel_rao_distance_op",
        archs = [60],
     ),
]

def arch_headers(op_instance):
    include_headers ="\n".join([
        f"#include <raft/distance/detail/pairwise_matrix/dispatch_sm{arch}.cuh>"
        for arch in op_instance["archs"]
    ])
    return include_headers



for op in op_instances:
    for dt in data_type_instances:
        DataT, AccT, OutT, IdxT = (dt[k] for k in ["DataT", "AccT", "OutT", "IdxT"]);
        path = f"dispatch_{op['path_prefix']}_{DataT}_{AccT}_{OutT}_{IdxT}.cu"
        with open(path, "w") as f:
            f.write(header)
            f.write(arch_headers(op))
            f.write(macro)

            OpT = op['OpT']
            FinOpT = "raft::identity_op"
            f.write(f"\ninstantiate_raft_distance_detail_pairwise_matrix_dispatch({OpT}, {DataT}, {AccT}, {OutT}, {FinOpT}, {IdxT});\n")
            f.write("\n#undef instantiate_raft_distance_detail_pairwise_matrix_dispatch\n")
