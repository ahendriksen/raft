#!/usr/bin/env python3

header = """
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

#include <cstdint>
#include <raft/neighbors/brute_force-inl.cuh>

#define instantiate_raft_neighbors_brute_force_knn(idx_t, value_t, matrix_idx, index_layout, search_layout, epilogue_op) \\
    template void raft::neighbors::brute_force::knn<idx_t, value_t, matrix_idx, index_layout, search_layout, epilogue_op>( \\
        raft::device_resources const& handle,                           \\
        std::vector<raft::device_matrix_view<const value_t, matrix_idx, index_layout>> index, \\
        raft::device_matrix_view<const value_t, matrix_idx, search_layout> search, \\
        raft::device_matrix_view<idx_t, matrix_idx, row_major> indices, \\
        raft::device_matrix_view<value_t, matrix_idx, row_major> distances, \\
        raft::distance::DistanceType metric,                            \\
        std::optional<float> metric_arg,                                \\
        std::optional<idx_t> global_id_offset,                          \\
        epilogue_op distance_epilogue);


#define instantiate_raft_neighbors_brute_force_fused_l2_knn(value_t, idx_t, idx_layout, query_layout) \\
    template void raft::neighbors::brute_force::fused_l2_knn(    \\
        raft::device_resources const& handle,                           \\
        raft::device_matrix_view<const value_t, idx_t, idx_layout> index, \\
        raft::device_matrix_view<const value_t, idx_t, query_layout> query, \\
        raft::device_matrix_view<idx_t, idx_t, row_major> out_inds,     \\
        raft::device_matrix_view<value_t, idx_t, row_major> out_dists,  \\
        raft::distance::DistanceType metric);

"""

trailer = """

#undef instantiate_raft_neighbors_brute_force_knn
#undef instantiate_raft_neighbors_brute_force_fused_l2_knn
"""

knn_types = dict(
    int64_t_float_uint32_t=("int64_t","float","uint32_t"),
    int64_t_float_int64_t=("int64_t","float","int64_t"),
    int_float_int=("int","float","int"),
    uint32_t_float_uint32_t=("uint32_t","float","uint32_t"),
)

fused_l2_knn_types = dict(
    float_int64_t=("float", "int64_t"),
)

# knn
for type_path, (idx_t, value_t, matrix_idx) in knn_types.items():
    path = f"brute_force_knn_{type_path}.cu"
    with open(path, "w") as f:
        f.write(header)
        f.write(f"instantiate_raft_neighbors_brute_force_knn({idx_t},{value_t},{matrix_idx},raft::row_major,raft::row_major,raft::identity_op);\n")
        f.write(trailer)
    # For pasting into CMakeLists.txt
    print(f"src/neighbors/{path}")

#fused_l2_knn
for type_path, (value_t, idx_t) in fused_l2_knn_types.items():
    path = f"brute_force_fused_l2_knn_{type_path}.cu"
    with open(path, "w") as f:
        f.write(header)
        f.write(f"instantiate_raft_neighbors_brute_force_fused_l2_knn({value_t},{idx_t},raft::row_major,raft::row_major);\n")
        f.write(trailer)
    # For pasting into CMakeLists.txt
    print(f"src/neighbors/{path}")
