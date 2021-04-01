 /*
  * Copyright (c) 2021, NVIDIA CORPORATION.
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

 #include "base_strategy.cuh"
 #include "bloom_filter_strategy.cuh"

 #include <cuco/static_map.cuh>

 namespace raft {
 namespace sparse {
 namespace distance {

 template <typename value_idx, typename value_t, int tpb>
 class hash_strategy : public coo_spmv_strategy<value_idx, value_t> {
  public:
   // namespace cg = cooperative_groups;
   using insert_type =
     typename cuco::static_map<value_idx, value_t,
                               cuda::thread_scope_block>::device_mutable_view;
   using smem_type = typename insert_type::slot_type *;
   using find_type =
     typename cuco::static_map<value_idx, value_t,
                               cuda::thread_scope_block>::device_view;

   hash_strategy(const distances_config_t<value_idx, value_t> &config_, float capacity_threshold_ = 0.5)
     : coo_spmv_strategy<value_idx, value_t>(config_), mask_indptr(1), capacity_threshold(capacity_threshold_) {
     this->smem = raft::getSharedMemPerBlock();
   }

   bool chunking_needed(const value_idx *indptr, const value_idx n_rows) {
     auto widest_row =
       max_degree<value_idx, true>(indptr, n_rows, this->config.allocator,
                                   this->config.stream, capacity_threshold * map_size());

     // figure out if chunking strategy needs to be enabled
     // operating at 50% of hash table size
     if (widest_row.first > capacity_threshold * map_size()) {
       chunking = true;
       more_rows = widest_row.second;
       less_rows = n_rows - more_rows;
       mask_indptr = rmm::device_vector<value_idx>(n_rows);

       fits_in_hash_table<true> fits_functor(indptr, capacity_threshold);
       thrust::copy_if(thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(n_rows),
                       mask_indptr.begin(), fits_functor);
       fits_in_hash_table<false> not_fits_functor(indptr, capacity_threshold);
       thrust::copy_if(thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(n_rows),
                       mask_indptr.begin() + less_rows, not_fits_functor);
       // printv(mask_indptr, "mask_indptr");
     } else {
       chunking = false;
     }
     return chunking;
   }

   template <typename product_f, typename accum_f, typename write_f>
   void dispatch(value_t *out_dists, value_idx *coo_rows_b,
                 product_f product_func, accum_f accum_func, write_f write_func,
                 int chunk_size) {
     auto need = chunking_needed(this->config.a_indptr, this->config.a_nrows);

     auto n_blocks_per_row = raft::ceildiv(this->config.b_nnz, chunk_size * tpb);

     if (need) {
       mask_row_it<value_idx> less(this->config.a_indptr, less_rows,
                                   mask_indptr.data().get());
       // mask_row_it<value_idx> more(this->config.a_indptr, more_rows,
       //   mask_indptr.data().get() + less_rows);
       // bloom_filter_strategy<value_idx, value_t, tpb> bf_strategy(this->config, more);
       chunked_mask_row_it<value_idx> more(
         this->config.a_indptr, more_rows, mask_indptr.data().get() + less_rows,
         capacity_threshold * map_size(), this->config.stream);
       more.init();
       // cudaStreamSynchronize(this->config.stream);

       auto n_less_blocks = less_rows * n_blocks_per_row;
       if (less_rows > 0) {
         int smem = map_size();
         int smem_dim = smem / sizeof(typename insert_type::slot_type);
         this->_dispatch_base(*this, smem, smem_dim, less,
                              out_dists, coo_rows_b,
                             product_func, accum_func, write_func, chunk_size,
                             n_less_blocks, n_blocks_per_row);
         // cudaStreamSynchronize(this->config.stream);
       }
       // bf_strategy.dispatch(out_dists, coo_rows_b, product_func, accum_func, write_func, chunk_size);
       auto n_more_blocks = more.total_row_blocks * n_blocks_per_row;
       int smem = map_size();
       int smem_dim = smem / sizeof(typename insert_type::slot_type);
       this->_dispatch_base(*this, smem, smem_dim, more, out_dists, coo_rows_b,
                            product_func, accum_func, write_func, chunk_size,
                            n_more_blocks, n_blocks_per_row);
       // cudaStreamSynchronize(this->config.stream);
     } else {
       mask_row_it<value_idx> less(this->config.a_indptr, this->config.a_nrows);

       auto n_blocks = this->config.a_nrows * n_blocks_per_row;
       int smem = map_size();
       int smem_dim = smem / sizeof(typename insert_type::slot_type);
       this->_dispatch_base(*this, smem, smem_dim, less, out_dists, coo_rows_b,
                            product_func, accum_func, write_func, chunk_size,
                            n_blocks, n_blocks_per_row);
     }
   }

   template <typename product_f, typename accum_f, typename write_f>
   void dispatch_rev(value_t *out_dists, value_idx *coo_rows_a,
                     product_f product_func, accum_f accum_func,
                     write_f write_func, int chunk_size) {
     throw raft::exception("Cannot perform reverse on hash table strategy");
   }

   __device__ inline insert_type init_insert(smem_type cache,
                                             value_idx &cache_size) {
     return insert_type::make_from_uninitialized_slots(
       cooperative_groups::this_thread_block(), cache, map_size(), -1, 0);
   }

   __device__ inline void insert(insert_type cache, value_idx &key,
                                 value_t &value, int &size) {
     auto success = cache.insert(thrust::make_pair(key, value));
   }

   __device__ inline find_type init_find(smem_type cache) {
     return find_type(cache, map_size(), -1, 0);
   }

   __device__ inline value_t find(find_type cache, value_idx &key, value_idx *indices, value_t *data, value_idx start_offset, value_idx stop_offset, int &size) {
     auto a_pair = cache.find(key);

     value_t a_col = 0.0;
     if (a_pair != cache.end()) {
       a_col = a_pair->second;
     }
     return a_col;
   }

   template <bool fits>
   struct fits_in_hash_table {
     fits_in_hash_table(const value_idx *indptr_, float capacity_threshold_) : indptr(indptr_), capacity_threshold(capacity_threshold_) {}

     __host__ __device__ bool operator()(const value_idx &i) {
       auto degree = indptr[i + 1] - indptr[i];

       if (fits) {
         return degree <= capacity_threshold * hash_strategy::map_size();
       } else {
         return degree > capacity_threshold * hash_strategy::map_size();
       }
     }

    private:
     float capacity_threshold;
     const value_idx *indptr;
   };

  private:
   float capacity_threshold;
   __host__ __device__ constexpr static int map_size() {
     return (48000 - ((tpb / raft::warp_size()) * sizeof(value_t))) /
            sizeof(typename insert_type::slot_type);
     // return 2;
   }

   bool chunking = false;
   value_idx less_rows, more_rows;
   rmm::device_vector<value_idx> mask_indptr;
 };

 }  // namespace distance
 }  // namespace sparse
 }  // namespace raft