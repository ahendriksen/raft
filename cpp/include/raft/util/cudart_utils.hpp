/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

/**
 * This file is deprecated and will be removed in release 22.06.
 * Please use raft_runtime/cudart_utils.hpp instead.
 */

#ifndef __RAFT_RT_CUDART_UTILS_H
#define __RAFT_RT_CUDART_UTILS_H

#pragma once

#include <cassert>
#include <raft/core/error.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace raft {

/**
 * @brief Exception thrown when a CUDA error is encountered.
 */
struct cuda_error : public raft::exception {
  explicit cuda_error(char const* const message) : raft::exception(message) {}
  explicit cuda_error(std::string const& message) : raft::exception(message) {}
};

}  // namespace raft

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 *
 */
#define RAFT_CUDA_TRY(call)                        \
  do {                                             \
    cudaError_t const status = call;               \
    if (status != cudaSuccess) {                   \
      cudaGetLastError();                          \
      std::string msg{};                           \
      SET_ERROR_MSG(msg,                           \
                    "CUDA error encountered at: ", \
                    "call='%s', Reason=%s:%s",     \
                    #call,                         \
                    cudaGetErrorName(status),      \
                    cudaGetErrorString(status));   \
      throw raft::cuda_error(msg);                 \
    }                                              \
  } while (0)

// FIXME: Remove after consumers rename
#ifndef CUDA_TRY
#define CUDA_TRY(call) RAFT_CUDA_TRY(call)
#endif


namespace raft {

/** Helper method to get to know warp size in device code */


/** helper method to get multi-processor count parameter */
inline int getMultiProcessorCount()
{
  int devId;
  RAFT_CUDA_TRY(cudaGetDevice(&devId));
  int mpCount;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&mpCount, cudaDevAttrMultiProcessorCount, devId));
  return mpCount;
}

}  // namespace raft

#endif
