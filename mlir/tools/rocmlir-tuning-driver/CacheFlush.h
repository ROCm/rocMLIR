//===- CacheFlush.h - Cache flush helpers -----------------------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCMLIR_TUNING_DRIVER_CACHE_FLUSH_H
#define ROCMLIR_TUNING_DRIVER_CACHE_FLUSH_H

#include "mlir/Support/LogicalResult.h"

#include <hip/hip_runtime.h>

namespace rocmlir::tuningdriver {

/// \brief Flushes the L2 cache by performing a memory write operation.
/// \param stream The HIP stream to use for the flush operation.
/// \return success() if the flush succeeds, failure() otherwise.
mlir::LogicalResult flushL2Cache(hipStream_t stream);

/// \brief Flushes the instruction cache to ensure that any modified code is
/// visible to the device.
/// \param stream The HIP stream to use for the flush operation.
/// \return success() if the flush succeeds, failure() otherwise.
mlir::LogicalResult flushInstructionCache(hipStream_t stream);

/// \brief Cleans up any artifacts created during cache flush operations.
/// \return success() if cleanup succeeds, failure() otherwise.
mlir::LogicalResult cleanupCacheFlushArtifacts();

} // namespace rocmlir::tuningdriver

#endif // ROCMLIR_TUNING_DRIVER_CACHE_FLUSH_H
