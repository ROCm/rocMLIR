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

/// \brief Builds the artifacts that the flush helpers below would otherwise
/// create lazily on their first call: the hiprtc-compiled instruction-cache
/// invalidation kernel and the cache-sized flush buffer.
///
/// Both are one-time, host-side and expensive (a runtime compile plus a large
/// ``hipMalloc``). Callers that time the flush helpers must build them up
/// front, or that setup cost is attributed to the first timed iteration.
/// \param useLastLevelCacheSize Sizing for the flush buffer; see flushCache.
/// \return success() if the artifacts are ready, failure() otherwise.
mlir::LogicalResult prepareCacheFlushArtifacts(bool useLastLevelCacheSize);

/// \brief Flushes the cache by performing a memory write operation.
/// \param stream The HIP stream to use for the flush operation.
/// \param useLastLevelCacheSize When true, size the flush buffer to the
/// architecture's last-level cache (e.g. AMD Infinity Cache) instead of the
/// per-XCD L2 cache size reported by the HIP runtime.
/// \return success() if the flush succeeds, failure() otherwise.
mlir::LogicalResult flushCache(hipStream_t stream,
                               bool useLastLevelCacheSize = false);

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
