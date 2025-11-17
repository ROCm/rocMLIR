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

mlir::LogicalResult flushL2Cache(hipStream_t stream);
mlir::LogicalResult flushInstructionCache(hipStream_t stream);
mlir::LogicalResult cleanupCacheFlushArtifacts();

} // namespace rocmlir::tuningdriver

#endif // ROCMLIR_TUNING_DRIVER_CACHE_FLUSH_H
