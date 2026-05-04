//===- QuickTuningDb.h - MLIR tuning parameter lookup ---------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Public API of the quick-tuning database: a static table mapping
// (arch, op, dtype) to a set-cover of perfconfigs, with optional per-problem
// overrides.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_QUICK_TUNING_DB_H
#define MLIR_DIALECT_ROCK_QUICK_TUNING_DB_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace mlir {
namespace rock {
namespace QuickTuningDb {

// Returns the perfconfigs for (arch, op, dataType, isAccel). When
// `problemKeyHash` matches a per-problem entry, returns only that
// subset; otherwise returns the full set-cover. Capped at
// ROCMLIR_QUICK_TUNING_LIST_MAX entries (default 30).
SmallVector<StringRef>
lookup(StringRef arch, KernelType op, Type dataType, bool isAccel,
       std::optional<uint64_t> problemKeyHash = std::nullopt);

// Resolves the table key for (arch, op, dataType, isAccel). Returns the
// input key on an exact match, otherwise the closest variant in the
// same gfx family (or in gfx1*+f32 when `isAccel` is false). bf16
// retries with f16. Returns an empty StringRef when no candidate
// exists.
StringRef resolveKey(StringRef arch, KernelType op, Type dataType,
                     bool isAccel);

// Hashes `op`'s problem signature. Fails if `op` is not a recognised
// gemm/conv/attention or its shape metadata is incomplete.
FailureOr<uint64_t> computeProblemKeyHash(Operation *op);

// -- Database invariants -----------------------------------------------------
//
// Always true; exposed for tests.

// True if the entries are sorted lexicographically by key.
bool isSortedByKey();

// True if every entry's per-problem map is sorted ascending by hash.
bool problemMapsAreSortedByHash();

} // namespace QuickTuningDb
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_QUICK_TUNING_DB_H
