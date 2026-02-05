//===- ParamLookupTable.h - MLIR tuning parameter lookup ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR tuning parameter lookup
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_PARAM_LOOKUP_TABLE_H
#define MLIR_DIALECT_ROCK_PARAM_LOOKUP_TABLE_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace rock {

template <typename ParamsType>
class ParamLookupTable {
public:
  static ArrayRef<StringRef> lookup(StringRef arch, KernelType op,
                                    Type dataType);

  // Finds the lexicographically closest architecture variant when the exact
  // target key is not found in the lookup table.
  //
  // A "relative" entry must have:
  // - Same suffix (operation + data type, e.g., "_gemm_f16")
  // - Same architecture prefix (e.g., "gfx9" for gfx908, gfx90a, gfx942)
  //
  // Example: If target "gfx1151_gemm_f16" is missing but "gfx1101_gemm_f16"
  // and "gfx1201_gemm_f16" exist, this picks the lexicographically closest one
  // (gfx1101_gemm_f16). This enables graceful fallback between similar GPU
  // architectures.
  static StringRef findFallback(StringRef target);

private:
  static constexpr char separator = '_';

  static std::string makeKey(StringRef arch, KernelType op, Type dataType) {
    return (Twine(arch) + Twine(separator) + getKernelTypeString(op) +
            Twine(separator) + getDataTypeString(dataType))
        .str();
  }

  static const std::map<StringRef, ArrayRef<StringRef>> &getTable() {
    static const std::map<StringRef, ArrayRef<StringRef>> table = buildTable();
    return table;
  }

  static std::map<StringRef, ArrayRef<StringRef>> buildTable();

  static StringRef normalizeArch(StringRef arch);

  static std::string getKernelTypeString(KernelType kernelType);

  static std::string getDataTypeString(Type dataType);

  // Get all related entries sorted lexicographically
  static SmallVector<StringRef, 12> getRelatives(StringRef target);
};

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_PARAM_LOOKUP_TABLE_H
