//=== Utils.h - functions that often come up during lowering
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MHAL_UTILITY_UTILS_H
#define MHAL_UTILITY_UTILS_H

#include "mlir/Dialect/MHAL/IR/MHAL.h"

namespace mlir {

namespace LLVM {
class LLVMFuncOp;
}

namespace mhal {

class PrefillAttr;

// Return `mhal::PrefillAttr` attributes for a given function
SmallVector<PrefillAttr> getStoredPrefillAttributes(LLVM::LLVMFuncOp func);

} // namespace mhal
} // end namespace mlir
#endif
