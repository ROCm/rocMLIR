//===- FmaInsnGroup.h - MLIR to C++ for Rock conversion
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file implements code selection logic for Fma instructions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_FMA_INSN_GROUP_H
#define MLIR_FMA_INSN_GROUP_H

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace rock {

struct FmaInsn {
  Type argTypeA;
  Type argTypeB;
  Type retType;

  public:
    static FailureOr<FmaInsn> select(Type elementTypeA, Type elementTypeB, StringRef arch);
};



} // namespace rock
} // namespace mlir

#endif // MLIR_FMA_INSN_GROUP_H