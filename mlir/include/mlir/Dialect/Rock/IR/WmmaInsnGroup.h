//===- WmmaInsnGroup.h - MLIR to C++ for Rock conversion
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file implements code selection logic for Wmma instructions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_WMMA_INSN_GROUP_H
#define MLIR_WMMA_INSN_GROUP_H

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace rock {

struct WmmaInsn {
  StringRef insn;
  int64_t inputLen;
  int64_t outputLen;
  int64_t outputStride;
  int64_t mRepeats;
  int64_t nRepeats;
  VectorType argTypeA;
  VectorType argTypeB;
  VectorType retType;

public:
  bool isCoherentWithK(int64_t kpack, int64_t kPerBlock);
  static FailureOr<WmmaInsn> select(Type elementTypeA, Type elementTypeB,
                                    int64_t waveSize, int64_t mPerWave,
                                    int64_t nPerWave);
};
} // namespace rock
} // namespace mlir

#endif // MLIR_WMMA_INSN_GROUP_H
