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

// Type IDs for WMMA instruction selection
// Naming convention: InputType(s)_To_OutputType_TyId
enum class WmmaTypeId : uint32_t {
  // F32 output
  F32_To_F32_TyId = 0, // F32 x F32 -> F32
  F16_To_F32_TyId,     // F16 x F16 -> F32
  Bf16_To_F32_TyId,    // BF16 x BF16 -> F32
  Fp8Fp8_To_F32_TyId,  // FP8 x FP8 -> F32
  Fp8Bf8_To_F32_TyId,  // FP8 x BF8 -> F32
  Bf8Fp8_To_F32_TyId,  // BF8 x FP8 -> F32
  Bf8Bf8_To_F32_TyId,  // BF8 x BF8 -> F32

  // F16 output
  F16_To_F16_TyId,    // F16 x F16 -> F16
  Fp8Fp8_To_F16_TyId, // FP8 x FP8 -> F16
  Fp8Bf8_To_F16_TyId, // FP8 x BF8 -> F16
  Bf8Fp8_To_F16_TyId, // BF8 x FP8 -> F16
  Bf8Bf8_To_F16_TyId, // BF8 x BF8 -> F16

  // BF16 output
  Bf16_To_Bf16_TyId,    // BF16 x BF16 -> BF16
  Bf16_To_Bf16F32_TyId, // BF16 x BF16 -> mixed BF16/F32

  // Integer output
  I8_To_I32_TyId, // I8 x I8 -> I32
  I4_To_I32_TyId, // I4 x I4 -> I32

  // Small float scaled WMMA: FP8/BF8/FP6/FP4 x FP8/BF8/FP6/FP4 -> F32
  // "Small floats" are sub-byte or 8-bit floating point formats:
  //   - FP8 (E4M3), BF8 (E5M2): 8-bit floats
  //   - FP6 (E2M3, E3M2): 6-bit floats
  //   - FP4 (E2M1): 4-bit float
  // All combinations map to wmma_scale_f32_*_f8f6f4
  SmallFloat_To_F32_TyId
};

// Information needed to create a WMMA instruction
struct WmmaInsnInfo {
  StringRef insn;
  int64_t inputVectorLen;
  int64_t outputVectorLen;
  int64_t mPerAccel;
  int64_t nPerAccel;
  bool isScaled;
};

// Key type for WMMA instruction maps: (TypeId, K dimension)
using WmmaInsnKey = std::pair<WmmaTypeId, int64_t>;

struct WmmaInsn {
  StringRef insn;
  int64_t mPerAccel;
  int64_t nPerAccel;
  int64_t kDim;
  int64_t outputStride;
  int64_t mRepeats;
  int64_t nRepeats;
  VectorType argTypeA;
  VectorType argTypeB;
  VectorType retType;
  bool isScaled;

public:
  bool isCoherentWithK(int64_t kpack, int64_t kPerBlock);
  /// Select a WMMA instruction based on element types and architecture.
  static FailureOr<WmmaInsn> select(Type elementTypeA, Type elementTypeB,
                                    int64_t waveSize, StringRef arch,
                                    int64_t mPerWave, int64_t nPerWave,
                                    int64_t kPack, int64_t kPackPerBlock,
                                    bool forScaledOp);
};
} // namespace rock
} // namespace mlir

#endif // MLIR_WMMA_INSN_GROUP_H
