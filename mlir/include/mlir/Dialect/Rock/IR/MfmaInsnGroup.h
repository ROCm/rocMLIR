//===- MfmaInsnGroup.h - MLIR to C++ for Rock conversion
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file implements code selection logic for Mfma instructions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MFMA_INSN_GROUP_H
#define MLIR_MFMA_INSN_GROUP_H

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace rock {

enum class MfmaTypeId : uint32_t {
  Fp32TyId = 0,
  Fp16TyId,
  Bf16TyId,
  I8TyId,
  Fp8Fp8TyId,
  Fp8Bf8TyId,
  Bf8Fp8TyId,
  Bf8Bf8TyId
};

struct MfmaInsnInfo {
  MfmaTypeId type;
  int64_t mfmaNonKDim;
  int64_t k;
  int64_t blocksMfma;
};

struct MfmaInsnAttr {
  int64_t mfmaNonKDim;
  int64_t k;
  int64_t blocksMfma;

  int64_t nInputsToMfma;
  int64_t k_base;
  int64_t inputSpanLen;
  int64_t inputSpansPerMfmaIn;

  int64_t nOutputsOfMfma;
  int64_t rowGroupSize;
  int64_t rowsPerMfmaOutput;
  int64_t blocksPerMfmaOutput;
  int64_t rowGroupsPerBlock;
  int64_t blocksInOutRegs;
};

class MfmaInsn {
private:
  MfmaInsnAttr attr;

public:
  static FailureOr<MfmaInsn> select(StringRef mfmaInsn);
  MfmaInsn(const MfmaInsnAttr &mfmaInsnAttr);

  MfmaInsnAttr getAttr();
  Type getArgTypeFor(Type elementTypeA);
  VectorType getRetType(Type elementType);
  bool isCoherentWithK(int64_t kPack, int64_t kPerBlock);
};

template <typename T>
constexpr typename std::underlying_type<T>::type cast_as_underlying(T t) {
  return static_cast<typename std::underlying_type<T>::type>(t);
}

struct MfmaInsnGroupSelectKey {
  MfmaTypeId type;
  int64_t mPerWave;
  int64_t nPerWave;
};

struct MfmaInsnGroupSelectKeyInfo
    : public llvm::DenseMapInfo<MfmaInsnGroupSelectKey> {
  static inline MfmaInsnGroupSelectKey getEmptyKey() {
    return {MfmaTypeId::Fp32TyId, 0, 0};
  }

  static inline MfmaInsnGroupSelectKey getTombstoneKey() {
    return {MfmaTypeId::Fp32TyId, -1, -1};
  }

  static inline bool isEqual(const MfmaInsnGroupSelectKey &lhs,
                             const MfmaInsnGroupSelectKey &rhs) {
    return lhs.type == rhs.type && lhs.mPerWave == rhs.mPerWave &&
           lhs.nPerWave == rhs.nPerWave;
  }

  static unsigned getHashValue(const MfmaInsnGroupSelectKey &key) {
    return llvm::detail::combineHashValue(
        cast_as_underlying(key.type),
        llvm::detail::combineHashValue(key.mPerWave, key.nPerWave));
  }
};

struct MFMAParams {
  // log_2 of the number of blocks to chop the lanes of A in to for broadcast
  uint32_t cbsz;
  // Which of the said groups should be broadcast
  uint32_t abid;
  amdgpu::MFMAPermB blgp;
};

struct MfmaInsnGroupAttr {
  SmallString<16> insn;
  SmallVector<MFMAParams, 2> imms;
  // Reduction constructor
  MfmaInsnGroupAttr(const SmallString<16> &insn)
      : insn{insn}, imms{{{0, 0, amdgpu::MFMAPermB::none}}} {}
  // Broadcast constructor
  MfmaInsnGroupAttr(const SmallString<16> &insn,
                    const SmallVector<MFMAParams, 2> &imms)
      : insn{insn}, imms{imms} {}
};

class MfmaInsnGroup {
private:
  Type elementTypeA;
  Type elementTypeB;
  MfmaInsn insn;
  MfmaInsnGroupAttr groupAttr;

public:
  static FailureOr<MfmaInsnGroup> select(Type elementTypeA, Type elementTypeB,
                                         StringRef arch, int64_t mPerWave,
                                         int64_t nPerWave);
  MfmaInsnGroup(Type elementTypeA, Type elementTypeB, const MfmaInsn &insn,
                const MfmaInsnGroupAttr &groupAttr);
  int64_t getMRepeats(int64_t mPerWave);
  int64_t getNRepeats(int64_t nPerWave);
  static int64_t getLenPerMfmaGroup(int64_t lenPerWave);
  SmallVector<mlir::rock::MFMAParams, 2> getImms();

  MfmaInsnAttr getInsnAttr();
  Type getArgTypeA();
  Type getArgTypeB();
  VectorType getRetType();
  bool isCoherentWithK(int64_t kPack, int64_t kPerBlock);
};

} // namespace rock
} // namespace mlir

#endif // MLIR_MFMA_INSN_GROUP_H
