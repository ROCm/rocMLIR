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

#include <set>

#include "mlir/Dialect/AMDGPU/AMDGPUDialect.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace rock {

enum class MfmaTypeId { Fp32TyId = 0, Fp16TyId, Bf16TyId, I8TyId };

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
  static const llvm::StringMap<MfmaInsnInfo> mfmaInsnInfoMap;
  static const llvm::StringMap<MfmaInsnAttr> mfmaInsnAttrMap;
  static MfmaInsnAttr deriveAttr(MfmaInsnInfo info);
  MfmaInsnAttr attr;

public:
  static FailureOr<MfmaInsn> select(StringRef mfmaInsn);
  MfmaInsn(const MfmaInsnAttr &mfmaInsnAttr);

  MfmaInsnAttr getAttr();
  Type getArgType(Type elementType);
  VectorType getRetType(Type elementType);
  bool isCohereantWithK(int64_t kPack, int64_t kPerBlock);
};

struct MfmaInsnGroupSelectKey {
  MfmaTypeId type;
  int64_t MPerWave;
  int64_t NPerWave;

  inline bool operator<(const MfmaInsnGroupSelectKey rhs) const {
    return static_cast<int>(type) < static_cast<int>(rhs.type) ||
           (type == rhs.type && MPerWave < rhs.MPerWave) ||
           (type == rhs.type && MPerWave == rhs.MPerWave &&
            NPerWave < rhs.NPerWave);
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
  int64_t MRepeats;
  int64_t NRepeats;
  SmallVector<MFMAParams, 2> imms;
};

class MfmaInsnGroup {
private:
  static const std::map<MfmaInsnGroupSelectKey, MfmaInsnGroupAttr>
      mfmaInsnGroupAttrMap;
  Type elementType;
  MfmaInsn insn;
  MfmaInsnGroupAttr groupAttr;

public:
  static FailureOr<MfmaInsnGroup> select(Type elementType, int64_t mPerWave,
                                         int64_t nPerWave);
  MfmaInsnGroup(Type elementType, const MfmaInsn &insn,
                const MfmaInsnGroupAttr &groupAttr);
  int64_t getMRepeats();
  int64_t getNRepeats();
  SmallVector<mlir::rock::MFMAParams, 2> getImms();

  MfmaInsnAttr getInsnAttr();
  Type getArgType();
  VectorType getRetType();
  bool isCohereantWithK(int64_t kPack, int64_t kPerBlock);
};

} // namespace rock
} // namespace mlir

#endif // MLIR_MFMA_INSN_GROUP_H
