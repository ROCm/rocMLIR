//===- XdlopsCodeSelection.h - MLIR to C++ for Rock conversion
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements code selection logic for XDLOPS instructions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_XDLOPS_CODE_SELECTION_H
#define MLIR_XDLOPS_CODE_SELECTION_H

#include "mlir/Dialect/AMDGPU/AMDGPUDialect.h"

namespace mlir {
struct MFMAParams {
  // log_2 of the number of blocks to chop the lanes of A in to for broadcast
  uint32_t cbsz;
  // Which of the said groups should be broadcast
  uint32_t abid;
  amdgpu::MFMAPermB blgp;
};

struct XdlopsCodeSelection {
  int64_t mfmaNonKDim;
  // k for MFMA, and, by extension, for the xdlops as a whole
  int64_t k;
  int64_t blocksMfma;

  int64_t MRepeats;
  int64_t NRepeats;
  int64_t nResultVectors;

  VectorType vectorType;
  SmallVector<MFMAParams, 2> imms;
  Type argType;

  int64_t k_base;

  int64_t rowGroupSize;
  int64_t rowsPerMfmaOutput;
  int64_t blocksPerMfmaOutput;
  int64_t rowGroupsPerBlock;
  int64_t blocksInOutRegs;

  int64_t inputSpanLen;
  int64_t inputSpansPerMfmaIn;

  static void dumpUnsupported(const mlir::Type &dataType, int64_t MPerWave,
                              int64_t NPerWave);
  static XdlopsCodeSelection get(mlir::Type dataType, int64_t MPerWave,
                                 int64_t NPerWave);
  bool isValid(int64_t kpack, int64_t kPerBlock);
};
} // namespace mlir

#endif // MLIR_XDLOPS_CODE_SELECTION_H
