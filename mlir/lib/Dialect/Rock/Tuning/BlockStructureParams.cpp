//===- BlockStructureParams.h - Block structure derived parameters ----*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines the BlockStructureParams structure, which describes how
// the workitems of a workgroup are grouped into smaller parts based on.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/BlockStructureParams.h"

using namespace mlir;
using namespace mlir::rock;

FailureOr<BlockStructureParams>
mlir::rock::blockStructureParams(uint32_t blockSize) {
  BlockStructureParams ret;

  if (blockSize == 64) {
    ret.mThreadsPerCuwave = 4;
    ret.nThreadsPerCuwave = 4;
    ret.mCuwavesPerBlock = 2;
    ret.nCuwavesPerBlock = 2;
  } else if (blockSize == 128) {
    ret.mThreadsPerCuwave = 4;
    ret.nThreadsPerCuwave = 4;
    ret.mCuwavesPerBlock = 4;
    ret.nCuwavesPerBlock = 2;
  } else if (blockSize == 256) {
    ret.mThreadsPerCuwave = 4;
    ret.nThreadsPerCuwave = 4;
    ret.mCuwavesPerBlock = 4;
    ret.nCuwavesPerBlock = 4;
  } else {
    return failure();
  }
  return ret;
}
