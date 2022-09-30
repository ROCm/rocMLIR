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

#ifndef MLIR_DIALECT_ROCK_TUNING_BLOCKSTRUCTUREPARAMS_H
#define MLIR_DIALECT_ROCK_TUNING_BLOCKSTRUCTUREPARAMS_H

#include "mlir/Support/LogicalResult.h"
#include <cstdint>

namespace mlir {
namespace rock {
/// This structure defines the following fields, which are all a function of
/// the block size selected for a kernel
/// - mThreadsPerCuwave: The length of the m dimension in the grid the 16
/// threads
///   of a CU wave are arranged into
/// - nThreadsPerCuwave: The length of the n dimension of said thread grid
/// - mCuwavesPerBlock: The length of the m dimension of the grid the CU waves
///   composing each block is arranged in to
/// - nCuwavesPerBlock: The n dimension of said grid of CU waves
struct BlockStructureParams {
  int64_t mThreadsPerCuwave;
  int64_t nThreadsPerCuwave;
  int64_t mCuwavesPerBlock;
  int64_t nCuwavesPerBlock;
};

FailureOr<BlockStructureParams> blockStructureParams(uint32_t blockSize);
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_TUNING_BLOCKSTRUCTUREPARAMS_H
