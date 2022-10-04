//===- GeneralGemmBlockStructure.h - Gemm block structure non-xdlops --*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines the GeneralGemmBlockStructure structure, which describes
// how the workitems of a workgroup are grouped into smaller parts based on.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_TUNING_GENERALGEMMBLOCKSTRUCTURE_H
#define MLIR_DIALECT_ROCK_TUNING_GENERALGEMMBLOCKSTRUCTURE_H

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
struct GeneralGemmBlockStructure {
  int64_t mThreadsPerCuwave;
  int64_t nThreadsPerCuwave;
  int64_t mCuwavesPerBlock;
  int64_t nCuwavesPerBlock;
};

/// Gen the GeneralGemmBlockStructure for a given blockSize and return failure()
/// if one cannot be found.
FailureOr<GeneralGemmBlockStructure>
deriveGeneralGemmBlockStructure(uint32_t blockSize);
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_TUNING_GENERALGEMMBLOCKSTRUCTURE_H
