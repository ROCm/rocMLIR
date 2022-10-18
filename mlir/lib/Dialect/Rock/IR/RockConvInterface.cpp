//===- RockGemmWrapperInterface.cpp - ops that wrap rock.gemm -------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockConvInterface, which groups common access methods
// conv-like operations
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/RockConvInterface.h"

namespace mlir {
namespace rock {
#include "mlir/Dialect/Rock/IR/RockConvInterface.cpp.inc"
} // namespace rock
} // namespace mlir
