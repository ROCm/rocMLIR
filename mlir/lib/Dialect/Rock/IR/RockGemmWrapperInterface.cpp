//===- RockGemmWrapperInterface.cpp - ops that wrap rock.gemm -------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockGemmWrapperInterface, which abstracts convolutions and
// matrix multiplies to allow code to operate on them generically.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"

namespace mlir {
namespace rock {
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.cpp.inc"
} // namespace rock
} // namespace mlir
