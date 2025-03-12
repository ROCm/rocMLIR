//===- RockGemmGemmWrapperInterface.cpp - ops that wrap rock.attention
//-------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockGemmGemmWrapperInterface, which abstracts attention and
// gemm+gemm to allow code to operate on them generically.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"

namespace mlir {
namespace rock {
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.cpp.inc"
} // namespace rock
} // namespace mlir
