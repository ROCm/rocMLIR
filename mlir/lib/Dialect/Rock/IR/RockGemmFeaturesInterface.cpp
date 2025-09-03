//===------------------- RockGemmFeaturesInterface.cpp --------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockGemmFeaturesInterface, which abstracts Rock ops for
// which we would expect to extract GemmFeatures for.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"

namespace mlir {
namespace rock {
#include "mlir/Dialect/Rock/IR/RockGemmFeaturesInterface.cpp.inc"
} // namespace rock
} // namespace mlir
