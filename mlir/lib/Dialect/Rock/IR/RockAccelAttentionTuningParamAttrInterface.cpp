//=- RockAccelAttentionTuningParamAttrInterface.cpp - tuning params for the rock
//ops -=//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockAccelAttentionTuningParamAttrInterface, which provides
// a common interface for tuning parameters for different attention
// accelerators.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/RockAccelAttentionTuningParamAttrInterface.h"

namespace mlir {
namespace rock {
#include "mlir/Dialect/Rock/IR/RockAccelAttentionTuningParamAttrInterface.cpp.inc"

} // namespace rock
} // namespace mlir
