//===- RockTuningParamAttrInterface.cpp - tuning params for the rock ops ---===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockTuningParamAttrInterface, which abstracts definitions
// and methods of different tuning parameters for rock dialect ops.
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"

namespace mlir {
namespace rock {
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.cpp.inc"


} // namespace rock
} // namespace mlir
