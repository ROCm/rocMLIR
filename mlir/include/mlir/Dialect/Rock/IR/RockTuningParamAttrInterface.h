//===- RockTuningParamAttrInterface.h - tuning params for the rock ops ---===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockTuningParamAttrInterface, which abstracts definitions
// and methods of different tuning parameters for rock dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_IR_ROCK_TUNINGPARAM_ATTR_INTERFACE_H
#define MLIR_DIALECT_ROCK_IR_ROCK_TUNINGPARAM_ATTR_INTERFACE_H

#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h.inc"

#endif // MLIR_DIALECT_ROCK_IR_ROCK_TUNINGPARAM_ATTR_INTERFACE_H
