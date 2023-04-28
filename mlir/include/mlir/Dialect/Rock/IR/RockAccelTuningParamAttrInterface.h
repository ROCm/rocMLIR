//===- RockTuningParamAttrInterface.h - tuning params for accelerated rock ops
//---===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockAccelTuningParamAttrInterface, which abstracts
// definitions and methods of different tuning parameters for accelerated rock
// dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_IR_ROCK_ACCEL_TUNINGPARAM_ATTR_INTERFACE_H
#define MLIR_DIALECT_ROCK_IR_ROCK_ACCEL_TUNINGPARAM_ATTR_INTERFACE_H

#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Dialect/Rock/IR/RockAccelTuningParamAttrInterface.h.inc"

#endif // MLIR_DIALECT_ROCK_IR_ROCK_ACCEL_TUNINGPARAM_ATTR_INTERFACE_H
