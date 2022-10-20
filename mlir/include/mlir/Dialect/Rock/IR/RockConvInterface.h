//===- RockConvInterface.td - common ConvOps access methods -*- C++ -*-
//---------=== //
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

#ifndef MLIR_DIALECT_ROCK_IR_ROCKCONVINTERFACE_H
#define MLIR_DIALECT_ROCK_IR_ROCKCONVINTERFACE_H

#include "mlir/IR/OpDefinition.h"

#include "mlir/Dialect/Rock/IR/RockTypes.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "mlir/Dialect/Rock/IR/RockConvInterface.h.inc"

#endif // MLIR_DIALECT_ROCK_IR_ROCKCONVINTERFACE_H