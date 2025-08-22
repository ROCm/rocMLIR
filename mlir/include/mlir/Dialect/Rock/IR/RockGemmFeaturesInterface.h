//===---- RockGemmFeaturesInterface.h - ops that wrap rock.gemm -*- C++ -*-===//
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

#ifndef MLIR_DIALECT_ROCK_IR_ROCKGEMMFEATURESINTERFACE_H
#define MLIR_DIALECT_ROCK_IR_ROCKGEMMFEATURESINTERFACE_H

#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Dialect/Rock/IR/RockTypes.h"

#include "mlir/Dialect/Rock/IR/RockGemmFeaturesInterface.h.inc"

#endif // MLIR_DIALECT_ROCK_IR_ROCKGEMMFEATURESINTERFACE_H
