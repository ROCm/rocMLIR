//===- RockGemmGemmWrapperInterface.h - ops that wrap rock.attention -*- C++
//-*-===//
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

#ifndef MLIR_DIALECT_ROCK_IR_ROCKGEMMGEMMWRAPPERINTERFACE_H
#define MLIR_DIALECT_ROCK_IR_ROCKGEMMGEMMWRAPPERINTERFACE_H

#include "mlir/Dialect/Rock/IR/GemmGemmSize.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Dialect/Rock/IR/RockTypes.h"

#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h.inc"

#endif // MLIR_DIALECT_ROCK_IR_ROCKGEMMGEMMWRAPPERINTERFACE_H
