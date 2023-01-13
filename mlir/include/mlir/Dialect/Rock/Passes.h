//===- Passes.h - Linalg pass entry points ----------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_PASSES_H_
#define MLIR_DIALECT_ROCK_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace rock {

#define GEN_PASS_DECL_ROCKAFFIXTUNINGPARAMETERSPASS
#define GEN_PASS_DECL_ROCKAFFIXTUNINGPARAMETERSPASS
#define GEN_PASS_DECL_ROCKBLOCKWISEGEMMTOTHREADWISEPASS
#define GEN_PASS_DECL_ROCKBUFFERLOADMERGEPASS
#define GEN_PASS_DECL_ROCKCLEANMATHPASS
#define GEN_PASS_DECL_ROCKCONVTOGEMMPASS
#define GEN_PASS_DECL_ROCKCOPYOPTPASS
#define GEN_PASS_DECL_ROCKREGULARIZEKERNELPASS
#define GEN_PASS_DECL_ROCKGEMMTOGRIDWISEPASS
#define GEN_PASS_DECL_ROCKGRIDWISEGEMMTOBLOCKWISEPASS
#define GEN_PASS_DECL_ROCKLINALGALIGNPASS
#define GEN_PASS_DECL_ROCKLOOPSTOCFPASS
#define GEN_PASS_DECL_ROCKSUGARTOLOOPSPASS
#define GEN_PASS_DECL_ROCKTHREADWISEGEMMLOWERINGPASS
#define GEN_PASS_DECL_ROCKLOWERREDUCEPASS

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Rock/Passes.h.inc"

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_PASSES_H_
