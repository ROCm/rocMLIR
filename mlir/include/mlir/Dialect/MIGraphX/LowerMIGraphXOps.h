//===- LowerMIGraphXOps.h - MIGraphX conversion ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the lowering pass for the MIGraphX.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIGRAPHX_LOWERMIGRAPHXOPS_H
#define MLIR_DIALECT_MIGRAPHX_LOWERMIGRAPHXOPS_H

#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/MIGraphX/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::migraphx;



#endif // MLIR_DIALECT_MIGRAPHX_LOWERMIGRAPHXOPS_H
