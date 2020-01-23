//===- MIOpenOps.cpp - MIOpen MLIR Operations -----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpenOps/MIOpenOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/SideEffectsInterface.h"

using namespace mlir;
using namespace mlir::miopen;

//===----------------------------------------------------------------------===//
// MIOpenOpsDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {

} // namespace

//===----------------------------------------------------------------------===//
// MIOpenOpsDialect
//===----------------------------------------------------------------------===//

MIOpenOpsDialect::MIOpenOpsDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
//  addOperations<
//#define GET_OP_LIST
//#include "mlir/Dialect/MIOpenOps/MIOpenOps.cpp.inc"
//      >();

  //addInterfaces<LoopSideEffectsInterface>();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

//#define GET_OP_CLASSES
//#include "mlir/Dialect/MIOpenOps/MIOpenOps.cpp.inc"
