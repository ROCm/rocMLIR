//====----- Bufferize.cpp - Bufferization of shape ops  ---------*- C++-*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MHAL/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MHAL/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_MHALBUFFERIZE
#include "mlir/Dialect/MHAL/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace bufferization;

namespace {
struct MHALBufferizePass
    : public impl::MHALBufferizeBase<MHALBufferizePass> {
  void runOnOperation() override {
    BufferizationOptions options = getPartialBufferizationOptions();
    options.opFilter.allowDialect<mhal::MHALDialect>();

    if (failed(bufferizeOp(getOperation(), options)))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    mhal::MHALDialect>();
    mhal::registerBufferizableOpInterfaceExternalModels(registry);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createMHALBufferizePass() {
  return std::make_unique<MHALBufferizePass>();
}
