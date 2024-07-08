//===- GPUToMIGraphX.cpp - Lowering GPU to MIGraphX Dialect ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the GPU to the MIGraphX dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToMIGraphX/GPUToMIGraphX.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

class FuncToCOBJPattern : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto results = op->getResults();
    auto resultType = cast<MemRefType>(results[0].getType());
    auto fnAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee");
    SmallVector<Value, 8> mrOperands(op.getOperands());
    SmallVector<Value, 8> cobjArgs;

    SmallVector<IntegerAttr, 5> globalSizeAttr;
    SmallVector<IntegerAttr, 5> localSizeAttr;
    SymbolRefAttr kernelRefAttr;
    auto fusedFuncOp =
        op->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(
            fnAttr.getValue());

    // Insert alloc for result buffer
    rewriter.setInsertionPoint(op);
    auto resultAlloc = rewriter.create<memref::AllocOp>(loc, resultType);
    mrOperands.push_back(resultAlloc);

    fusedFuncOp.walk([&](gpu::LaunchFuncOp Lop) {
      // x, y, z
      auto gridSize = Lop.getGridSizeOperandValues();
      auto blockSize = Lop.getBlockSizeOperandValues();

      // FIXME: Better way to import attribute as I64?
      globalSizeAttr.push_back(rewriter.getI64IntegerAttr(
          (((gridSize.z.getDefiningOp())->getAttrOfType<IntegerAttr>("value")))
              .getInt()));
      globalSizeAttr.push_back(rewriter.getI64IntegerAttr(
          (((gridSize.y.getDefiningOp())->getAttrOfType<IntegerAttr>("value")))
              .getInt()));
      globalSizeAttr.push_back(rewriter.getI64IntegerAttr(
          (((gridSize.x.getDefiningOp())->getAttrOfType<IntegerAttr>("value")))
              .getInt()));
      localSizeAttr.push_back(rewriter.getI64IntegerAttr(
          (((blockSize.z.getDefiningOp())->getAttrOfType<IntegerAttr>("value")))
              .getInt()));
      localSizeAttr.push_back(rewriter.getI64IntegerAttr(
          (((blockSize.y.getDefiningOp())->getAttrOfType<IntegerAttr>("value")))
              .getInt()));
      localSizeAttr.push_back(rewriter.getI64IntegerAttr(
          (((blockSize.x.getDefiningOp())->getAttrOfType<IntegerAttr>("value")))
              .getInt()));

      kernelRefAttr = Lop->getAttrOfType<SymbolRefAttr>("kernel");

      // Lowering memref structure
      for (auto arg : mrOperands) {
        // Sending the reference to the memref itself because we're sending this
        // to an excution engine which will handle the allocation. allocation
        // ptr
        cobjArgs.push_back(arg);
        // aligned ptr
        cobjArgs.push_back(arg);
        ValueRange noArgs({});

        // offset
        auto offsetOp = rewriter.create<mlir::migraphx::LiteralOp>(
            loc, migraphx::MIXRShapedType::get({1}, {0}, rewriter.getI64Type()),
            noArgs);
        uint64_t zero = 0;
        offsetOp->setAttr(
            "value",
            DenseIntElementsAttr::get(
                RankedTensorType::get({1}, rewriter.getI64Type()), zero));
        cobjArgs.push_back(offsetOp);
        // shape
        auto argType = cast<MemRefType>(arg.getType());
        auto argShape = argType.getShape();
        for (auto it = argShape.rbegin(); it != argShape.rend(); ++it) {
          auto constOp = rewriter.create<mlir::migraphx::LiteralOp>(
              loc,
              migraphx::MIXRShapedType::get({1}, {1}, rewriter.getI64Type()),
              noArgs);
          constOp->setAttr(
              "value",
              DenseIntElementsAttr::get(
                  RankedTensorType::get({1}, rewriter.getI64Type()), *it));
          cobjArgs.push_back(constOp);
        }
        // stride
        uint64_t stride = 1;
        for (auto it = argShape.rbegin(); it != argShape.rend(); ++it) {
          auto constOp = rewriter.create<mlir::migraphx::LiteralOp>(
              loc,
              migraphx::MIXRShapedType::get({1}, {0}, rewriter.getI64Type()),
              noArgs);
          constOp->setAttr(
              "value",
              DenseIntElementsAttr::get(
                  RankedTensorType::get({1}, rewriter.getI64Type()), stride));
          cobjArgs.push_back(constOp);
          stride *= (*it);
        }
      }

      // insert the result buffer at the end again to specify the output buffer
      cobjArgs.push_back(mrOperands.back());
    });

    auto cop =
        rewriter.create<mlir::migraphx::CodeObjOp>(loc, resultType, cobjArgs);
    cop->setAttr("kernel", kernelRefAttr);
    cop->setAttr("globalSize",
                 rewriter.getArrayAttr(ArrayRef<Attribute>(
                     globalSizeAttr.begin(), globalSizeAttr.end())));
    cop->setAttr("localSize", rewriter.getArrayAttr(ArrayRef<Attribute>(
                                  localSizeAttr.begin(), localSizeAttr.end())));

    rewriter.replaceOp(op, cop->getResults());
    return success();
  }
};

} // namespace

void mlir::migraphx::populateFuncToCOBJPatterns(MLIRContext *context,
                                                RewritePatternSet &patterns) {
  patterns.add<FuncToCOBJPattern>(context);
}
