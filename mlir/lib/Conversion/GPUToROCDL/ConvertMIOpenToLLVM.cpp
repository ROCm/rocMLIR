//===- VectorToLLVM.cpp - Conversion from Vector to the LLVM dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToROCDL/ConvertMIOpenToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::miopen;

class MIOpenDataConvertOpConversion : public ConvertToLLVMPattern {
public:
  explicit MIOpenDataConvertOpConversion(MLIRContext *context,
                                     LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(miopen::DataConvertOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto adaptor = miopen::DataConvertOpOperandAdaptor(operands);
   // auto dataConvertOp = cast<miopen::DataConvertOp>(op);
   // auto operand = dataConvertOp.getOperand();
   // auto operandType = operand.getType();
   // auto returnType = dataConvertOp.getType();
   
    Type castedVectorType = VectorType::get({2}, rewriter.getIntegerType(16));
    Type llvmType = typeConverter.convertType(castedVectorType);
    if (!llvmType)
      return failure();

    auto bitcastop = rewriter.create<LLVM::BitcastOp>(loc,llvmType,adaptor.in());
    auto idxType = rewriter.getIndexType();
    auto constant = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter.convertType(idxType),
        rewriter.getIntegerAttr(idxType, 1));
    auto extractorOp = rewriter.create<LLVM::ExtractElementOp>(loc, llvmType, bitcastop,
                                                   constant);
    rewriter.replaceOp(op, {extractorOp});
    return success();
  }
};

void mlir::populateMIOpenToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  MLIRContext *ctx = converter.getDialect()->getContext();
  patterns.insert<MIOpenDataConvertOpConversion>(ctx, converter);
}