//===- LegalizeForExport.cpp - Prepare for translation to LLVM IR ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/SoftwareBF16.h"
#include "PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace LLVM {
/// Rewrites bf16 constants to their i16 equivalents
/// This is relying on the fact that the vector, i16, and bf16 types used in the
/// LLVM dialect are the standard ones and not weird custom wrappers
struct BF16ConstCasting : OpRewritePattern<LLVM::ConstantOp> {
  explicit BF16ConstCasting(MLIRContext *context) : OpRewritePattern(context) {}

  llvm::APInt toInt(llvm::APFloat value) const {
    assert(&value.getSemantics() == &llvm::APFloat::BFloat() &&
           "Must cast bf16 only");
    APInt ret = value.bitcastToAPInt();
    assert(ret.getBitWidth() == 16 && "bf16 conversion should make i16");
    return ret;
  }

  LogicalResult matchAndRewrite(LLVM::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    Attribute val = op.getValueAttr();
    Operation *rawOp = op.getOperation();
    Type bf16 = rewriter.getBF16Type();
    Type retType = op.getRes().getType();
    Type retElemType = retType;

    if (auto retTypeShaped = retType.dyn_cast<ShapedType>())
      retElemType = retTypeShaped.getElementType();

    if (auto valFloat = val.dyn_cast<mlir::FloatAttr>()) {
      if (valFloat.getType() != bf16)
        return failure();
      APInt newVal = toInt(valFloat.getValue());
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          rawOp, retType, rewriter.getIntegerAttr(retType, newVal));
      return success();
    }

    if (auto valDense = val.dyn_cast<mlir::DenseElementsAttr>()) {
      if (valDense.getElementType() != bf16)
        return failure();
      DenseElementsAttr newVal = valDense.bitcast(retElemType);
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(rawOp, retType, newVal);
      return success();
    }

    if (auto valSparse = val.dyn_cast<mlir::SparseElementsAttr>()) {
      if (valSparse.getElementType() != bf16)
        return failure();
      DenseElementsAttr values = valSparse.getValues();
      DenseElementsAttr newValues = values.bitcast(retElemType);
      auto newVal = SparseElementsAttr::get(retType.cast<ShapedType>(),
                                            valSparse.getIndices(), newValues);
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(rawOp, retType, newVal);
      return success();
    }
    // No match otherwise
    return failure();
  }
};

template <typename Op>
struct BF16AsF32 : OpRewritePattern<Op> {
  explicit BF16AsF32(MLIRContext *context) : OpRewritePattern<Op>(context) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Type opdType = op->getOperand(0).getType();
    Type opdElementType = opdType;
    Type i16 = rewriter.getIntegerType(16);

    if (auto opdShaped = opdType.dyn_cast<ShapedType>()) {
      opdElementType = opdShaped.getElementType();
    }

    Type resType = op.getResult().getType();
    Type extType = rewriter.getF32Type();
    Type resElementType = resType;

    if (auto resShaped = resType.dyn_cast<ShapedType>()) {
      extType = resShaped.clone(extType);
      resElementType = resShaped.getElementType();
    }

    if (resElementType != i16 && opdElementType != i16)
      return failure();

    llvm::SmallVector<Value, 2> extended;
    if (isa<LLVM::SIToFPOp>(op) || isa<LLVM::UIToFPOp>(op)) {
      extended.push_back(op->getOperand(0));
    } else {
      for (Value v : op->getOperands()) {
        extended.push_back(
            rewriter.create<LLVM::FPExtOp>(loc, extType, v)); // i16->f32
      }
    }

    if (resElementType == i16) {
      Op operation =
          rewriter.create<Op>(loc, extType, extended, op->getAttrs());
      rewriter.replaceOpWithNewOp<LLVM::FPTruncOp>(op, resType,
                                                   operation.getResult());
    } else { // FCmp
      Op operation =
          rewriter.create<Op>(loc, resType, extended, op->getAttrs());
      rewriter.replaceOp(op.getOperation(), {operation});
    }

    return success();
  }
};

Value getLlvmI32Const(Location loc, PatternRewriter &rewriter, Type type,
                      int32_t value) {
  Attribute ret = rewriter.getI32IntegerAttr(value);
  if (LLVM::isCompatibleVectorType(type))
    ret = SplatElementsAttr::get(type.cast<ShapedType>(), ret);
  return rewriter.create<LLVM::ConstantOp>(loc, type, ret);
}

struct SoftwareBF16Ext : OpRewritePattern<LLVM::FPExtOp> {
  explicit SoftwareBF16Ext(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(LLVM::FPExtOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Type srcType = op.getArg().getType();
    Type destType = op.getResult().getType();
    Type srcElemType = srcType;
    if (auto shaped = srcType.dyn_cast<ShapedType>())
      srcElemType = shaped.getElementType();

    Type i16 = rewriter.getIntegerType(16);
    if (srcElemType != i16)
      return failure();

    Type extType = rewriter.getI32Type();
    if (auto srcShaped = srcType.dyn_cast<ShapedType>())
      extType = srcShaped.clone(extType);

    Type f32 = rewriter.getF32Type();
    if (auto destShaped = destType.dyn_cast<ShapedType>()) {
      if (destShaped.getElementType() != f32)
        return failure();
    } else if (destType != f32)
      return failure();

    Value extended = rewriter.create<LLVM::ZExtOp>(loc, extType, op.getArg());
    Value shifted = rewriter.create<LLVM::ShlOp>(
        loc, extended, getLlvmI32Const(loc, rewriter, extType, 16));
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, destType, shifted);

    return success();
  }
};

/// Rewrites truncation to bfloat as a series of integer operations.
struct SoftwareBF16Trunc : OpRewritePattern<LLVM::FPTruncOp> {
  explicit SoftwareBF16Trunc(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(LLVM::FPTruncOp op,
                                PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();

    Type srcType = op.getArg().getType();
    Type destType = op.getRes().getType();
    Type srcElemType = srcType;
    if (auto shaped = srcType.dyn_cast<ShapedType>())
      srcElemType = shaped.getElementType();

    Type f32 = rewriter.getF32Type();
    if (srcElemType != f32)
      return failure();

    Type bitcastType = rewriter.getI32Type();
    if (auto srcShaped = srcType.dyn_cast<ShapedType>())
      bitcastType = srcShaped.clone(bitcastType);

    Type i16 = rewriter.getIntegerType(16);
    if (auto destShaped = destType.dyn_cast<ShapedType>()) {
      if (destShaped.getElementType() != i16)
        return failure();
    } else if (destType != i16)
      return failure();

    // a = bitcast f32 value to i32
    // b = (a + 32767) << 16
    // c = ((a << 16) & 1)
    // d = b + c
    // truncate (d << 16) to i16 and return this i16
    Value bitcastop =
        rewriter.create<LLVM::BitcastOp>(loc, bitcastType, op.getArg());
    Value constantSixteen = getLlvmI32Const(loc, rewriter, bitcastType, 16);
    Value shiftValue = rewriter.create<LLVM::LShrOp>(
        loc, bitcastType, bitcastop, constantSixteen);

    Value constantOne = getLlvmI32Const(loc, rewriter, bitcastType, 1);
    Value andValue = rewriter.create<LLVM::AndOp>(loc, shiftValue, constantOne);

    Value constantBig = getLlvmI32Const(loc, rewriter, bitcastType, 32767);
    Value addBigValue =
        rewriter.create<LLVM::AddOp>(loc, bitcastop, constantBig);
    Value addValue = rewriter.create<LLVM::AddOp>(loc, andValue, addBigValue);

    Value shiftBeforeTruncValue = rewriter.create<LLVM::LShrOp>(
        loc, bitcastType, addValue, constantSixteen);
    Value truncValue =
        rewriter.create<LLVM::TruncOp>(loc, destType, shiftBeforeTruncValue);
    rewriter.replaceOp(op.getOperation(), {truncValue});

    return success();
  }
};

void replaceBF16InOp(Operation *op, LLVMTypeConverter &converter);

void replaceBF16InBlock(Block &block, LLVMTypeConverter &converter) {
  for (Operation &op : block.getOperations())
    replaceBF16InOp(&op, converter);
}

void replaceBF16InRegion(Region &region, LLVMTypeConverter &converter) {
  for (Block &block : region.getBlocks())
    replaceBF16InBlock(block, converter);
}

void replaceBF16InOp(Operation *op, LLVMTypeConverter &converter) {
  if (auto func = dyn_cast<LLVM::LLVMFuncOp>(op)) {
    auto funcType = func.getFunctionType();
    func.setType(converter.convertType(funcType));
    for (auto arg : func.getArguments())
      arg.setType(converter.convertType(arg.getType()));
  } else {
    for (unsigned idx = 0; idx < op->getNumOperands(); idx++) {
      auto type = converter.convertType(op->getOperand(idx).getType());
      op->getOperand(idx).setType(type);
    }
    for (unsigned idx = 0; idx < op->getNumResults(); idx++) {
      auto type = converter.convertType(op->getResult(idx).getType());
      op->getResult(0).setType(type);
    }
  }
  for (Region &region : op->getRegions())
    replaceBF16InRegion(region, converter);
  return;
}

} // namespace LLVM
} // namespace mlir

void mlir::LLVM::populateBF16ToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  MLIRContext *ctx = converter.getDialect()->getContext();
  // AMD GPUs don't have a backend that understands bfloat, even though LLVM's
  // frontend does. Remove this if/when that changes. Note that adding
  // conversions after the default constructor runs gives them priority
  // over the defaults.
  Type llvmI16 = converter.convertType(IntegerType::get(ctx, 16));
  Type bf16 = mlir::BFloat16Type::get(ctx);
  converter.addConversion(
      [llvmI16](mlir::BFloat16Type type) -> Type { return llvmI16; });

  // Override for vector/struct types since they get caught by
  // isCompatibleType(), which doesn't convert the element type
  converter.addConversion(
      [llvmI16, bf16](mlir::VectorType type) -> Optional<Type> {
        if (type.getElementType() == bf16 && type.getRank() == 1)
          return type.clone(llvmI16);
        return llvm::None; // continue search
      });

  converter.addConversion([&](LLVM::LLVMStructType type) -> Optional<Type> {
    bool converted = false;
    SmallVector<Type> convertedElemTypes;
    convertedElemTypes.reserve(type.getBody().size());
    for (auto t : type.getBody()) {
      SmallVector<Type, 1> element;
      if (failed(converter.convertType(t, element)))
        return llvm::None;
      convertedElemTypes.push_back(element[0]);
      if (t != element[0])
        converted = true;
    }

    if (!converted)
      return type;

    if (type.isIdentified()) {
      auto convertedType = LLVM::LLVMStructType::getIdentified(
          type.getContext(), ("_Converted_" + type.getName()).str());
      unsigned counter = 1;
      while (convertedType.isInitialized()) {
        convertedType = LLVM::LLVMStructType::getIdentified(
            type.getContext(),
            ("_Converted_" + std::to_string(counter++) + type.getName()).str());
      }
      if (failed(convertedType.setBody(convertedElemTypes, type.isPacked())))
        return llvm::None;
      return convertedType;
    } else
      return LLVM::LLVMStructType::getLiteral(
          type.getContext(), convertedElemTypes, type.isPacked());
  });

  patterns.add<LLVM::BF16ConstCasting, LLVM::SoftwareBF16Trunc,
               LLVM::SoftwareBF16Ext>(ctx);

  patterns.add<LLVM::BF16AsF32<LLVM::FAddOp>, LLVM::BF16AsF32<LLVM::FCmpOp>,
               LLVM::BF16AsF32<LLVM::FDivOp>, LLVM::BF16AsF32<LLVM::FMulOp>,
               LLVM::BF16AsF32<LLVM::FNegOp>, LLVM::BF16AsF32<LLVM::FRemOp>,
               LLVM::BF16AsF32<LLVM::FSubOp>, LLVM::BF16AsF32<LLVM::FPToSIOp>,
               LLVM::BF16AsF32<LLVM::FPToUIOp>, LLVM::BF16AsF32<LLVM::SIToFPOp>,
               LLVM::BF16AsF32<LLVM::UIToFPOp>, LLVM::BF16AsF32<LLVM::FAbsOp>,
               LLVM::BF16AsF32<LLVM::FCeilOp>, LLVM::BF16AsF32<LLVM::FFloorOp>,
               LLVM::BF16AsF32<LLVM::FMAOp>, LLVM::BF16AsF32<LLVM::FMulAddOp>>(
      ctx);
}

namespace {
struct SoftwareBF16Pass : public SoftwareBF16Base<SoftwareBF16Pass> {
  void runOnOperation() override {
    auto m = getOperation();
    MLIRContext *ctx = m->getContext();
    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter converter(ctx, options);
    RewritePatternSet bf16fixupPatterns(ctx);
    RewritePatternSet llvmPatterns(ctx);
    LLVMConversionTarget target(getContext());

    LLVM::populateBF16ToLLVMConversionPatterns(converter, bf16fixupPatterns);
    // replace BF16 types in an operation with Int8 types
    LLVM::replaceBF16InOp(m, converter);

    if (failed(applyPatternsAndFoldGreedily(m, std::move(bf16fixupPatterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> LLVM::createSoftwareBF16Pass() {
  return std::make_unique<SoftwareBF16Pass>();
}
