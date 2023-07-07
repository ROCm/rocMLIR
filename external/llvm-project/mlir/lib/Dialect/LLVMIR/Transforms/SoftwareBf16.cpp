//===- SoftwareBf16.cpp - Prepare for translation to LLVM IR ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_SOFTWAREBF16
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;

static APInt castBF16toInt(APFloat value) {
  assert(&value.getSemantics() == &APFloat::BFloat() && "Must cast bf16 only");
  APInt ret = value.bitcastToAPInt();
  assert(ret.getBitWidth() == 16 && "bf16 conversion should make i16");
  return ret;
}

static Value getLlvmI32Const(Location loc, PatternRewriter &rewriter, Type type,
                             int32_t value) {
  Attribute ret = rewriter.getI32IntegerAttr(value);
  if (LLVM::isCompatibleVectorType(type))
    ret = SplatElementsAttr::get(type.cast<ShapedType>(), ret);
  return rewriter.createOrFold<LLVM::ConstantOp>(loc, type, ret);
}

namespace {
/// Rewrites bf16 constants to their i16 equivalents
/// This is relying on the fact that the vector, i16, and bf16 types used in the
/// LLVM dialect are the standard ones and not weird custom wrappers
struct BF16ConstCasting : OpRewritePattern<LLVM::ConstantOp> {
  explicit BF16ConstCasting(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(LLVM::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    Attribute val = op.getValueAttr();
    Operation *rawOp = op.getOperation();
    Type bf16 = rewriter.getBF16Type();
    Type retType = op.getRes().getType();
    Type retElemType = retType;

    if (auto retTypeShaped = retType.dyn_cast<ShapedType>())
      retElemType = retTypeShaped.getElementType();

    if (auto valFloat = val.dyn_cast<FloatAttr>()) {
      if (valFloat.getType() != bf16)
        return failure();
      APInt newVal = castBF16toInt(valFloat.getValue());
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          rawOp, retType, rewriter.getIntegerAttr(retType, newVal));
      return success();
    }

    if (auto valDense = val.dyn_cast<DenseElementsAttr>()) {
      if (valDense.getElementType() != bf16)
        return failure();
      DenseElementsAttr newVal = valDense.bitcast(retElemType);
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(rawOp, retType, newVal);
      return success();
    }

    if (auto valSparse = val.dyn_cast<SparseElementsAttr>()) {
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
      rewriter.replaceOpWithNewOp<Op>(op, resType, extended, op->getAttrs());
    }

    return success();
  }
};

struct SoftwareBF16Ext : OpRewritePattern<LLVM::FPExtOp> {
  explicit SoftwareBF16Ext(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(LLVM::FPExtOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

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

    Location loc = op.getLoc();

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
    // b = (a + 32767) >> 16
    // c = ((a >> 16) & 1)
    // d = b + c
    // truncate d to i16 and return this i16
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
    Value shiftBigValue = rewriter.create<LLVM::LShrOp>(
        loc, bitcastType, addBigValue, constantSixteen);

    Value addValue = rewriter.create<LLVM::AddOp>(loc, andValue, shiftBigValue);

    Value truncValue = rewriter.create<LLVM::TruncOp>(loc, destType, addValue);
    rewriter.replaceOp(op.getOperation(), {truncValue});

    return success();
  }
};

} // namespace

static void replaceBF16WithI16(Operation *op, TypeConverter &converter) {
  if (auto func = dyn_cast<LLVM::LLVMFuncOp>(op)) {
    auto funcType = func.getFunctionType();
    func.setType(converter.convertType(funcType));
    for (Value arg : func.getArguments())
      arg.setType(converter.convertType(arg.getType()));
  } else if (auto globalOp = dyn_cast<LLVM::GlobalOp>(op)) {
    Type globalType = globalOp.getType();
    globalOp.setGlobalTypeAttr(
        TypeAttr::get(converter.convertType(globalType)));
  } else {
    for (unsigned idx = 0; idx < op->getNumOperands(); idx++) {
      auto type = converter.convertType(op->getOperand(idx).getType());
      op->getOperand(idx).setType(type);
    }
    for (unsigned idx = 0; idx < op->getNumResults(); idx++) {
      auto type = converter.convertType(op->getResult(idx).getType());
      op->getResult(idx).setType(type);
    }
  }
}

static void populateSoftwareBF16Patterns(MLIRContext *ctx,
                                         TypeConverter &converter,
                                         RewritePatternSet &patterns) {
  // AMD GPUs don't have a backend that understands bfloat, even though LLVM's
  // frontend does. Remove this if/when that changes. Note that adding
  // conversions after the default constructor runs gives them priority
  // over the defaults.
  Type llvmI16 = IntegerType::get(ctx, 16);

  converter.addConversion([](Type type) { return type; });

  converter.addConversion(
      [llvmI16](BFloat16Type type) -> Type { return llvmI16; });

  converter.addConversion([&](VectorType type) -> std::optional<Type> {
    if (auto element = converter.convertType(type.getElementType()))
      return type.clone(element);
    return std::nullopt;
  });

  converter.addConversion(
      [&](LLVM::LLVMPointerType type) -> std::optional<Type> {
        if (type.isOpaque())
          return type;
        if (auto pointee = converter.convertType(type.getElementType()))
          return LLVM::LLVMPointerType::get(pointee, type.getAddressSpace());
        return std::nullopt;
      });

  converter.addConversion(
      [&](LLVM::LLVMStructType type, SmallVectorImpl<Type> &results,
          ArrayRef<Type> callStack) -> std::optional<LogicalResult> {
        bool converted = false;
        SmallVector<Type> convertedElemTypes;
        convertedElemTypes.reserve(type.getBody().size());
        for (auto t : type.getBody()) {
          SmallVector<Type, 1> element;
          if (failed(converter.convertType(t, element))) {
            return std::nullopt;
          }
          assert(element.size() == 1);
          convertedElemTypes.push_back(element[0]);
          if (t != element[0])
            converted = true;
        }

        if (!converted) {
          results.push_back(type);
          return success();
        }

        // Identified StructType
        if (type.isIdentified()) {
          auto convertedType = LLVM::LLVMStructType::getIdentified(
              type.getContext(), ("_Converted_" + type.getName()).str());
          unsigned counter = 1;
          while (convertedType.isInitialized()) {
            convertedType = LLVM::LLVMStructType::getIdentified(
                type.getContext(),
                ("_Converted_" + Twine(counter++) + type.getName()).str());
          }
          if (llvm::count(callStack, type) > 1) {
            results.push_back(convertedType);
            return success();
          }
          if (failed(
                  convertedType.setBody(convertedElemTypes, type.isPacked())))
            return std::nullopt;
          results.push_back(convertedType);
          return success();
        }

        // Literal StructType
        results.push_back(LLVM::LLVMStructType::getLiteral(
            type.getContext(), convertedElemTypes, type.isPacked()));
        return success();
      });

  converter.addConversion([&](LLVM::LLVMArrayType type) -> std::optional<Type> {
    if (auto element = converter.convertType(type.getElementType()))
      return LLVM::LLVMArrayType::get(element, type.getNumElements());
    return std::nullopt;
  });

  converter.addConversion(
      [&](LLVM::LLVMFunctionType type) -> std::optional<Type> {
        Type convertedResType = converter.convertType(type.getReturnType());
        if (!convertedResType)
          return std::nullopt;

        SmallVector<Type> convertedArgTypes;
        convertedArgTypes.reserve(type.getNumParams());
        if (failed(converter.convertTypes(type.getParams(), convertedArgTypes)))
          return std::nullopt;
        return LLVM::LLVMFunctionType::get(convertedResType, convertedArgTypes,
                                           type.isVarArg());
      });

  patterns.add<BF16ConstCasting, SoftwareBF16Trunc, SoftwareBF16Ext>(ctx);

  patterns.add<BF16AsF32<LLVM::FAddOp>, BF16AsF32<LLVM::FCmpOp>,
               BF16AsF32<LLVM::FDivOp>, BF16AsF32<LLVM::FMulOp>,
               BF16AsF32<LLVM::FNegOp>, BF16AsF32<LLVM::FRemOp>,
               BF16AsF32<LLVM::FSubOp>, BF16AsF32<LLVM::FPToSIOp>,
               BF16AsF32<LLVM::FPToUIOp>, BF16AsF32<LLVM::SIToFPOp>,
               BF16AsF32<LLVM::UIToFPOp>, BF16AsF32<LLVM::FAbsOp>,
               BF16AsF32<LLVM::FCeilOp>, BF16AsF32<LLVM::FFloorOp>,
               BF16AsF32<LLVM::FMAOp>, BF16AsF32<LLVM::FMulAddOp>>(ctx);
}

namespace {
struct SoftwareBF16Pass
    : public LLVM::impl::SoftwareBF16Base<SoftwareBF16Pass> {
  void runOnOperation() override {
    auto m = getOperation();
    MLIRContext *ctx = m->getContext();
    TypeConverter converter;
    RewritePatternSet bf16fixupPatterns(ctx);

    populateSoftwareBF16Patterns(ctx, converter, bf16fixupPatterns);
    // Replace BF16 types in an operation with I16 types
    m->walk([&converter](Operation *op) { replaceBF16WithI16(op, converter); });

    if (failed(applyPatternsAndFoldGreedily(m, std::move(bf16fixupPatterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> LLVM::createSoftwareBF16Pass() {
  return std::make_unique<SoftwareBF16Pass>();
}
