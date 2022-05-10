//===- LowerGpuOpsToROCDLOps.cpp - MLIR GPU to ROCDL lowering passes ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate ROCDLIR operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"

#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/Runtimes.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToROCDL/VectorToROCDL.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/FormatVariadic.h"
#include <iterator>

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/IndexIntrinsicsOpLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"
#include "../PassDetail.h"

using namespace mlir;

namespace {

/// Import the GPU Ops to ROCDL Patterns.
#include "GPUToROCDL.cpp.inc"

// A pass that replaces all occurrences of GPU device operations with their
// corresponding ROCDL equivalent.
//
// This pass only handles device code and is not meant to be run on GPU host
// code.
struct LowerGpuOpsToROCDLOpsPass
    : public ConvertGpuOpsToROCDLOpsBase<LowerGpuOpsToROCDLOpsPass> {
  LowerGpuOpsToROCDLOpsPass() = default;
  LowerGpuOpsToROCDLOpsPass(unsigned indexBitwidth, gpu::amd::Runtime runtime) {
    this->indexBitwidth = indexBitwidth;
    this->runtime = runtime;
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(
        m.getContext(),
        DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    options.emitCWrappers = true;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    LLVMTypeConverter converter(ctx, options);
    RewritePatternSet patterns(ctx);
    RewritePatternSet llvmPatterns(ctx);
    RewritePatternSet bf16fixupPatterns(ctx);

    populateGpuRewritePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(m, std::move(patterns));

    vector::populateVectorMaskMaterializationPatterns(llvmPatterns, true);
    vector::populateVectorTransferLoweringPatterns(llvmPatterns);
    mlir::arith::populateArithmeticToLLVMConversionPatterns(converter,
                                                            llvmPatterns);
    populateAMDGPUToROCDLConversionPatterns(converter, llvmPatterns);
    populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
    populateVectorToROCDLConversionPatterns(converter, llvmPatterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
    populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
    populateMemRefToLLVMConversionPatterns(converter, llvmPatterns);
    populateGpuToROCDLConversionPatterns(converter, llvmPatterns, runtime);
    // Keep last to prioritize newly-added type conversions, just in case
    // Note that since these patterns touch LLVM ops, they'll need to run after
    // conversion
    populateBF16ToROCDLConversionPatterns(converter, bf16fixupPatterns);

    LLVMConversionTarget target(getContext());
    configureGpuToROCDLConversionLegality(target);
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();
    if (failed(applyPatternsAndFoldGreedily(m, std::move(bf16fixupPatterns))))
      signalPassFailure();
  }
};
} // anonymous namespace

void mlir::configureGpuToROCDLConversionLegality(ConversionTarget &target) {
  target.addIllegalOp<FuncOp>();
  target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<::mlir::ROCDL::ROCDLDialect>();
  target.addIllegalDialect<gpu::GPUDialect>();
  target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::Exp2Op, LLVM::FAbsOp,
                      LLVM::FCeilOp, LLVM::FFloorOp, LLVM::LogOp, LLVM::Log10Op,
                      LLVM::Log2Op, LLVM::PowOp, LLVM::SinOp, LLVM::SqrtOp>();

  // TODO: Remove once we support replacing non-root ops.
  target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp, gpu::ModuleEndOp>();
}

namespace mlir {
struct MFMAOpLowering : ConvertOpToLLVMPattern<gpu::MFMAOp> {
  using ConvertOpToLLVMPattern<gpu::MFMAOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::MFMAOp op, gpu::MFMAOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Obtain MFMA instruction be used.
    StringRef mfmaInstr = "mfma_f32_32x32x1f32";
    if (op->getAttr("instr"))
      mfmaInstr = op->getAttr("instr").cast<StringAttr>().getValue();

    // Obtain immediate values be used.
    ArrayAttr immArrayAttr = rewriter.getArrayAttr({
        rewriter.getI32IntegerAttr(0),
        rewriter.getI32IntegerAttr(0),
        rewriter.getI32IntegerAttr(0),
    });
    if (op->getAttr("imm"))
      immArrayAttr = op->getAttr("imm").cast<ArrayAttr>();

    SmallVector<Value, 3> immValues;
    for (Attribute immAttr : immArrayAttr)
      immValues.push_back(rewriter.create<LLVM::ConstantOp>(
          loc, typeConverter->convertType(rewriter.getIntegerType(32)),
          immAttr));

    Value sourceA = adaptor.sourceA();
    Value sourceB = adaptor.sourceB();
    Value destC = adaptor.destC();
    Type retType = typeConverter->convertType(destC.getType());

    if (mfmaInstr.endswith("f32")) {
      // F32.
      if (mfmaInstr == "mfma_f32_32x32x1f32")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x1f32>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
      else if (mfmaInstr == "mfma_f32_32x32x2f32")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x2f32>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
      else if (mfmaInstr == "mfma_f32_16x16x4f32")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x4f32>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
      else if (mfmaInstr == "mfma_f32_16x16x1f32")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x1f32>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
      else if (mfmaInstr == "mfma_f32_4x4x1f32")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_4x4x1f32>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
    } else if (mfmaInstr.endswith("f16") && !mfmaInstr.endswith("bf16")) {
      // F16.
      if (mfmaInstr == "mfma_f32_32x32x4f16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x4f16>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
      else if (mfmaInstr == "mfma_f32_32x32x8f16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x8f16>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
      else if (mfmaInstr == "mfma_f32_16x16x16f16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x16f16>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
      else if (mfmaInstr == "mfma_f32_16x16x4f16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x4f16>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
      else if (mfmaInstr == "mfma_f32_4x4x4f16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_4x4x4f16>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
    } else if (mfmaInstr.endswith("bf16")) {
      // BF16.
      if (mfmaInstr == "mfma_f32_32x32x2bf16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x2bf16>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
      else if (mfmaInstr == "mfma_f32_32x32x4bf16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x4bf16>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
      else if (mfmaInstr == "mfma_f32_16x16x8bf16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x8bf16>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
      else if (mfmaInstr == "mfma_f32_16x16x2bf16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x2bf16>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
      else if (mfmaInstr == "mfma_f32_4x4x2bf16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_4x4x2bf16>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
    } else if (mfmaInstr.endswith("i8")) {
      if (mfmaInstr == "mfma_i32_32x32x8i8")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_i32_32x32x8i8>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
      else if (mfmaInstr == "mfma_i32_16x16x16i8")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_i32_16x16x16i8>(
            op, retType,
            ValueRange{sourceA, sourceB, destC, immValues[0], immValues[1],
                       immValues[2]});
      else {
        return failure();
      }
    }

    return success();
  }
};

struct WarpSwizzleOpLowering : ConvertOpToLLVMPattern<gpu::WarpSwizzleOp> {
  using ConvertOpToLLVMPattern<gpu::WarpSwizzleOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::WarpSwizzleOp op, gpu::WarpSwizzleOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto mlirI32Type = rewriter.getI32Type();
    auto llvmI32Type = typeConverter->convertType(mlirI32Type);
    auto llvmI1Type = typeConverter->convertType(rewriter.getI1Type());

    int32_t permConst = 0;
    const ArrayRef<mlir::Attribute> selector = adaptor.selector().getValue();
    for (auto v = selector.rbegin(); v != selector.rend(); ++v) {
      permConst = (permConst << 2) |
                  v->cast<mlir::IntegerAttr>().getValue().getZExtValue();
    }

    auto dppCtrlConstImm = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type, rewriter.getI32IntegerAttr(permConst));

    // DPP instructions support a mask that allows selectively disabling
    // rows and/or banks of VGPRs during the shuffle. Since we want to shuffle
    // all lanes, we use all 1s to avoid disabling any writes
    auto noMaskConstImm = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type, rewriter.getI32IntegerAttr(0b1111));
    auto noBoundsControlConstImm = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI1Type, rewriter.getBoolAttr(true));

    auto intrinsic = rewriter.create<ROCDL::DPPMovOp>(
        loc, llvmI32Type, adaptor.in(), dppCtrlConstImm, noMaskConstImm,
        noMaskConstImm, noBoundsControlConstImm);
    rewriter.replaceOp(op, {intrinsic});
    return success();
  }
};

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
    Type resType = op.getResult().getType();
    Type extType = rewriter.getF32Type();
    Type resElementType = resType;

    if (auto resShaped = resType.dyn_cast<ShapedType>()) {
      extType = resShaped.clone(extType);
      resElementType = resShaped.getElementType();
    }
    Type i16 = rewriter.getIntegerType(16);
    if (resElementType != i16)
      return failure();

    llvm::SmallVector<Value, 2> extended;
    for (Value v : op.getOperands()) {
      extended.push_back(rewriter.create<LLVM::FPExtOp>(loc, extType, v));
    }
    Op operation = rewriter.create<Op>(loc, extType, extended, op->getAttrs());
    rewriter.replaceOpWithNewOp<LLVM::FPTruncOp>(op, resType,
                                                 operation.getResult());
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

/// Rewrites extension of bfloat as a bitshift. This is needed since the ROCDL
/// target doesn't support the bfloat type even though LLVM in general does.
struct SoftwareBF16Ext : OpRewritePattern<LLVM::FPExtOp> {
  explicit SoftwareBF16Ext(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(LLVM::FPExtOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Type srcType = op.getArg().getType();
    Type destType = op.getRes().getType();
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
/// This is needed since the ROCDL target doesn't support the bfloat type,
/// even though LLVM in general does
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

struct LDSBarrierOpLowering : public ConvertOpToLLVMPattern<gpu::LDSBarrierOp> {
  using ConvertOpToLLVMPattern<gpu::LDSBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::LDSBarrierOp op, gpu::LDSBarrierOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto asmDialectAttr = LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                                    LLVM::AsmDialect::AD_ATT);
    const auto *asmStr = "s_waitcnt lgkmcnt(0) \n s_barrier";
    const auto *asmCstr = "";
    SmallVector<Value> asmVals{};
    SmallVector<Type> types{};
    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op,
        /*resultTypes=*/types, /*operands=*/asmVals, /*asm_string=*/asmStr,
        /*constraints=*/asmCstr, /*has_side_effects=*/true,
        /*is_align_stack=*/false, /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/ArrayAttr());
    return success();
  }
};
} // namespace mlir

void mlir::populateGpuToROCDLConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    mlir::gpu::amd::Runtime runtime) {
  using mlir::gpu::amd::Runtime;

  populateWithGenerated(patterns);
  patterns
      .add<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, ROCDL::ThreadIdXOp,
                                       ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>,
           GPUIndexIntrinsicOpLowering<gpu::BlockDimOp, ROCDL::BlockDimXOp,
                                       ROCDL::BlockDimYOp, ROCDL::BlockDimZOp>,
           GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, ROCDL::BlockIdXOp,
                                       ROCDL::BlockIdYOp, ROCDL::BlockIdZOp>,
           GPUIndexIntrinsicOpLowering<gpu::GridDimOp, ROCDL::GridDimXOp,
                                       ROCDL::GridDimYOp, ROCDL::GridDimZOp>,
           GPUReturnOpLowering>(converter);
  patterns.add<GPUFuncOpLowering>(
      converter, /*allocaAddrSpace=*/5,
      StringAttr::get(&converter.getContext(),
                      ROCDL::ROCDLDialect::getKernelFuncAttrName()));
  if (Runtime::HIP == runtime) {
    patterns.add<GPUPrintfOpToHIPLowering>(converter);
  } else if (Runtime::OpenCL == runtime) {
    // Use address space = 4 to match the OpenCL definition of printf()
    patterns.add<GPUPrintfOpToLLVMCallLowering>(converter, /*addressSpace=*/4);
  }

  patterns.add<OpToFuncCallLowering<math::AbsOp>>(converter, "__ocml_fabs_f32",
                                                  "__ocml_fabs_f64");
  patterns.add<OpToFuncCallLowering<math::AtanOp>>(converter, "__ocml_atan_f32",
                                                   "__ocml_atan_f64");
  patterns.add<OpToFuncCallLowering<math::Atan2Op>>(
      converter, "__ocml_atan2_f32", "__ocml_atan2_f64");
  patterns.add<OpToFuncCallLowering<math::CeilOp>>(converter, "__ocml_ceil_f32",
                                                   "__ocml_ceil_f64");
  patterns.add<OpToFuncCallLowering<math::CosOp>>(converter, "__ocml_cos_f32",
                                                  "__ocml_cos_f64");
  patterns.add<OpToFuncCallLowering<math::ExpOp>>(converter, "__ocml_exp_f32",
                                                  "__ocml_exp_f64");
  patterns.add<OpToFuncCallLowering<math::Exp2Op>>(converter, "__ocml_exp2_f32",
                                                   "__ocml_exp2_f64");
  patterns.add<OpToFuncCallLowering<math::ExpM1Op>>(
      converter, "__ocml_expm1_f32", "__ocml_expm1_f64");
  patterns.add<OpToFuncCallLowering<math::FloorOp>>(
      converter, "__ocml_floor_f32", "__ocml_floor_f64");
  patterns.add<OpToFuncCallLowering<math::LogOp>>(converter, "__ocml_log_f32",
                                                  "__ocml_log_f64");
  patterns.add<OpToFuncCallLowering<math::Log10Op>>(
      converter, "__ocml_log10_f32", "__ocml_log10_f64");
  patterns.add<OpToFuncCallLowering<math::Log1pOp>>(
      converter, "__ocml_log1p_f32", "__ocml_log1p_f64");
  patterns.add<OpToFuncCallLowering<math::Log2Op>>(converter, "__ocml_log2_f32",
                                                   "__ocml_log2_f64");
  patterns.add<OpToFuncCallLowering<math::PowFOp>>(converter, "__ocml_pow_f32",
                                                   "__ocml_pow_f64");
  patterns.add<OpToFuncCallLowering<math::RsqrtOp>>(
      converter, "__ocml_rsqrt_f32", "__ocml_rsqrt_f64");
  patterns.add<OpToFuncCallLowering<math::SinOp>>(converter, "__ocml_sin_f32",
                                                  "__ocml_sin_f64");
  patterns.add<OpToFuncCallLowering<math::SqrtOp>>(converter, "__ocml_sqrt_f32",
                                                   "__ocml_sqrt_f64");
  patterns.add<OpToFuncCallLowering<math::TanhOp>>(converter, "__ocml_tanh_f32",
                                                   "__ocml_tanh_f64");

  patterns.add<MFMAOpLowering, WarpSwizzleOpLowering, LDSBarrierOpLowering>(
      converter);
}

void mlir::populateBF16ToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns) {
  MLIRContext *ctx = converter.getDialect()->getContext();
  // AMD GPUs don't have a backend that understands bfloat, even though LLVM's
  // frontend does. Remove this if/when that changes. Note that adding
  // conversions after the default constructor runs gives them priority
  // over the defaults.
  Type llvmI16 = converter.convertType(IntegerType::get(ctx, 16));
  Type bf16 = mlir::BFloat16Type::get(ctx);
  converter.addConversion(
      [llvmI16](mlir::BFloat16Type type) -> Type { return llvmI16; });
  // Override for vector types since they get caught by isCompatibleType(),
  // which doesn't convert the element type
  converter.addConversion(
      [llvmI16, bf16](mlir::VectorType type) -> Optional<Type> {
        if (type.getElementType() == bf16 && type.getRank() == 1)
          return type.clone(llvmI16);
        return llvm::None; // continue search
      });
  patterns.add<BF16ConstCasting, SoftwareBF16Trunc, SoftwareBF16Ext,
               BF16AsF32<LLVM::FAddOp>, BF16AsF32<LLVM::FMulOp>,
               BF16AsF32<LLVM::FMAOp>>(ctx);
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerGpuOpsToROCDLOpsPass(unsigned indexBitwidth,
                                      gpu::amd::Runtime runtime) {
  return std::make_unique<LowerGpuOpsToROCDLOpsPass>(indexBitwidth, runtime);
}
