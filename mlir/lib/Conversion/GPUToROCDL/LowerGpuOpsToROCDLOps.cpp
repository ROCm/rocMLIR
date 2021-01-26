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

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"

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
class LowerGpuOpsToROCDLOpsPass
    : public ConvertGpuOpsToROCDLOpsBase<LowerGpuOpsToROCDLOpsPass> {
public:
  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    LLVMTypeConverterCustomization customs;
    customs.indexBitwidth = 32;
    LLVMTypeConverter converter(m.getContext(), customs);

    OwningRewritePatternList patterns;

    populateGpuRewritePatterns(m.getContext(), patterns);
    applyPatternsAndFoldGreedily(m, patterns);
    patterns.clear();

    populateVectorToLLVMConversionPatterns(converter, patterns);
    populateStdToLLVMConversionPatterns(converter, patterns);
    populateGpuToROCDLConversionPatterns(converter, patterns);
    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<gpu::GPUDialect>();
    target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::FAbsOp, LLVM::FCeilOp,
                        LLVM::LogOp, LLVM::Log10Op, LLVM::Log2Op>();
    target.addIllegalOp<FuncOp>();
    target.addLegalDialect<ROCDL::ROCDLDialect>();
    // TODO(whchung): Remove once we support replacing non-root ops.
    target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp, gpu::ModuleEndOp>();
    if (failed(applyPartialConversion(m, target, patterns, &converter)))
      signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
struct MubufLoadOpLowering : ConvertToLLVMPattern {
  explicit MubufLoadOpLowering(MLIRContext *context,
                               LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(gpu::MubufLoadOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto mubufLoadOp = cast<gpu::MubufLoadOp>(op);
    auto adaptor = gpu::MubufLoadOpOperandAdaptor(operands);
    auto loc = mubufLoadOp.getLoc();

    MemRefType srcMemRefType = mubufLoadOp.memref().getType().cast<MemRefType>();
    Type srcElementType = srcMemRefType.getElementType();
    auto srcShape = srcMemRefType.getShape();
    auto adaptorIndices = adaptor.indices();

    Type resultType = mubufLoadOp.result().getType();
    Type LLVMResultType = typeConverter.convertType(resultType);

    // use standard load for:
    // 1) loading scalar f16 and i16 (bf16) from global (addrspace 0).
    if ((srcElementType == rewriter.getIntegerType(16) || srcElementType == rewriter.getF16Type()) &&
        (srcMemRefType.getMemorySpace() == 0 && !resultType.isa<VectorType>())) {
      Value dataPtr = getDataPtr(op->getLoc(), srcMemRefType, adaptor.memref(),
                                 adaptor.indices(), rewriter, getModule());
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, dataPtr);
      return success();
    }

    // use standard load for:
    // 2) loading scalar and vector f16 and i16 (bf16) from LDS (addrspace 3).
    if (srcMemRefType.getMemorySpace() == 3) {
      Value dataPtr = getDataPtr(op->getLoc(), srcMemRefType, adaptor.memref(),
                                 adaptor.indices(), rewriter, getModule());
      if (!resultType.isa<VectorType>()) {
        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, dataPtr);
	return success();
      } else {
        // bitcast in case the result type is a vector.
	LLVM::LLVMType LLVMResultPointerType = LLVMResultType.cast<LLVM::LLVMType>().getPointerTo(srcMemRefType.getMemorySpace());
        Value dataPtrBitcasted = rewriter.create<LLVM::BitcastOp>(loc, LLVMResultPointerType, dataPtr);
        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, dataPtrBitcasted);
	return success();
      }
    }

    // for all other cases, use rocdl.mubuf_load.

    Type I1Type = rewriter.getI1Type();
    Type LLVMI1Type = typeConverter.convertType(I1Type);

    Type I32Type = rewriter.getIntegerType(32);
    Type LLVMI32Type = typeConverter.convertType(I32Type);

    Type I64Type = rewriter.getIntegerType(64);
    Type LLVMI64Type = typeConverter.convertType(I64Type);

    Type rsrcVectorType = VectorType::get({4}, I32Type);
    Type LLVMRsrcVectorType = typeConverter.convertType(rsrcVectorType);

    Type I32x2Type = VectorType::get({2}, I32Type);
    Type LLVMI32x2Type = typeConverter.convertType(I32x2Type);

    // word 0-1: pointer to memref.
    MemRefDescriptor memrefDescriptor(adaptor.memref());
    Value ptr = memrefDescriptor.alignedPtr(rewriter, loc);
    Value ptrToInt = rewriter.create<LLVM::PtrToIntOp>(loc, LLVMI64Type, ptr);

    // word 0-1: pointer to memref.
    Value ptrBitcasted = rewriter.create<LLVM::BitcastOp>(loc, LLVMI32x2Type, ptrToInt);
    Value rsrcUndefTwoItems = rewriter.create<LLVM::UndefOp>(loc, LLVMI32x2Type);
    Value rsrcFirstTwoItems = rewriter.create<LLVM::ShuffleVectorOp>(loc, ptrBitcasted, rsrcUndefTwoItems, rewriter.getI32ArrayAttr({0, 1, -1, -1}));

    Value rsrcUndef = rewriter.create<LLVM::UndefOp>(loc, LLVMRsrcVectorType);
    // word 2: fixed as -1 .
    Value constant2 = rewriter.create<LLVM::ConstantOp>(loc, LLVMI32Type, rewriter.getI32IntegerAttr(2));
    Value minusOne = rewriter.create<LLVM::ConstantOp>(loc, LLVMI32Type, rewriter.getI32IntegerAttr(-1));
    Value rsrc2 = rewriter.create<LLVM::InsertElementOp>(loc, LLVMRsrcVectorType, rsrcUndef, minusOne, constant2);

    // word 3: fixed as 0x00027000 .
    Value constant3 = rewriter.create<LLVM::ConstantOp>(loc, LLVMI32Type, rewriter.getI32IntegerAttr(3));
    Value bufferLoadConstant = rewriter.create<LLVM::ConstantOp>(loc, LLVMI32Type, rewriter.getI32IntegerAttr(0x00027000));
    Value rsrcLastTwoItems = rewriter.create<LLVM::InsertElementOp>(loc, LLVMRsrcVectorType, rsrc2, bufferLoadConstant, constant3);

    Value rsrc = rewriter.create<LLVM::ShuffleVectorOp>(loc, rsrcFirstTwoItems, rsrcLastTwoItems, rewriter.getI32ArrayAttr({0, 1, 6, 7}));

    // populate vindex : fixed as 0 of type i32.
    Value vindex = rewriter.create<LLVM::ConstantOp>(loc, LLVMI32Type, rewriter.getI32IntegerAttr(0));

    // populate voffset.
    SmallVector<Value, 4> indices;
    SmallVector<Value, 4> allocSizes;
    for (unsigned i = 0; i < srcShape.size(); ++i) {
      indices.push_back(adaptorIndices[i]);
      allocSizes.push_back(rewriter.create<LLVM::ConstantOp>(loc, LLVMI32Type, rewriter.getI32IntegerAttr(srcShape[i])));
    }
    Value voffsetElements = linearizeSubscripts(rewriter, loc, ArrayRef<Value>{indices.begin(), indices.end()}, ArrayRef<Value>{allocSizes.begin(), allocSizes.end()});

    // vindex is counted in bytes. Times size of element type.
    Value elementBytes = rewriter.create<LLVM::ConstantOp>(loc, LLVMI32Type, rewriter.getI32IntegerAttr(srcMemRefType.getElementTypeBitWidth() / 8));
    Value voffset = rewriter.create<LLVM::MulOp>(loc, LLVMI32Type, ArrayRef<Value>{voffsetElements, elementBytes});

    // populate slc : fixed as 0 of type i1.
    Value slc = rewriter.create<LLVM::ConstantOp>(loc, LLVMI1Type, rewriter.getIntegerAttr(I1Type, 0));

    // populate glc : fixed as 0 of type i1.
    Value glc = rewriter.create<LLVM::ConstantOp>(loc, LLVMI1Type, rewriter.getIntegerAttr(I1Type, 0));

    if (srcElementType == rewriter.getIntegerType(16) || srcElementType == rewriter.getF16Type()) {
      // for f16 and i16 (bf16) types, use f32 buffer_load and bitcast the result.
      assert(resultType.isa<VectorType>());
      // deduce the interim type for f16 / i16 (bf16).
      VectorType vectorResultType = resultType.template cast<VectorType>();
      auto vectorShape = vectorResultType.getShape();

      SmallVector<int64_t, 1> interimShape;
      for (unsigned iter = 0; iter < vectorShape.size() - 1; ++iter)
        interimShape.push_back(vectorShape[iter]);
      interimShape.push_back(vectorShape[vectorShape.size() - 1] >> 1);
      bool useScalarF32 = (vectorShape.size() == 1) && (vectorShape[0] == 2);
      Type interimResultType;
      if (useScalarF32)
        interimResultType = rewriter.getF32Type();
      else
	interimResultType = VectorType::get(interimShape, rewriter.getF32Type());
      Type interimLLVMResultType = typeConverter.convertType(interimResultType);

      Value interimLoad = rewriter.create<ROCDL::MubufLoadOp>(loc, interimLLVMResultType, rsrc, vindex, voffset, slc, glc);

      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, LLVMResultType, interimLoad);
    } else {
      rewriter.replaceOpWithNewOp<ROCDL::MubufLoadOp>(
              op, LLVMResultType, rsrc, vindex, voffset, slc, glc);
    }

    return success();
  }
};

struct MubufStoreOpLowering : ConvertToLLVMPattern {
  explicit MubufStoreOpLowering(MLIRContext *context,
                                LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(gpu::MubufStoreOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto mubufStoreOp = cast<gpu::MubufStoreOp>(op);
    auto adaptor = gpu::MubufStoreOpOperandAdaptor(operands);
    auto loc = mubufStoreOp.getLoc();

    MemRefType dstMemRefType = mubufStoreOp.memref().getType().cast<MemRefType>();
    Type dstElementType = dstMemRefType.getElementType();
    auto dstShape = dstMemRefType.getShape();
    auto adaptorIndices = adaptor.indices();
    auto adaptorValue = adaptor.value();

    Type valueType = mubufStoreOp.value().getType();
    Type LLVMValueType = typeConverter.convertType(valueType);

    // use standard store for storing scalar f16 and i16 (bf16).
    if ((dstElementType == rewriter.getIntegerType(16) || dstElementType == rewriter.getF16Type()) &&
        !valueType.isa<VectorType>()) {
      Value dataPtr = getDataPtr(op->getLoc(), dstMemRefType, adaptor.memref(),
                                 adaptor.indices(), rewriter, getModule());
      rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptorValue, dataPtr);
      return success();
    }

    // for all other cases, use rocdl.mubuf_store.

    Type I1Type = rewriter.getI1Type();
    Type LLVMI1Type = typeConverter.convertType(I1Type);

    Type I32Type = rewriter.getIntegerType(32);
    Type LLVMI32Type = typeConverter.convertType(I32Type);

    Type I64Type = rewriter.getIntegerType(64);
    Type LLVMI64Type = typeConverter.convertType(I64Type);

    Type rsrcVectorType = VectorType::get({4}, I32Type);
    Type LLVMRsrcVectorType = typeConverter.convertType(rsrcVectorType);

    Type I32x2Type = VectorType::get({2}, I32Type);
    Type LLVMI32x2Type = typeConverter.convertType(I32x2Type);

    // word 0-1: pointer to memref.
    MemRefDescriptor memrefDescriptor(adaptor.memref());
    Value ptr = memrefDescriptor.alignedPtr(rewriter, loc);
    Value ptrToInt = rewriter.create<LLVM::PtrToIntOp>(loc, LLVMI64Type, ptr);

    // word 0-1: pointer to memref.
    Value ptrBitcasted = rewriter.create<LLVM::BitcastOp>(loc, LLVMI32x2Type, ptrToInt);
    Value rsrcUndefTwoItems = rewriter.create<LLVM::UndefOp>(loc, LLVMI32x2Type);
    Value rsrcFirstTwoItems = rewriter.create<LLVM::ShuffleVectorOp>(loc, ptrBitcasted, rsrcUndefTwoItems, rewriter.getI32ArrayAttr({0, 1, -1, -1}));

    Value rsrcUndef = rewriter.create<LLVM::UndefOp>(loc, LLVMRsrcVectorType);
    // word 2: fixed as -1 .
    Value constant2 = rewriter.create<LLVM::ConstantOp>(loc, LLVMI32Type, rewriter.getI32IntegerAttr(2));
    Value minusOne = rewriter.create<LLVM::ConstantOp>(loc, LLVMI32Type, rewriter.getI32IntegerAttr(-1));
    Value rsrc2 = rewriter.create<LLVM::InsertElementOp>(loc, LLVMRsrcVectorType, rsrcUndef, minusOne, constant2);

    // word 3: fixed as 0x00027000 .
    Value constant3 = rewriter.create<LLVM::ConstantOp>(loc, LLVMI32Type, rewriter.getI32IntegerAttr(3));
    Value bufferLoadConstant = rewriter.create<LLVM::ConstantOp>(loc, LLVMI32Type, rewriter.getI32IntegerAttr(0x00027000));
    Value rsrcLastTwoItems = rewriter.create<LLVM::InsertElementOp>(loc, LLVMRsrcVectorType, rsrc2, bufferLoadConstant, constant3);

    Value rsrc = rewriter.create<LLVM::ShuffleVectorOp>(loc, rsrcFirstTwoItems, rsrcLastTwoItems, rewriter.getI32ArrayAttr({0, 1, 6, 7}));

    // populate vindex : fixed as 0 of type i32.
    Value vindex = rewriter.create<LLVM::ConstantOp>(loc, LLVMI32Type, rewriter.getI32IntegerAttr(0));

    // populate voffset.
    SmallVector<Value, 4> indices;
    SmallVector<Value, 4> allocSizes;
    for (unsigned i = 0; i < dstShape.size(); ++i) {
      indices.push_back(adaptorIndices[i]);
      allocSizes.push_back(rewriter.create<LLVM::ConstantOp>(loc, LLVMI32Type, rewriter.getI32IntegerAttr(dstShape[i])));
    }
    Value voffsetElements = linearizeSubscripts(rewriter, loc, ArrayRef<Value>{indices.begin(), indices.end()}, ArrayRef<Value>{allocSizes.begin(), allocSizes.end()});

    // vindex is counted in bytes. Times size of element type.
    Value elementBytes = rewriter.create<LLVM::ConstantOp>(loc, LLVMI32Type, rewriter.getI32IntegerAttr(dstMemRefType.getElementTypeBitWidth() / 8));
    Value voffset = rewriter.create<LLVM::MulOp>(loc, LLVMI32Type, ArrayRef<Value>{voffsetElements, elementBytes});

    // populate slc : fixed as 0 of type i1.
    Value slc = rewriter.create<LLVM::ConstantOp>(loc, LLVMI1Type, rewriter.getIntegerAttr(I1Type, 0));

    // populate glc : fixed as 0 of type i1.
    Value glc = rewriter.create<LLVM::ConstantOp>(loc, LLVMI1Type, rewriter.getIntegerAttr(I1Type, 0));

    if (dstElementType == rewriter.getIntegerType(16) || dstElementType == rewriter.getF16Type()) {
      // for f16 and i16 (bf16) types, use f32 buffer_store and bitcast the result.
      assert(valueType.isa<VectorType>());
      // deduce the interim type for f16 / i16 (bf16).
      VectorType vectorResultType = valueType.template cast<VectorType>();
      auto vectorShape = vectorResultType.getShape();

      SmallVector<int64_t, 1> interimShape;
      for (unsigned iter = 0; iter < vectorShape.size() - 1; ++iter)
        interimShape.push_back(vectorShape[iter]);
      interimShape.push_back(vectorShape[vectorShape.size() - 1] >> 1);
      bool useScalarF32 = (vectorShape.size() == 1) && (vectorShape[0] == 2);
      Type interimValueType;
      if (useScalarF32)
        interimValueType = rewriter.getF32Type();
      else
        interimValueType = VectorType::get(interimShape, rewriter.getF32Type());
      Type interimLLVMValueType = typeConverter.convertType(interimValueType);

      Value bitcastedValue = rewriter.create<LLVM::BitcastOp>(loc, interimLLVMValueType, adaptorValue);
      rewriter.replaceOpWithNewOp<ROCDL::MubufStoreOp>(op, bitcastedValue, rsrc, vindex, voffset, slc, glc);
    } else {
      rewriter.replaceOpWithNewOp<ROCDL::MubufStoreOp>(
              op, adaptorValue, rsrc, vindex, voffset, slc, glc);
    }

    return success();
  }
};

struct MFMAOpLowering : ConvertToLLVMPattern {
  explicit MFMAOpLowering(MLIRContext *context,
                          LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(gpu::MFMAOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto mfmaOp = cast<gpu::MFMAOp>(op);
    auto adaptor = gpu::MFMAOpOperandAdaptor(operands);
    auto loc = mfmaOp.getLoc();

    // Obtain MFMA instruction be used.
    StringRef mfmaInstr = "mfma_f32_32x32x1f32";
    if (mfmaOp.getAttr("instr"))
      mfmaInstr = mfmaOp.getAttr("instr").cast<StringAttr>().getValue();

    // Obtain immediate values be used.
    ArrayAttr immArrayAttr = rewriter.getArrayAttr({
        rewriter.getI32IntegerAttr(0),
        rewriter.getI32IntegerAttr(0),
        rewriter.getI32IntegerAttr(0),
    });
    if (mfmaOp.getAttr("imm"))
      immArrayAttr = mfmaOp.getAttr("imm").cast<ArrayAttr>();

    SmallVector<Value, 3> immValues;
    for (unsigned iter = 0; iter < immArrayAttr.size(); ++iter)
      immValues.push_back(rewriter.create<LLVM::ConstantOp>(
          loc, typeConverter.convertType(rewriter.getIntegerType(32)),
          immArrayAttr[iter]));

    if (mfmaInstr.endswith("f32")) {
      // F32.
      if (mfmaInstr == "mfma_f32_32x32x1f32")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x1f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(),
                       immValues[0], immValues[1], immValues[2]});
      else if (mfmaInstr == "mfma_f32_32x32x2f32")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x2f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(),
                       immValues[0], immValues[1], immValues[2]});
      else if (mfmaInstr == "mfma_f32_16x16x4f32")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x4f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(),
                       immValues[0], immValues[1], immValues[2]});
      else if (mfmaInstr == "mfma_f32_16x16x1f32")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x1f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(),
                       immValues[0], immValues[1], immValues[2]});
      else if (mfmaInstr == "mfma_f32_4x4x1f32")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_4x4x1f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(),
                       immValues[0], immValues[1], immValues[2]});
    } else if (mfmaInstr.endswith("f16") && !mfmaInstr.endswith("bf16")) {
      // F16.
      if (mfmaInstr == "mfma_f32_32x32x4f16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x4f16>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(),
                       immValues[0], immValues[1], immValues[2]});
      else if (mfmaInstr == "mfma_f32_32x32x8f16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x8f16>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(),
                       immValues[0], immValues[1], immValues[2]});
      else if (mfmaInstr == "mfma_f32_16x16x16f16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x16f16>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(),
                       immValues[0], immValues[1], immValues[2]});
      else if (mfmaInstr == "mfma_f32_16x16x4f16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x4f16>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(),
                       immValues[0], immValues[1], immValues[2]});
      else if (mfmaInstr == "mfma_f32_4x4x4f16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_4x4x4f16>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(),
                       immValues[0], immValues[1], immValues[2]});
    } else if (mfmaInstr.endswith("bf16")) {
      // BF16.
      Type castedVectorType = VectorType::get({2}, rewriter.getIntegerType(16));
      Type castedLLVMVectorType = typeConverter.convertType(castedVectorType);
      Value castedSourceA = rewriter.create<LLVM::BitcastOp>(
          op->getLoc(), castedLLVMVectorType, adaptor.sourceA());
      Value castedSourceB = rewriter.create<LLVM::BitcastOp>(
          op->getLoc(), castedLLVMVectorType, adaptor.sourceB());
      if (mfmaInstr == "mfma_f32_32x32x2bf16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x2bf16>(
            op, adaptor.destC().getType(),
            ValueRange{castedSourceA, castedSourceB, adaptor.destC(),
                       immValues[0], immValues[1], immValues[2]});
      else if (mfmaInstr == "mfma_f32_32x32x4bf16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x4bf16>(
            op, adaptor.destC().getType(),
            ValueRange{castedSourceA, castedSourceB, adaptor.destC(),
                       immValues[0], immValues[1], immValues[2]});
      else if (mfmaInstr == "mfma_f32_16x16x8bf16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x8bf16>(
            op, adaptor.destC().getType(),
            ValueRange{castedSourceA, castedSourceB, adaptor.destC(),
                       immValues[0], immValues[1], immValues[2]});
      else if (mfmaInstr == "mfma_f32_16x16x2bf16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x2bf16>(
            op, adaptor.destC().getType(),
            ValueRange{castedSourceA, castedSourceB, adaptor.destC(),
                       immValues[0], immValues[1], immValues[2]});
      else if (mfmaInstr == "mfma_f32_4x4x2bf16")
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_4x4x2bf16>(
            op, adaptor.destC().getType(),
            ValueRange{castedSourceA, castedSourceB, adaptor.destC(),
                       immValues[0], immValues[1], immValues[2]});
    }

    return success();
  }
};
} // namespace mlir

void mlir::populateGpuToROCDLConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  populateWithGenerated(converter.getDialect()->getContext(), &patterns);
  patterns.insert<
      GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, ROCDL::ThreadIdXOp,
                                  ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>,
      GPUIndexIntrinsicOpLowering<gpu::BlockDimOp, ROCDL::BlockDimXOp,
                                  ROCDL::BlockDimYOp, ROCDL::BlockDimZOp>,
      GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, ROCDL::BlockIdXOp,
                                  ROCDL::BlockIdYOp, ROCDL::BlockIdZOp>,
      GPUIndexIntrinsicOpLowering<gpu::GridDimOp, ROCDL::GridDimXOp,
                                  ROCDL::GridDimYOp, ROCDL::GridDimZOp>,
      GPUFuncOpLowering<5>, GPUReturnOpLowering>(converter);
  patterns.insert<OpToFuncCallLowering<AbsFOp>>(converter, "__ocml_fabs_f32",
                                                "__ocml_fabs_f64");
  patterns.insert<OpToFuncCallLowering<CeilFOp>>(converter, "__ocml_ceil_f32",
                                                 "__ocml_ceil_f64");
  patterns.insert<OpToFuncCallLowering<CosOp>>(converter, "__ocml_cos_f32",
                                               "__ocml_cos_f64");
  patterns.insert<OpToFuncCallLowering<ExpOp>>(converter, "__ocml_exp_f32",
                                               "__ocml_exp_f64");
  patterns.insert<OpToFuncCallLowering<LogOp>>(converter, "__ocml_log_f32",
                                               "__ocml_log_f64");
  patterns.insert<OpToFuncCallLowering<Log10Op>>(converter, "__ocml_log10_f32",
                                                 "__ocml_log10_f64");
  patterns.insert<OpToFuncCallLowering<Log2Op>>(converter, "__ocml_log2_f32",
                                                "__ocml_log2_f64");
  patterns.insert<OpToFuncCallLowering<TanhOp>>(converter, "__ocml_tanh_f32",
                                                "__ocml_tanh_f64");

  patterns.insert<MFMAOpLowering>(converter.getDialect()->getContext(),
                                  converter);

  patterns.insert<MubufLoadOpLowering>(converter.getDialect()->getContext(),
                                       converter);
  patterns.insert<MubufStoreOpLowering>(converter.getDialect()->getContext(),
                                        converter);
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerGpuOpsToROCDLOpsPass() {
  return std::make_unique<LowerGpuOpsToROCDLOpsPass>();
}
