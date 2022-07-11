//===- AMDGPUToROCDL.cpp - AMDGPU to ROCDL dialect conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/AMDGPU/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/TypeRange.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::amdgpu;

static Value createI32Constant(ConversionPatternRewriter &rewriter,
                               Location loc, int32_t value) {
  IntegerAttr valAttr = rewriter.getI32IntegerAttr(value);
  Type llvmI32 = rewriter.getI32Type();
  return rewriter.createOrFold<LLVM::ConstantOp>(loc, llvmI32, valAttr);
}

namespace {
/// Define lowering patterns for raw buffer ops
template <typename GpuOp, typename Intrinsic>
struct RawBufferOpLowering : public ConvertOpToLLVMPattern<GpuOp> {
  RawBufferOpLowering(LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<GpuOp>(converter), chipset(chipset) {}

  Chipset chipset;
  static constexpr uint32_t maxVectorOpWidth = 128;

  LogicalResult
  matchAndRewrite(GpuOp gpuOp, typename GpuOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = gpuOp.getLoc();
    Value memref = adaptor.memref();
    Value unconvertedMemref = gpuOp.memref();
    MemRefType memrefType = unconvertedMemref.getType().cast<MemRefType>();

    if (chipset.majorVersion < 9)
      return gpuOp.emitOpError("Raw buffer ops require GCN or higher");

    Value storeData = adaptor.getODSOperands(0)[0];
    if (storeData == memref) // no write component to this op
      storeData = Value();
    Type wantedDataType;
    if (storeData)
      wantedDataType = storeData.getType();
    else
      wantedDataType = gpuOp.getODSResults(0)[0].getType();

    Type llvmWantedDataType = this->typeConverter->convertType(wantedDataType);

    Type i32 = rewriter.getI32Type();
    Type llvmI32 = this->typeConverter->convertType(i32);

    int64_t elementByteWidth = memrefType.getElementTypeBitWidth() / 8;
    Value byteWidthConst = createI32Constant(rewriter, loc, elementByteWidth);

    // If we want to load a vector<NxT> with total size <= 32
    // bits, use a scalar load and bitcast it. Similarly, if bitsize(T) < 32
    // and the total load size is >= 32, use a vector load of N / (bitsize(T) /
    // 32) x i32 and bitcast.
    Type llvmBufferValType = llvmWantedDataType;
    if (auto dataVector = wantedDataType.dyn_cast<VectorType>()) {
      uint32_t elemBits = dataVector.getElementTypeBitWidth();
      uint32_t totalBits = elemBits * dataVector.getNumElements();
      if (totalBits > maxVectorOpWidth)
        return gpuOp.emitOpError(
            "Total width of loads or stores must be no more than " +
            Twine(maxVectorOpWidth) + " bits, but we call for " +
            Twine(totalBits) +
            " bits. This should've been caught in validation");
      if (elemBits < 32) {
        if (totalBits > 32) {
          if (totalBits % 32 != 0)
            return gpuOp.emitOpError("Load or store of more than 32-bits that "
                                     "doesn't fit into words. Can't happen\n");
          llvmBufferValType = this->typeConverter->convertType(
              VectorType::get(totalBits / 32, i32));
        } else {
          llvmBufferValType = this->typeConverter->convertType(
              rewriter.getIntegerType(totalBits));
        }
      }
    }

    SmallVector<Value, 6> args;
    if (storeData) {
      if (llvmBufferValType != llvmWantedDataType) {
        Value castForStore =
            rewriter.create<LLVM::BitcastOp>(loc, llvmBufferValType, storeData);
        args.push_back(castForStore);
      } else {
        args.push_back(storeData);
      }
    }

    // Construct buffer descriptor from memref, attributes
    int64_t offset = 0;
    SmallVector<int64_t, 5> strides;
    if (failed(getStridesAndOffset(memrefType, strides, offset)))
      return gpuOp.emitOpError("Can't lower non-stride-offset memrefs");

    // Resource descriptor
    // bits 0-47: base address
    // bits 48-61: stride (0 for raw buffers)
    // bit 62: texture cache coherency (always 0)
    // bit 63: enable swizzles (always off for raw buffers)
    // bits 64-95 (word 2): Number of records, units of stride
    // bits 96-127 (word 3): See below

    Type llvm4xI32 = this->typeConverter->convertType(VectorType::get(4, i32));
    MemRefDescriptor memrefDescriptor(memref);
    Type llvmI64 = this->typeConverter->convertType(rewriter.getI64Type());
    Type llvm2xI32 = this->typeConverter->convertType(VectorType::get(2, i32));

    Value resource = rewriter.create<LLVM::UndefOp>(loc, llvm4xI32);

    Value ptr = memrefDescriptor.alignedPtr(rewriter, loc);
    Value ptrAsInt = rewriter.create<LLVM::PtrToIntOp>(loc, llvmI64, ptr);
    Value ptrAsInts =
        rewriter.create<LLVM::BitcastOp>(loc, llvm2xI32, ptrAsInt);
    for (int64_t i = 0; i < 2; ++i) {
      Value idxConst = this->createIndexConstant(rewriter, loc, i);
      Value part =
          rewriter.create<LLVM::ExtractElementOp>(loc, ptrAsInts, idxConst);
      resource = rewriter.create<LLVM::InsertElementOp>(
          loc, llvm4xI32, resource, part, idxConst);
    }

    Value numRecords;
    if (memrefType.hasStaticShape()) {
      numRecords = createI32Constant(
          rewriter, loc,
          static_cast<int32_t>(memrefType.getNumElements() * elementByteWidth));
    } else {
      Value maxIndex;
      for (uint32_t i = 0, e = memrefType.getRank(); i < e; ++i) {
        Value size = memrefDescriptor.size(rewriter, loc, i);
        Value stride = memrefDescriptor.stride(rewriter, loc, i);
        stride = rewriter.create<LLVM::MulOp>(loc, stride, byteWidthConst);
        Value maxThisDim = rewriter.create<LLVM::MulOp>(loc, size, stride);
        maxIndex = maxIndex ? rewriter.create<LLVM::MaximumOp>(loc, maxIndex,
                                                               maxThisDim)
                            : maxThisDim;
      }
      numRecords = rewriter.create<LLVM::TruncOp>(loc, llvmI32, maxIndex);
    }
    resource = rewriter.create<LLVM::InsertElementOp>(
        loc, llvm4xI32, resource, numRecords,
        this->createIndexConstant(rewriter, loc, 2));

    // Final word:
    // bits 0-11: dst sel, ignored by these intrinsics
    // bits 12-14: data format (ignored, must be nonzero, 7=float)
    // bits 15-18: data format (ignored, must be nonzero, 4=32bit)
    // bit 19: In nested heap (0 here)
    // bit 20: Behavior on unmap (0 means  "return 0 / ignore")
    // bits 21-22: Index stride for swizzles (N/A)
    // bit 23: Add thread ID (0)
    // bit 24: Reserved to 1 (RDNA) or 0 (CDNA)
    // bits 25-26: Reserved (0)
    // bit 27: Buffer is non-volatile (CDNA only)
    // bits 28-29: Out of bounds select (0 = structured, 1 = check index, 2 =
    //  none, 3 = either swizzles or testing against offset field) RDNA only
    // bits 30-31: Type (must be 0)
    uint32_t word3 = (7 << 12) | (4 << 15);
    if (chipset.majorVersion == 10) {
      word3 |= (1 << 24);
      uint32_t oob = adaptor.boundsCheck() ? 3 : 2;
      word3 |= (oob << 28);
    }
    Value word3Const = createI32Constant(rewriter, loc, word3);
    resource = rewriter.create<LLVM::InsertElementOp>(
        loc, llvm4xI32, resource, word3Const,
        this->createIndexConstant(rewriter, loc, 3));
    args.push_back(resource);

    // Indexing (voffset)
    Value voffset;
    for (auto &pair : llvm::enumerate(adaptor.indices())) {
      size_t i = pair.index();
      Value index = pair.value();
      Value strideOp;
      if (ShapedType::isDynamicStrideOrOffset(strides[i])) {
        strideOp = rewriter.create<LLVM::MulOp>(
            loc, memrefDescriptor.stride(rewriter, loc, i), byteWidthConst);
      } else {
        strideOp =
            createI32Constant(rewriter, loc, strides[i] * elementByteWidth);
      }
      index = rewriter.create<LLVM::MulOp>(loc, index, strideOp);
      voffset =
          voffset ? rewriter.create<LLVM::AddOp>(loc, voffset, index) : index;
    }
    if (adaptor.indexOffset().hasValue()) {
      int32_t indexOffset = *gpuOp.indexOffset() * elementByteWidth;
      Value extraOffsetConst = createI32Constant(rewriter, loc, indexOffset);
      voffset =
          voffset ? rewriter.create<LLVM::AddOp>(loc, voffset, extraOffsetConst)
                  : extraOffsetConst;
    }
    args.push_back(voffset);

    Value sgprOffset = adaptor.sgprOffset();
    if (!sgprOffset)
      sgprOffset = createI32Constant(rewriter, loc, 0);
    if (ShapedType::isDynamicStrideOrOffset(offset))
      sgprOffset = rewriter.create<LLVM::AddOp>(
          loc, memrefDescriptor.offset(rewriter, loc), sgprOffset);
    else if (offset > 0)
      sgprOffset = rewriter.create<LLVM::AddOp>(
          loc, sgprOffset, createI32Constant(rewriter, loc, offset));
    args.push_back(sgprOffset);

    // bit 0: GLC = 0 (atomics drop value, less coherency)
    // bits 1-2: SLC, DLC = 0 (similarly)
    // bit 3: swizzled (0 for raw)
    args.push_back(createI32Constant(rewriter, loc, 0));

    llvm::SmallVector<Type, 1> resultTypes(gpuOp->getNumResults(),
                                           llvmBufferValType);
    Operation *lowered = rewriter.create<Intrinsic>(loc, resultTypes, args,
                                                    ArrayRef<NamedAttribute>());
    if (lowered->getNumResults() == 1) {
      Value replacement = lowered->getResult(0);
      if (llvmBufferValType != llvmWantedDataType) {
        replacement = rewriter.create<LLVM::BitcastOp>(loc, llvmWantedDataType,
                                                       replacement);
      }
      rewriter.replaceOp(gpuOp, replacement);
    } else {
      rewriter.eraseOp(gpuOp);
    }
    return success();
  }
};

struct LDSBarrierOpLowering : public ConvertOpToLLVMPattern<LDSBarrierOp> {
  using ConvertOpToLLVMPattern<LDSBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LDSBarrierOp op, LDSBarrierOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto asmDialectAttr = LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                                    LLVM::AsmDialect::AD_ATT);
    const char *asmStr = "s_waitcnt lgkmcnt(0)\ns_barrier";
    const char *constraints = "";
    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op,
        /*resultTypes=*/TypeRange(), /*operands=*/ValueRange(),
        /*asm_string=*/asmStr, constraints, /*has_side_effects=*/true,
        /*is_align_stack=*/false, /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/ArrayAttr());
    return success();
  }
};

} // end anonymous namespace

/// If `input` is a vector of bytes, concatentate those bytes in little-endian
/// order to form a single integer of size 8 * [vector length]. This works
/// around a wart in the AMDGPU intrinsics where operations that logically take
/// vectors of bytes instead integers. Since we do not want to expose this
/// implementation detail to MLIR, we correct for it here.
static Value mfmaConcatIfNeeded(ConversionPatternRewriter &rewriter,
                                Location loc, Value input) {
  Type inputType = input.getType();
  if (auto vectorType = inputType.dyn_cast<VectorType>()) {
    if (vectorType.getElementType() != rewriter.getI8Type())
      return input;
    int64_t numBytes = vectorType.getNumElements();
    Type destType = rewriter.getIntegerType(numBytes * 8);
    Value result = rewriter.createOrFold<LLVM::ConstantOp>(
        loc, destType, rewriter.getIntegerAttr(destType, 0));
    for (int64_t i = 0; i < numBytes; ++i) {
      Value idxConst = createI32Constant(rewriter, loc, i);
      Value element =
          rewriter.create<LLVM::ExtractElementOp>(loc, input, idxConst);
      Value extended = rewriter.create<LLVM::ZExtOp>(loc, destType, element);
      Value shiftConst = rewriter.createOrFold<LLVM::ConstantOp>(
          loc, destType, rewriter.getIntegerAttr(destType, i * 8));
      Value shifted = rewriter.create<LLVM::ShlOp>(loc, extended, shiftConst);
      result = rewriter.create<LLVM::OrOp>(loc, result, shifted);
    }
    return result;
  }
  return input;
}

/// Return the `rocdl` intrinsic corresponding to a `MFMAInstr` value.
/// This conversion happens here to allow code up the stack to handle the choice
/// of mfma by picking between enum variants, which is much more ergonomic than
/// picking between ops, at the cost of some long switch statements in this
/// pass.
static StringRef mfmaInstrToIntrinsicName(MFMAInstr instr) {
#define LOWERING_CASE(type)                                                    \
  case MFMAInstr::type:                                                        \
    return ROCDL::mfma_##type::getOperationName();
  switch (instr) {
    LOWERING_CASE(f32_32x32x1f32)
    LOWERING_CASE(f32_16x16x1f32)
    LOWERING_CASE(f32_4x4x1f32)
    LOWERING_CASE(f32_32x32x2f32)
    LOWERING_CASE(f32_16x16x4f32)
    LOWERING_CASE(f32_32x32x4f16)
    LOWERING_CASE(f32_16x16x4f16)
    LOWERING_CASE(f32_4x4x4f16)
    LOWERING_CASE(f32_32x32x8f16)
    LOWERING_CASE(f32_16x16x16f16)
    LOWERING_CASE(i32_32x32x4i8)
    LOWERING_CASE(i32_16x16x4i8)
    LOWERING_CASE(i32_4x4x4i8)
    LOWERING_CASE(i32_32x32x8i8)
    LOWERING_CASE(i32_16x16x16i8)
    LOWERING_CASE(f32_32x32x2bf16)
    LOWERING_CASE(f32_16x16x2bf16)
    LOWERING_CASE(f32_4x4x2bf16)
    LOWERING_CASE(f32_32x32x4bf16)
    LOWERING_CASE(f32_16x16x8bf16)
    LOWERING_CASE(f32_32x32x4bf16_1k)
    LOWERING_CASE(f32_16x16x4bf16_1k)
    LOWERING_CASE(f32_4x4x4bf16_1k)
    LOWERING_CASE(f32_32x32x8bf16_1k)
    LOWERING_CASE(f32_16x16x16bf16_1k)
    LOWERING_CASE(f64_16x16x4f64)
    LOWERING_CASE(f64_4x4x4f64)
    LOWERING_CASE(i32_16x16x32_i8)
    LOWERING_CASE(i32_32x32x16_i8)
    LOWERING_CASE(f32_16x16x8_xf32)
    LOWERING_CASE(f32_32x32x4_xf32)
  }
#undef LOWERING_CASE
}

namespace {
struct MFMAOpLowering : public ConvertOpToLLVMPattern<MFMAOp> {
  MFMAOpLowering(LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<MFMAOp>(converter), chipset(chipset) {}

  Chipset chipset;

  LogicalResult
  matchAndRewrite(MFMAOp op, MFMAOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outType = typeConverter->convertType(op.destD().getType());

    if (chipset.majorVersion != 9 || chipset.minorVersion < 0x08)
      return op->emitOpError("MFMA only supported on gfx908+");
    OperationState loweredOp(loc, mfmaInstrToIntrinsicName(op.instr()));
    loweredOp.addTypes(outType);
    loweredOp.addOperands({mfmaConcatIfNeeded(rewriter, loc, adaptor.sourceA()),
                           mfmaConcatIfNeeded(rewriter, loc, adaptor.sourceB()),
                           adaptor.destC(),
                           createI32Constant(rewriter, loc, op.cbsz()),
                           createI32Constant(rewriter, loc, op.abid()),
                           createI32Constant(rewriter, loc, op.blgp())});
    Operation *lowered = rewriter.create(loweredOp);
    rewriter.replaceOp(op, lowered->getResults());
    return success();
  }
};

struct ConvertAMDGPUToROCDLPass
    : public ConvertAMDGPUToROCDLBase<ConvertAMDGPUToROCDLPass> {
  ConvertAMDGPUToROCDLPass() = default;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    FailureOr<Chipset> maybeChipset = Chipset::parse(chipset);
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(ctx), "Invalid chipset name: " + chipset);
      return signalPassFailure();
    }

    RewritePatternSet patterns(ctx);
    LLVMTypeConverter converter(ctx);
    populateAMDGPUToROCDLConversionPatterns(converter, patterns, *maybeChipset);
    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<::mlir::amdgpu::AMDGPUDialect>();
    target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<::mlir::ROCDL::ROCDLDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace

void mlir::populateAMDGPUToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                                   RewritePatternSet &patterns,
                                                   Chipset chipset) {
  patterns.add<LDSBarrierOpLowering>(converter);
  patterns.add<
      RawBufferOpLowering<RawBufferLoadOp, ROCDL::RawBufferLoadOp>,
      RawBufferOpLowering<RawBufferStoreOp, ROCDL::RawBufferStoreOp>,
      RawBufferOpLowering<RawBufferAtomicFaddOp, ROCDL::RawBufferAtomicFAddOp>,
      MFMAOpLowering>(converter, chipset);
}

std::unique_ptr<Pass> mlir::createConvertAMDGPUToROCDLPass() {
  return std::make_unique<ConvertAMDGPUToROCDLPass>();
}
