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

    LLVMTypeConverter converter(m.getContext());

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

struct BFOpLowering : ConvertToLLVMPattern {
  explicit BFOpLowering(MLIRContext *context,
                          LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(gpu::BFConvertOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bfOp = cast<gpu::BFConvertOp>(op);
    auto adaptor = gpu::BFConvertOpOperandAdaptor(operands);
    auto loc = bfOp.getLoc();

    Type castedI32Type = rewriter.getIntegerType(32);
    Type castedI16Type = rewriter.getIntegerType(16);
    Type llvmI32Type = typeConverter.convertType(castedI32Type);
    Type llvmI16Type = typeConverter.convertType(castedI16Type);

    auto bitcastop = rewriter.create<LLVM::BitcastOp>(loc, llvmI32Type, adaptor.in());
    auto constantSixteen = rewriter.create<LLVM::ConstantOp>(loc,
                           llvmI32Type,rewriter.getIntegerAttr(castedI32Type, 16));
    auto ShiftValue = rewriter.create<LLVM::LShrOp>(loc, llvmI32Type, bitcastop, constantSixteen);

    auto constantOne = rewriter.create<LLVM::ConstantOp>(loc,
                           llvmI32Type,rewriter.getIntegerAttr(castedI32Type, 1));
    auto andValue = rewriter.create<LLVM::AndOp>(loc, ShiftValue, constantOne);

    auto constantBig = rewriter.create<LLVM::ConstantOp>(loc,
                           llvmI32Type,rewriter.getIntegerAttr(castedI32Type, 32767));
    auto addBigValue = rewriter.create<LLVM::AddOp>(loc, bitcastop, constantBig);
    auto addValue = rewriter.create<LLVM::AddOp>(loc, andValue, addBigValue);

    auto ShiftBeforeTruncValue = rewriter.create<LLVM::LShrOp>(loc, llvmI32Type, addValue, constantSixteen);
    auto truncValue = rewriter.create<LLVM::TruncOp>(loc,  llvmI16Type  , ShiftBeforeTruncValue);
    rewriter.replaceOp(op, {truncValue});
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
  patterns.insert<BFOpLowering>(converter.getDialect()->getContext(),
                                  converter);
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerGpuOpsToROCDLOpsPass() {
  return std::make_unique<LowerGpuOpsToROCDLOpsPass>();
}
