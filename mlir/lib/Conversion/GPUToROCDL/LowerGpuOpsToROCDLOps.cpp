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

    // Retrieve data_type.
    auto dataType = mfmaOp.sourceA().getType();

    // Retrieve m_per_wave and n_per_wave attribute.
    int64_t MPerWave = 64;
    int64_t NPerWave = 64;
    if (mfmaOp.getAttr("m_per_wave"))
      MPerWave = mfmaOp.getAttr("m_per_wave").cast<IntegerAttr>().getInt();
    if (mfmaOp.getAttr("n_per_wave"))
      NPerWave = mfmaOp.getAttr("n_per_wave").cast<IntegerAttr>().getInt();

    auto i32Zero = rewriter.create<LLVM::ConstantOp>(
      loc, typeConverter.convertType(rewriter.getIntegerType(32)),
      rewriter.getI32IntegerAttr(0));
    auto i32One = rewriter.create<LLVM::ConstantOp>(
      loc, typeConverter.convertType(rewriter.getIntegerType(32)),
      rewriter.getI32IntegerAttr(1));
    auto i32Two = rewriter.create<LLVM::ConstantOp>(
      loc, typeConverter.convertType(rewriter.getIntegerType(32)),
      rewriter.getI32IntegerAttr(2));
    auto i32Four = rewriter.create<LLVM::ConstantOp>(
      loc, typeConverter.convertType(rewriter.getIntegerType(32)),
      rewriter.getI32IntegerAttr(4));

    if (dataType == rewriter.getF32Type()) {
      if (MPerWave == 64 && NPerWave == 64) {
        // Original C++ logic:
        // __device__ void gcnasm_mfma_f32_32x32x1f32<64, 64>(const float& reg_a, const float& reg_b, float32_t* reg_c)
        //  reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 1, 0, 0);
        //  reg_c[1] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[1], 1, 1, 0);

        // reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 1, 0, 0);
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x1f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(), i32One, i32Zero, i32Zero});
        // TBD: figure out how to do reg_c[1].
        // reg_c[1] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[1], 1, 1, 0);
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x1f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(), i32One, i32One, i32Zero});
      } else if (MPerWave == 32 && NPerWave == 64) {
        // Original C++ logic:
        // __device__ void gcnasm_mfma_f32_32x32x1f32<32, 64>(const float& reg_a, const float& reg_b, float32_t* reg_c)
        //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 1, 0, 0);

        //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 1, 0, 0);
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x1f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(), i32One, i32Zero, i32Zero});
      } else if (MPerWave == 64 && NPerWave == 32) {
        // Original C++ logic:
        // __device__ void gcnasm_mfma_f32_32x32x1f32<64, 32>(const float& reg_a, const float& reg_b, float32_t* reg_c)
        //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 0, 0, 1);

        //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x1f32(reg_a, reg_b, reg_c[0], 0, 0, 1);
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x1f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(), i32Zero, i32Zero, i32One});
      } else if (MPerWave == 32 && NPerWave == 32) {
        // Original C++ logic:
        // __device__ void gcnasm_mfma_f32_32x32x2f32(const float& reg_a, const float& reg_b, float16_t* reg_c)
        //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2f32(reg_a, reg_b, reg_c[0], 0, 0, 0);

        //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_32x32x2f32(reg_a, reg_b, reg_c[0], 0, 0, 0);
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_32x32x2f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(), i32Zero, i32Zero, i32Zero});
      } else if (MPerWave == 16 && NPerWave == 16) {
        // Original C++ logic:
        // __device__ void gcnasm_mfma_f32_16x16x4f32(const float& reg_a, const float& reg_b, float4_t* reg_c)
        //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x4f32(reg_a, reg_b, reg_c[0], 0, 0, 0);

        //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x4f32(reg_a, reg_b, reg_c[0], 0, 0, 0);
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x4f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(), i32Zero, i32Zero, i32Zero});
      } else if (MPerWave == 16 && NPerWave == 64) {
        // Original C++ logic:
        // __device__ void gcnasm_mfma_f32_16x16x1f32<16, 64>(const float& reg_a, const float& reg_b, float16_t* reg_c)
        //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x1f32(reg_a, reg_b, reg_c[0], 2, 0, 0);

        //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x1f32(reg_a, reg_b, reg_c[0], 2, 0, 0);
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x1f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(), i32Two, i32Zero, i32Zero});
      } else if (MPerWave == 64 && NPerWave == 16) {
        // Original C++ logic:
        // __device__ void gcnasm_mfma_f32_16x16x1f32<64, 16>(const float& reg_a, const float& reg_b, float16_t* reg_c)
        //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x1f32(reg_a, reg_b, reg_c[0], 0, 0, 4);

        //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x1f32(reg_a, reg_b, reg_c[0], 0, 0, 4);
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_16x16x1f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(), i32Zero, i32Zero, i32Four});
      } else if (MPerWave == 8 && NPerWave == 64) {
        // Original C++ logic:
        // __device__ void gcnasm_mfma_f32_4x4x1f32<4, 64>(const float& reg_a, const float& reg_b, float4_t* reg_c)
        //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a, reg_b, reg_c[0], 4, 0, 0);

        //   reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a, reg_b, reg_c[0], 4, 0, 0);
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_4x4x1f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(), i32Four, i32Zero, i32Zero});
      } else if (MPerWave == 4 && NPerWave == 64) {
        // Original C++ logic:
        // __device__ void gcnasm_mfma_f32_4x4x1f32<8, 64>(const float& reg_a, const float& reg_b, float4_t* reg_c)
        //     reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a, reg_b, reg_c[0], 4, 0, 0);
        //     reg_c[1] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a, reg_b, reg_c[1], 4, 1, 0);

        //     reg_c[0] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a, reg_b, reg_c[0], 4, 0, 0);
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_4x4x1f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(), i32Four, i32Zero, i32Zero});
        // TBD: figure out how to do reg_c[1].
        //     reg_c[1] = llvm_intrin_amdgcn_mfma_f32_4x4x1f32(reg_a, reg_b, reg_c[1], 4, 1, 0);
        rewriter.replaceOpWithNewOp<ROCDL::mfma_f32_4x4x1f32>(
            op, adaptor.destC().getType(),
            ValueRange{adaptor.sourceA(), adaptor.sourceB(), adaptor.destC(), i32Four, i32One, i32Zero});
      }
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
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerGpuOpsToROCDLOpsPass() {
  return std::make_unique<LowerGpuOpsToROCDLOpsPass>();
}
