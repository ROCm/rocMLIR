//===- LowerRockOpsToROCDL - MLIR Rock ops lowering passes -----===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// This pass lowers specific rock ops to ROCDL ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_CONVERTROCKOPSTOROCDLOPS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-to-rocdl"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct AsyncWaitOpConversion
    : public ConvertOpToLLVMPattern<rock::AsyncWaitOp> {
  using ConvertOpToLLVMPattern<rock::AsyncWaitOp>::ConvertOpToLLVMPattern;

  AsyncWaitOpConversion(const LLVMTypeConverter &converter,
                        amdgpu::Chipset chipset, bool supportsDirectToLDS)
      : ConvertOpToLLVMPattern<rock::AsyncWaitOp>(converter), chipset(chipset),
        supportsDirectToLDS(supportsDirectToLDS) {
  }

  mlir::amdgpu::Chipset chipset;
  bool supportsDirectToLDS;

  LogicalResult
  matchAndRewrite(rock::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Erase the op when DirectToLDS is not supported
    if (!supportsDirectToLDS) {
      rewriter.eraseOp(op);
      return success();
    }

    auto loc = op->getLoc();

    // Clamp vmcnt to 6bits; a lower vmcnt will produce a conservative wait
    unsigned vmCnt = std::min(63u, op.getNumInst());

    // Extract low and high bits and combine while setting all other bits to 1
    unsigned lowBits = vmCnt & 0xF;
    unsigned highBits = vmCnt >> 4 << 14;
    unsigned otherCnts = ~0xC00F; // C00F has bits 15:14 and 3:0 set
    unsigned waitValue = lowBits | highBits | otherCnts;

    // NOTE: The fence op should not be neccesary, but without the
    // alias analysis attributes, the backend will generate incorrect waitcnts.
    // So we must use the fence op to ensure correct waitcnts.
    Attribute mmra =
        rewriter.getAttr<LLVM::MMRATagAttr>("amdgpu-synchronize-as", "local");
    StringRef scope = "workgroup";
    auto relFence = LLVM::FenceOp::create(rewriter, loc,
                                          LLVM::AtomicOrdering::release, scope);
    relFence->setDiscardableAttr(LLVM::LLVMDialect::getMmraAttrName(), mmra);

    ROCDL::SWaitcntOp::create(rewriter, loc, waitValue);
    ROCDL::SBarrierOp::create(rewriter, loc);

    auto acqFence = LLVM::FenceOp::create(rewriter, loc,
                                          LLVM::AtomicOrdering::acquire, scope);
    acqFence->setDiscardableAttr(LLVM::LLVMDialect::getMmraAttrName(), mmra);
    rewriter.replaceOp(op, acqFence);

    return success();
  }
};

struct LowerRockOpsToROCDLOpsPass final
    : public rock::impl::ConvertRockOpsToROCDLOpsBase<
          LowerRockOpsToROCDLOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    LLVM::LLVMFuncOp op = getOperation();
    MLIRContext *ctx = op.getContext();

    LLVMTypeConverter converter(ctx);
    RewritePatternSet patterns(ctx);

    FailureOr<amdgpu::Chipset> maybeChipset = amdgpu::Chipset::parse(chipset);
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(ctx), "Invalid chipset name: " + chipset);
      return signalPassFailure();
    }

    // Check if the GPU supports DirectToLDS
    AmdArchInfo archInfo = lookupArchInfo(chipset);
    bool supportsDirectToLDS = isDirectToLDSSupported(archInfo.defaultFeatures);

    LLVMConversionTarget target(getContext());
    target.addIllegalOp<rock::AsyncWaitOp>();
    patterns.add<AsyncWaitOpConversion>(converter, *maybeChipset,
                                        supportsDirectToLDS);
    target.addLegalDialect<ROCDL::ROCDLDialect>();

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace
