//===- RockWaveReduce.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers the `rock.wave_reduce` operation into architecture-specific
// intrinsics such as DPP or other wavefront-level instructions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKWAVEREDUCELOWERINGPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-wave-reduce-lowering"

using namespace mlir;
using namespace mlir::rock;
using namespace mlir::amdgpu;

namespace {

struct RockWaveReduceLoweringPass
    : public rock::impl::RockWaveReduceLoweringPassBase<
          RockWaveReduceLoweringPass> {
  using RockWaveReduceLoweringPassBase::RockWaveReduceLoweringPassBase;
  void runOnOperation() override;
};

} // namespace

struct RockWaveReduceRewritePattern : public OpRewritePattern<WaveReductionOp> {
  using OpRewritePattern<WaveReductionOp>::OpRewritePattern;
  RockWaveReduceRewritePattern(MLIRContext *context, Chipset chipset)
      : OpRewritePattern<WaveReductionOp>(context), chipset(chipset) {}
  Chipset chipset;

  Value createReducingOp(WaveReductionOp op, Value input, Value acc,
                         OpBuilder &builder) const {

    ReduceMethod rMethod = op.getReduceMethod();
    Location loc = op.getLoc();
    auto vecType = dyn_cast<VectorType>(op.getInput().getType());
    assert(vecType && "Expected input to be a vector type");
    Type elementType = vecType.getElementType();
    if (rMethod == ReduceMethod::Sum) {
      Value reduced;
      if (elementType.isIntOrIndex()) {
        reduced = builder.create<arith::AddIOp>(loc, acc, input);
      } else {
        reduced = builder.create<arith::AddFOp>(loc, acc, input);
      }
      return reduced;
    } else {
      assert(rMethod == ReduceMethod::Max);
      Value reduced;
      if (elementType.isIntOrIndex()) {
        reduced = builder.create<arith::MaxSIOp>(loc, acc, input);
      } else {
        reduced = builder.create<arith::MaximumFOp>(loc, acc, input);
      }
      return reduced;
    }
  }

  LogicalResult matchAndRewrite(WaveReductionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value defaultVal = op.getInit();
    auto vecType = dyn_cast<VectorType>(input.getType());
    auto vecLen = vecType.getNumElements();
    Value ReducedAll;

    // Perform intra-wavefront reduction using DPP row_shr shifts.
    // Reduce within the row using shifts {1, 2, 3, 4, 8}.
    // First shifts (1–3) use the input value, others use accumulated
    // result. The result is combined using `createReducingOp`.
    // Bank mask (0xF) enables all four banks (each 4 lanes) in the row.
    // Row mask (0xF) enables all 4 rows in the wavefront (each row has 16
    // lanes).

    for (int64_t i = 0; i < vecLen; ++i) {
      Value scalarVal = rewriter.create<vector::ExtractElementOp>(
          loc, input, rewriter.create<arith::ConstantIndexOp>(loc, i));
      Value setInactiveScalar = rewriter.create<ROCDL::SetInactiveOp>(
          loc, vecType.getElementType(), scalarVal, defaultVal);
      std::array<int, 5> row_shifts = {1, 2, 3, 4, 8};
      Value dppResult = setInactiveScalar;
      Value BrodcastAll;

      for (int shift : row_shifts) {
        Value input = (shift <= 3) ? setInactiveScalar : dppResult;
        auto dppOp = rewriter.create<amdgpu::DPPOp>(
            loc, vecType.getElementType(), input, input,
            amdgpu::DPPPermAttr::get(rewriter.getContext(),
                                     amdgpu::DPPPerm::row_shr),
            rewriter.getI32IntegerAttr(shift), rewriter.getI32IntegerAttr(0xF),
            rewriter.getI32IntegerAttr(0xF), rewriter.getBoolAttr(false));

        dppResult = createReducingOp(op, dppResult, dppOp, rewriter);
      }

      // Broadcast the reduced value across the entire wavefront.
      // Chipset version determines the broadcast method used.
      if (chipset.majorVersion == 9) {
        auto makeDPP = [&](amdgpu::DPPPerm perm, int rowMask, int bankMask) {
          auto dppOp = rewriter.create<amdgpu::DPPOp>(
              loc, vecType.getElementType(), dppResult, dppResult,
              amdgpu::DPPPermAttr::get(rewriter.getContext(), perm), nullptr,
              rewriter.getI32IntegerAttr(rowMask),
              rewriter.getI32IntegerAttr(bankMask),
              rewriter.getBoolAttr(false));
          dppResult = createReducingOp(op, dppResult, dppOp, rewriter);
          return dppOp;
        };
        makeDPP(amdgpu::DPPPerm::row_bcast_15, 0xA, 0xF);
        makeDPP(amdgpu::DPPPerm::row_bcast_31, 0xC, 0xF);
        BrodcastAll = makeDPP(amdgpu::DPPPerm::wave_ror, 0xF, 0xF);

      } else if (chipset.majorVersion >= 10) {
        Value src1Value = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI32Type(),
            rewriter.getIntegerAttr(rewriter.getI32Type(), -1));
        Value src2Value = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI32Type(),
            rewriter.getIntegerAttr(rewriter.getI32Type(), -1));

        BrodcastAll = rewriter.create<ROCDL::PermlaneX16Op>(
            loc, vecType.getElementType(), dppResult, dppResult, src1Value,
            src2Value, rewriter.getBoolAttr(false),
            rewriter.getBoolAttr(false));

      } else if (chipset.majorVersion == 0) {
        return failure();
      }

      // Final result is reduced into lane 0 using WWM, then broadcast to all
      // lanes.
      ReducedAll = rewriter.create<ROCDL::StrictWWMOp>(
          loc, vecType.getElementType(), BrodcastAll);
      ReducedAll = rewriter.create<ROCDL::ReadlaneOp>(
          loc, vecType.getElementType(), ReducedAll,
          rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 32));
    }
    rewriter.replaceOp(op, ReducedAll);
    return success();
  }
};

void RockWaveReduceLoweringPass::runOnOperation() {
  FailureOr<Chipset> maybeChipset = Chipset::parse(chipset);
  if (failed(maybeChipset)) {
    emitError(UnknownLoc::get(&getContext()),
              "Invalid chipset name: " + chipset);
    return signalPassFailure();
  }
  RewritePatternSet patterns(&getContext());
  patterns.add<RockWaveReduceRewritePattern>(&getContext(), *maybeChipset);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}