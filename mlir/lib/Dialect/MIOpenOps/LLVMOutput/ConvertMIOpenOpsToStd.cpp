//===- ConvertMIOpenOpsToStd.cpp - MLIR MIOpen ops lowering passes ---------------===//
//
// Copyright 2020 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This pass converts miopen ops to std dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpenOps/ConvertMIOpenOpsToStd.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MIOpenOps/MIOpenOps.h"
#include "mlir/Dialect/MIOpenOps/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {
struct LowerMIOpenOpsToStdPass : public ModulePass<LowerMIOpenOpsToStdPass> {
  void runOnModule() override;
};
} // end anonymous namespace

void LowerMIOpenOpsToStdPass::runOnModule() {
  auto m = getModule();

  for (auto func : m.getOps<FuncOp>()) {
    LLVMTypeConverter converter(&getContext());

    func.walk([&](miopen::TransformOp op) {
      op.replaceAllUsesWith(op.input());
      op.erase();
    });

    func.walk([&](miopen::ThreadwiseGemmOp op) {
      op.erase();
    });

    func.walk([&](miopen::ThreadwiseCopyOp op) {
      op.erase();
    });

    func.walk([&](miopen::GpuAllocOp op) {
      auto loc = op.getLoc();
      auto type = op.output().getType().cast<MemRefType>();

      OpBuilder b(op.getContext());
      b.setInsertionPoint(op);
      if (type.getMemorySpace() == 5) {
        // TBD. rebase with latest MLIR and switch to std.alloca.
        auto allocated = b.create<AllocOp>(loc, type);
        op.replaceAllUsesWith(allocated.getResult());
      } else if (type.getMemorySpace() == 3) {
        auto allocated = b.create<AllocOp>(loc, type);
        op.replaceAllUsesWith(allocated.getResult());
      }
      op.erase();
    });

    func.walk([&](miopen::SubviewOp op) {
      auto loc = op.getLoc();
      auto outputType = op.output().getType().cast<MemRefType>();
      auto outputShape = outputType.getShape();
      auto inputType = op.input().getType().cast<MemRefType>();
      auto inputShape = inputType.getShape();

      OpBuilder b(op.getContext());
      b.setInsertionPoint(op);

      if (outputShape.size() == 2) {
        auto viewOp = b.create<ViewOp>(loc, op.output().getType(), op.input(), ArrayRef<Value>{});
        op.replaceAllUsesWith(viewOp.getResult());
        op.erase();
      } else {
        auto subviewOp = b.create<SubViewOp>(loc, op.output().getType(), op.input());
        op.replaceAllUsesWith(subviewOp.getResult());
        op.erase();
      }
    });

    func.walk([&](miopen::LdsBarrierOp op) {
      OpBuilder b(op.getContext());
      auto loc = op.getLoc();
      if (!getModule().lookupSymbol<FuncOp>("lds_barrier")) {
        auto funcType = b.getFunctionType({}, {});

        StringRef funcName = "lds_barrier";
        b.setInsertionPoint(getModule().getBody(), getModule().getBody()->begin());
        auto func = b.create<FuncOp>(loc, funcName, funcType, ArrayRef<NamedAttribute>{});
      }
      auto barrierFunc = getModule().lookupSymbol<FuncOp>("lds_barrier");
      b.setInsertionPoint(op);
      b.create<CallOp>(loc, ArrayRef<Type>{},
                       b.getSymbolRefAttr(barrierFunc),
                       ArrayRef<Value>{});
      op.erase();
    });
  }
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::miopen::createLowerMIOpenOpsToStdPass() {
  return std::make_unique<LowerMIOpenOpsToStdPass>();
}

static PassRegistration<LowerMIOpenOpsToStdPass>
    lowerMIOpenOpsToStdPass("miopen-lowering-step4",
                       "Lower MIOpen ops to std dialect.");
