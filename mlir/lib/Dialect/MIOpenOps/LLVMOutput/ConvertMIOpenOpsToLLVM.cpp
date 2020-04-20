//===- ConvertMIOpenOpsToLLVM.cpp - MLIR MIOpen ops lowering passes ---------------===//
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
// This pass converts miopen ops to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpenOps/ConvertMIOpenOpsToLLVM.h"

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
struct LowerMIOpenOpsToLLVMPass : public FunctionPass<LowerMIOpenOpsToLLVMPass> {
  void runOnFunction() override;
};
} // end anonymous namespace

void LowerMIOpenOpsToLLVMPass::runOnFunction() {
  FuncOp func = getFunction();
  LLVMTypeConverter converter(&getContext());

  func.walk([&](miopen::TransformOp op) {
    op.replaceAllUsesWith(op.input());
    op.erase();
  });

  func.walk([&](miopen::FillOp op) {
    op.erase();
  });

  func.walk([&](miopen::LdsBarrierOp op) {
    op.erase();
  });

  func.walk([&](miopen::SubviewOp op) {
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
    auto sizeBytes = op.sizeBytes().getDefiningOp()->getAttr("value").dyn_cast<IntegerAttr>().getInt();
    auto type = op.output().getType().cast<MemRefType>();

    OpBuilder b(op.getContext());

    if (type.getMemorySpace() == 5) {
      // Create llvm.mlir.alloca for VGPRs.
      b.setInsertionPointToStart(op.getOperation()->getBlock());
      auto ptrType = converter.convertType(type.getElementType())
                         .cast<LLVM::LLVMType>().getPointerTo();

      auto *llvmDialect = b.getContext()->getRegisteredDialect<LLVM::LLVMDialect>();

      auto int64Ty = LLVM::LLVMType::getInt64Ty(llvmDialect);
      auto numElements = b.create<LLVM::ConstantOp>(loc, int64Ty, b.getIntegerAttr(b.getIndexType(), sizeBytes));
      auto allocated = b.create<LLVM::AllocaOp>(loc, ptrType, numElements, 0);
      op.replaceAllUsesWith(allocated.res());
    } else if (type.getMemorySpace() == 3) {
      // Create llvm.mlir.global for LDS.
      b.setInsertionPointToStart(op.getOperation()->getParentOp()->getBlock());
      auto elementType = converter.convertType(type.getElementType()).cast<LLVM::LLVMType>();
      auto arrayType = LLVM::LLVMType::getArrayTy(elementType, sizeBytes);
      StringRef name = "lds_buffer";
      auto globalOp = b.create<LLVM::GlobalOp>(loc, arrayType.cast<LLVM::LLVMType>(),
                                               /*isConstant=*/false, LLVM::Linkage::Internal, name,
                                               /*value=*/Attribute(), 3);
      b.setInsertionPoint(op);
      auto addrOfOp = b.create<LLVM::AddressOfOp>(loc, globalOp);
      op.replaceAllUsesWith(addrOfOp.res());
    }
    op.erase();
  });
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::miopen::createLowerMIOpenOpsToLLVMPass() {
  return std::make_unique<LowerMIOpenOpsToLLVMPass>();
}

static PassRegistration<LowerMIOpenOpsToLLVMPass>
    lowerMIOpenOpsToLLVMPass("miopen-lowering-llvm",
                       "Lower MIOpen ops to LLVM dialect.");
