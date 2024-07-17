//===- IREEMerge.cpp - Merge an IREE binary into rocMLIR ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass searches for a single `gpu.binary` and tries to substitute the
// object of the binary with the object stored in the module attribute
// `iree.export`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

#define LINE_NUM STRINGIFY_HELPER(__LINE__)
#define STRINGIFY_HELPER(x) STRINGIFY(x)
#define STRINGIFY(x) #x

#define CHECK(conditionMacro, msgMacro)                                        \
  [](bool condition, bool debug, StringRef msg) -> bool {                      \
    if (debug && condition)                                                    \
      llvm::errs() << "Error[iree-merge:" LINE_NUM "]: " << msg << "\n";       \
    return condition;                                                          \
  }(conditionMacro, debug, msgMacro)

/// Get the IREE_DEBUG environment variable.
static bool getIREEDebug() {
  if (const char *var = std::getenv("IREE_DEBUG"))
    return std::atoi(var) != 0;
  return false;
}

/// Get the IREE_DISABLE environment variable.
static bool getIREEDisable() {
  if (const char *var = std::getenv("IREE_DISABLE"))
    return std::atoi(var) != 0;
  return false;
}

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKIREEMERGEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

namespace {
struct RockIREEMergePass
    : public rock::impl::RockIREEMergePassBase<RockIREEMergePass> {
public:
  void runOnOperation() override;
};
} // namespace

static gpu::ObjectAttr getObject(Builder &builder, gpu::ObjectAttr rockBin,
                                 gpu::ObjectAttr ireeBin, StringAttr rockName,
                                 StringAttr ireeName, DenseI64ArrayAttr gridDim,
                                 DenseI64ArrayAttr blockDim,
                                 IntegerAttr sharedMem) {
  Attribute target = rockBin.getTarget();
  gpu::CompilationTarget format = rockBin.getFormat();
  StringAttr binary = ireeBin.getObject();
  // Get the roc binary properties.
  DictionaryAttr rockProps = rockBin.getProperties();
  assert(rockProps && "invalid object properties");
  gpu::KernelTableAttr rockKernels = rockBin.getKernels();
  assert(rockKernels && "invalid kernel table");
  gpu::KernelAttr rockKernel;
  // Find the roc kernel that needs substitution.
  if (rockName) {
    rockKernel = rockKernels.lookup(rockName);
  } else {
    assert(rockKernels.size() == 1 &&
           "invalid binary, expected a single kernel");
    auto [name, kernelTmp] = *rockKernels.begin();
    rockKernel = kernelTmp;
  }
  assert(rockKernel && "invalid rock kernel");
  // Build the new kernel attribute.
  NamedAttrList kernelAttrs;
  kernelAttrs.append("rock.shared_buffer_size", sharedMem);
  kernelAttrs.append("lds_size", sharedMem);
  kernelAttrs.append("grid_size", gridDim);
  kernelAttrs.append("block_size", blockDim);
  if (Attribute originalFunc = rockKernel.getAttr("original_func"))
    kernelAttrs.append("original_func", originalFunc);
  // Create the kernel attribute.
  auto ireeKernel = gpu::KernelAttr::get(
      ireeName, rockKernel.getFunctionType(), rockKernel.getArgAttrs(),
      builder.getDictionaryAttr(kernelAttrs));
  // Get any possible module attributes.
  DictionaryAttr moduleAttrs;
  if (auto kernelModProps = rockProps.get(rockName)) {
    moduleAttrs =
        builder.getDictionaryAttr({NamedAttribute(ireeName, kernelModProps)});
  }
  // Create the new GPU object.
  return builder.getAttr<gpu::ObjectAttr>(
      target, format, binary, moduleAttrs,
      gpu::KernelTableAttr::get(
          builder.getDictionaryAttr({NamedAttribute(ireeName, ireeKernel)})));
}

void RockIREEMergePass::runOnOperation() {
  // Don't run if `IREE_DISABLE` was set.
  if (getIREEDisable())
    return;
  ModuleOp op = getOperation();
  // Get whether to print debug messages. This variable is required by the CHECK
  // macro.
  bool debug = getIREEDebug();
  // Search for the `iree.export` attribute.
  auto ireeObject = op->getAttrOfType<gpu::ObjectAttr>("iree.export");
  // Return if `iree.export` is not present or is not a `#gpu.object`, this is
  // not a fatal error.
  if (CHECK(!ireeObject, "failed to find the `iree.export` attribute"))
    return;
  // Collect all the binaries in the module.
  SmallVector<gpu::BinaryOp> binaries;
  llvm::append_range(binaries, op.getBody()->getOps<gpu::BinaryOp>());
  // Return if there are no binaries in the module, this is not a fatal error.
  if (CHECK(binaries.size() > 1, "expected a single binary") ||
      CHECK(binaries.empty(), "expected at least one binary"))
    return;
  // Get the binary to replace.
  gpu::BinaryOp binary = binaries[0];
  // Get all the LaunchFunc ops to update.
  SmallVector<gpu::LaunchFuncOp> launchFuncOps;
  StringAttr rockName{};
  if (op.walk([&](gpu::LaunchFuncOp op) -> WalkResult {
          // Abort if there's more than one binary.
          if (op.getKernelModuleName() != binary.getSymNameAttr())
            return WalkResult::interrupt();
          if (rockName == nullptr)
            rockName = op.getKernelName();
          // Abort if there's more than one kernel.
          if (rockName != op.getKernelName())
            return WalkResult::interrupt();
          launchFuncOps.push_back(op);
          return WalkResult::advance();
        }).wasInterrupted()) {
    CHECK(true, "expected at most one kernel");
    return;
  }
  // Get the IREE object properties.
  DictionaryAttr ireeProps = ireeObject.getProperties();
  assert(ireeProps && "expected a property dictionary");
  // Get IREE's parameters.
  auto ireeMem = ireeProps.getAs<IntegerAttr>("workgroup_memory");
  assert(ireeMem && "invalid `workgroup_memory` field");
  auto ireeGrid = ireeProps.getAs<DenseI64ArrayAttr>("workgroup_count");
  assert(ireeGrid && ireeGrid.size() == 3 && "invalid `workgroup_count` field");
  auto ireeBlock = ireeProps.getAs<DenseI64ArrayAttr>("workgroup_sizes");
  assert(ireeBlock && ireeBlock.size() == 3 &&
         "invalid `workgroup_sizes` field");
  auto ireeName = ireeProps.getAs<StringAttr>("kernel");
  assert(ireeName && "invalid `kernel` field");
  // Update ops:
  OpBuilder builder(op.getContext());
  // Get the new symbol ireeName.
  SymbolRefAttr binRef = SymbolRefAttr::get(binary.getSymNameAttr(),
                                            {FlatSymbolRefAttr::get(ireeName)});
  // Update all the launch ops.
  for (auto op : launchFuncOps) {
    builder.setInsertionPoint(op);
    bool isIndex = op.getGridSizeX().getType().isIndex();
    auto getC = [&](IntegerAttr attr) -> Value {
      if (isIndex)
        return builder.create<arith::ConstantOp>(op.getLoc(), attr);
      return builder.create<LLVM::ConstantOp>(op.getLoc(), attr);
    };
    op.getGridSizeXMutable().assign(getC(builder.getIndexAttr(ireeGrid[0])));
    op.getGridSizeYMutable().assign(getC(builder.getIndexAttr(ireeGrid[1])));
    op.getGridSizeZMutable().assign(getC(builder.getIndexAttr(ireeGrid[2])));
    op.getBlockSizeXMutable().assign(getC(builder.getIndexAttr(ireeBlock[0])));
    op.getBlockSizeYMutable().assign(getC(builder.getIndexAttr(ireeBlock[1])));
    op.getBlockSizeZMutable().assign(getC(builder.getIndexAttr(ireeBlock[2])));
    op.getDynamicSharedMemorySizeMutable().assign(
        getC(builder.getI32IntegerAttr(ireeMem.getValue().getSExtValue())));
    op.setKernelAttr(binRef);
  }
  // Update the binary.
  binary.setObjectsAttr(builder.getArrayAttr({getObject(
      builder, cast<gpu::ObjectAttr>(binary.getObjects()[0]), ireeObject,
      rockName, ireeName, ireeGrid, ireeBlock, ireeMem)}));
  op->removeAttr("iree.export");
  if (debug)
    llvm::errs() << "[iree-merge]: success\n";
}
